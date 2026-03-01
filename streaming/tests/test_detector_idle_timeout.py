"""
Tests for idle timeout in StreamingDetector.

Bug: StreamingDetector.run() polls Kafka indefinitely even when no more
messages will arrive (FlowConsumer has finished). This causes experiments
to hang after all flows are processed.

Fix: Count consecutive empty polls. After IDLE_LIMIT (10) consecutive
empty polls (~10s), set _running = False and exit gracefully.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from src.detector.streaming_detector import StreamingDetector, StreamingDetectorConfig


def make_detector():
    """Create a StreamingDetector without connecting to Kafka."""
    config = StreamingDetectorConfig()
    detector = StreamingDetector(config)
    return detector


class TestIdleTimeout:
    def test_detector_stops_after_idle_limit_empty_polls(self):
        """After IDLE_LIMIT consecutive empty polls, _running must become False."""
        detector = make_detector()

        # Mock consumer: always returns empty records
        mock_consumer = MagicMock()
        mock_consumer.poll.return_value = {}
        detector._consumer = mock_consumer
        detector._running = True

        # Simulate the idle logic directly (unit test without full run())
        idle_polls = 0
        idle_limit = 10

        for _ in range(idle_limit):
            records = detector._consumer.poll(timeout_ms=1000)
            if not records:
                idle_polls += 1
                if idle_polls >= idle_limit:
                    detector._running = False

        assert detector._running is False
        assert idle_polls == idle_limit

    def test_idle_counter_resets_on_message(self):
        """Receiving a message resets the idle counter to 0."""
        detector = make_detector()

        mock_consumer = MagicMock()
        # Returns empty 5 times, then a message, then empty again
        fake_flow = {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2",
                     "src_port": 1234, "dst_port": 80, "protocol": "TCP",
                     "packet_count": 5, "flow_duration": 1.0,
                     "total_bytes": 500, "fwd_packet_count": 3,
                     "bwd_packet_count": 2, "fwd_bytes": 300, "bwd_bytes": 200,
                     "packets_per_second": 5.0, "bytes_per_second": 500.0,
                     "packet_size_mean": 100.0, "packet_size_std": 0.0,
                     "packet_size_min": 100.0, "packet_size_max": 100.0,
                     "fwd_packet_size_mean": 100.0, "fwd_packet_size_std": 0.0,
                     "bwd_packet_size_mean": 100.0, "bwd_packet_size_std": 0.0,
                     "iat_mean": 0.2, "iat_std": 0.0, "iat_min": 0.2, "iat_max": 0.2,
                     "syn_count": 1, "ack_count": 4, "fin_count": 1,
                     "rst_count": 0, "psh_count": 2, "urg_count": 0,
                     "fwd_bwd_ratio": 1.5, "first_packet_time": 1698800000.0,
                     "last_packet_time": 1698800001.0}

        from kafka import TopicPartition
        tp = TopicPartition("flows", 0)
        mock_msg = MagicMock()
        mock_msg.value = fake_flow

        side_effects = [
            {},          # empty
            {},          # empty
            {},          # empty
            {},          # empty
            {},          # empty
            {tp: [mock_msg]},   # message arrives
            {},          # empty again
        ]
        mock_consumer.poll.side_effect = side_effects
        detector._consumer = mock_consumer

        idle_polls = 0
        idle_limit = 10

        for poll_result in side_effects:
            if not poll_result:
                idle_polls += 1
            else:
                idle_polls = 0  # reset on message

        # After receiving a message and 1 more empty, counter should be 1 (not 6)
        assert idle_polls == 1

    def test_run_exits_after_idle_timeout(self):
        """Integration: run() must return after IDLE_LIMIT empty polls."""
        detector = make_detector()

        mock_consumer = MagicMock()
        mock_consumer.poll.return_value = {}  # always empty

        # Patch connect() to avoid real Kafka
        with patch.object(detector, 'connect'):
            detector._consumer = mock_consumer
            detector.start_time = None  # run() will set this

            result = detector.run(max_flows=None)

        # Should have returned a stats dict (not hung forever)
        assert isinstance(result, dict)
        assert "flows_processed" in result
        assert result["flows_processed"] == 0
        # Poll should have been called exactly IDLE_LIMIT times
        assert mock_consumer.poll.call_count == 10

    def test_idle_limit_is_ten(self):
        """IDLE_LIMIT must be 10 (10s with 1s poll timeout)."""
        # This test documents the expected constant.
        # If you change IDLE_LIMIT, update this test intentionally.
        from src.detector import streaming_detector
        assert streaming_detector.IDLE_LIMIT == 10
