"""
Tests for idle timeout in StreamingDetector.

Bug: StreamingDetector.run() polls Kafka indefinitely even when no more
messages will arrive (FlowConsumer has finished). This causes experiments
to hang after all flows are processed.

Fix: Count consecutive empty polls. After IDLE_LIMIT (10) consecutive
empty polls (~10s), set _running = False and exit gracefully.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.detector.streaming_detector import StreamingDetector, StreamingDetectorConfig, IDLE_LIMIT


def make_detector():
    """Create a StreamingDetector without connecting to Kafka."""
    config = StreamingDetectorConfig()
    detector = StreamingDetector(config)
    return detector


class TestIdleTimeout:
    def test_detector_stops_after_idle_limit_empty_polls(self):
        """After IDLE_LIMIT consecutive empty polls, run() must exit with _running=False."""
        detector = make_detector()

        mock_consumer = MagicMock()
        mock_consumer.poll.return_value = {}  # always empty

        with patch.object(detector, 'connect'):
            detector._consumer = mock_consumer
            result = detector.run(max_flows=None)

        assert mock_consumer.poll.call_count == IDLE_LIMIT
        assert isinstance(result, dict)
        assert result["flows_processed"] == 0

    def test_idle_counter_resets_on_message(self):
        """Receiving a message before idle limit resets counter — detector stays alive."""
        detector = make_detector()

        # Build a fake flow message
        from kafka import TopicPartition
        tp = TopicPartition("flows", 0)
        mock_msg = MagicMock()
        mock_msg.value = {
            "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2",
            "src_port": 1234, "dst_port": 80, "protocol": "TCP",
            "packet_count": 5, "flow_duration": 1.0,
            "total_bytes": 500, "fwd_packet_count": 3, "bwd_packet_count": 2,
            "fwd_bytes": 300, "bwd_bytes": 200,
            "packets_per_second": 5.0, "bytes_per_second": 500.0,
            "packet_size_mean": 100.0, "packet_size_std": 0.0,
            "packet_size_min": 100.0, "packet_size_max": 100.0,
            "fwd_packet_size_mean": 100.0, "fwd_packet_size_std": 0.0,
            "bwd_packet_size_mean": 100.0, "bwd_packet_size_std": 0.0,
            "iat_mean": 0.2, "iat_std": 0.0, "iat_min": 0.2, "iat_max": 0.2,
            "syn_count": 1, "ack_count": 4, "fin_count": 1,
            "rst_count": 0, "psh_count": 2, "urg_count": 0,
            "fwd_bwd_ratio": 1.5, "first_packet_time": 1698800000.0,
            "last_packet_time": 1698800001.0,
        }

        mock_consumer = MagicMock()
        # Pattern: 5 empty polls, then 1 message, then IDLE_LIMIT empty polls to terminate
        # Total polls = 5 + 1 + IDLE_LIMIT
        side_effects = [{}] * 5 + [{tp: [mock_msg]}] + [{}] * IDLE_LIMIT
        mock_consumer.poll.side_effect = side_effects

        with patch.object(detector, 'connect'):
            detector._consumer = mock_consumer
            result = detector.run(max_flows=None)

        # The message was received and one flow was processed
        assert result["flows_processed"] == 1
        # Total polls = 5 (empty) + 1 (message) + IDLE_LIMIT (empty again = terminates)
        assert mock_consumer.poll.call_count == 5 + 1 + IDLE_LIMIT

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
