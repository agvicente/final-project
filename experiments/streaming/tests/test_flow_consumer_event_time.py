"""
Tests for event-time flow timeout in FlowConsumer.

Bug: _check_flow_timeouts() used time.time() (wall clock ~2026) to compare
against PCAP packet timestamps (~2023), causing flows to expire immediately.

Fix: FlowConsumer tracks _pcap_clock = max(packet timestamps seen).
     _check_flow_timeouts() uses _pcap_clock instead of time.time().
"""
import pytest
from unittest.mock import MagicMock, patch
from src.consumer.flow_consumer import FlowConsumer
from src.consumer.config import ConsumerConfig


def make_consumer():
    """Creates a FlowConsumer without connecting to Kafka."""
    config = ConsumerConfig()
    config.flow.flow_timeout_seconds = 60.0
    config.flow.min_packets_per_flow = 1
    consumer = FlowConsumer(config)
    return consumer


def make_packet(src_ip, dst_ip, src_port, dst_port, protocol, timestamp, length=100):
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "timestamp": timestamp,
        "length": length,
        "tcp_flags": 0,
    }


class TestPcapClockInitialization:
    def test_pcap_clock_starts_unset(self):
        consumer = make_consumer()
        assert consumer._pcap_clock is None

    def test_pcap_clock_updated_on_first_packet(self):
        consumer = make_consumer()
        pkt = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=1698800000.0)
        consumer._process_packet(pkt)
        assert consumer._pcap_clock == 1698800000.0

    def test_pcap_clock_advances_to_max(self):
        consumer = make_consumer()
        pkt1 = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=1698800000.0)
        pkt2 = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=1698800050.0)
        consumer._process_packet(pkt1)
        consumer._process_packet(pkt2)
        assert consumer._pcap_clock == 1698800050.0

    def test_pcap_clock_does_not_go_backwards(self):
        consumer = make_consumer()
        pkt1 = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=1698800100.0)
        pkt2 = make_packet("3.3.3.3", "4.4.4.4", 5678, 443, "TCP", timestamp=1698800000.0)
        consumer._process_packet(pkt1)
        consumer._process_packet(pkt2)
        assert consumer._pcap_clock == 1698800100.0  # max, not overwritten by older pkt

    def test_pcap_clock_ignores_zero_timestamp_packet(self):
        """A packet with timestamp=0 should not count as clock initialized."""
        consumer = make_consumer()
        pkt = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=0.0)
        consumer._process_packet(pkt)
        assert consumer._pcap_clock is None  # clock not set by zero-timestamp packet


class TestEventTimeTimeout:
    def test_flow_not_closed_before_timeout(self):
        """Flow with 1 packet should NOT close if pcap_clock hasn't advanced 60s."""
        consumer = make_consumer()
        t0 = 1698800000.0
        pkt = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=t0)
        consumer._process_packet(pkt)

        # Advance pcap_clock by only 30s (less than flow_timeout=60s)
        consumer._pcap_clock = t0 + 30.0
        consumer._check_flow_timeouts()

        assert consumer.flows_completed == 0
        assert len(consumer._active_flows) == 1

    def test_flow_closed_after_timeout(self):
        """Flow should close when pcap_clock advances past last_packet_time + timeout."""
        consumer = make_consumer()
        t0 = 1698800000.0
        pkt = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=t0)
        consumer._producer = MagicMock()  # avoid real Kafka
        consumer._process_packet(pkt)

        # Advance pcap_clock by 61s (past flow_timeout=60s)
        consumer._pcap_clock = t0 + 61.0
        consumer._check_flow_timeouts()

        assert consumer.flows_completed == 1
        assert len(consumer._active_flows) == 0

    def test_old_pcap_timestamps_dont_cause_immediate_expiry(self):
        """Core regression: 2023 PCAP timestamps must NOT expire immediately in 2026."""
        consumer = make_consumer()
        consumer._producer = MagicMock()

        # Simulate 2023 PCAP timestamp
        pcap_ts = 1698800000.0  # Nov 2023

        pkt1 = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=pcap_ts)
        pkt2 = make_packet("1.1.1.1", "2.2.2.2", 1234, 80, "TCP", timestamp=pcap_ts + 1.0)

        consumer._process_packet(pkt1)
        consumer._process_packet(pkt2)

        # _check_flow_timeouts runs — pcap_clock is only 1s ahead, NOT 2026
        consumer._check_flow_timeouts()

        # Flow should NOT be closed yet (only 1s elapsed in PCAP time, timeout=60s)
        assert consumer.flows_completed == 0

    def test_timeout_not_triggered_when_pcap_clock_zero(self):
        """If no packets processed yet, _check_flow_timeouts should be a no-op."""
        consumer = make_consumer()
        # _pcap_clock is None by default (no packets seen)
        # Manually insert a flow to test
        from src.consumer.flow_consumer import FlowData
        key = ("1.1.1.1", "2.2.2.2", 1234, 80, "TCP")
        consumer._active_flows[key] = FlowData(*key)
        consumer._check_flow_timeouts()
        # Should not close anything when pcap_clock is None
        assert len(consumer._active_flows) == 1
