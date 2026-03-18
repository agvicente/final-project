"""
Tests para WindowAggregator.

Verifica:
  - Acumulacao de flows por src_ip
  - Emissao quando janela fecha
  - Resposta ao min_flows_per_window
  - Feature vectors corretos
  - Flush final
"""

import numpy as np
import pytest

from src.detector.window_aggregator import (
    WindowAggregator, WindowRecord, WINDOW_FEATURES, WINDOW_FEATURES_V2,
    _shannon_entropy,
)


def make_flow(src_ip="1.2.3.4", dst_ip="5.6.7.8", dst_port=80,
              timestamp=100.0, packet_count=10, total_bytes=500,
              flow_duration=1.0, packet_size_mean=50.0,
              fwd_bwd_ratio=1.0, iat_mean=0.1, syn_count=1,
              ack_count=1, bwd_packet_count=5):
    """Helper para criar flows de teste."""
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": 12345,
        "dst_port": dst_port,
        "first_packet_time": timestamp,
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "flow_duration": flow_duration,
        "packet_size_mean": packet_size_mean,
        "packet_size_std": 10.0,
        "fwd_bwd_ratio": fwd_bwd_ratio,
        "iat_mean": iat_mean,
        "syn_count": syn_count,
        "ack_count": ack_count,
        "bwd_packet_count": bwd_packet_count,
    }


class TestWindowRecord:
    """Tests para WindowRecord."""

    def test_add_flow(self):
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        record.add_flow(make_flow())
        assert record.flow_count == 1

    def test_to_feature_vector_shape(self):
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(timestamp=100.0 + i))
        vector = record.to_feature_vector()
        assert vector.shape == (len(WINDOW_FEATURES),)
        assert vector.dtype == np.float64

    def test_to_feature_vector_values(self):
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(3):
            record.add_flow(make_flow(
                dst_ip=f"10.0.0.{i}",
                dst_port=80 + i,
                packet_count=10,
                total_bytes=500,
            ))
        vector = record.to_feature_vector()

        # flow_count
        assert vector[0] == 3.0
        # total_packets
        assert vector[1] == 30.0
        # total_bytes
        assert vector[2] == 1500.0
        # unique_dst_ips
        assert vector[3] == 3.0
        # unique_dst_ports
        assert vector[4] == 3.0

    def test_empty_record(self):
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        vector = record.to_feature_vector()
        assert np.all(vector == 0)

    def test_metadata(self):
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        record.add_flow(make_flow())
        meta = record.to_metadata()
        assert meta["src_ip"] == "1.2.3.4"
        assert meta["window_start"] == 100.0
        assert meta["flow_count"] == 1


class TestWindowAggregator:
    """Tests para WindowAggregator."""

    def test_no_emission_within_window(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=3)
        # Todos na mesma janela (100-110)
        for i in range(5):
            result = agg.add_flow(make_flow(timestamp=100.0 + i))
            assert result == []  # Nenhuma emissao — janela nao fechou

    def test_emission_on_window_change(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=3)
        # 5 flows na janela [100, 110)
        for i in range(5):
            agg.add_flow(make_flow(timestamp=100.0 + i))

        # Flow na proxima janela → fecha anterior
        result = agg.add_flow(make_flow(timestamp=110.0))
        assert len(result) == 1
        vector, metadata = result[0]
        assert vector.shape == (len(WINDOW_FEATURES),)
        assert metadata["src_ip"] == "1.2.3.4"

    def test_min_flows_filter(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=5)
        # Apenas 3 flows (abaixo de min_flows=5) → nao emite
        for i in range(3):
            agg.add_flow(make_flow(timestamp=100.0 + i))

        result = agg.add_flow(make_flow(timestamp=110.0))
        assert result == []  # Abaixo do minimo

    def test_multiple_ips_in_window(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=2)
        # IP A: 3 flows, IP B: 2 flows na mesma janela
        for i in range(3):
            agg.add_flow(make_flow(src_ip="1.1.1.1", timestamp=100.0 + i))
        for i in range(2):
            agg.add_flow(make_flow(src_ip="2.2.2.2", timestamp=100.0 + i))

        # Fecha janela
        result = agg.add_flow(make_flow(src_ip="1.1.1.1", timestamp=110.0))
        assert len(result) == 2  # Ambos IPs emitiram (>= min_flows=2)

        ips = {meta["src_ip"] for _, meta in result}
        assert ips == {"1.1.1.1", "2.2.2.2"}

    def test_flush(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=2)
        for i in range(5):
            agg.add_flow(make_flow(timestamp=100.0 + i))

        # Flush final
        result = agg.flush()
        assert len(result) == 1

    def test_statistics(self):
        agg = WindowAggregator(window_size_seconds=10.0, min_flows_per_window=2)
        for i in range(5):
            agg.add_flow(make_flow(timestamp=100.0 + i))
        agg.add_flow(make_flow(timestamp=110.0))

        stats = agg.get_statistics()
        assert stats["total_flows_received"] == 6
        assert stats["total_windows_emitted"] == 1
        assert stats["total_vectors_emitted"] == 1

    def test_syn_ratio(self):
        """SYN ratio should be total_syns / total_packets."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        record.add_flow(make_flow(packet_count=10, syn_count=5))
        record.add_flow(make_flow(packet_count=10, syn_count=5))
        vector = record.to_feature_vector()

        # syn_ratio = 10 / 20 = 0.5
        syn_ratio_idx = WINDOW_FEATURES.index("syn_ratio")
        assert vector[syn_ratio_idx] == pytest.approx(0.5)


class TestWindowFeaturesV2:
    """Tests para window features v2 (behavioral)."""

    def test_v2_feature_vector_shape(self):
        """v2 should produce 19 features."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(timestamp=100.0 + i))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        assert vector.shape == (len(WINDOW_FEATURES_V2),)
        assert vector.shape == (19,)
        assert vector.dtype == np.float64

    def test_v1_backward_compatible(self):
        """v1 should still produce 12 features."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(timestamp=100.0 + i))
        vector = record.to_feature_vector(feature_version="v1")
        assert vector.shape == (len(WINDOW_FEATURES),)
        assert vector.shape == (12,)

    def test_flows_per_second(self):
        """10 flows in 10s window -> 1.0 flows/s."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(10):
            record.add_flow(make_flow(timestamp=100.0 + i))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        fps_idx = WINDOW_FEATURES_V2.index("flows_per_second")
        assert vector[fps_idx] == pytest.approx(1.0)

    def test_dst_port_entropy_uniform(self):
        """5 flows to 5 different ports -> log2(5)."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(dst_port=80 + i))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        ent_idx = WINDOW_FEATURES_V2.index("dst_port_entropy")
        assert vector[ent_idx] == pytest.approx(np.log2(5), rel=1e-6)

    def test_dst_port_entropy_concentrated(self):
        """5 flows all to port 80 -> entropy 0."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(dst_port=80))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        ent_idx = WINDOW_FEATURES_V2.index("dst_port_entropy")
        assert vector[ent_idx] == pytest.approx(0.0)

    def test_dst_ip_entropy(self):
        """3 flows to 3 different IPs -> log2(3)."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(3):
            record.add_flow(make_flow(dst_ip=f"10.0.0.{i}"))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        ent_idx = WINDOW_FEATURES_V2.index("dst_ip_entropy")
        assert vector[ent_idx] == pytest.approx(np.log2(3), rel=1e-6)

    def test_unanswered_ratio(self):
        """Mix of SYN+ACK and SYN-only flows."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        # 3 flows with SYN but no ACK (unanswered)
        for i in range(3):
            record.add_flow(make_flow(syn_count=1, ack_count=0))
        # 2 flows with SYN and ACK (answered)
        for i in range(2):
            record.add_flow(make_flow(syn_count=1, ack_count=1))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        idx = WINDOW_FEATURES_V2.index("unanswered_ratio")
        assert vector[idx] == pytest.approx(3 / 5)

    def test_payload_std_uniform(self):
        """Flows with identical bytes -> std ~= 0."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        for i in range(5):
            record.add_flow(make_flow(total_bytes=500))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        idx = WINDOW_FEATURES_V2.index("payload_std")
        assert vector[idx] == pytest.approx(0.0, abs=1e-10)

    def test_small_flow_ratio(self):
        """Mix of small (<=3 pkts) and large flows."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        # 3 small flows
        for i in range(3):
            record.add_flow(make_flow(packet_count=2))
        # 2 large flows
        for i in range(2):
            record.add_flow(make_flow(packet_count=10))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        idx = WINDOW_FEATURES_V2.index("small_flow_ratio")
        assert vector[idx] == pytest.approx(3 / 5)

    def test_fwd_only_ratio(self):
        """Mix of unidirectional and bidirectional flows."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        # 4 forward-only (bwd_packet_count=0)
        for i in range(4):
            record.add_flow(make_flow(bwd_packet_count=0))
        # 1 bidirectional
        record.add_flow(make_flow(bwd_packet_count=5))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        idx = WINDOW_FEATURES_V2.index("fwd_only_ratio")
        assert vector[idx] == pytest.approx(4 / 5)

    def test_empty_record_v2(self):
        """Empty record in v2 mode should produce zeros with correct shape."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        assert vector.shape == (19,)
        assert np.all(vector == 0)

    def test_aggregator_v2_emission(self):
        """WindowAggregator with v2 should emit 19-feature vectors."""
        agg = WindowAggregator(
            window_size_seconds=10.0, min_flows_per_window=3,
            window_feature_version="v2",
        )
        for i in range(5):
            agg.add_flow(make_flow(timestamp=100.0 + i))
        result = agg.add_flow(make_flow(timestamp=110.0))
        assert len(result) == 1
        vector, _ = result[0]
        assert vector.shape == (19,)

    def test_no_nan_inf_v2(self):
        """v2 features should never produce NaN or Inf."""
        record = WindowRecord(src_ip="1.2.3.4", window_start=100.0)
        # Single flow — edge case
        record.add_flow(make_flow(packet_count=1, total_bytes=0, syn_count=0, ack_count=0))
        vector = record.to_feature_vector(feature_version="v2", window_size_seconds=10.0)
        assert not np.any(np.isnan(vector))
        assert not np.any(np.isinf(vector))
