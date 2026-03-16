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

from src.detector.window_aggregator import WindowAggregator, WindowRecord, WINDOW_FEATURES


def make_flow(src_ip="1.2.3.4", dst_ip="5.6.7.8", dst_port=80,
              timestamp=100.0, packet_count=10, total_bytes=500,
              flow_duration=1.0, packet_size_mean=50.0,
              fwd_bwd_ratio=1.0, iat_mean=0.1, syn_count=1):
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
