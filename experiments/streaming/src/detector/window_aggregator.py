"""
Window Aggregator - Agrega flows por IP em janelas temporais.

Muda a granularidade de deteccao de "flow individual" para
"comportamento agregado por IP em janela temporal".

Racional:
    Um flow DDoS individual e indistinguivel de um heartbeat IoT.
    Mas um IP atacante gera centenas de flows similares em poucos
    segundos — esse padrao agregado e anomalo.

Arquitetura:
    flows -> WindowAggregator -> features_por_IP_por_janela -> MicroTEDAclus

Features por IP/janela (~12):
    flow_count, total_packets, total_bytes, unique_dst_ips,
    unique_dst_ports, mean_flow_duration, mean_packets_per_flow,
    mean_packet_size, std_packet_size, fwd_bwd_ratio_mean,
    syn_ratio, mean_iat
"""

from collections import Counter

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# Features emitidas pelo WindowAggregator
WINDOW_FEATURES = [
    "flow_count",
    "total_packets",
    "total_bytes",
    "unique_dst_ips",
    "unique_dst_ports",
    "mean_flow_duration",
    "mean_packets_per_flow",
    "mean_packet_size",
    "std_packet_size",
    "fwd_bwd_ratio_mean",
    "syn_ratio",
    "mean_iat",
]

# v2: 19 features = 12 base + 7 behavioral (entropy, ratios, rates)
WINDOW_FEATURES_V2 = WINDOW_FEATURES + [
    "flows_per_second",
    "dst_port_entropy",
    "dst_ip_entropy",
    "unanswered_ratio",
    "payload_std",
    "small_flow_ratio",
    "fwd_only_ratio",
]


def _shannon_entropy(values: list) -> float:
    """Compute Shannon entropy (bits) of a list of discrete values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)


@dataclass
class WindowRecord:
    """Acumulador de flows para um src_ip dentro de uma janela."""
    src_ip: str
    window_start: float
    flows: List[Dict[str, Any]] = field(default_factory=list)

    def add_flow(self, flow: Dict[str, Any]) -> None:
        self.flows.append(flow)

    @property
    def flow_count(self) -> int:
        return len(self.flows)

    def to_feature_vector(
        self,
        feature_version: str = "v1",
        window_size_seconds: float = 10.0,
    ) -> np.ndarray:
        """Agrega flows em um vetor de features.

        Args:
            feature_version: "v1" (12 base) or "v2" (19 = base + behavioral)
            window_size_seconds: Window duration in seconds (used by v2 for flows_per_second)
        """
        flows = self.flows
        n = len(flows)
        if n == 0:
            num_features = len(WINDOW_FEATURES_V2 if feature_version == "v2" else WINDOW_FEATURES)
            return np.zeros(num_features, dtype=np.float64)

        total_packets = sum(f.get("packet_count", 0) for f in flows)
        total_bytes = sum(f.get("total_bytes", 0) for f in flows)
        dst_ips = {f.get("dst_ip") for f in flows} - {None}
        dst_ports = {f.get("dst_port") for f in flows} - {None}

        durations = [f.get("flow_duration", 0) for f in flows]
        pkt_counts = [f.get("packet_count", 0) for f in flows]
        pkt_sizes = [f.get("packet_size_mean", 0) for f in flows]
        fwd_bwd = [f.get("fwd_bwd_ratio", 0) for f in flows]
        iats = [f.get("iat_mean", 0) for f in flows]
        syn_counts = [f.get("syn_count", 0) for f in flows]

        total_syn = sum(syn_counts)
        syn_ratio = total_syn / total_packets if total_packets > 0 else 0

        base_features = [
            float(n),                                       # flow_count
            float(total_packets),                           # total_packets
            float(total_bytes),                             # total_bytes
            float(len(dst_ips)),                            # unique_dst_ips
            float(len(dst_ports)),                          # unique_dst_ports
            float(np.mean(durations)) if durations else 0,  # mean_flow_duration
            float(np.mean(pkt_counts)) if pkt_counts else 0,# mean_packets_per_flow
            float(np.mean(pkt_sizes)) if pkt_sizes else 0, # mean_packet_size
            float(np.std(pkt_sizes)) if pkt_sizes else 0,  # std_packet_size
            float(np.mean(fwd_bwd)) if fwd_bwd else 0,     # fwd_bwd_ratio_mean
            float(syn_ratio),                               # syn_ratio
            float(np.mean(iats)) if iats else 0,            # mean_iat
        ]

        if feature_version == "v2":
            # 7 behavioral features
            flows_per_second = n / max(window_size_seconds, 0.001)

            dst_ports_list = [f.get("dst_port") for f in flows if f.get("dst_port") is not None]
            dst_port_entropy = _shannon_entropy(dst_ports_list)

            dst_ips_list = [f.get("dst_ip") for f in flows if f.get("dst_ip") is not None]
            dst_ip_entropy = _shannon_entropy(dst_ips_list)

            unanswered = sum(1 for f in flows if f.get("syn_count", 0) > 0 and f.get("ack_count", 0) == 0)
            unanswered_ratio = unanswered / n

            bytes_list = [f.get("total_bytes", 0) for f in flows]
            payload_std = float(np.std(bytes_list)) if len(bytes_list) > 1 else 0.0

            small_flows = sum(1 for f in flows if f.get("packet_count", 0) <= 3)
            small_flow_ratio = small_flows / n

            fwd_only = sum(1 for f in flows if f.get("bwd_packet_count", 0) == 0)
            fwd_only_ratio = fwd_only / n

            base_features.extend([
                float(flows_per_second),
                float(dst_port_entropy),
                float(dst_ip_entropy),
                float(unanswered_ratio),
                float(payload_std),
                float(small_flow_ratio),
                float(fwd_only_ratio),
            ])

        return np.array(base_features, dtype=np.float64)

    def to_metadata(self) -> Dict[str, Any]:
        """Retorna metadata para rastreamento de ground truth."""
        return {
            "src_ip": self.src_ip,
            "window_start": self.window_start,
            "flow_count": self.flow_count,
        }


class WindowAggregator:
    """
    Agrega flows por src_ip em janelas temporais.

    Quando uma janela fecha (novo flow chega com timestamp alem da janela),
    emite vetores de features agregados para cada src_ip que atingiu
    min_flows_per_window.

    Args:
        window_size_seconds: Tamanho da janela em segundos (default: 10)
        min_flows_per_window: Minimo de flows por IP/janela para emitir (default: 5)
    """

    def __init__(
        self,
        window_size_seconds: float = 10.0,
        min_flows_per_window: int = 5,
        window_feature_version: str = "v1",
    ):
        self.window_size = window_size_seconds
        self.min_flows = min_flows_per_window
        self.feature_version = window_feature_version

        # Estado: {src_ip: WindowRecord}
        self._current_windows: Dict[str, WindowRecord] = {}
        self._current_window_start: Optional[float] = None

        # Estatisticas
        self.total_flows_received = 0
        self.total_windows_emitted = 0
        self.total_vectors_emitted = 0

    def _get_window_start(self, timestamp: float) -> float:
        """Calcula o inicio da janela para um dado timestamp."""
        return (timestamp // self.window_size) * self.window_size

    def add_flow(self, flow: Dict[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Adiciona um flow ao agregador.

        Args:
            flow: Dicionario com dados do flow (precisa de src_ip e first_packet_time)

        Returns:
            Lista de (feature_vector, metadata) para janelas que fecharam.
            Vazia se nenhuma janela fechou.
        """
        self.total_flows_received += 1

        timestamp = flow.get("first_packet_time", 0)
        src_ip = flow.get("src_ip", "unknown")
        window_start = self._get_window_start(timestamp)

        emitted = []

        # Se mudou de janela, emite os vetores da janela anterior
        if self._current_window_start is not None and window_start > self._current_window_start:
            emitted = self._flush_window()

        self._current_window_start = window_start

        # Adiciona flow ao acumulador do src_ip
        if src_ip not in self._current_windows:
            self._current_windows[src_ip] = WindowRecord(
                src_ip=src_ip,
                window_start=window_start,
            )
        self._current_windows[src_ip].add_flow(flow)

        return emitted

    def _flush_window(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Emite vetores de features para IPs que atingiram min_flows na janela atual.

        Returns:
            Lista de (feature_vector, metadata)
        """
        emitted = []

        for src_ip, record in self._current_windows.items():
            if record.flow_count >= self.min_flows:
                vector = record.to_feature_vector(
                    feature_version=self.feature_version,
                    window_size_seconds=self.window_size,
                )
                metadata = record.to_metadata()
                emitted.append((vector, metadata))
                self.total_vectors_emitted += 1

        if emitted:
            self.total_windows_emitted += 1

        # Limpa janela
        self._current_windows.clear()

        return emitted

    def flush(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Flush final — emite janela atual (para fim de stream)."""
        return self._flush_window()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do agregador."""
        return {
            "total_flows_received": self.total_flows_received,
            "total_windows_emitted": self.total_windows_emitted,
            "total_vectors_emitted": self.total_vectors_emitted,
            "window_size_seconds": self.window_size,
            "min_flows_per_window": self.min_flows,
            "feature_version": self.feature_version,
        }
