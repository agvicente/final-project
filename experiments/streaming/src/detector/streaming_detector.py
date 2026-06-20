"""
Streaming Detector - Integra Consumer Kafka com TEDA/MicroTEDAclus.

Este modulo conecta o pipeline de streaming com detectores de anomalias:
    1. Consome flows do topico 'flows'
    2. Extrai features numericas de cada flow
    3. Passa para TEDADetector ou MicroTEDAclus
    4. Publica alertas no topico 'alerts'

Arquitetura:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Kafka     │────►│  Streaming  │────►│   Kafka     │
    │  (flows)    │     │  Detector   │     │  (alerts)   │
    └─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  TEDA / MicroTEDAclus│
                   │     (selecionavel)   │
                   └─────────────────────┘

Algoritmos disponiveis:
    - TEDA: Detector basico, single-center (vulneravel a contaminacao)
    - MicroTEDAclus: Multi-cluster evolutivo (robusto a contaminacao)

Uso:
    python -m src.detector.streaming_detector --verbose
    python -m src.detector.streaming_detector --algorithm micro_teda
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import time
import logging
import signal
import sys
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from .teda import TEDADetector, TEDAResult
from .micro_teda import MicroTEDAclus, MicroTEDAResult
from .window_aggregator import WindowAggregator, WINDOW_FEATURES, WINDOW_FEATURES_V2

# Optional detector imports — deferred to avoid hard dependency chains.
# Actual import happens in __init__ when the algorithm is selected.


# ============================================================
# ENUMS
# ============================================================

class DetectorAlgorithm(Enum):
    """Algoritmo de deteccao a usar."""
    TEDA = "teda"
    MICRO_TEDA = "micro_teda"
    ORIGINAL_MICRO_TEDA = "original_micro_teda"
    ISOLATION_FOREST = "isolation_forest"
    OCSVM = "ocsvm"
    VARIANT_MICRO_TEDA = "variant_micro_teda"
    HALFSPACE_TREES = "halfspace_trees"
    LOF = "lof"


class DetectionGranularity(Enum):
    """Granularidade de deteccao."""
    FLOW = "flow"      # Per-flow: cada flow individual
    WINDOW = "window"  # Per-window: agregado por IP em janela temporal

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of consecutive empty Kafka polls before the detector self-terminates.
# With poll(timeout_ms=1000), this equals ~10 seconds of silence.
IDLE_LIMIT = 10


# ============================================================
# CONFIGURACAO
# ============================================================

@dataclass
class StreamingDetectorConfig:
    """
    Configuracoes do StreamingDetector.

    Agrupa configuracoes de Kafka e detectores em um unico objeto.
    """
    # Kafka
    bootstrap_servers: str = "localhost:9092"
    topic_flows: str = "flows"
    topic_alerts: str = "alerts"
    group_id: str = "teda-detector"
    auto_offset_reset: str = "earliest"

    # Algoritmo de deteccao
    algorithm: DetectorAlgorithm = DetectorAlgorithm.MICRO_TEDA  # Default: robusto

    # TEDA basico
    teda_m: float = 3.0  # Desvios padrao para threshold
    teda_min_samples: int = 10  # Minimo de amostras antes de detectar

    # MicroTEDAclus
    micro_teda_r0: float = 0.1  # Variancia minima (calibrar para escala dos dados)
    micro_teda_min_samples: int = 10  # Minimo de amostras antes de detectar

    # Isolation Forest (batch-adapted-to-streaming)
    if_n_estimators: int = 100  # Numero de arvores
    if_contamination: float = 0.1  # Fracao esperada de anomalias
    if_buffer_size: int = 200  # Pontos acumulados antes de retreinar

    # One-Class SVM (batch-adapted-to-streaming)
    svm_nu: float = 0.1  # Upper bound na fracao de outliers
    svm_kernel: str = "rbf"  # Tipo de kernel
    svm_gamma: str = "scale"  # Coeficiente do kernel

    # Half-Space Trees (river — genuinamente incremental)
    hst_n_trees: int = 25  # Numero de arvores
    hst_height: int = 15  # Altura maxima
    hst_window_size: int = 250  # Janela de referencia
    hst_threshold: float = 0.5  # Limiar anomalia (0-1)

    # LOF (river — genuinamente incremental)
    lof_n_neighbors: int = 20  # Numero de vizinhos
    lof_threshold: float = 1.5  # Limiar anomalia

    # Variantes ablation (teda-high-dim V0-V7)
    variant_name: str = "V7_full_corrected"  # Variante a usar

    # Features a usar (None = todas numericas)
    feature_names: Optional[List[str]] = None

    # Granularidade de deteccao
    granularity: DetectionGranularity = DetectionGranularity.FLOW
    window_size_seconds: float = 10.0
    min_flows_per_window: int = 5
    window_feature_version: str = "v1"

    # Comportamento
    publish_alerts: bool = True
    publish_all_results: bool = False  # Se True, publica normal tambem
    verbose: bool = False
    log_interval: int = 100  # Log a cada N flows


# ============================================================
# FEATURES PARA TEDA
# ============================================================

# Features numericas extraidas do flow que serao usadas pelo TEDA
# Estas features foram selecionadas por serem:
# 1. Numericas (TEDA trabalha com vetores)
# 2. Relevantes para deteccao de anomalias
# 3. Disponiveis no Consumer v0.1

FEATURES_V1 = [
    # Contadores basicos
    "packet_count",
    "total_bytes",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_bytes",
    "bwd_bytes",

    # Taxas
    "packets_per_second",
    "bytes_per_second",

    # Estatisticas de tamanho
    "packet_size_mean",
    "packet_size_std",

    # Inter-arrival time
    "iat_mean",
    "iat_std",

    # TCP Flags (importantes para detectar scans)
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",

    # Ratio
    "fwd_bwd_ratio",
]

# v2: 25 features = v1 (17) + 8 curadas para melhorar separabilidade ataque/benigno
FEATURES_V2 = FEATURES_V1 + [
    "flow_duration",          # Distingue bursts curtos de conexoes longas
    "packet_size_min",        # Ataques DDoS tem pacotes uniformes (min ≈ max)
    "packet_size_max",        # Complementa min — revela uniformidade
    "fwd_packet_size_mean",   # Assimetria direcional forte em ataques
    "bwd_packet_size_mean",   # DDoS tipicamente sem trafego de retorno
    "iat_min",                # Floods tem IAT minimo muito baixo
    "iat_max",                # Combinado com iat_min revela burstiness
    "psh_count",              # Padrao PSH difere entre tipos de ataque
]

# v3: 32 features = v2 (25) + 7 adicionais (variabilidade direcional + IAT direcional)
FEATURES_V3 = FEATURES_V2 + [
    "fwd_packet_size_std",    # Variabilidade direcional forward
    "bwd_packet_size_std",    # Variabilidade direcional backward
    "urg_count",              # Completude de flags
    "fwd_iat_mean",           # IAT direcional forward
    "fwd_iat_std",            # Variabilidade IAT forward
    "bwd_iat_mean",           # IAT direcional backward
    "bwd_iat_std",            # Variabilidade IAT backward
]

FEATURE_SETS = {
    "v1": FEATURES_V1,
    "v2": FEATURES_V2,
    "v3": FEATURES_V3,
}

DEFAULT_FEATURES = FEATURES_V1


# ============================================================
# STREAMING DETECTOR
# ============================================================

class StreamingDetector:
    """
    Detector de anomalias em streaming usando TEDA ou MicroTEDAclus.

    Consome flows do Kafka, aplica detector de anomalias,
    e publica alertas.

    Fluxo:
        1. Conecta ao Kafka (consumer para 'flows', producer para 'alerts')
        2. Para cada flow recebido:
           a. Extrai features numericas
           b. Normaliza features (opcional)
           c. Passa para detector (TEDA ou MicroTEDAclus)
           d. Se anomalia, publica alerta
        3. Mantem estatisticas de execucao

    Algoritmos:
        - TEDA: Detector basico single-center (vulneravel a contaminacao)
        - MicroTEDAclus: Multi-cluster evolutivo (robusto a contaminacao)

    Uso:
        detector = StreamingDetector()
        detector.run()  # Loop infinito
        # ou
        detector.run(max_flows=1000)  # Processa 1000 flows
    """

    def __init__(
        self,
        config: Optional[StreamingDetectorConfig] = None,
    ):
        """
        Inicializa o StreamingDetector.

        Args:
            config: Configuracoes (None = defaults, usa MicroTEDAclus)
        """
        self.config = config or StreamingDetectorConfig()

        # Kafka clients
        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaProducer] = None

        # Detector de anomalias (TEDA ou MicroTEDAclus)
        self._detector: Union[TEDADetector, MicroTEDAclus]
        self._algorithm = self.config.algorithm

        if self._algorithm == DetectorAlgorithm.TEDA:
            self._detector = TEDADetector(
                m=self.config.teda_m,
                min_samples=self.config.teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.ORIGINAL_MICRO_TEDA:
            from .original_micro_teda import OriginalMicroTEDAclus
            self._detector = OriginalMicroTEDAclus(
                r0=self.config.micro_teda_r0,
                min_samples=self.config.micro_teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.ISOLATION_FOREST:
            from .isolation_forest_detector import IsolationForestDetector
            self._detector = IsolationForestDetector(
                n_estimators=self.config.if_n_estimators,
                contamination=self.config.if_contamination,
                buffer_size=self.config.if_buffer_size,
                min_samples=self.config.micro_teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.OCSVM:
            from .ocsvm_detector import OneClassSVMDetector
            self._detector = OneClassSVMDetector(
                nu=self.config.svm_nu,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
                buffer_size=self.config.if_buffer_size,
                min_samples=self.config.micro_teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.VARIANT_MICRO_TEDA:
            from .variant_micro_teda import VariantMicroTEDAclus
            self._detector = VariantMicroTEDAclus(
                variant_name=self.config.variant_name,
                r0=self.config.micro_teda_r0,
                min_samples=self.config.micro_teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.HALFSPACE_TREES:
            from .halfspace_trees_detector import HalfSpaceTreesDetector
            self._detector = HalfSpaceTreesDetector(
                n_trees=self.config.hst_n_trees,
                height=self.config.hst_height,
                window_size=self.config.hst_window_size,
                seed=42,
                threshold=self.config.hst_threshold,
                min_samples=self.config.micro_teda_min_samples,
            )
        elif self._algorithm == DetectorAlgorithm.LOF:
            from .lof_detector import LOFDetector
            self._detector = LOFDetector(
                n_neighbors=self.config.lof_n_neighbors,
                threshold=self.config.lof_threshold,
                min_samples=self.config.micro_teda_min_samples,
            )
        else:  # MicroTEDAclus (default)
            self._detector = MicroTEDAclus(
                r0=self.config.micro_teda_r0,
                min_samples=self.config.micro_teda_min_samples,
            )

        # Features a usar
        self._granularity = self.config.granularity
        if self._granularity == DetectionGranularity.WINDOW:
            wfv = self.config.window_feature_version
            self._feature_names = WINDOW_FEATURES_V2 if wfv == "v2" else WINDOW_FEATURES
            self._window_aggregator = WindowAggregator(
                window_size_seconds=self.config.window_size_seconds,
                min_flows_per_window=self.config.min_flows_per_window,
                window_feature_version=wfv,
            )
        else:
            self._feature_names = self.config.feature_names or DEFAULT_FEATURES
            self._window_aggregator = None

        # Estatisticas
        self.flows_processed = 0
        self.anomalies_detected = 0
        self.start_time: Optional[float] = None

        # Controle
        self._running = False

        # Resultados de detecção (para avaliação externa pelo orquestrador)
        self._detection_results: List[Dict] = []

    def connect(self) -> None:
        """Conecta ao Kafka."""
        logger.info(f"Conectando ao Kafka em {self.config.bootstrap_servers}...")

        # Consumer para flows
        self._consumer = KafkaConsumer(
            self.config.topic_flows,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.group_id,
            auto_offset_reset=self.config.auto_offset_reset,
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
        )
        logger.info(f"Consumer conectado ao topico '{self.config.topic_flows}'")

        # Producer para alerts
        if self.config.publish_alerts:
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            logger.info(f"Producer conectado ao topico '{self.config.topic_alerts}'")

    def close(self) -> None:
        """Fecha conexoes."""
        self._running = False

        if self._consumer:
            self._consumer.close()
            logger.info("Consumer fechado")

        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Producer fechado")

    def _extract_features(self, flow: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extrai features numericas de um flow.

        Args:
            flow: Dicionario com dados do flow (do Consumer)

        Returns:
            Array numpy com features, ou None se erro
        """
        try:
            features = []
            for name in self._feature_names:
                value = flow.get(name, 0)
                # Converte para float, tratando None
                if value is None:
                    value = 0.0
                features.append(float(value))

            return np.array(features, dtype=np.float64)

        except Exception as e:
            logger.warning(f"Erro extraindo features: {e}")
            return None

    def _create_alert(
        self,
        flow: Dict[str, Any],
        result: Union[TEDAResult, MicroTEDAResult]
    ) -> Dict[str, Any]:
        """
        Cria mensagem de alerta para publicar no Kafka.

        Args:
            flow: Dados originais do flow
            result: Resultado do detector (TEDAResult ou MicroTEDAResult)

        Returns:
            Dicionario com alerta formatado
        """
        # Base comum para ambos os tipos
        alert = {
            # Identificacao do flow
            "flow_id": f"{flow.get('src_ip', '?')}:{flow.get('src_port', '?')}->"
                      f"{flow.get('dst_ip', '?')}:{flow.get('dst_port', '?')}",
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "src_port": flow.get("src_port"),
            "dst_port": flow.get("dst_port"),
            "protocol": flow.get("protocol"),

            # Metricas do flow
            "packet_count": flow.get("packet_count"),
            "total_bytes": flow.get("total_bytes"),
            "flow_duration": flow.get("flow_duration"),

            # Resultado comum
            "eccentricity": float(result.eccentricity),
            "typicality": float(result.typicality),
            "is_anomaly": bool(result.is_anomaly),

            # Contexto
            "sample_number": result.sample_count,
            "detected_at": datetime.now().isoformat(),

            # Algoritmo usado
            "algorithm": self._algorithm.value,

            # Severidade
            "severity": self._calculate_severity(result),
        }

        # Campos especificos do algoritmo
        if isinstance(result, TEDAResult):
            alert["threshold"] = float(result.threshold)
            alert["normalized_eccentricity"] = float(result.normalized_eccentricity)
        elif isinstance(result, MicroTEDAResult):
            alert["cluster_id"] = result.cluster_id
            alert["num_clusters"] = result.num_clusters
            alert["new_cluster_created"] = result.new_cluster_created

        return alert

    def _calculate_severity(
        self, result: Union[TEDAResult, MicroTEDAResult]
    ) -> str:
        """
        Calcula severidade da anomalia baseada na eccentricity.

        Para TEDA: ratio de eccentricity sobre threshold
        Para MicroTEDAclus: baseado na eccentricity absoluta (novo cluster criado)

        Returns:
            "low", "medium", "high", ou "critical"
        """
        if not result.is_anomaly:
            return "normal"

        if isinstance(result, TEDAResult):
            # TEDA: ratio sobre threshold
            ratio = result.normalized_eccentricity / result.threshold
        elif isinstance(result, MicroTEDAResult):
            # MicroTEDAclus: usa eccentricity direta
            # Anomalias sao novos clusters, entao eccentricity=1.0 sempre
            # Severidade baseada em quantos clusters ja existem (padroes anteriores)
            # Mais clusters = mais padroes conhecidos = anomalia mais significativa
            if result.num_clusters <= 2:
                ratio = 1.2  # Poucos clusters, pode ser ruido
            elif result.num_clusters <= 5:
                ratio = 1.8
            elif result.num_clusters <= 10:
                ratio = 2.5
            else:
                ratio = 3.5  # Muitos clusters, novo padrao e significativo
        else:
            ratio = 1.5  # Default

        if ratio < 1.5:
            return "low"
        elif ratio < 2.0:
            return "medium"
        elif ratio < 3.0:
            return "high"
        else:
            return "critical"

    def _process_window_vectors(
        self, vectors: list
    ) -> None:
        """Processa vetores agregados emitidos pelo WindowAggregator."""
        for feature_vector, metadata in vectors:
            if self._algorithm == DetectorAlgorithm.TEDA:
                result = self._detector.update(feature_vector)
            else:
                result = self._detector.process(feature_vector)

            self.flows_processed += 1

            self._detection_results.append({
                "is_anomaly": result.is_anomaly,
                "first_packet_time": metadata.get("window_start", time.time()),
                "src_ip": metadata.get("src_ip"),
                "dst_ip": None,  # Window mode: ground truth por src_ip
            })

            if result.is_anomaly:
                self.anomalies_detected += 1

    def _process_flow(
        self, flow: Dict[str, Any]
    ) -> Optional[Union[TEDAResult, MicroTEDAResult]]:
        """
        Processa um flow: extrai features, aplica detector, publica alerta.

        Em modo WINDOW: acumula flows no WindowAggregator; quando janela
        fecha, processa vetores agregados no detector.

        Args:
            flow: Dados do flow do Kafka

        Returns:
            TEDAResult ou MicroTEDAResult, ou None se erro/window mode
        """
        # Modo WINDOW: acumula no agregador
        if self._granularity == DetectionGranularity.WINDOW:
            emitted = self._window_aggregator.add_flow(flow)
            if emitted:
                self._process_window_vectors(emitted)
            return None

        # Modo FLOW (original): extrai features e processa
        features = self._extract_features(flow)
        if features is None:
            return None

        # Aplica detector (TEDA ou MicroTEDAclus)
        if self._algorithm == DetectorAlgorithm.TEDA:
            result = self._detector.update(features)  # TEDADetector.update()
        else:
            result = self._detector.process(features)  # MicroTEDAclus.process()

        self.flows_processed += 1

        # Coleta resultado para avaliação externa (ground truth aplicado pelo orquestrador)
        # G3: estado de regime por fluxo (rho = variance/r0) para a série temporal de drift.
        _rec = {
            "is_anomaly": result.is_anomaly,
            "first_packet_time": flow.get("first_packet_time", time.time()),
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "flow_index": self.flows_processed,
            "eccentricity": float(getattr(result, "eccentricity", 0.0) or 0.0),
            "typicality": float(getattr(result, "typicality", 0.0) or 0.0),
            "num_clusters": int(getattr(result, "num_clusters", 0) or 0),
            "new_cluster_created": bool(getattr(result, "new_cluster_created", False)),
        }
        # rho por fluxo (so MicroTEDAclus tem micro_clusters/r0)
        _det = self._detector
        if hasattr(_det, "micro_clusters") and hasattr(_det, "r0") and _det.r0 > 0:
            _vars = [mc.variance for mc in _det.micro_clusters]
            if _vars:
                _rhos = [v / _det.r0 for v in _vars]
                _rec["rho_mean"] = float(np.mean(_rhos))
                _rec["rho_max"] = float(np.max(_rhos))
                _rec["rho_frac_above_1"] = float(np.mean([r > 1.0 for r in _rhos]))
                _rec["n_singletons"] = int(sum(1 for mc in _det.micro_clusters if mc.n == 1))
            else:
                _rec["rho_mean"] = _rec["rho_max"] = _rec["rho_frac_above_1"] = 0.0
                _rec["n_singletons"] = 0
        self._detection_results.append(_rec)

        # Log verbose
        if self.config.verbose:
            status = "ANOMALIA!" if result.is_anomaly else "normal"
            if isinstance(result, TEDAResult):
                logger.info(
                    f"Flow {self.flows_processed}: "
                    f"ξ={result.eccentricity:.4f}, "
                    f"τ={result.typicality:.4f}, "
                    f"threshold={result.threshold:.4f} → {status}"
                )
            elif isinstance(result, MicroTEDAResult):
                cluster_info = f"cluster={result.cluster_id}"
                if result.new_cluster_created:
                    cluster_info = f"NEW cluster={result.cluster_id}"
                logger.info(
                    f"Flow {self.flows_processed}: "
                    f"ξ={result.eccentricity:.4f}, "
                    f"τ={result.typicality:.4f}, "
                    f"{cluster_info}, "
                    f"total_clusters={result.num_clusters} → {status}"
                )

        # Se anomalia, publica alerta
        if result.is_anomaly:
            self.anomalies_detected += 1
            alert = self._create_alert(flow, result)

            if self._producer and self.config.publish_alerts:
                self._producer.send(
                    self.config.topic_alerts,
                    key=alert["flow_id"],
                    value=alert,
                )

            if self.config.verbose:
                extra_info = ""
                if isinstance(result, MicroTEDAResult):
                    extra_info = f", clusters={result.num_clusters}"
                logger.warning(
                    f"ANOMALIA DETECTADA: {alert['flow_id']} "
                    f"(ξ={result.eccentricity:.4f}, severity={alert['severity']}{extra_info})"
                )

        # Publica todos os resultados (se configurado)
        elif self._producer and self.config.publish_all_results:
            alert = self._create_alert(flow, result)
            self._producer.send(
                self.config.topic_alerts,
                key=alert["flow_id"],
                value=alert,
            )

        return result

    def run(self, max_flows: Optional[int] = None) -> Dict[str, Any]:
        """
        Executa o detector em loop.

        Args:
            max_flows: Limite de flows (None = infinito)

        Returns:
            Estatisticas de execucao
        """
        if not self._consumer:
            self.connect()

        self._running = True
        self.start_time = time.time()
        idle_polls = 0

        logger.info("=" * 60)
        logger.info("Streaming Detector iniciado")
        logger.info(f"  Algoritmo: {self._algorithm.value}")
        logger.info(f"  Topico entrada: {self.config.topic_flows}")
        logger.info(f"  Topico alertas: {self.config.topic_alerts}")
        if self._algorithm == DetectorAlgorithm.TEDA:
            logger.info(f"  TEDA: m={self.config.teda_m}, min_samples={self.config.teda_min_samples}")
        elif self._algorithm == DetectorAlgorithm.ISOLATION_FOREST:
            logger.info(f"  IF: n_estimators={self.config.if_n_estimators}, contamination={self.config.if_contamination}, buffer={self.config.if_buffer_size}")
        elif self._algorithm == DetectorAlgorithm.OCSVM:
            logger.info(f"  OC-SVM: nu={self.config.svm_nu}, kernel={self.config.svm_kernel}, gamma={self.config.svm_gamma}, buffer={self.config.if_buffer_size}")
        elif self._algorithm == DetectorAlgorithm.VARIANT_MICRO_TEDA:
            logger.info(f"  Variant: {self.config.variant_name}, r0={self.config.micro_teda_r0}, min_samples={self.config.micro_teda_min_samples}")
        elif self._algorithm == DetectorAlgorithm.HALFSPACE_TREES:
            logger.info(f"  HST: n_trees={self.config.hst_n_trees}, height={self.config.hst_height}, window={self.config.hst_window_size}, threshold={self.config.hst_threshold}")
        elif self._algorithm == DetectorAlgorithm.LOF:
            logger.info(f"  LOF: n_neighbors={self.config.lof_n_neighbors}, threshold={self.config.lof_threshold}")
        else:
            logger.info(f"  MicroTEDAclus: r0={self.config.micro_teda_r0}, min_samples={self.config.micro_teda_min_samples}")
        logger.info(f"  Features: {len(self._feature_names)}")
        logger.info(f"  Granularidade: {self._granularity.value}")
        if self._granularity == DetectionGranularity.WINDOW:
            logger.info(f"  Janela: {self.config.window_size_seconds}s, min_flows={self.config.min_flows_per_window}")
        logger.info("=" * 60)

        try:
            while self._running:
                # Poll com timeout de 1s
                records = self._consumer.poll(timeout_ms=1000)

                if not records:
                    idle_polls += 1
                    if idle_polls >= IDLE_LIMIT:
                        logger.info(f"Sem mensagens por {IDLE_LIMIT}s — encerrando")
                        self._running = False
                    continue

                idle_polls = 0  # reset on any message

                for topic_partition, messages in records.items():
                    for message in messages:
                        self._process_flow(message.value)

                        # Verifica limite
                        if max_flows and self.flows_processed >= max_flows:
                            logger.info(f"Limite de {max_flows} flows atingido")
                            self._running = False
                            break

                    if not self._running:
                        break

                # Log de progresso periodico
                if (self.flows_processed > 0 and
                    self.flows_processed % self.config.log_interval == 0):
                    elapsed = time.time() - self.start_time
                    rate = self.flows_processed / elapsed if elapsed > 0 else 0
                    anomaly_rate = (self.anomalies_detected / self.flows_processed * 100
                                   if self.flows_processed > 0 else 0)
                    logger.info(
                        f"Progresso: {self.flows_processed} flows, "
                        f"{self.anomalies_detected} anomalias ({anomaly_rate:.1f}%), "
                        f"{rate:.1f} flows/s"
                    )

        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuario (Ctrl+C)")

        finally:
            # Flush da janela final (modo window)
            if self._window_aggregator is not None:
                remaining = self._window_aggregator.flush()
                if remaining:
                    self._process_window_vectors(remaining)

            self.close()

        # Estatisticas finais
        elapsed = time.time() - self.start_time if self.start_time else 0
        stats = {
            "flows_processed": self.flows_processed,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": (self.anomalies_detected / self.flows_processed * 100
                           if self.flows_processed > 0 else 0),
            "elapsed_seconds": elapsed,
            "flows_per_second": self.flows_processed / elapsed if elapsed > 0 else 0,
            "algorithm": self._algorithm.value,
            "detector_stats": self._detector.get_statistics(),
            "detection_results": self._detection_results,
        }

        return stats


# ============================================================
# SIGNAL HANDLERS
# ============================================================

_detector_instance: Optional[StreamingDetector] = None


def _signal_handler(signum, frame):
    """Handler para SIGINT e SIGTERM."""
    logger.info(f"Recebido sinal {signum}, iniciando shutdown...")
    if _detector_instance:
        _detector_instance._running = False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    """
    Execucao direta do StreamingDetector.

    Uso:
        cd streaming
        python -m src.detector.streaming_detector

    Ou com argumentos:
        python -m src.detector.streaming_detector --max-flows 100 --verbose
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming Detector - TEDA para deteccao de anomalias em tempo real"
    )
    parser.add_argument(
        "--max-flows",
        type=int,
        default=None,
        help="Numero maximo de flows a processar (default: infinito)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra cada flow)"
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="teda-detector",
        help="ID do consumer group (default: teda-detector)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["teda", "micro_teda"],
        default="micro_teda",
        help="Algoritmo de deteccao (default: micro_teda)"
    )
    parser.add_argument(
        "--m",
        type=float,
        default=3.0,
        help="Parametro m do TEDA basico (desvios padrao, default: 3.0)"
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=0.1,
        help="Parametro r0 do MicroTEDAclus (variancia minima, default: 0.1)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimo de amostras antes de detectar (default: 10)"
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Nao publica alertas no Kafka"
    )

    args = parser.parse_args()

    # Seleciona algoritmo
    algorithm = (DetectorAlgorithm.TEDA
                 if args.algorithm == "teda"
                 else DetectorAlgorithm.MICRO_TEDA)

    # Configuracao
    config = StreamingDetectorConfig(
        group_id=args.group_id,
        algorithm=algorithm,
        teda_m=args.m,
        teda_min_samples=args.min_samples,
        micro_teda_r0=args.r0,
        micro_teda_min_samples=args.min_samples,
        publish_alerts=not args.no_publish,
        verbose=args.verbose,
    )

    # Signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Executa
    detector = StreamingDetector(config)
    _detector_instance = detector

    logger.info("=" * 60)
    logger.info(f"Streaming Detector v0.2 - {args.algorithm.upper()}")
    logger.info("=" * 60)

    try:
        stats = detector.run(max_flows=args.max_flows)

        logger.info("=" * 60)
        logger.info("Estatisticas finais:")
        logger.info(f"  Flows processados: {stats['flows_processed']}")
        logger.info(f"  Anomalias detectadas: {stats['anomalies_detected']}")
        logger.info(f"  Taxa de anomalias: {stats['anomaly_rate']:.2f}%")
        logger.info(f"  Tempo total: {stats['elapsed_seconds']:.2f}s")
        logger.info(f"  Taxa: {stats['flows_per_second']:.1f} flows/s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise
