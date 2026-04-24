"""
Detector module - Anomaly detection algorithms for IoT IDS.

Este modulo contem implementacoes de algoritmos de deteccao de anomalias
para uso em streaming de dados IoT.

Modulos:
    teda: TEDA basico (Typicality and Eccentricity Data Analytics) - Angelov 2014
    micro_teda: MicroTEDAclus (Multi-cluster evolutivo) - Maia et al. 2020
    streaming_detector: Integracao Kafka + TEDA/MicroTEDAclus para deteccao em tempo real

Uso:
    # TEDA basico (vulneravel a contaminacao)
    from src.detector import TEDADetector, TEDAResult

    detector = TEDADetector(m=3.0)
    result = detector.update(features)
    if result.is_anomaly:
        print("Anomalia detectada!")

    # MicroTEDAclus (robusto a contaminacao)
    from src.detector import MicroTEDAclus, MicroTEDAResult

    detector = MicroTEDAclus(r0=0.1, min_samples=10)
    result = detector.process(features)
    if result.is_anomaly:
        print(f"Anomalia! Novo cluster {result.cluster_id} criado")

    # Streaming completo (usa MicroTEDAclus por padrao)
    from src.detector import StreamingDetector

    detector = StreamingDetector()
    detector.run()  # Consome flows do Kafka e detecta anomalias
"""

from .teda import TEDADetector, TEDAResult, calculate_eccentricity_batch
from .micro_teda import MicroCluster, MicroTEDAclus, MicroTEDAResult

# Optional dependencies — import with try/except to allow partial installs
try:
    from .original_micro_teda import OriginalMicroTEDAclus
except ImportError:
    OriginalMicroTEDAclus = None  # type: ignore[misc,assignment]

try:
    from .isolation_forest_detector import IsolationForestDetector
except ImportError:
    IsolationForestDetector = None  # type: ignore[misc,assignment]

try:
    from .ocsvm_detector import OneClassSVMDetector
except ImportError:
    OneClassSVMDetector = None  # type: ignore[misc,assignment]

try:
    from .variant_micro_teda import VariantMicroTEDAclus
except ImportError:
    VariantMicroTEDAclus = None  # type: ignore[misc,assignment]

try:
    from .halfspace_trees_detector import HalfSpaceTreesDetector
except ImportError:
    HalfSpaceTreesDetector = None  # type: ignore[misc,assignment]

try:
    from .lof_detector import LOFDetector
except ImportError:
    LOFDetector = None  # type: ignore[misc,assignment]

from .streaming_detector import (
    StreamingDetector,
    StreamingDetectorConfig,
    DetectorAlgorithm,
)

__all__ = [
    # TEDA basico
    "TEDADetector",
    "TEDAResult",
    "calculate_eccentricity_batch",
    # MicroTEDAclus
    "MicroCluster",
    "MicroTEDAclus",
    "MicroTEDAResult",
    # MicroTEDAclus original (Maia 2020)
    "OriginalMicroTEDAclus",
    # Baselines (batch-adapted-to-streaming)
    "IsolationForestDetector",
    "OneClassSVMDetector",
    # Variantes ablation (teda-high-dim)
    "VariantMicroTEDAclus",
    # Streaming baselines (river — genuinamente incrementais)
    "HalfSpaceTreesDetector",
    "LOFDetector",
    # Streaming
    "StreamingDetector",
    "StreamingDetectorConfig",
    "DetectorAlgorithm",
]
