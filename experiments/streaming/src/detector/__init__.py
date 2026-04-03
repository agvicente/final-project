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
from .original_micro_teda import OriginalMicroTEDAclus
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
    # Streaming
    "StreamingDetector",
    "StreamingDetectorConfig",
    "DetectorAlgorithm",
]
