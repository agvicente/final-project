"""
Detector module - Anomaly detection algorithms for IoT IDS.

Este modulo contem implementacoes de algoritmos de deteccao de anomalias
para uso em streaming de dados IoT.

Modulos:
    teda: TEDA (Typicality and Eccentricity Data Analytics) - Angelov 2014
    streaming_detector: Integracao Kafka + TEDA para deteccao em tempo real

Uso:
    # TEDA basico
    from src.detector import TEDADetector, TEDAResult

    detector = TEDADetector(m=3.0)
    result = detector.update(features)
    if result.is_anomaly:
        print("Anomalia detectada!")

    # Streaming completo
    from src.detector import StreamingDetector

    detector = StreamingDetector()
    detector.run()  # Consome flows do Kafka e detecta anomalias
"""

from .teda import TEDADetector, TEDAResult, calculate_eccentricity_batch
from .streaming_detector import StreamingDetector, StreamingDetectorConfig

__all__ = [
    "TEDADetector",
    "TEDAResult",
    "calculate_eccentricity_batch",
    "StreamingDetector",
    "StreamingDetectorConfig",
]
