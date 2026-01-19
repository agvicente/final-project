"""
Detector module - Anomaly detection algorithms for IoT IDS.

Este modulo contem implementacoes de algoritmos de deteccao de anomalias
para uso em streaming de dados IoT.

Modulos:
    teda: TEDA (Typicality and Eccentricity Data Analytics) - Angelov 2014

Uso:
    from src.detector import TEDADetector, TEDAResult

    detector = TEDADetector(m=3.0)
    result = detector.update(features)
    if result.is_anomaly:
        print("Anomalia detectada!")
"""

from .teda import TEDADetector, TEDAResult, calculate_eccentricity_batch

__all__ = [
    "TEDADetector",
    "TEDAResult",
    "calculate_eccentricity_batch",
]
