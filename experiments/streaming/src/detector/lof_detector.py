"""
Adapter para river LocalOutlierFactor no pipeline de streaming.

Wrapper que adapta o LOF incremental (river) para a interface process()
usada pelo StreamingDetector, permitindo comparacao direta com
MicroTEDAclus sob avaliacao prequential identica.

Diferente do LOF batch do sklearn, a versao river e incremental:
    - Atualiza vizinhos a cada ponto
    - Genuinamente streaming (sem buffer/retreino)

Referencia:
    Breunig, M.M., Kriegel, H.-P., Ng, R.T., Sander, J. (2000).
    "LOF: Identifying Density-Based Local Outliers." ACM SIGMOD,
    pp. 93-104.
"""

import numpy as np
from typing import Dict, Any
import logging

try:
    from river.anomaly import LocalOutlierFactor
except ImportError:
    raise ImportError(
        "river is required for LOFDetector. "
        "Install it with: pip install river"
    )

from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class LOFDetector:
    """
    Adapter do LocalOutlierFactor (river) para a interface do StreamingDetector.

    Genuinamente incremental — atualiza vizinhos por ponto.
    Usa score_one() ANTES de learn_one() para avaliacao prequential
    (test-then-train).

    O score do LOF: valores maiores indicam mais anomalo.
    O threshold padrao de 1.5 pode ser ajustado (LOF classico usa 1.0,
    mas na pratica 1.5 reduz falsos positivos).

    Atributos:
        n_neighbors: Numero de vizinhos para calculo de LOF.
        threshold: Limiar para classificacao de anomalia.
        min_samples: Amostras minimas antes de reportar anomalias.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        threshold: float = 1.5,
        min_samples: int = 10,
    ):
        """
        Args:
            n_neighbors: Numero de vizinhos para LOF (default: 20).
            threshold: Limiar para anomalia (default: 1.5).
            min_samples: Amostras minimas antes de detectar (default: 10).
        """
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.min_samples = min_samples

        # Modelo river
        self._model = LocalOutlierFactor(n_neighbors=n_neighbors)

        # Contadores
        self.total_samples = 0
        self.anomaly_count = 0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self._model = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        self.total_samples = 0
        self.anomaly_count = 0

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Logica prequential (test-then-train):
        1. Converte array numpy para dict (formato river)
        2. score_one() ANTES de learn_one()
        3. Classifica como anomalia se score > threshold

        Args:
            x: Vetor de features do ponto.

        Returns:
            MicroTEDAResult compativel com o pipeline existente.
        """
        x = np.asarray(x, dtype=np.float64)
        self.total_samples += 1

        # river espera dict como input
        x_dict = {f'f{i}': float(v) for i, v in enumerate(x)}

        # Prequential: score (test) ANTES de learn (train)
        score = self._model.score_one(x_dict)
        self._model.learn_one(x_dict)

        # Classificacao
        is_anomaly = (
            score > self.threshold
            and self.total_samples >= self.min_samples
        )

        if is_anomaly:
            self.anomaly_count += 1

        # Mapear para MicroTEDAResult
        # LOF score: >1 = outlier. Normalizar para 0-1 range aprox.
        eccentricity = min(max(float(score), 0.0), 1.0)
        typicality = 1.0 - eccentricity

        return MicroTEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            cluster_id=-1,  # LOF nao tem clusters
            is_anomaly=is_anomaly,
            num_clusters=self.n_neighbors,  # Vizinhos como proxy
            sample_count=self.total_samples,
            new_cluster_created=False,
            cluster_typicalities=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        return {
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "n_neighbors": self.n_neighbors,
            "threshold": self.threshold,
            "min_samples": self.min_samples,
        }
