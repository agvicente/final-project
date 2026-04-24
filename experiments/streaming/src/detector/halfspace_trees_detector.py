"""
Adapter para river HalfSpaceTrees no pipeline de streaming.

Wrapper que adapta o Half-Space Trees (Tan et al. 2011) para a interface
process() usada pelo StreamingDetector, permitindo comparacao direta
com MicroTEDAclus sob avaliacao prequential identica.

Diferente de IF e OC-SVM, HST e genuinamente incremental:
    - O(1) por ponto (score_one + learn_one)
    - Sem buffer, sem retreino batch
    - Nativo para streaming

Referencia:
    Tan, S.C., Ting, K.M., Liu, T.F. (2011). "Fast Anomaly Detection
    for Streaming Data." IJCAI, pp. 1511-1516.
"""

import numpy as np
from typing import Dict, Any
import logging

try:
    from river.anomaly import HalfSpaceTrees
except ImportError:
    raise ImportError(
        "river is required for HalfSpaceTreesDetector. "
        "Install it with: pip install river"
    )

from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class HalfSpaceTreesDetector:
    """
    Adapter do HalfSpaceTrees (river) para a interface do StreamingDetector.

    Genuinamente incremental — O(1) por ponto, sem buffer ou retreino.
    Usa score_one() ANTES de learn_one() para avaliacao prequential
    (test-then-train).

    O score do HST varia de 0 (normal) a 1 (anomalo).
    O threshold padrao de 0.5 pode ser ajustado.

    Atributos:
        n_trees: Numero de arvores no ensemble.
        height: Altura maxima de cada arvore.
        window_size: Janela de referencia para scores.
        seed: Seed para reprodutibilidade.
        threshold: Limiar para classificacao de anomalia.
        min_samples: Amostras minimas antes de reportar anomalias.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 15,
        window_size: int = 250,
        seed: int = 42,
        threshold: float = 0.5,
        min_samples: int = 10,
    ):
        """
        Args:
            n_trees: Numero de arvores no ensemble (default: 25).
            height: Altura maxima de cada arvore (default: 15).
            window_size: Janela de referencia para scores (default: 250).
            seed: Seed para reprodutibilidade (default: 42).
            threshold: Limiar para anomalia, 0-1 (default: 0.5).
            min_samples: Amostras minimas antes de detectar (default: 10).
        """
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.seed = seed
        self.threshold = threshold
        self.min_samples = min_samples

        # Modelo river
        self._model = HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=seed,
        )

        # Contadores
        self.total_samples = 0
        self.anomaly_count = 0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self._model = HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            seed=self.seed,
        )
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
        # score do HST: 0 (normal) a 1 (anomalo) → eccentricity
        eccentricity = float(score)
        typicality = 1.0 - eccentricity

        return MicroTEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            cluster_id=-1,  # HST nao tem clusters
            is_anomaly=is_anomaly,
            num_clusters=self.n_trees,  # Arvores como proxy
            sample_count=self.total_samples,
            new_cluster_created=False,
            cluster_typicalities=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        return {
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "n_trees": self.n_trees,
            "height": self.height,
            "window_size": self.window_size,
            "seed": self.seed,
            "threshold": self.threshold,
            "min_samples": self.min_samples,
        }
