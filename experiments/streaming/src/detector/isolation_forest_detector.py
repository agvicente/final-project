"""
Adapter para sklearn IsolationForest no pipeline de streaming.

Wrapper que adapta o IsolationForest (Liu et al. 2008) para a interface
process() usada pelo StreamingDetector, permitindo comparacao direta
com MicroTEDAclus sob avaliacao prequential identica.

Estrategia de adaptacao (batch-adapted-to-streaming):
    - Acumula pontos em buffer de tamanho fixo
    - Classifica cada ponto ANTES de adicionar ao buffer (test-then-train)
    - Retreina modelo a cada buffer_size pontos
    - Pratica aceita na literatura (Gama 2013, Losing 2018)

Nota: IF nao e incremental nativo. A comparacao foca em acuracia
(FPR/Recall) sob mesmo protocolo prequential, nao em eficiencia.

Referencia:
    Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). "Isolation Forest."
    IEEE ICDM, pp. 413-422.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    raise ImportError(
        "scikit-learn is required for IsolationForestDetector. "
        "Install it with: pip install scikit-learn"
    )

from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Adapter do IsolationForest para a interface do StreamingDetector.

    Usa buffer-based retraining para simular comportamento em streaming:
    pontos sao classificados antes de serem adicionados ao buffer
    (prequential). Quando o buffer atinge buffer_size, o modelo e
    retreinado e o buffer e limpo.

    A deteccao de anomalia usa o predict() do sklearn:
        - predict(x) == -1 → anomalia
        - predict(x) ==  1 → normal

    Atributos:
        n_estimators: Numero de arvores no ensemble.
        contamination: Fracao esperada de anomalias (afeta threshold).
        buffer_size: Tamanho do buffer para retreino.
        seed: Seed para reprodutibilidade.
        min_samples: Amostras minimas antes de reportar anomalias.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        buffer_size: int = 200,
        seed: int = 42,
        min_samples: int = 10,
    ):
        """
        Args:
            n_estimators: Numero de arvores no ensemble (default: 100).
            contamination: Fracao esperada de anomalias (default: 0.1).
            buffer_size: Pontos acumulados antes de retreinar (default: 200).
            seed: Seed para reprodutibilidade (default: 42).
            min_samples: Amostras minimas antes de detectar (default: 10).
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.buffer_size = buffer_size
        self.seed = seed
        self.min_samples = min_samples

        # Modelo sklearn
        self._model: Optional[IsolationForest] = None
        self._buffer: List[np.ndarray] = []

        # Contadores
        self.total_samples = 0
        self.anomaly_count = 0
        self._retrain_count = 0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self._model = None
        self._buffer = []
        self.total_samples = 0
        self.anomaly_count = 0
        self._retrain_count = 0

    def _build_model(self) -> IsolationForest:
        """Cria nova instancia do IsolationForest."""
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,
        )

    def _retrain(self) -> None:
        """Retreina o modelo com dados do buffer."""
        if len(self._buffer) < 2:
            return

        X = np.array(self._buffer)
        self._model = self._build_model()
        self._model.fit(X)
        self._retrain_count += 1
        self._buffer = []

        logger.debug(
            f"IF retreinado (#{self._retrain_count}) com {len(X)} pontos"
        )

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Logica prequential (test-then-train):
        1. Se buffer insuficiente (< min_samples), retorna nao-anomalia
        2. Classifica usando modelo atual (predict + score_samples)
        3. Adiciona ao buffer
        4. Se buffer cheio, retreina e limpa

        Args:
            x: Vetor de features do ponto.

        Returns:
            MicroTEDAResult compativel com o pipeline existente.
        """
        x = np.asarray(x, dtype=np.float64)
        self.total_samples += 1

        # Classificacao prequential: classifica ANTES de adicionar ao buffer
        is_anomaly = False
        anomaly_score = 0.0

        if self._model is not None and self.total_samples >= self.min_samples:
            # Classificar com modelo atual
            x_2d = x.reshape(1, -1)
            prediction = self._model.predict(x_2d)[0]
            anomaly_score = -self._model.score_samples(x_2d)[0]

            is_anomaly = prediction == -1
        elif self.total_samples < self.min_samples:
            # Warmup: nao classifica
            is_anomaly = False
            anomaly_score = 0.0

        if is_anomaly:
            self.anomaly_count += 1

        # Adiciona ao buffer (train)
        self._buffer.append(x.copy())

        # Retreina se buffer cheio
        if len(self._buffer) >= self.buffer_size:
            self._retrain()

        # Mapear para MicroTEDAResult
        # anomaly_score do IF: maior = mais anomalo
        # Mapeamos para eccentricity (0-1 range aproximado)
        eccentricity = min(max(anomaly_score, 0.0), 1.0)
        typicality = 1.0 - eccentricity

        return MicroTEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            cluster_id=-1,  # IF nao tem clusters
            is_anomaly=is_anomaly,
            num_clusters=self.n_estimators,  # Arvores como proxy
            sample_count=self.total_samples,
            new_cluster_created=False,
            cluster_typicalities=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        return {
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "buffer_size": self.buffer_size,
            "seed": self.seed,
            "min_samples": self.min_samples,
            "retrain_count": self._retrain_count,
            "current_buffer_size": len(self._buffer),
            "model_fitted": self._model is not None,
        }
