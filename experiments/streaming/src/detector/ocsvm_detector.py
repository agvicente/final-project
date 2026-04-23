"""
Adapter para sklearn OneClassSVM no pipeline de streaming.

Wrapper que adapta o One-Class SVM (Scholkopf et al. 2001) para a
interface process() usada pelo StreamingDetector, permitindo comparacao
direta com MicroTEDAclus sob avaliacao prequential identica.

Estrategia de adaptacao (batch-adapted-to-streaming):
    - Acumula pontos em buffer de tamanho fixo
    - Classifica cada ponto ANTES de adicionar ao buffer (test-then-train)
    - Retreina modelo a cada buffer_size pontos
    - Pratica aceita na literatura (Gama 2013, Losing 2018)

Nota: OC-SVM nao e incremental nativo. A comparacao foca em acuracia
(FPR/Recall) sob mesmo protocolo prequential, nao em eficiencia.

Referencia:
    Scholkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J.,
    Williamson, R.C. (2001). "Estimating the Support of a
    High-Dimensional Distribution." Neural Computation 13(7):1443-1471.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

try:
    from sklearn.svm import OneClassSVM
except ImportError:
    raise ImportError(
        "scikit-learn is required for OneClassSVMDetector. "
        "Install it with: pip install scikit-learn"
    )

from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class OneClassSVMDetector:
    """
    Adapter do OneClassSVM para a interface do StreamingDetector.

    Usa buffer-based retraining para simular comportamento em streaming:
    pontos sao classificados antes de serem adicionados ao buffer
    (prequential). Quando o buffer atinge buffer_size, o modelo e
    retreinado e o buffer e limpo.

    A deteccao de anomalia usa o predict() do sklearn:
        - predict(x) == -1 → anomalia
        - predict(x) ==  1 → normal

    Atributos:
        nu: Upper bound na fracao de erros de treino (e lower bound na
            fracao de support vectors). Equivalente a contamination.
        kernel: Kernel do SVM (default: 'rbf').
        gamma: Coeficiente do kernel RBF (default: 'scale').
        buffer_size: Tamanho do buffer para retreino.
        seed: Seed para reprodutibilidade (afeta apenas amostragem).
        min_samples: Amostras minimas antes de reportar anomalias.
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        buffer_size: int = 200,
        seed: int = 42,
        min_samples: int = 10,
    ):
        """
        Args:
            nu: Upper bound na fracao de outliers (default: 0.1).
            kernel: Tipo de kernel SVM (default: 'rbf').
            gamma: Coeficiente do kernel (default: 'scale').
            buffer_size: Pontos acumulados antes de retreinar (default: 200).
            seed: Seed para reprodutibilidade (default: 42).
            min_samples: Amostras minimas antes de detectar (default: 10).
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.seed = seed
        self.min_samples = min_samples

        # Modelo sklearn
        self._model: Optional[OneClassSVM] = None
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

    def _build_model(self) -> OneClassSVM:
        """Cria nova instancia do OneClassSVM."""
        return OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
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
            f"OC-SVM retreinado (#{self._retrain_count}) com {len(X)} pontos"
        )

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Logica prequential (test-then-train):
        1. Se buffer insuficiente (< min_samples), retorna nao-anomalia
        2. Classifica usando modelo atual (predict + decision_function)
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
        decision_value = 0.0

        if self._model is not None and self.total_samples >= self.min_samples:
            # Classificar com modelo atual
            x_2d = x.reshape(1, -1)
            prediction = self._model.predict(x_2d)[0]
            decision_value = self._model.decision_function(x_2d)[0]

            is_anomaly = prediction == -1
        elif self.total_samples < self.min_samples:
            # Warmup: nao classifica
            is_anomaly = False
            decision_value = 0.0

        if is_anomaly:
            self.anomaly_count += 1

        # Adiciona ao buffer (train)
        self._buffer.append(x.copy())

        # Retreina se buffer cheio
        if len(self._buffer) >= self.buffer_size:
            self._retrain()

        # Mapear para MicroTEDAResult
        # decision_function do OC-SVM: positivo = normal, negativo = anomalia
        # Mapeamos para eccentricity: anomalo → alto, normal → baixo
        eccentricity = min(max(-decision_value, 0.0), 1.0)
        typicality = 1.0 - eccentricity

        return MicroTEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            cluster_id=-1,  # OC-SVM nao tem clusters
            is_anomaly=is_anomaly,
            num_clusters=0,
            sample_count=self.total_samples,
            new_cluster_created=False,
            cluster_typicalities=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        n_support = 0
        if self._model is not None and hasattr(self._model, "support_vectors_"):
            n_support = len(self._model.support_vectors_)

        return {
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "seed": self.seed,
            "min_samples": self.min_samples,
            "retrain_count": self._retrain_count,
            "current_buffer_size": len(self._buffer),
            "model_fitted": self._model is not None,
            "n_support_vectors": n_support,
        }
