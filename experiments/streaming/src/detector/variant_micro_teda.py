"""
Adapter para variantes ablation do CorrectedMicroTEDAclus (teda-high-dim).

Wrapper que adapta as variantes V0-V7 do CorrectedMicroTEDAclus
para a interface process() usada pelo StreamingDetector, permitindo
comparacao direta no pipeline de streaming com avaliacao prequential.

Variantes disponiveis:
    V0_original:        Todas as flags OFF (implementacao original)
    V1_welford_var:     Apenas variancia Welford ON
    V2_consistent_ecc:  Apenas eccentricity consistente ON
    V3_welford_and_ecc: Welford + eccentricity ON
    V4_selective_update: Apenas update seletivo ON
    V5_n1_guard:        Apenas guard n=1 ON
    V6_n2_guard:        Apenas guard n=2 ON
    V7_full_corrected:  Todas as flags ON (implementacao corrigida)

Referencia:
    teda-high-dim/src/teda_hd/algorithms/variants.py
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

try:
    from teda_hd.algorithms.variants import create_variant, VARIANT_CONFIGS
    from teda_hd.algorithms.base import ClusteringResult
except ImportError as e:
    raise ImportError(
        "teda_hd package not found. Install it with: "
        "cd experiments/teda-high-dim && pip install -e .\n"
        f"Original error: {e}"
    )

from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class VariantMicroTEDAclus:
    """
    Adapter das variantes ablation V0-V7 para o pipeline de streaming.

    Mapeia ClusteringResult do teda-high-dim para MicroTEDAResult
    do pipeline existente, mantendo compatibilidade total.

    Uso:
        detector = VariantMicroTEDAclus(variant_name="V7_full_corrected", r0=0.1)
        result = detector.process(feature_vector)
        if result.is_anomaly:
            print("Anomalia detectada!")
    """

    # Nomes validos para referencia rapida
    VALID_VARIANTS = list(VARIANT_CONFIGS.keys())

    def __init__(
        self,
        variant_name: str = "V7_full_corrected",
        r0: float = 0.1,
        min_samples: int = 10,
    ):
        """
        Args:
            variant_name: Nome da variante (V0_original .. V7_full_corrected).
            r0: Limite de variancia para teste de Chebyshev (default: 0.1).
            min_samples: Amostras minimas antes de detectar (default: 10).
        """
        if variant_name not in VARIANT_CONFIGS:
            raise ValueError(
                f"Variante desconhecida: '{variant_name}'. "
                f"Opcoes: {self.VALID_VARIANTS}"
            )

        self.variant_name = variant_name
        self.r0 = r0
        self.min_samples = min_samples

        # Cria instancia da variante configurada
        self._algo = create_variant(variant_name, r0=r0)

        # Contadores expostos para compatibilidade com o pipeline
        self.total_samples = 0

        logger.info(
            f"VariantMicroTEDAclus inicializado: {variant_name} (r0={r0})"
        )

    @property
    def anomaly_count(self) -> int:
        return self._algo.anomaly_count

    @property
    def micro_clusters(self):
        """Expoe os micro-clusters da variante (corrected.py) sob o MESMO nome que
        MicroTEDAclus usa. Permite ao streaming_detector computar o sinal de regime
        rho = variance / r0 por fluxo de forma identica nas duas linhagens (cada
        CorrectedMicroCluster tem .variance e .n, como o micro_cluster original).
        Sem isto, o re-run do Exp A nao gravaria rho_mean/rho_max/... e a Tabela 5
        / H4 ficariam sem dados. Ver streaming_detector._process_flow (bloco rho)."""
        return self._algo._clusters

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self._algo.reset()

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Delega para o CorrectedMicroTEDAclus da variante selecionada
        e mapeia ClusteringResult → MicroTEDAResult.

        Args:
            x: Vetor de features do ponto.

        Returns:
            MicroTEDAResult compativel com o pipeline existente.
        """
        x = np.asarray(x, dtype=np.float64)
        self.total_samples += 1

        # Processar via variante
        result: ClusteringResult = self._algo.process(x)

        # Aplicar min_samples override
        # O CorrectedMicroTEDAclus usa min_samples=3 internamente,
        # mas no pipeline de streaming usamos o valor configurado
        is_anomaly = result.is_anomaly
        if result.new_cluster_created and result.sample_count < self.min_samples:
            is_anomaly = False

        # Mapear ClusteringResult → MicroTEDAResult
        return MicroTEDAResult(
            eccentricity=result.eccentricity,
            typicality=result.typicality,
            cluster_id=result.cluster_id,
            is_anomaly=is_anomaly,
            num_clusters=result.num_clusters,
            sample_count=result.sample_count,
            new_cluster_created=result.new_cluster_created,
            cluster_typicalities=None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        clusters = self._algo.get_clusters()
        clusters_info = []
        for mc in clusters:
            clusters_info.append({
                "cluster_id": mc.cluster_id,
                "n": mc.n,
                "mean": mc.mean.tolist() if hasattr(mc.mean, "tolist") else mc.mean,
                "variance": float(mc.variance),
            })

        return {
            "total_samples": self._algo.total_samples,
            "num_clusters": len(clusters),
            "anomaly_count": self._algo.anomaly_count,
            "variant_name": self.variant_name,
            "r0": self.r0,
            "min_samples": self.min_samples,
            "clusters": clusters_info,
        }
