"""
Adapter para o EvolvingClustering original (Maia 2020).

Wrapper que adapta a implementacao original do autor (evolclustering package)
para a interface process() usada pelo StreamingDetector.

Diferencas em relacao a implementacao propria (micro_teda.py):
1. Atualiza TODOS os clusters que aceitam (nao so o melhor)
2. Macro-clusters via grafo de conectividade (NetworkX)
3. Pruning com life decay para clusters inativos
4. Formula de variancia diferente (norm-based vs Welford)
5. Sem caso especial para n=1 (threshold=13) ou n=2 (variance >= r0)

Referencia:
    Maia, J. et al. (2020). "Evolving clustering algorithm based on
    mixture of typicalities for stream data mining." Future Generation
    Computer Systems, 106, pp.672-684.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from evolving.EvolvingClustering import EvolvingClustering
from .micro_teda import MicroTEDAResult

logger = logging.getLogger(__name__)


class OriginalMicroTEDAclus:
    """
    Adapter do EvolvingClustering original para a interface do StreamingDetector.

    Mapeia:
        - variance_limit ←→ r0 (controla criacao de novos clusters)
        - decay ←→ controle de pruning (life decay)
        - new micro-cluster creation → is_anomaly=True

    A deteccao de anomalia segue a mesma logica: ponto rejeitado por todos
    os clusters existentes → novo cluster criado → anomalia.
    """

    def __init__(
        self,
        r0: float = 0.001,
        min_samples: int = 10,
        decay: int = 100,
        macro_cluster_update: int = 50,
    ):
        """
        Args:
            r0: Mapeado para variance_limit do EvolvingClustering.
            min_samples: Amostras minimas antes de reportar anomalias.
            decay: Fator de decay para pruning (fading_factor = 1/decay).
            macro_cluster_update: Frequencia de atualizacao de macro-clusters
                                  (a cada N amostras).
        """
        self.r0 = r0
        self.min_samples = min_samples
        self.decay = decay
        self.macro_cluster_update_freq = macro_cluster_update

        self._model = EvolvingClustering(
            variance_limit=r0,
            decay=decay,
            macro_cluster_update=macro_cluster_update,
            verbose=0,
            debug=False,
            plot_graph=False,
        )

        # Contadores
        self.total_samples = 0
        self.anomaly_count = 0

        # Rastreamento para detectar criacao de novos clusters
        self._prev_num_clusters = 0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self._model = EvolvingClustering(
            variance_limit=self.r0,
            decay=self.decay,
            macro_cluster_update=self.macro_cluster_update_freq,
            verbose=0,
            debug=False,
            plot_graph=False,
        )
        self.total_samples = 0
        self.anomaly_count = 0
        self._prev_num_clusters = 0

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Internamente:
        1. Chama update_micro_clusters(x) do EvolvingClustering original
        2. Detecta se novo cluster foi criado (comparando contagem)
        3. Aplica prune_micro_clusters()
        4. Periodicamente atualiza macro_clusters

        Args:
            x: Vetor de features do ponto.

        Returns:
            MicroTEDAResult compativel com o pipeline existente.
        """
        x = np.asarray(x, dtype=np.float64)
        self.total_samples += 1

        # Numero de micro-clusters antes
        num_before = len(self._model.micro_clusters)

        # Processar ponto (update_micro_clusters faz toda a logica interna)
        self._model.update_micro_clusters(x)
        self._model.total_num_samples += 1

        # Pruning
        self._model.prune_micro_clusters()

        # Atualizar macro-clusters periodicamente
        if self.total_samples % self.macro_cluster_update_freq == 0:
            self._model.update_macro_clusters()

        # Numero de micro-clusters depois
        num_after = len(self._model.micro_clusters)
        new_cluster_created = num_after > num_before

        # Anomalia = novo cluster criado (apos min_samples de warmup)
        is_anomaly = new_cluster_created and self.total_samples >= self.min_samples

        if is_anomaly:
            self.anomaly_count += 1

        # Calcular eccentricity/typicality para o ponto
        # Encontra o cluster mais proximo (maior typicality)
        best_cluster_id = -1
        best_typicality = float('-inf')
        cluster_typicalities = {}

        for mc in self._model.micro_clusters:
            if mc["num_samples"] > 0 and mc["variance"] >= 0:
                try:
                    typ = 1.0 - EvolvingClustering.get_normalized_eccentricity(
                        x, mc["num_samples"], mc["mean"], mc["variance"]
                    )
                except Exception:
                    typ = 0.0
                cluster_typicalities[mc["id"]] = typ
                if typ > best_typicality:
                    best_typicality = typ
                    best_cluster_id = mc["id"]

        eccentricity = 1.0 - best_typicality if best_typicality > float('-inf') else 1.0
        typicality = best_typicality if best_typicality > float('-inf') else 0.0

        return MicroTEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            cluster_id=best_cluster_id if not new_cluster_created else -1,
            is_anomaly=is_anomaly,
            num_clusters=num_after,
            sample_count=self.total_samples,
            new_cluster_created=new_cluster_created,
            cluster_typicalities=cluster_typicalities,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        clusters_info = []
        for mc in self._model.micro_clusters:
            clusters_info.append({
                "cluster_id": mc["id"],
                "n": mc["num_samples"],
                "mean": mc["mean"].tolist() if hasattr(mc["mean"], "tolist") else mc["mean"],
                "variance": float(mc["variance"]),
                "density": float(mc["density"]),
                "active": mc["active"],
                "life": float(mc["life"]),
            })

        return {
            "total_samples": self.total_samples,
            "num_clusters": len(self._model.micro_clusters),
            "num_macro_clusters": len(self._model.macro_clusters),
            "num_active_macro_clusters": len(self._model.active_macro_clusters),
            "anomaly_count": self.anomaly_count,
            "r0": self.r0,
            "decay": self.decay,
            "min_samples": self.min_samples,
            "clusters": clusters_info,
        }
