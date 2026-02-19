"""
MicroTEDAclus v0.1 - Evolving Clustering with Multiple Micro-clusters

Implementacao do algoritmo MicroTEDAclus (Maia et al., 2020) para deteccao
de anomalias em streaming com resistencia a contaminacao de estatisticas.

Este modulo implementa:
- Multiplos micro-clusters com estatisticas isoladas
- Threshold dinamico m(k) que cresce com o tamanho do cluster
- Rejeicao via Chebyshev → criacao de novos clusters
- Mixture of typicalities para atribuicao de clusters

Arquitetura:
    Flow (vetor de features)
          │
          ▼
    ┌─────────────────────────────────────┐
    │         MicroTEDAclus               │
    │  ┌───────┐ ┌───────┐ ┌───────┐     │
    │  │  MC₁  │ │  MC₂  │ │  MC₃  │ ... │
    │  │ μ,σ²,n│ │ μ,σ²,n│ │ μ,σ²,n│     │
    │  └───────┘ └───────┘ └───────┘     │
    │       │         │         │         │
    │       └─────────┴─────────┘         │
    │              │                      │
    │     Chebyshev Test (aceita/rejeita) │
    └─────────────────────────────────────┘
          │
          ▼
    (cluster_id, is_anomaly, typicality)

Vantagens sobre TEDA basico:
- Outliers NAO contaminam clusters existentes
- Cada cluster mantem estatisticas isoladas
- Adapta-se a concept drift criando novos clusters

Referencia:
    Maia, J. et al. (2020). "Evolving clustering algorithm based on
    mixture of typicalities for stream data mining." Future Generation
    Computer Systems, 106, pp.672-684.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================
# MICRO TEDA RESULT
# ============================================================

@dataclass
class MicroTEDAResult:
    """
    Resultado da classificacao MicroTEDAclus para um ponto.

    Diferente do TEDAResult basico, inclui informacoes sobre
    qual micro-cluster o ponto foi atribuido.
    """
    # Eccentricity em relacao ao cluster atribuido
    eccentricity: float

    # Typicality em relacao ao cluster atribuido
    typicality: float

    # ID do micro-cluster atribuido (-1 se novo cluster criado)
    cluster_id: int

    # Decisao: True = anomalia (novo cluster criado), False = normal
    is_anomaly: bool

    # Numero total de micro-clusters apos processamento
    num_clusters: int

    # Numero de amostras processadas
    sample_count: int

    # Se um novo cluster foi criado para este ponto
    new_cluster_created: bool

    # Tipicalidades para todos os clusters (mixture of typicalities)
    cluster_typicalities: Optional[Dict[int, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario (para JSON)."""
        return {
            "eccentricity": self.eccentricity,
            "typicality": self.typicality,
            "cluster_id": self.cluster_id,
            "is_anomaly": self.is_anomaly,
            "num_clusters": self.num_clusters,
            "sample_count": self.sample_count,
            "new_cluster_created": self.new_cluster_created,
            "cluster_typicalities": self.cluster_typicalities,
        }


# ============================================================
# MICRO CLUSTER
# ============================================================

class MicroCluster:
    """
    Um micro-cluster individual com estatisticas isoladas.

    Cada micro-cluster mantem sua propria media (μ), variancia (σ²),
    e contagem de amostras (n). Isso evita que outliers contaminem
    as estatisticas de clusters existentes.

    Atributos:
        cluster_id: Identificador unico do cluster
        n: Numero de amostras no cluster
        mean: Centro do cluster (μ)
        variance: Variancia do cluster (σ²)
        density: Densidade do cluster (para mixture of typicalities)
    """

    def __init__(self, cluster_id: int, initial_point: np.ndarray):
        """
        Inicializa micro-cluster com um ponto.

        Args:
            cluster_id: ID unico do cluster
            initial_point: Primeiro ponto do cluster
        """
        self.cluster_id = cluster_id
        self.n = 1
        self.mean = initial_point.copy().astype(np.float64)
        self.variance = 0.0
        self.density = 1.0  # Densidade inicial

        # Estatisticas adicionais
        self._var_sum = 0.0  # Para calculo incremental de variancia

    def dynamic_m(self) -> float:
        """
        Calcula threshold dinamico m(k) baseado no tamanho do cluster.

        Formula: m(k) = 3 / (1 + e^{-0.007(k-100)})

        Comportamento:
        - k pequeno (cluster jovem): m ≈ 1 (restritivo)
        - k grande (cluster maduro): m → 3 (permissivo)

        Isso protege clusters jovens de serem contaminados por outliers.

        Returns:
            Valor de m para usar no teste de Chebyshev
        """
        return 3.0 / (1.0 + math.exp(-0.007 * (self.n - 100)))

    def calculate_eccentricity(self, x: np.ndarray, min_variance: float = 0.001) -> float:
        """
        Calcula eccentricity de um ponto em relacao a este cluster.

        Formula: ξ(x) = 1/n + ||x - μ||² / (n × σ²)

        Para clusters jovens (variance < min_variance), usa min_variance
        como piso para evitar rejeicao excessiva.

        Args:
            x: Ponto para calcular eccentricity
            min_variance: Variancia minima para calculo (default: 0.001)

        Returns:
            Eccentricity do ponto (0 = tipico, >1 = outlier)
        """
        if self.n == 0:
            return float('inf')

        diff = x - self.mean
        dist_squared = np.sum(diff ** 2)

        # Usa variancia minima para clusters jovens
        effective_variance = max(self.variance, min_variance)

        return (1.0 / self.n) + (dist_squared / (self.n * effective_variance))

    def calculate_normalized_eccentricity(self, x: np.ndarray, min_variance: float = 0.001) -> float:
        """
        Calcula eccentricity normalizada (ζ = ξ/2).

        A normalizacao garante que a soma sobre todos os pontos = 1.

        Args:
            x: Ponto para calcular
            min_variance: Variancia minima para calculo

        Returns:
            Eccentricity normalizada
        """
        return self.calculate_eccentricity(x, min_variance) / 2.0

    def calculate_typicality(self, x: np.ndarray) -> float:
        """
        Calcula typicality de um ponto em relacao a este cluster.

        Formula: τ(x) = 1 - ξ(x)

        Args:
            x: Ponto para calcular typicality

        Returns:
            Typicality do ponto (1 = muito tipico, <0 = outlier)
        """
        return 1.0 - self.calculate_eccentricity(x)

    def calculate_normalized_typicality(self, x: np.ndarray) -> float:
        """
        Calcula typicality normalizada (t = τ/(n-2)).

        Args:
            x: Ponto para calcular

        Returns:
            Typicality normalizada
        """
        if self.n <= 2:
            return self.calculate_typicality(x)
        return self.calculate_typicality(x) / (self.n - 2)

    def chebyshev_threshold(self) -> float:
        """
        Calcula threshold de Chebyshev para este cluster.

        Formula: threshold = (m(k)² + 1) / (2n)

        Para n=1, usa threshold intermediario para permitir crescimento
        inicial do cluster enquanto ainda rejeita outliers extremos.

        Returns:
            Threshold para eccentricity normalizada
        """
        m = self.dynamic_m()

        if self.n == 1:
            # Para n=1, usar threshold permissivo para crescimento inicial
            # Equivalente a m=5 (aceita ate ~5 "desvios padrao" do centro)
            # threshold = (5² + 1) / 2 = 13
            # Isso permite que pontos razoavelmente proximos sejam aceitos
            return 13.0

        return (m ** 2 + 1) / (2 * self.n)

    def chebyshev_accepts(self, x: np.ndarray, r0: float = 0.001) -> bool:
        """
        Testa se o ponto e aceito por este cluster via Chebyshev.

        Condicao de aceitacao:
        - ζ(x) ≤ threshold
        - Para n=2: tambem verifica σ² < r0

        Args:
            x: Ponto para testar
            r0: Limite de variancia para n=2 (default: 0.001)

        Returns:
            True se o ponto e aceito, False se e outlier
        """
        # Usa r0 como variancia minima para eccentricity
        zeta = self.calculate_normalized_eccentricity(x, min_variance=r0)
        threshold = self.chebyshev_threshold()

        if self.n == 2:
            # Condicao especial para clusters com apenas 2 pontos
            # Rejeita se e outlier E variancia ja e grande
            return not (zeta > threshold and self.variance >= r0)

        return zeta <= threshold

    def update(self, x: np.ndarray) -> None:
        """
        Atualiza estatisticas do cluster com novo ponto.

        Usa formulas recursivas de Welford para O(1) por atualizacao:
        - μ_k = ((k-1)/k) × μ_{k-1} + x_k/k
        - σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²

        Args:
            x: Novo ponto para adicionar ao cluster
        """
        x = np.asarray(x, dtype=np.float64)
        self.n += 1

        if self.n == 1:
            # Primeiro ponto (nao deveria acontecer, mas por seguranca)
            self.mean = x.copy()
            self.variance = 0.0
            return

        # Atualiza media (Welford)
        delta = x - self.mean
        self.mean = self.mean + delta / self.n

        # Atualiza variancia (Welford)
        delta2 = x - self.mean
        self._var_sum += np.dot(delta, delta2)

        if self.n > 1:
            self.variance = self._var_sum / (self.n - 1)

        # Atualiza densidade
        self._update_density(x)

    def _update_density(self, x: np.ndarray) -> None:
        """Atualiza densidade do cluster."""
        if self.n > 0:
            zeta = self.calculate_normalized_eccentricity(x)
            # Densidade e a eccentricity normalizada do primeiro ponto dividida por n
            # Simplificacao: usamos a media das eccentricities
            self.density = zeta / self.n if self.n > 0 else 1.0

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do cluster."""
        return {
            "cluster_id": self.cluster_id,
            "n": self.n,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "variance": self.variance,
            "density": self.density,
            "dynamic_m": self.dynamic_m(),
            "threshold": self.chebyshev_threshold(),
        }


# ============================================================
# MICRO TEDA CLUS - Orquestrador
# ============================================================

class MicroTEDAclus:
    """
    Algoritmo MicroTEDAclus para clustering evolutivo.

    Gerencia multiplos micro-clusters, decidindo quando:
    - Atualizar clusters existentes (ponto aceito)
    - Criar novos clusters (ponto rejeitado por todos)

    Isso evita contaminacao de estatisticas por outliers.

    Atributos:
        micro_clusters: Lista de micro-clusters ativos
        r0: Limite de variancia para teste com n=2
        min_samples: Minimo de amostras antes de classificar anomalias
    """

    def __init__(
        self,
        r0: float = 0.001,
        min_samples: int = 3,
    ):
        """
        Inicializa MicroTEDAclus.

        Args:
            r0: Limite de variancia para clusters com n=2 (default: 0.001)
            min_samples: Amostras minimas antes de detectar anomalias
        """
        self.r0 = r0
        self.min_samples = min_samples

        # Lista de micro-clusters
        self.micro_clusters: List[MicroCluster] = []

        # Contador para IDs de clusters
        self._next_cluster_id = 0

        # Estatisticas globais
        self.total_samples = 0
        self.anomaly_count = 0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self.micro_clusters = []
        self._next_cluster_id = 0
        self.total_samples = 0
        self.anomaly_count = 0

    def _create_micro_cluster(self, x: np.ndarray) -> MicroCluster:
        """
        Cria novo micro-cluster com o ponto dado.

        Args:
            x: Ponto inicial do cluster

        Returns:
            Novo MicroCluster
        """
        cluster = MicroCluster(self._next_cluster_id, x)
        self._next_cluster_id += 1
        self.micro_clusters.append(cluster)
        return cluster

    def _find_accepting_clusters(self, x: np.ndarray) -> List[MicroCluster]:
        """
        Encontra todos os clusters que aceitam o ponto.

        Args:
            x: Ponto para testar

        Returns:
            Lista de clusters que aceitam o ponto
        """
        accepting = []
        for mc in self.micro_clusters:
            if mc.chebyshev_accepts(x, self.r0):
                accepting.append(mc)
        return accepting

    def _calculate_cluster_typicalities(
        self, x: np.ndarray
    ) -> Dict[int, float]:
        """
        Calcula typicality do ponto para cada cluster.

        Args:
            x: Ponto para calcular

        Returns:
            Dicionario {cluster_id: typicality}
        """
        typicalities = {}
        for mc in self.micro_clusters:
            typicalities[mc.cluster_id] = mc.calculate_typicality(x)
        return typicalities

    def _find_best_cluster(self, x: np.ndarray) -> Optional[MicroCluster]:
        """
        Encontra o cluster com maior typicality para o ponto.

        Usado para atribuicao quando multiplos clusters aceitam.

        Args:
            x: Ponto para atribuir

        Returns:
            Cluster com maior typicality, ou None se nao houver clusters
        """
        if not self.micro_clusters:
            return None

        best_cluster = None
        best_typicality = float('-inf')

        for mc in self.micro_clusters:
            typ = mc.calculate_typicality(x)
            if typ > best_typicality:
                best_typicality = typ
                best_cluster = mc

        return best_cluster

    def process(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Processa um novo ponto no stream.

        Algoritmo:
        1. Se nao ha clusters, cria o primeiro
        2. Encontra clusters que aceitam o ponto (Chebyshev)
        3. Se algum aceita: atualiza o de maior typicality
        4. Se nenhum aceita: cria novo cluster (anomalia)

        Args:
            x: Vetor de features do ponto

        Returns:
            MicroTEDAResult com informacoes da classificacao
        """
        x = np.asarray(x, dtype=np.float64)
        self.total_samples += 1

        # Caso especial: primeiro ponto
        if not self.micro_clusters:
            cluster = self._create_micro_cluster(x)
            return MicroTEDAResult(
                eccentricity=1.0,
                typicality=0.0,
                cluster_id=cluster.cluster_id,
                is_anomaly=False,  # Primeiro ponto nunca e anomalia
                num_clusters=1,
                sample_count=self.total_samples,
                new_cluster_created=True,
                cluster_typicalities={cluster.cluster_id: 0.0},
            )

        # Encontra clusters que aceitam o ponto
        accepting_clusters = self._find_accepting_clusters(x)

        # Calcula typicalities para todos os clusters
        all_typicalities = self._calculate_cluster_typicalities(x)

        if accepting_clusters:
            # Ponto aceito por pelo menos um cluster
            # Atualiza o cluster com maior typicality
            best_cluster = max(
                accepting_clusters,
                key=lambda mc: mc.calculate_typicality(x)
            )

            eccentricity = best_cluster.calculate_eccentricity(x)
            typicality = best_cluster.calculate_typicality(x)

            # Atualiza estatisticas do cluster
            best_cluster.update(x)

            return MicroTEDAResult(
                eccentricity=eccentricity,
                typicality=typicality,
                cluster_id=best_cluster.cluster_id,
                is_anomaly=False,
                num_clusters=len(self.micro_clusters),
                sample_count=self.total_samples,
                new_cluster_created=False,
                cluster_typicalities=all_typicalities,
            )
        else:
            # Ponto rejeitado por todos os clusters → novo cluster
            new_cluster = self._create_micro_cluster(x)

            # Determina se e anomalia (apos min_samples)
            is_anomaly = self.total_samples >= self.min_samples

            if is_anomaly:
                self.anomaly_count += 1

            # Atualiza typicalities com novo cluster
            all_typicalities[new_cluster.cluster_id] = 0.0

            return MicroTEDAResult(
                eccentricity=1.0,  # Primeiro ponto do cluster
                typicality=0.0,
                cluster_id=new_cluster.cluster_id,
                is_anomaly=is_anomaly,
                num_clusters=len(self.micro_clusters),
                sample_count=self.total_samples,
                new_cluster_created=True,
                cluster_typicalities=all_typicalities,
            )

    def predict(self, x: np.ndarray) -> MicroTEDAResult:
        """
        Classifica um ponto sem atualizar estatisticas.

        Util para avaliacao ou quando nao queremos que o ponto
        afete os clusters.

        Args:
            x: Vetor de features do ponto

        Returns:
            MicroTEDAResult com classificacao
        """
        x = np.asarray(x, dtype=np.float64)

        if not self.micro_clusters:
            return MicroTEDAResult(
                eccentricity=1.0,
                typicality=0.0,
                cluster_id=-1,
                is_anomaly=True,
                num_clusters=0,
                sample_count=self.total_samples,
                new_cluster_created=False,
                cluster_typicalities={},
            )

        # Encontra clusters que aceitariam o ponto
        accepting_clusters = self._find_accepting_clusters(x)
        all_typicalities = self._calculate_cluster_typicalities(x)

        if accepting_clusters:
            # Seria atribuido ao cluster com maior typicality
            best_cluster = max(
                accepting_clusters,
                key=lambda mc: mc.calculate_typicality(x)
            )

            return MicroTEDAResult(
                eccentricity=best_cluster.calculate_eccentricity(x),
                typicality=best_cluster.calculate_typicality(x),
                cluster_id=best_cluster.cluster_id,
                is_anomaly=False,
                num_clusters=len(self.micro_clusters),
                sample_count=self.total_samples,
                new_cluster_created=False,
                cluster_typicalities=all_typicalities,
            )
        else:
            # Seria criado novo cluster (anomalia)
            return MicroTEDAResult(
                eccentricity=1.0,
                typicality=0.0,
                cluster_id=-1,
                is_anomaly=self.total_samples >= self.min_samples,
                num_clusters=len(self.micro_clusters),
                sample_count=self.total_samples,
                new_cluster_created=False,
                cluster_typicalities=all_typicalities,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas do detector."""
        return {
            "total_samples": self.total_samples,
            "num_clusters": len(self.micro_clusters),
            "anomaly_count": self.anomaly_count,
            "r0": self.r0,
            "min_samples": self.min_samples,
            "clusters": [mc.get_statistics() for mc in self.micro_clusters],
        }

    def get_cluster_centers(self) -> np.ndarray:
        """Retorna centros de todos os clusters como array."""
        if not self.micro_clusters:
            return np.array([])
        return np.array([mc.mean for mc in self.micro_clusters])

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Retorna tamanho de cada cluster."""
        return {mc.cluster_id: mc.n for mc in self.micro_clusters}
