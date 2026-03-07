"""
TEDA Detector v0.1 - Typicality and Eccentricity Data Analytics

Implementacao do algoritmo TEDA (Angelov, 2014) para deteccao de anomalias.

Este modulo implementa o TEDA basico (sem micro-clusters), que:
- Calcula eccentricity e typicality de cada ponto
- Detecta anomalias usando threshold de Chebyshev
- Atualiza estatisticas de forma recursiva (O(1) por ponto)

Arquitetura:
    Flow (vetor de features)
          │
          ▼
    ┌─────────────┐
    │   TEDA      │ ← Atualiza μ, σ² recursivamente
    │  Detector   │
    └─────────────┘
          │
          ▼
    Normal / Anomalia

Referencia:
    Angelov, P. (2014). "Outside the box: an alternative data analytics
    framework." Journal of Automation Mobile Robotics and Intelligent
    Systems, 8(2), pp.29-35.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# TEDA RESULT - Resultado da classificacao
# ============================================================

@dataclass
class TEDAResult:
    """
    Resultado da classificacao TEDA para um ponto.

    Contem metricas de eccentricity, typicality e decisao.
    """
    # Eccentricity: quao "excentrico" (longe dos outros) o ponto e
    # ξ alto = ponto anomalo
    eccentricity: float

    # Typicality: quao "tipico" (proximo ao padrao) o ponto e
    # τ = 1 - ξ
    typicality: float

    # Normalized eccentricity (soma = 1 sobre todos os pontos)
    # ζ = ξ / 2
    normalized_eccentricity: float

    # Threshold usado para classificacao
    threshold: float

    # Decisao: True = anomalia, False = normal
    is_anomaly: bool

    # Numero de amostras processadas ate este ponto
    sample_count: int

    # Distancia ao centro (para debug)
    distance_to_mean: float

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario (para JSON)."""
        return {
            "eccentricity": self.eccentricity,
            "typicality": self.typicality,
            "normalized_eccentricity": self.normalized_eccentricity,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "sample_count": self.sample_count,
            "distance_to_mean": self.distance_to_mean,
        }


# ============================================================
# TEDA DETECTOR - Classe principal
# ============================================================

class TEDADetector:
    """
    Detector de anomalias baseado em TEDA (Angelov 2014).

    TEDA calcula eccentricity (quao diferente) e typicality (quao tipico)
    de cada ponto usando apenas estatisticas recursivas (media e variancia).

    Formulas principais (distancia Euclidiana):

        Atualizacao recursiva da media:
            μ_k = ((k-1)/k) × μ_{k-1} + x_k/k

        Atualizacao recursiva da variancia:
            σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²

        Eccentricity:
            ξ(x_k) = 1/k + ||x_k - μ_k||² / (k × σ²_k)

        Typicality:
            τ(x_k) = 1 - ξ(x_k)

        Threshold de anomalia (Chebyshev):
            Anomalia se: ξ > (m² + 1) / (2k)
            onde m = numero de desvios padrao (default: 3)

    Uso:
        detector = TEDADetector()
        for flow in flows:
            result = detector.update(flow_features)
            if result.is_anomaly:
                print(f"Anomalia detectada! ξ={result.eccentricity:.3f}")

    Comparacao com Java:
        Similar a um detector estateful que mantem media movel.
        Em Java seria uma classe com campos privados para μ, σ², k.
    """

    def __init__(
        self,
        m: float = 3.0,
        min_samples: int = 3,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Inicializa o detector TEDA.

        Args:
            m: Numero de desvios padrao para threshold Chebyshev (default: 3)
               - m=2: ~75% dos dados considerados normais (Chebyshev)
               - m=3: ~89% dos dados considerados normais (Chebyshev)
               - m=3: ~99.7% se distribuicao Gaussiana (3-sigma rule)

            min_samples: Minimo de amostras antes de detectar anomalias.
                        Com k < 3, as estimativas nao sao confiaveis.

            feature_names: Nomes das features (opcional, para debug).
        """
        # --------------------------------------------------------
        # PARAMETROS
        # --------------------------------------------------------
        self.m = m
        self.min_samples = min_samples
        self.feature_names = feature_names

        # --------------------------------------------------------
        # ESTATISTICAS RECURSIVAS
        # --------------------------------------------------------
        # Estas sao as unicas variaveis que precisamos manter!
        # Nao precisamos armazenar todos os pontos (eficiencia de memoria).
        #
        # k: numero de amostras processadas
        # μ: media (vetor de dimensao d)
        # σ²: variancia (escalar, media das variancias por dimensao)
        # --------------------------------------------------------
        self.k: int = 0
        self.mean: Optional[np.ndarray] = None  # μ
        self.variance: float = 0.0  # σ²

        # --------------------------------------------------------
        # ESTATISTICAS DE MONITORAMENTO
        # --------------------------------------------------------
        self.anomaly_count: int = 0
        self.total_eccentricity: float = 0.0

    def reset(self) -> None:
        """Reseta o detector para estado inicial."""
        self.k = 0
        self.mean = None
        self.variance = 0.0
        self.anomaly_count = 0
        self.total_eccentricity = 0.0

    def _calculate_threshold(self) -> float:
        """
        Calcula threshold de anomalia usando desigualdade de Chebyshev.

        Chebyshev: P(|X - μ| ≥ mσ) ≤ 1/m²

        Para TEDA, o threshold e:
            threshold = (m² + 1) / (2k)

        Isso garante que pontos com ξ > threshold sao anomalos
        com confianca relacionada a m.

        Returns:
            Threshold para eccentricity normalizada (ζ)
        """
        if self.k < self.min_samples:
            return float('inf')  # Nao classifica como anomalia

        return (self.m ** 2 + 1) / (2 * self.k)

    def _update_statistics(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Atualiza estatisticas recursivas (μ, σ²) com novo ponto.

        Implementa as formulas de atualizacao do Angelov (2014):

            μ_k = ((k-1)/k) × μ_{k-1} + x_k/k
            σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²

        Args:
            x: Novo ponto (vetor de features)

        Returns:
            Tupla (distancia_ao_centro, variancia_atualizada)
        """
        self.k += 1

        if self.k == 1:
            # Primeiro ponto: inicializa media, variancia = 0
            self.mean = x.copy()
            self.variance = 0.0
            return 0.0, 0.0

        # --------------------------------------------------------
        # ATUALIZACAO RECURSIVA DA MEDIA
        # --------------------------------------------------------
        # μ_k = ((k-1)/k) × μ_{k-1} + x_k/k
        #
        # Equivalente a: μ_k = μ_{k-1} + (x_k - μ_{k-1}) / k
        # Esta forma e mais estavel numericamente.
        # --------------------------------------------------------
        old_mean = self.mean.copy()
        self.mean = old_mean + (x - old_mean) / self.k

        # --------------------------------------------------------
        # DISTANCIA AO CENTRO (apos atualizacao da media)
        # --------------------------------------------------------
        # ||x_k - μ_k||² = soma dos quadrados das diferencas
        # --------------------------------------------------------
        diff = x - self.mean
        distance_squared = np.sum(diff ** 2)

        # --------------------------------------------------------
        # ATUALIZACAO RECURSIVA DA VARIANCIA
        # --------------------------------------------------------
        # A formula do paper e:
        #   σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²
        #
        # NOTA: Para k=2, temos (k-1)=1, entao:
        #   σ²_2 = (1/2) × 0 + 1 × ||x_2 - μ_2||² = ||x_2 - μ_2||²
        #
        # Usamos uma forma numericamente estavel (Welford's algorithm).
        # --------------------------------------------------------
        if self.k == 2:
            # Caso especial k=2: variancia inicial
            self.variance = distance_squared
        else:
            # k >= 3: atualizacao recursiva
            # σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²
            self.variance = ((self.k - 1) / self.k) * self.variance + \
                           (1 / (self.k - 1)) * distance_squared

        return np.sqrt(distance_squared), self.variance

    def _calculate_eccentricity(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Calcula eccentricity de um ponto.

        Formula (Angelov 2014):
            ξ(x_k) = 1/k + ||x_k - μ_k||² / (k × σ²_k)

        Args:
            x: Ponto para calcular eccentricity

        Returns:
            Tupla (eccentricity, distancia_ao_centro)
        """
        if self.k < 1 or self.mean is None:
            return 0.0, 0.0

        # Distancia ao centro
        diff = x - self.mean
        distance_squared = np.sum(diff ** 2)
        distance = np.sqrt(distance_squared)

        # Eccentricity
        if self.k == 1:
            # Primeiro ponto: eccentricity = 1 (maximo)
            eccentricity = 1.0
        elif self.variance <= 0:
            # Variancia zero: todos os pontos sao iguais
            # Se ponto igual a media: eccentricity = 1/k
            # Se ponto diferente: eccentricity alta
            if distance_squared < 1e-10:
                eccentricity = 1.0 / self.k
            else:
                eccentricity = 1.0  # Muito excentrico
        else:
            # Formula normal
            eccentricity = (1.0 / self.k) + \
                          (distance_squared / (self.k * self.variance))

        return eccentricity, distance

    def update(self, x: np.ndarray) -> TEDAResult:
        """
        Processa um novo ponto e retorna resultado da classificacao.

        Este e o metodo principal do detector. Para cada novo ponto:
        1. Atualiza estatisticas (μ, σ²)
        2. Calcula eccentricity e typicality
        3. Compara com threshold
        4. Retorna decisao

        Args:
            x: Vetor de features do ponto (numpy array)

        Returns:
            TEDAResult com eccentricity, typicality e decisao

        Exemplo:
            detector = TEDADetector()
            features = np.array([1.0, 2.0, 3.0])
            result = detector.update(features)
            print(f"Anomalia: {result.is_anomaly}")
        """
        # Converte para numpy array se necessario
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        # Atualiza estatisticas
        distance, _ = self._update_statistics(x)

        # Calcula eccentricity
        eccentricity, distance = self._calculate_eccentricity(x)

        # Typicality = 1 - eccentricity
        typicality = 1.0 - eccentricity

        # Normalized eccentricity (ζ = ξ / 2, soma = 1)
        normalized_eccentricity = eccentricity / 2.0

        # Threshold
        threshold = self._calculate_threshold()

        # Decisao: anomalia se ζ > threshold
        is_anomaly = normalized_eccentricity > threshold

        # Atualiza contadores
        if is_anomaly:
            self.anomaly_count += 1
        self.total_eccentricity += eccentricity

        return TEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            normalized_eccentricity=normalized_eccentricity,
            threshold=threshold,
            is_anomaly=is_anomaly,
            sample_count=self.k,
            distance_to_mean=distance,
        )

    def predict(self, x: np.ndarray) -> TEDAResult:
        """
        Classifica um ponto SEM atualizar as estatisticas.

        Util para:
        - Classificar pontos de teste sem contaminar o modelo
        - Avaliar multiplos pontos com mesmo estado

        Args:
            x: Vetor de features do ponto

        Returns:
            TEDAResult com classificacao
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        # Calcula eccentricity sem atualizar
        eccentricity, distance = self._calculate_eccentricity(x)
        typicality = 1.0 - eccentricity
        normalized_eccentricity = eccentricity / 2.0
        threshold = self._calculate_threshold()
        is_anomaly = normalized_eccentricity > threshold

        return TEDAResult(
            eccentricity=eccentricity,
            typicality=typicality,
            normalized_eccentricity=normalized_eccentricity,
            threshold=threshold,
            is_anomaly=is_anomaly,
            sample_count=self.k,
            distance_to_mean=distance,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatisticas atuais do detector.

        Util para monitoramento e debug.

        Returns:
            Dicionario com estatisticas
        """
        return {
            "sample_count": self.k,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "variance": self.variance,
            "std_dev": np.sqrt(self.variance) if self.variance > 0 else 0.0,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": self.anomaly_count / self.k if self.k > 0 else 0.0,
            "avg_eccentricity": self.total_eccentricity / self.k if self.k > 0 else 0.0,
            "current_threshold": self._calculate_threshold(),
            "m_parameter": self.m,
        }


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def calculate_eccentricity_batch(X: np.ndarray) -> np.ndarray:
    """
    Calcula eccentricity para um batch de pontos.

    Esta funcao calcula a eccentricity de cada ponto em relacao
    ao conjunto completo (nao streaming).

    Util para:
    - Analise inicial de dados
    - Comparacao com versao streaming
    - Visualizacao

    Args:
        X: Matriz de dados (n_samples, n_features)

    Returns:
        Array de eccentricities (n_samples,)

    Exemplo:
        X = np.array([[1, 2], [2, 3], [3, 4], [10, 10]])
        ecc = calculate_eccentricity_batch(X)
        # ecc[3] sera alto (ponto [10,10] e anomalo)
    """
    n = len(X)
    if n < 2:
        return np.ones(n)

    # Calcula media
    mean = np.mean(X, axis=0)

    # Distancia de cada ponto a media
    distances_squared = np.sum((X - mean) ** 2, axis=1)

    # Variancia (media das distancias quadradas)
    variance = np.mean(distances_squared)

    # Eccentricity
    if variance > 0:
        eccentricity = (1.0 / n) + (distances_squared / (n * variance))
    else:
        eccentricity = np.ones(n) / n

    return eccentricity


# ============================================================
# MAIN - Exemplo de uso
# ============================================================

if __name__ == "__main__":
    """
    Exemplo de uso do TEDADetector.

    Demonstra:
    1. Criacao do detector
    2. Processamento de pontos normais
    3. Deteccao de anomalia
    """
    import random

    print("=" * 60)
    print("TEDA Detector v0.1 - Exemplo")
    print("=" * 60)

    # Cria detector
    detector = TEDADetector(m=3.0, min_samples=3)

    # Gera dados normais (cluster em torno de [5, 5])
    np.random.seed(42)
    normal_data = np.random.randn(20, 2) + [5, 5]

    # Adiciona uma anomalia
    anomaly = np.array([15, 15])  # Longe do cluster

    print("\nProcessando pontos normais...")
    for i, point in enumerate(normal_data):
        result = detector.update(point)
        status = "ANOMALIA!" if result.is_anomaly else "normal"
        print(f"  Ponto {i+1}: ξ={result.eccentricity:.4f}, "
              f"τ={result.typicality:.4f}, threshold={result.threshold:.4f} → {status}")

    print("\nProcessando anomalia [15, 15]...")
    result = detector.update(anomaly)
    status = "ANOMALIA!" if result.is_anomaly else "normal"
    print(f"  Anomalia: ξ={result.eccentricity:.4f}, "
          f"τ={result.typicality:.4f}, threshold={result.threshold:.4f} → {status}")

    print("\nEstatisticas finais:")
    stats = detector.get_statistics()
    print(f"  Amostras: {stats['sample_count']}")
    print(f"  Anomalias: {stats['anomaly_count']} ({stats['anomaly_rate']*100:.1f}%)")
    print(f"  Media: [{stats['mean'][0]:.2f}, {stats['mean'][1]:.2f}]")
    print(f"  Desvio padrao: {stats['std_dev']:.4f}")

    print("\n" + "=" * 60)
