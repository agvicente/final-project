# Notas de Implementação: MicroTEDAclus v0.1

**Data:** 2026-01-25
**Autor:** Augusto (com assistência de Claude)
**Capítulo da Dissertação:** 4 - Metodologia / Apêndice

---

## 1. Visão Geral

Este documento registra os desafios, soluções e lições aprendidas durante a implementação do MicroTEDAclus, um algoritmo de clustering evolutivo para detecção de anomalias em streaming.

**Objetivo:** Resolver o problema de contaminação de estatísticas identificado no TEDA básico (v0.1).

**Resultado:** 67 testes passando (36 TEDA básico + 31 MicroTEDAclus).

---

## 2. Problema Original: Contaminação de Estatísticas

### 2.1 Comportamento do TEDA Básico

No TEDA básico (single-center), outliers contaminam as estatísticas globais:

```
Antes do outlier:          Depois do outlier:
μ = [5, 5]                  μ = [13.6, 13.6]  ← DESLOCOU
σ² = 0.25                   σ² = 810          ← EXPLODIU
```

**Consequências:**
1. Segundo outlier pode não ser detectado
2. Pontos normais podem parecer anomalias
3. Detector fica "cego" após poucos outliers

### 2.2 Testes que Documentam o Problema

```python
# streaming/tests/test_teda.py - class TestStatisticsContamination

def test_outlier_contaminates_variance(self, trained_detector):
    """Demonstra aumento de 50x+ na variância após outlier."""
    var_before = trained_detector.variance
    trained_detector.update(np.array([100.0, 100.0]))
    var_after = trained_detector.variance
    assert var_after > var_before * 50  # PASSA - variance explode

def test_second_outlier_may_not_be_detected(self, trained_detector):
    """Após primeiro outlier, segundo pode não ser detectado."""
    result1 = trained_detector.update(np.array([100.0, 100.0]))
    result2 = trained_detector.update(np.array([-100.0, -100.0]))
    # result2.is_anomaly pode ser False (comportamento documentado)
```

---

## 3. Arquitetura do MicroTEDAclus

### 3.1 Estrutura de Classes

```
MicroTEDAclus (Orquestrador)
├── micro_clusters: List[MicroCluster]
├── r0: float (variância mínima)
├── min_samples: int
└── process(x) → MicroTEDAResult

MicroCluster (Estatísticas Isoladas)
├── cluster_id: int
├── n: int (contagem)
├── mean: np.ndarray (μ)
├── variance: float (σ²)
├── dynamic_m() → float
├── chebyshev_threshold() → float
├── chebyshev_accepts(x, r0) → bool
└── update(x) → None
```

### 3.2 Fluxo de Processamento

```
Ponto x chega
     │
     ▼
Para cada micro-cluster:
     │
     ├── Calcular eccentricity com min_variance=r0
     │
     └── chebyshev_accepts(x)?
              │
         ┌────┴────┐
        SIM       NÃO
         │         │
         ▼         │
  Adicionar à      │
  lista de         │
  accepting        │
         │         │
         └────┬────┘
              │
              ▼
     Algum cluster aceita?
              │
         ┌────┴────┐
        SIM       NÃO
         │         │
         ▼         ▼
    Atualizar   Criar novo
    cluster     cluster
    c/ maior    (anomalia)
    typicality
```

---

## 4. Problemas Encontrados e Soluções

### 4.1 Problema: Divisão por Zero na Eccentricity

**Sintoma:** Para clusters com n=1, variance=0, causando divisão por zero.

**Código original:**
```python
def calculate_eccentricity(self, x):
    if self.variance <= 0:
        return 1.0 / self.n  # Retorna 1.0, aceita qualquer ponto!
```

**Problema:** Com eccentricity=1.0 e threshold alto, QUALQUER ponto era aceito.

**Solução Tentada 1 - Retornar infinito:**
```python
if self.variance <= 0:
    if dist_squared < 1e-10:
        return 1.0 / self.n
    else:
        return float('inf')  # Rejeita tudo!
```

**Resultado:** Agora TODOS os pontos eram rejeitados, criando 1 cluster por ponto.

**Solução Final - Variância mínima:**
```python
def calculate_eccentricity(self, x, min_variance=0.001):
    effective_variance = max(self.variance, min_variance)
    return (1.0/self.n) + (dist_squared / (self.n * effective_variance))
```

**Lição:** Usar variância mínima (r0) como "piso" evita divisão por zero sem ser muito restritivo.

---

### 4.2 Problema: Threshold Infinito para n=1

**Sintoma:** Clusters recém-criados (n=1) aceitavam qualquer ponto, mesmo outliers extremos.

**Código original:**
```python
def chebyshev_threshold(self):
    if self.n < 2:
        return float('inf')  # Aceita tudo!
```

**Problema:** Outlier [100, 100] criava cluster, depois [-100, -100] era aceito no mesmo cluster!

**Análise:**
- Com threshold=inf, qualquer zeta <= inf é True
- Segundo outlier era absorvido pelo primeiro
- Não detectado como anomalia separada

**Solução Tentada 1 - Threshold muito restritivo:**
```python
if self.n == 1:
    return (m**2 + 1) / 4  # ≈ 0.5
```

**Resultado:** Muito restritivo! Pontos normais a 0.03 unidades do centro eram rejeitados.

**Solução Final - Threshold permissivo para crescimento:**
```python
if self.n == 1:
    return 13.0  # Equivalente a m=5, aceita até ~5 "desvios"
```

**Lição:** Clusters jovens (n=1) precisam de threshold permissivo para crescer, mas não infinito.

---

### 4.3 Problema: r0 Incompatível com Escala dos Dados

**Sintoma:** Dados não-normalizados criavam muitos clusters (fragmentação excessiva).

**Análise com r0=0.001:**
```python
# Dados com variance ~0.25 (randn * 0.5)
# Para ponto a distância d=0.5 do centro:
ecc = 1/2 + 0.25/(2 * 0.001) = 0.5 + 125 = 125.5
zeta = 62.75
threshold ≈ 0.5

# 62.75 >> 0.5 → REJEITADO (mesmo sendo ponto normal!)
```

**Experimento de diagnóstico:**
```python
# Com r0=0.001: 10 pontos normais → 10 clusters (1 por ponto!)
# Com r0=0.1:   10 pontos normais → 3 clusters (razoável)
```

**Solução:**
```python
# Para dados normalizados (mean=0, std=1): r0=0.001 (como no paper)
# Para dados não-normalizados: r0 proporcional à variance esperada
detector = MicroTEDAclus(r0=0.1)  # Para dados com var ~0.25
```

**Lição:** O parâmetro r0 deve ser calibrado para a escala dos dados. Em produção, normalizar os dados OU usar r0 adaptativo.

---

### 4.4 Problema: Testes com Expectativas Rígidas

**Sintoma:** Testes falhavam mesmo com algoritmo funcionando corretamente.

**Exemplo de teste problemático:**
```python
def test_normal_point_after_outliers_still_normal(self):
    # ...
    assert result.cluster_id == 0  # FALHA: vai pro cluster 2!
```

**Análise:** O algoritmo corretamente atribuía ao cluster com maior typicality, mas o teste esperava cluster específico (0).

**Solução:**
```python
def test_normal_point_after_outliers_still_normal(self):
    normal_cluster_ids = {mc.cluster_id for mc in trained_detector.micro_clusters}
    # ...
    assert result.cluster_id in normal_cluster_ids  # Aceita qualquer normal
```

**Lição:** Testes devem verificar comportamento correto, não valores específicos que dependem de detalhes de implementação.

---

## 5. Parâmetros Críticos e Calibração

### 5.1 Tabela de Parâmetros

| Parâmetro | Valor Default | Impacto | Calibração |
|-----------|---------------|---------|------------|
| `r0` | 0.001 (normalizado) / 0.1 (raw) | Variância mínima para eccentricity | Proporcional à variance esperada dos dados |
| `min_samples` | 3 | Pontos antes de detectar anomalias | Depende do cold-start desejado |
| `threshold(n=1)` | 13.0 | Permissividade de clusters jovens | Fixo (equivalente a m=5) |
| `m(k)` | 1 → 3 (dinâmico) | Restritivo → permissivo | Fórmula do paper |

### 5.2 Fórmula do Threshold Dinâmico m(k)

```python
m(k) = 3 / (1 + e^{-0.007(k-100)})

# Comportamento:
# k=1:    m ≈ 1.01 (muito restritivo, mas usamos threshold=13 fixo)
# k=10:   m ≈ 1.07
# k=100:  m ≈ 1.50
# k=500:  m ≈ 2.86
# k=1000: m ≈ 2.99 (quase 3)
```

### 5.3 Recomendações para Produção

1. **Normalizar dados** antes de alimentar o detector (StandardScaler)
2. **Usar r0=0.001** para dados normalizados
3. **Cold-start:** Primeiros `min_samples` pontos nunca são anomalias
4. **Monitorar número de clusters:** Se explodir, r0 está muito pequeno

---

## 6. Comparação: TEDA Básico vs MicroTEDAclus

### 6.1 Teste de Contaminação

```python
def test_contamination_comparison(self):
    # Setup
    basic_teda = TEDADetector(m=3.0, min_samples=3)
    micro_teda = MicroTEDAclus(r0=0.1, min_samples=3)

    # Treinar com 10 pontos normais
    for point in normal_data:
        basic_teda.update(point)
        micro_teda.process(point)

    # Guardar variance antes
    basic_var_before = basic_teda.variance
    micro_var_before = micro_teda.micro_clusters[0].variance

    # Adicionar outlier
    basic_teda.update(np.array([100.0, 100.0]))
    micro_teda.process(np.array([100.0, 100.0]))

    # TEDA Básico: variance explode
    assert basic_teda.variance > basic_var_before * 50  # PASSA

    # MicroTEDAclus: cluster original inalterado!
    assert micro_teda.micro_clusters[0].variance == micro_var_before  # PASSA
```

### 6.2 Tabela Comparativa

| Cenário | TEDA Básico | MicroTEDAclus |
|---------|-------------|---------------|
| 1º outlier | ✓ Detecta, contamina | ✓ Detecta, novo cluster |
| 2º outlier | ✗ Pode não detectar | ✓ Detecta, outro cluster |
| Normal após outliers | ✗ Pode ser anomalia | ✓ Normal (cluster preservado) |
| Memória | O(1) | O(k) onde k = num clusters |
| Complexidade | O(1)/ponto | O(k)/ponto |

---

## 7. Estrutura de Testes

### 7.1 Organização dos Testes

```
tests/
├── test_teda.py           (36 testes - TEDA básico)
│   ├── TestInitialization
│   ├── TestFirstSample
│   ├── TestStatisticsUpdate
│   ├── TestEccentricity
│   ├── TestTypicality
│   ├── TestThreshold
│   ├── TestAnomalyDetection
│   ├── TestPredictWithoutUpdate
│   ├── TestEdgeCases
│   ├── TestBatchEccentricity
│   ├── TestStatisticsContamination  ← Documenta limitação
│   └── TestTEDAResult
│
└── test_micro_teda.py     (31 testes - MicroTEDAclus)
    ├── TestMicroClusterBasic
    ├── TestDynamicThreshold
    ├── TestChebyshevAcceptance
    ├── TestMicroClusterTypicality
    ├── TestMicroTEDAclusBasic
    ├── TestContaminationResistance  ← Valida solução
    ├── TestClusterAssignment
    ├── TestPredictWithoutUpdate
    ├── TestEdgeCases
    ├── TestMicroTEDAResult
    └── TestComparisonWithBasicTEDA  ← Comparação direta
```

### 7.2 Fixtures Importantes

```python
@pytest.fixture
def trained_detector():
    """MicroTEDAclus treinado com 10 pontos normais."""
    detector = MicroTEDAclus(r0=0.1, min_samples=3)
    np.random.seed(42)
    normal_data = np.random.randn(10, 2) * 0.5 + [5, 5]
    for point in normal_data:
        detector.process(point)
    return detector
    # Resultado: 3 clusters com n=3, n=3, n=4
```

---

## 8. Código Final Relevante

### 8.1 Cálculo de Eccentricity com Variância Mínima

```python
def calculate_eccentricity(self, x: np.ndarray, min_variance: float = 0.001) -> float:
    if self.n == 0:
        return float('inf')

    diff = x - self.mean
    dist_squared = np.sum(diff ** 2)

    # Usa variância mínima para clusters jovens
    effective_variance = max(self.variance, min_variance)

    return (1.0 / self.n) + (dist_squared / (self.n * effective_variance))
```

### 8.2 Threshold com Tratamento para n=1

```python
def chebyshev_threshold(self) -> float:
    m = self.dynamic_m()

    if self.n == 1:
        # Threshold permissivo para crescimento inicial
        return 13.0  # Equivalente a m=5

    return (m ** 2 + 1) / (2 * self.n)
```

### 8.3 Lógica Principal de Processamento

```python
def process(self, x: np.ndarray) -> MicroTEDAResult:
    x = np.asarray(x, dtype=np.float64)
    self.total_samples += 1

    # Primeiro ponto: criar cluster
    if not self.micro_clusters:
        cluster = self._create_micro_cluster(x)
        return MicroTEDAResult(...)

    # Encontrar clusters que aceitam
    accepting_clusters = self._find_accepting_clusters(x)

    if accepting_clusters:
        # Atualizar cluster com maior typicality
        best_cluster = max(accepting_clusters,
                          key=lambda mc: mc.calculate_typicality(x))
        best_cluster.update(x)
        return MicroTEDAResult(is_anomaly=False, ...)
    else:
        # Criar novo cluster (anomalia)
        new_cluster = self._create_micro_cluster(x)
        return MicroTEDAResult(is_anomaly=True, ...)
```

---

## 9. Lições Aprendidas

### 9.1 Sobre Implementação

1. **Variância zero é problemática** - Sempre usar variância mínima como piso
2. **Threshold infinito quebra a lógica** - Clusters jovens precisam de threshold finito
3. **Parâmetros dependem da escala** - r0 deve ser calibrado para os dados

### 9.2 Sobre Testes

1. **Testes devem ser flexíveis** - Verificar comportamento, não valores específicos
2. **Fixtures com seed fixo** - Reprodutibilidade é essencial para debugging
3. **Testar edge cases** - n=1, variance=0, primeiro ponto, etc.

### 9.3 Sobre o Algoritmo

1. **Trade-off fragmentação vs absorção** - Threshold muito baixo fragmenta, muito alto absorve outliers
2. **Clusters jovens são instáveis** - Precisam de tratamento especial
3. **Comparação direta é valiosa** - Testar lado a lado com baseline

---

## 10. Próximos Passos

1. **Integrar com StreamingDetector** - Substituir TEDADetector por MicroTEDAclus
2. **Testar com dados reais** - CICIoT2023 via Kafka
3. **Comparar com implementação oficial** - GitHub do paper
4. **Implementar merge/split de clusters** - Para clusters que crescem demais

---

## 11. Referências

- Maia et al. (2020) - "Evolving clustering algorithm based on mixture of typicalities"
- Angelov (2014) - "Outside the box: an alternative data analytics framework"
- `docs/paper-summaries/maia-2020-microtedaclus.md`
- `docs/theory/teda-contamination-problem.md`

---

**Este documento deve ser atualizado conforme o MicroTEDAclus evolui.**
