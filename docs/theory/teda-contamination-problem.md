# Problema de Contaminação de Estatísticas no TEDA Básico

**Criado:** 2026-01-21
**Capítulo da Dissertação:** 2 - Fundamentação Teórica / 4 - Metodologia

---

## 1. Descrição do Problema

O TEDA básico (single-center) possui uma vulnerabilidade fundamental: **outliers contaminam as estatísticas globais** (média μ e variância σ²), potencialmente cegando o detector para anomalias subsequentes.

### Sequência de Eventos

```
1. Detector treinado com dados normais
   μ = [5, 5], σ² = 0.25, k = 10

2. Outlier extremo [100, 100] chega
   → Detectado como anomalia ✓
   → ATUALIZA estatísticas: μ = [13.6, 13.6], σ² = 810

3. Segundo outlier [-100, -100] chega
   → Variância alta → threshold alto
   → Pode NÃO ser detectado como anomalia ✗

4. Ponto normal [5, 5] chega
   → Longe do novo centro [13.6, 13.6]
   → Pode ser detectado como anomalia! ✗
```

---

## 2. Demonstração Matemática

### Antes do Outlier
```
μ₁₀ = [5.0, 5.0]
σ²₁₀ = 0.25
k = 10
threshold = (m² + 1) / (2k) = (9 + 1) / 20 = 0.5
```

### Após Outlier [100, 100]
```
μ₁₁ = (10/11) × [5, 5] + (1/11) × [100, 100]
     = [4.55, 4.55] + [9.09, 9.09]
     = [13.64, 13.64]

δ = [100, 100] - [13.64, 13.64] = [86.36, 86.36]
||δ||² = 14920

σ²₁₁ = (10/11) × 0.25 + (1/10) × 14920
      = 0.227 + 1492
      ≈ 1492 (aumento de ~6000x!)
```

### Segundo Outlier [-100, -100]
```
k = 12
threshold = 10 / 24 ≈ 0.42

Para [-100, -100]:
d = [-100, -100] - [13.64, 13.64] = [-113.64, -113.64]
||d||² = 25828

ξ = 1/12 + 25828 / (12 × ~1500) ≈ 0.083 + 1.44 ≈ 1.52

Mas com a variância ainda maior após k=11...
O ponto pode passar pelo threshold.
```

---

## 3. Impactos na Detecção de Intrusão

### 3.1 Cenário: DDoS Ataque

| Fase | Evento | TEDA Básico |
|------|--------|-------------|
| 1 | Tráfego normal (1000 pacotes) | Aprende padrão normal |
| 2 | Início DDoS (primeiros 10 pacotes) | Detecta anomalias ✓ |
| 3 | DDoS continua (próximos 100 pacotes) | Estatísticas contaminadas |
| 4 | DDoS normalizado | Novos ataques não detectados ✗ |
| 5 | Tráfego volta ao normal | Detectado como anomalia! ✗ |

### 3.2 Cenário: Poisoning Attack

Um atacante pode deliberadamente:
1. Enviar padrões extremos espaçados
2. Contaminar gradualmente as estatísticas
3. Tornar o detector tolerante a ataques futuros

---

## 4. Como MicroTEDAclus Resolve

### 4.1 Rejeição ao Invés de Absorção

```python
# TEDA Básico (vulnerável)
def update(x):
    self.k += 1
    self.mean = update_mean(x)      # SEMPRE atualiza
    self.variance = update_variance(x)

# MicroTEDAclus (protegido)
def process(x):
    accepted = False
    for mc in micro_clusters:
        if mc.chebyshev_accepts(x):
            mc.update(x)            # Só atualiza se aceito
            accepted = True

    if not accepted:
        create_new_cluster(x)       # Outlier vira cluster separado
```

### 4.2 Isolamento de Estatísticas

```
Antes:                          Depois:
┌─────────────┐                 ┌─────────────┐  ┌─────────────┐
│ Normal      │                 │ Normal      │  │ Outliers    │
│ μ=[5,5]     │    Outliers     │ μ=[5,5]     │  │ μ=[100,100] │
│ σ²=0.25     │    ──────→      │ σ²=0.25     │  │ σ²=pequeno  │
│ k=10        │                 │ k=10        │  │ k=1         │
└─────────────┘                 └─────────────┘  └─────────────┘
                                 (inalterado!)    (cluster novo)
```

### 4.3 Threshold Dinâmico m(k)

```
m(k) = 3 / (1 + e^{-0.007(k-100)})

- Clusters jovens (k pequeno): m ≈ 1 → restritivo, rejeita mais
- Clusters maduros (k grande): m ≈ 3 → permissivo, confiante

Resultado: Clusters não são "envenenados" quando jovens.
```

---

## 5. Comportamento Comparativo

| Situação | TEDA Básico | MicroTEDAclus |
|----------|-------------|---------------|
| Primeiro outlier | ✓ Detecta, contamina stats | ✓ Detecta, novo cluster |
| Segundo outlier (similar) | ✗ Pode não detectar | ✓ Atualiza cluster de outliers |
| Terceiro outlier (diferente) | ✗ Provavelmente não detecta | ✓ Detecta (cluster normal intacto) |
| Normal após ataques | ✗ Pode parecer anomalia | ✓ Normal (cluster preservado) |
| Novo tipo de ataque | ✗ Tolerante demais | ✓ Detecta (compara com clusters limpos) |

---

## 6. Testes Implementados

Arquivo: `streaming/tests/test_teda.py`
Classe: `TestStatisticsContamination`

```python
def test_outlier_contaminates_variance(self, trained_detector):
    """Demonstra aumento de 50x+ na variância após outlier."""

def test_second_outlier_may_not_be_detected(self, trained_detector):
    """Documenta comportamento de falso negativo."""

def test_normal_point_after_contamination(self, trained_detector):
    """Demonstra inversão após contaminação."""
```

---

## 7. Implicações para a Implementação

### 7.1 TEDA v0.1 (Atual)
- ⚠️ Vulnerável a contaminação
- ✓ Útil para cold start e aprendizado inicial
- ✓ Base para entender o algoritmo
- ⚠️ Não usar em produção sem MicroTEDAclus

### 7.2 TEDA v0.2 (MicroTEDAclus) - Próximo
- ✓ Múltiplos micro-clusters
- ✓ Rejeição via Chebyshev → novos clusters
- ✓ Threshold dinâmico m(k)
- ✓ Robusto a contaminação

### 7.3 Parâmetros Críticos

| Parâmetro | Valor | Impacto na Contaminação |
|-----------|-------|------------------------|
| m (fixo) | 3 | Muito permissivo para k pequeno |
| m(k) dinâmico | 1→3 | Protege clusters jovens |
| r₀ (variância mínima) | 0.001 | Evita clusters "gigantes" |

---

## 8. Conexão com a Dissertação

### Capítulo 2 - Fundamentação Teórica
- Explicar TEDA básico e sua limitação
- Apresentar MicroTEDAclus como solução
- Justificar escolha de múltiplos clusters

### Capítulo 4 - Metodologia
- Documentar por que TEDA básico é insuficiente para IDS
- Descrever implementação de MicroTEDAclus
- Testes de robustez contra contaminação

### Capítulo 5 - Resultados
- Comparar TEDA básico vs MicroTEDAclus sob ataque
- Métricas de falsos negativos após contaminação
- Tempo de recuperação

---

## 9. Referências

- Angelov, P. (2014). "Outside the box: an alternative data analytics framework."
- Maia et al. (2020). "Evolving clustering algorithm based on mixture of typicalities for stream data mining."
- `docs/paper-summaries/angelov-2014-teda.md`
- `docs/paper-summaries/maia-2020-microtedaclus.md`

---

**Este documento será referenciado na dissertação para justificar a escolha de MicroTEDAclus sobre TEDA básico.**
