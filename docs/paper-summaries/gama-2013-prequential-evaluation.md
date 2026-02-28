# Fichamento: On Evaluating Stream Learning Algorithms

**Referência completa:**
Gama, J., Sebastião, R., & Rodrigues, P. P. (2013). On evaluating stream learning algorithms. *Machine Learning*, 90, 317-346.

**Data de leitura:** 2026-02-23
**Área:** Machine Learning (Stream Learning / Concept Drift)
**DOI:** [10.1007/s10994-012-5320-9](https://doi.org/10.1007/s10994-012-5320-9)

---

## 1. Objetivo do Artigo

Os autores se propõem a **estabelecer uma metodologia rigorosa para avaliar algoritmos de aprendizado em streams**, argumentando que métodos tradicionais de avaliação de ML (como holdout e cross-validation) não são apropriados para dados não estacionários com concept drift.

**Problema central:** Como avaliar um algoritmo que aprende de forma contínua quando a distribuição dos dados muda ao longo do tempo?

**Solução proposta:** Avaliação prequential (test-then-train) com múltiplos estimadores de erro adaptados a diferentes características do stream.

---

## 2. Metodologia

### 2.1 Prequential Evaluation (Test-Then-Train)

**Princípio:** Para cada novo exemplo no stream:
1. **Testar** - Usar o modelo atual para predizer o label
2. **Avaliar** - Comparar predição com label verdadeiro (ground truth)
3. **Treinar** - Atualizar o modelo com o novo exemplo

**Vantagens:**
- Uso máximo dos dados (não precisa reservar holdout)
- Avaliação contínua reflete performance em produção
- Detecta mudanças na performance ao longo do tempo

**Desvantagens:**
- Assume que ground truth está disponível imediatamente (delay pode ser problemático)
- Ordem dos dados importa (não é reproduzível como k-fold)

### 2.2 Estimadores de Erro Prequential

Os autores propõem **três estimadores complementares**:

#### 2.2.1 Erro Acumulado (Holdout Prequential)
```
P(n) = (1/n) * Σ(i=1 to n) e_i
```
- **Uso:** Baseline de performance histórica
- **Limitação:** Dá peso igual a todos os exemplos (não adapta a drift)
- **Quando usar:** Streams estacionários ou para comparar algoritmos

#### 2.2.2 Erro em Janela Deslizante (Sliding Window)
```
P_w(n) = (1/w) * Σ(i=n-w+1 to n) e_i
```
- **Uso:** Performance recente (últimos `w` exemplos)
- **Vantagem:** Adapta rapidamente a mudanças abruptas
- **Limitação:** Esquece completamente dados antigos (descontinuidade)
- **Quando usar:** Concept drift abrupto, w ≈ 1000-5000

#### 2.2.3 Erro com Fading Factor (Esquecimento Exponencial)
```
P_α(n) = α * e_n + (1 - α) * P_α(n-1)
```
- **Uso:** Performance recente com esquecimento suave
- **Vantagem:** Sem descontinuidade, adapta a drift gradual
- **Parâmetro:** α (0 < α ≤ 1)
  - α pequeno (0.001-0.01): drift lento, mais estável
  - α grande (0.1-0.5): drift rápido, mais reativo
- **Quando usar:** Concept drift gradual (RECOMENDADO para IoT)

**Recomendação dos autores:** Usar fading factor como métrica principal + janela para análise complementar.

---

## 3. Principais Contribuições

1. **Formalização da avaliação prequential** - Definição matemática rigorosa dos estimadores
2. **Comparação sistemática dos estimadores** - Análise teórica e empírica de propriedades
3. **Page-Hinkley Test para drift detection** - Método estatístico para detectar mudanças
4. **Guidelines práticas** - Recomendações para escolha de w e α
5. **Experimentos com datasets reais** - Validação empírica da metodologia

---

## 4. Resultados Chave

### 4.1 Propriedades dos Estimadores

| Estimador | Adaptação a Drift | Estabilidade | Custo Computacional | Custo de Memória |
|-----------|-------------------|--------------|---------------------|------------------|
| Acumulado | ❌ Nenhuma | ✅ Alta | ⚡ O(1) | 💾 O(1) |
| Janela | ✅ Rápida | ⚠️ Média | ⚡ O(1) | 💾 O(w) |
| Fading | ✅ Suave | ✅ Alta | ⚡ O(1) | 💾 O(1) |

**Conclusão:** Fading factor oferece melhor trade-off para a maioria dos casos.

### 4.2 Escolha de Parâmetros

**Window size (w):**
- Streams estacionários: w grande (5000-10000)
- Drift moderado: w médio (1000-2000)
- Drift rápido: w pequeno (100-500)

**Fading factor (α):**
- Drift lento: α = 0.001-0.01 (esquece 10-1% por exemplo)
- Drift moderado: α = 0.01-0.05
- Drift rápido: α = 0.05-0.1

**Regra prática:** Escolher α tal que `1/α` corresponda ao "tempo de memória" desejado.

### 4.3 Page-Hinkley Test

**Objetivo:** Detectar mudanças estatisticamente significativas na média do erro.

**Algoritmo:**
```
m_n = Σ(i=1 to n) (e_i - ē - δ)
M_n = max(m_k, k=1..n)
PH_n = M_n - m_n

Se PH_n > λ → concept drift detectado
```

**Parâmetros:**
- δ: magnitude mínima de mudança para detectar (0.005-0.01)
- λ: threshold de detecção (50-100)

**Uso prático:** Reset do modelo ou retraining quando drift é detectado.

---

## 5. Limitações Identificadas

1. **Ground truth delay:** Assume label disponível imediatamente (não realista em muitos domínios)
2. **Multi-class evaluation:** Paper foca em classificação binária
3. **Custo de treinamento:** Não considera custo computacional do update
4. **Drift detection:** Page-Hinkley é paramétrico (requer tuning de δ e λ)
5. **Reprodutibilidade:** Ordem dos dados afeta resultados (sem garantias estatísticas como k-fold)

---

## 6. Relação com Minha Pesquisa

### 6.1 Implementação Atual (v0.2)

**Já implementado em `prequential_metrics.py`:**
- ✅ Erro acumulado: `get_prequential_error_cumulative()`
- ✅ Erro em janela: `get_prequential_error_window()` (w=1000)
- ✅ Erro com fading: `get_prequential_error_fading()` (α=0.01)

**Parâmetros escolhidos (justificativa acadêmica):**
- **w = 1000:** Tráfego IoT com drift moderado (Neto et al., 2023 reporta ataques de 5-30min)
- **α = 0.01:** Drift gradual esperado em redes IoT (novos dispositivos, mudanças de padrão)

### 6.2 Próximos Passos (Semanas 6-7)

**Implementação futura:**
- ⏳ Page-Hinkley detector para drift detection automático
- ⏳ Métricas multi-class (quando avaliar tipos de ataque específicos)
- ⏳ Análise de sensibilidade dos parâmetros (w, α)

**Justificativa:** Semana 5 foca em validação básica (ataque vs. benigno). Page-Hinkley será útil na Semana 7 para comparar adaptação a drift entre algoritmos.

### 6.3 Uso na Dissertação

**Capítulo 2 (Fundamentação Teórica):**
- Seção sobre avaliação de algoritmos de stream learning
- Justificativa teórica para escolha de métricas prequential
- Comparação com métodos tradicionais (holdout, k-fold)

**Capítulo 4 (Metodologia):**
- Protocolo de avaliação dos experimentos
- Justificativa de parâmetros (w=1000, α=0.01)
- Escolha de métricas (fading error como principal)

**Capítulo 6 (Resultados):**
- Análise de adaptação a drift usando fading error
- Comparação de algoritmos usando os três estimadores
- Detecção de drift com Page-Hinkley (se implementado)

### 6.4 Diferenças do Nosso Caso de Uso

**Contexto IoT IDS vs. Paper:**

| Aspecto | Paper Gama et al. | Nosso Projeto |
|---------|------------------|---------------|
| Ground truth | Imediato | Heurístico (filename-based, Semana 5) |
| Tipo de problema | Classificação | Detecção de anomalias binária |
| Tipo de drift | Genérico | Conceitual (novos ataques, mudanças de dispositivo) |
| Volume de dados | Moderado | Alto (Kafka streaming, ~10k flows/s) |
| Custo de erro | Uniforme | Assimétrico (FN > FP em IDS) |

**Implicações:**
- Precisamos de **MTTD** além de accuracy (paper não menciona)
- Fading factor ideal pode ser diferente (requer análise de sensibilidade)
- Page-Hinkley pode ser muito sensível para tráfego IoT (muita variação natural)

---

## 7. Citações Importantes

> "The prequential evaluation provides a more realistic assessment of the learner's performance because it simulates the real-world scenario where the learner is continuously updated with new data."

> "The choice between holdout, sliding window, and fading factor depends on the characteristics of the stream: stationary streams favor holdout, abrupt drifts favor sliding window, and gradual drifts favor fading factor."

> "The Page-Hinkley test is a sequential analysis technique that can be used to detect changes in the mean of a signal. It is particularly well-suited for detecting abrupt changes in streaming data."

> "A major limitation of prequential evaluation is that it assumes immediate availability of the true labels, which is not always realistic in practical applications."

---

## 8. Referências Relevantes

**Outros papers citados que devo ler:**

| Referência | Tema | Prioridade |
|------------|------|------------|
| Dawid, A.P. (1984) - "Present position and potential developments: Some personal views on statistical theory" | Origem do termo "prequential" | Baixa (histórico) |
| Bifet, A. & Gavaldà, R. (2007) - "Learning from time-changing data with adaptive windowing" | ADWIN (janela adaptativa) | Alta (Semana 7) |
| Page, E.S. (1954) - "Continuous inspection schemes" | Page-Hinkley test original | Média (fundamentos) |
| Gama, J. et al. (2014) - "A survey on concept drift adaptation" | Survey de drift detection | Alta (revisão) |

---

## 9. Notas Técnicas para Implementação

### 9.1 Pseudo-código Page-Hinkley (Futuro)

```python
class PageHinkleyDetector:
    """Detecta concept drift usando Page-Hinkley test."""

    def __init__(self, delta: float = 0.005, lambda_: float = 50):
        self.delta = delta  # Magnitude mínima de mudança
        self.lambda_ = lambda_  # Threshold de detecção
        self.sum_errors = 0.0
        self.mean_error = 0.0
        self.max_sum = 0.0
        self.n = 0

    def update(self, error: float) -> bool:
        """
        Atualiza detector com novo erro.

        Returns:
            True se drift detectado, False caso contrário
        """
        self.n += 1
        self.mean_error += (error - self.mean_error) / self.n

        # m_n = Σ(e_i - ē - δ)
        self.sum_errors += error - self.mean_error - self.delta

        # M_n = max(m_k)
        self.max_sum = max(self.max_sum, self.sum_errors)

        # PH_n = M_n - m_n
        ph_value = self.max_sum - self.sum_errors

        if ph_value > self.lambda_:
            # Drift detectado - reset
            self.reset()
            return True

        return False

    def reset(self):
        """Reset após drift detectado."""
        self.sum_errors = 0.0
        self.mean_error = 0.0
        self.max_sum = 0.0
        self.n = 0
```

### 9.2 Análise de Sensibilidade de Parâmetros (Semana 7)

Experimento para validar escolha de α e w:

```python
# Testar diferentes configurações
alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
windows = [100, 500, 1000, 2000, 5000]

for alpha in alphas:
    for w in windows:
        metrics = PrequentialMetrics(window_size=w, alpha=alpha)
        # Rodar experimento com PCAP
        # Plotar erro ao longo do tempo
        # Comparar adaptação a drift
```

**Métricas de comparação:**
- Tempo para detectar drift (correlação com ground truth)
- Estabilidade (variância do erro)
- Responsividade (quão rápido adapta)

---

## 10. Perguntas para Discussão (Orientador)

1. **Ground truth delay:** Como lidar com delay de rotulação em cenário real? Considerar avaliação semi-supervisionada?
2. **Parâmetros:** Fazer análise de sensibilidade formal (α, w) ou usar valores do paper?
3. **Page-Hinkley:** Vale a pena implementar na Fase 2B ou deixar para Fase 3 (device-specific)?
4. **Métricas multi-class:** Quando transitar de binário (ataque vs. normal) para multi-class (tipos de ataque)?
5. **Custo assimétrico:** Como incorporar que FN é mais grave que FP em IDS?

---

## Histórico de Revisões

| Data | Versão | Mudanças |
|------|--------|----------|
| 2026-02-23 | 1.0 | Fichamento inicial para guiar implementação Semana 5-7 |

---

**Status:** ✅ Completo - Documento de referência pronto para consulta durante implementação e escrita da dissertação.

**Próximo passo:** Implementar `run_experiment.py` (Semana 5, Fase A) usando `PrequentialMetrics` já implementada.
