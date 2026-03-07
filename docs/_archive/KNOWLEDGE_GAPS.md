# Lacunas de Conhecimento - Reforço Necessário

**Objetivo:** Identificar e acompanhar conceitos fundamentais que precisam ser reforçados para facilitar a pesquisa.

**Última atualização:** 2026-01-05

---

## Como Usar Este Documento

1. Estude os tópicos por **prioridade** (Alta → Média → Baixa)
2. Marque com ✅ quando se sentir confortável
3. Claude atualizará este documento quando identificar novas lacunas

---

## 📊 Resumo por Área

| Área | Lacunas | Prioridade Média |
|------|---------|------------------|
| Estatística/Probabilidade | 4 | Alta |
| Álgebra Linear | 3 | Alta |
| Cálculo/Análise | 1 | Média |
| Teoria da Informação | 1 | Baixa |

---

## 🔴 Prioridade Alta

### 1. Estatística Básica

**Identificado em:** Sessão 2026-01-03/05 (Fichamento Angelov)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Variância e Desvio Padrão | Base da fórmula recursiva do TEDA: σ² = X - \|\|μ\|\|² | ⬜ |
| Média como centro de massa | Propriedade Σ(x_i - μ) = 0 usada na derivação | ⬜ |
| Esperança E[X] | Fórmula de König: Var = E[X²] - E[X]² | ⬜ |

**Recurso sugerido:** Khan Academy - Statistics and Probability

---

### 2. Teoria de Probabilidade

**Identificado em:** Sessão 2026-01-03 (Perguntas sobre frequentismo)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Abordagem Frequentista vs Bayesiana | Entender crítica do TEDA à probabilidade clássica | ⬜ |
| Distribuições (Gaussiana, etc.) | Por que TEDA não assume distribuição prévia | ⬜ |
| Função Densidade de Probabilidade (PDF) | Entender por que ζ "resembles PDF" | ⬜ |

**Recurso sugerido:** "Probability Theory: The Logic of Science" (Jaynes) - Cap 1-2

---

### 3. Álgebra Linear Básica

**Identificado em:** Sessão 2026-01-03/05 (Métricas de distância)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Norma de vetor \|\|x\|\| | Todas as fórmulas de distância usam normas | ⬜ |
| Produto interno (dot product) | Expansão: \|\|a-b\|\|² = \|\|a\|\|² - 2a·b + \|\|b\|\|² | ⬜ |
| Matriz de covariância | Distância de Mahalanobis: (x-μ)ᵀΣ⁻¹(x-μ) | ⬜ |

**Recurso sugerido:** 3Blue1Brown - "Essence of Linear Algebra" (YouTube)

---

### 4. Identidades Matemáticas Clássicas

**Identificado em:** Sessão 2026-01-05 (Derivação da fórmula recursiva)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Teorema de Huygens-Steiner | Base da simplificação O(n²) → O(n) no TEDA | ⬜ |
| Fórmula de König-Huygens | Var(X) = E[X²] - E[X]² — expressar variância recursivamente | ⬜ |
| Expansão do quadrado | \|\|a-b\|\|² = \|\|a\|\|² - 2a·b + \|\|b\|\|² | ⬜ |

**Recurso sugerido:** Wikipedia + exercícios manuais

---

## 🟡 Prioridade Média

### 5. Normalização e Escalas

**Identificado em:** Sessão 2026-01-03 (Por que ξ é π normalizado)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Tipos de normalização | Min-Max, Z-Score, por soma — escolher corretamente | ⬜ |
| Por que normalizar | Comparabilidade, interpretabilidade, estabilidade numérica | ⬜ |

**Recurso sugerido:** Scikit-learn documentation - Preprocessing

---

### 6. Métricas de Distância

**Identificado em:** Sessão 2026-01-03 (Euclidiana, Manhattan, Mahalanobis, Cosseno)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Quando usar cada métrica | Escolha correta para features de rede IoT | ⬜ |
| Propriedades métricas | Simetria, desigualdade triangular, positividade | ⬜ |
| Distância vs Similaridade | Cosseno é similaridade, precisa converter | ⬜ |

**Recurso sugerido:** "Pattern Recognition and Machine Learning" (Bishop) - Cap 2

---

## 🟢 Prioridade Baixa

### 7. Teoria da Possibilidade

**Identificado em:** Sessão 2026-01-03 (Belief functions, necessity)

| Conceito | Por que é necessário | Status |
|----------|---------------------|--------|
| Dempster-Shafer | Contexto histórico — por que TEDA é diferente | ⬜ |
| Possibilidade vs Necessidade | Entender alternativas à probabilidade | ⬜ |

**Recurso sugerido:** Só se houver tempo — não é essencial para implementação

---

## 📝 Histórico de Atualizações

| Data | Lacuna Identificada | Contexto |
|------|---------------------|----------|
| 2026-01-03 | Frequentismo, PDF, métricas de distância | Fichamento Angelov - conceitos básicos |
| 2026-01-03 | Normalização | Pergunta sobre ξ como π normalizado |
| 2026-01-05 | Variância, esperança, produto interno | Derivação da fórmula recursiva |
| 2026-01-05 | Huygens-Steiner, König-Huygens | Nome da identidade matemática |

---

## 🎯 Plano de Estudo Sugerido

**Semana típica (2-3h extras):**

1. **30min/dia:** Um vídeo 3Blue1Brown (Álgebra Linear)
2. **1h/semana:** Khan Academy - Estatística
3. **Conforme surgir:** Consultar este documento antes de perguntar

**Ordem recomendada:**
1. Álgebra Linear (normas, produto interno) — impacta tudo
2. Estatística (variância, esperança) — impacta TEDA
3. Identidades matemáticas — aprofundamento
4. Métricas de distância — aplicação prática
5. Teoria da possibilidade — só se sobrar tempo

---

*Este documento é atualizado automaticamente quando Claude identifica lacunas nas perguntas.*
