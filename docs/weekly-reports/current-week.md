# Weekly Report - Fase 2, Semana 1
**Week:** 2025-12-04 to 2025-12-11
**Phase:** Fase 2 - Evolutionary Clustering (Semana 1 de 10-12)
**Status:** 🟡 In Progress

---

## 📊 Week Overview

**Goal:** Estudar fundamentos de clustering e clustering evolutivo

**Focus:** 30% Teoria (Semanas 1-3 do roadmap)

**Planned Activities:**
- Revisar K-means, DBSCAN, clustering hierárquico
- Ler paper Maia et al. (2020) - Mixture of Typicalities
- Entender concept drift e adaptação evolutiva
- Criar design draft da arquitetura

---

## 📅 Sprint Plan

### Dias 1-2 (~4-6h): Fundamentos de Clustering
- [x] Revisar K-means: algoritmo, limitações, quando usar ✅
- [x] Revisar DBSCAN: density-based, parâmetros eps/min_samples ✅
- [x] Entender clustering particional vs density-based ✅
- [x] Relacionar com contexto IoT IDS ✅

### Dias 3-4 (~4-6h): Paper Maia et al. (2020)
- [x] Ler "Mixture of Typicalities" paper completo ✅
- [x] Extrair: algoritmo, pseudocódigo, parâmetros principais ✅
- [x] Entender: como lida com concept drift ✅
- [x] Identificar: adaptações necessárias para IoT IDS ✅

### Dias 5-6 (~4-6h): Síntese e Design
- [x] Criar resumo estruturado dos conceitos aprendidos ✅
- [ ] Esboçar design inicial da arquitetura ← PRÓXIMO
- [ ] Identificar gaps de conhecimento
- [ ] Preparar relatório semanal

### Dia 7 (~2h): Revisão e Planejamento
- [ ] Finalizar relatório semanal
- [ ] Atualizar SESSION_CONTEXT
- [ ] Planejar Semana 2

---

## 🎯 Entregáveis

1. **Resumo de Clustering Fundamentals** - documento com conceitos chave
2. **Resumo Paper Maia et al. 2020** - extração de pseudocódigo e parâmetros
3. **Design Draft** - esboço inicial da arquitetura de clustering evolutivo
4. **Relatório Semanal** - este documento finalizado

---

## 💻 Sessions Log

### Session 2025-12-09
**Duration:** ~2-3 horas
**Focus:** Sprint planning + Fundamentos de Clustering

**Progress:**
- ✅ Sprint iniciada oficialmente
- ✅ SESSION_CONTEXT.md atualizado
- ✅ Plano semanal definido e aprovado
- ✅ K-means: conceito, implementação, limitações com classes desbalanceadas
- ✅ Silhouette Score e Método do Cotovelo aprendidos
- ✅ Descoberta: CICIoT2023 tem ~8-10 clusters naturais
- ✅ DBSCAN: conceito, parâmetros eps/min_samples, comportamento não-linear
- ✅ Comparação K-means vs DBSCAN para IDS

**Experimentos Realizados:**
- K-means com K=2 em 500 amostras (accuracy 90%, mas problemas com classe minoritária)
- Silhouette Score para K=2 até K=10 (melhor em K=10)
- DBSCAN com eps variando de 0.3 a 7.0 (entendido comportamento de pico)

**Decisões:**
- K-means/DBSCAN servem para aprendizado, não precisam de experimentos publicáveis
- Fase 1 já tem baselines não-supervisionados (Isolation Forest, LOF)
- Foco principal será clustering evolutivo (contribuição)

**Notes:**
- Skill `evolutionary-clustering-guide` funcionou bem para aprendizado iterativo
- Abordagem "prática primeiro, teoria depois" eficaz
- Próximo: Leitura do paper Maia et al. (2020)

---

## 📈 Learning Progress

### Clustering Fundamentals
- [x] K-means understood ✅
- [x] DBSCAN understood ✅
- [ ] Hierarchical clustering (não prioritário)
- [x] Concept drift understood ✅
- [x] Mixture of Typicalities understood ✅

### TEDA Framework (Maia et al. 2020)
- [x] Eccentricidade e Tipicalidade ✅
- [x] Atualização recursiva (single-pass) ✅
- [x] Micro-clusters e Macro-clusters ✅
- [x] Teste de Chebyshev ✅
- [x] Tratamento de concept drift ✅

### Key Concepts Captured

**K-means:**
- Algoritmo iterativo: assign → update centroids → repeat
- Assume clusters esféricos e balanceados
- Silhouette Score: mede qualidade dos clusters (-1 a +1)
- Método do Cotovelo: encontrar K ótimo via inertia
- Limitação: não lida bem com classes desbalanceadas (16 vs 484 no CICIoT2023)
- CICIoT2023 tem ~8-10 clusters naturais (provavelmente tipos de ataque)

**DBSCAN:**
- Density-based: não precisa definir K
- Parâmetros: eps (raio vizinhança), min_samples (mínimo para cluster)
- Detecta outliers automaticamente (label=-1)
- Comportamento não-linear: existe "pico" de clusters em eps intermediário
- Alta dimensionalidade requer eps maior que intuitivo

**Para IDS:**
- K-means/DBSCAN são estáticos - não adaptam a concept drift
- Motivação clara para clustering evolutivo

**TEDA Framework:**
- Eccentricidade: mede "estranheza" de um ponto (0=típico, 1=outlier)
- Tipicalidade: 1 - eccentricidade (pertencimento ao cluster)
- Single-pass: atualização recursiva sem armazenar todos os dados
- Cold start: ~100-200 pontos para estabilizar
- Micro-clusters: múltiplos centros locais com tipicalidade própria
- Mixture of Typicalities: atribuir ao cluster com maior tipicalidade
- Chebyshev test: threshold adaptativo para aceitar/rejeitar pontos
- Clusters maduros são mais "exigentes" (threshold menor)
- Concept drift: novos padrões → novos clusters automaticamente

---

## 🧠 Insights & Decisions

**Decisão 001:** K-means/DBSCAN são para aprendizado, não publicação
- Fase 1 já tem baselines não-supervisionados rigorosos
- Foco deve ser no clustering evolutivo (contribuição)
- Experimentos exploratórios suficientes para fundamentação teórica

**Insight:** Abordagem "prática primeiro" funciona
- Experimentar antes de ler papers ajuda a entender as motivações dos autores
- Descobrir limitações na prática → entender por que soluções foram propostas

---

## 🚧 Blockers & Challenges

*None yet*

---

## 📝 Notes for Advisor Meeting

**Progress:**
- Fase 2 iniciada oficialmente
- Foco em fundamentos teóricos (semanas 1-3)
- Sistema de desenvolvimento acelerado em uso

**Discussion Points:**
- Validar abordagem de clustering evolutivo
- Discutir papers relevantes além de Maia et al.
- Timeline para primeiros experimentos

**Questions:**
*(To be added during the week)*

---

**Auto-updated by session hooks. Use `/finalize-week` to create final version.**
