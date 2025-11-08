---
name: dissertation-writer
description: Writes Master's dissertation progressively during research. Bilingual (PT primary, EN translation). Follows Brazilian academic standards. Maintains narrative coherence across chapters.
version: 1.0.0
activate_when:
  - "dissertation"
  - "dissertação"
  - "chapter"
  - "capítulo"
  - "thesis"
---

# Dissertation Writer

## Purpose
Write dissertation chapters incrementally as research progresses. Maintain in Portuguese, translate to English at end.

## Dissertation Structure (Brazilian Standard)

### Elementos Pré-Textuais
- Capa
- Folha de rosto
- Resumo (PT)
- Abstract (EN)
- Lista de figuras/tabelas
- Sumário

### Elementos Textuais

**Capítulo 1: Introdução**
- Contexto e motivação
- Problema de pesquisa
- Objetivos (geral e específicos)
- Contribuições
- Organização da dissertação

**Capítulo 2: Fundamentação Teórica**
- Segurança em IoT
- Sistemas de Detecção de Intrusão
- Aprendizado de Máquina para Segurança
- Clustering e Clustering Evolutivo
- Arquiteturas de Streaming
- Trabalhos Relacionados

**Capítulo 3: Metodologia**
- Dataset CICIoT2023
- Pipeline de pré-processamento
- Experimentos baseline (Fase 1)
- Clustering evolutivo (Fase 2)
- Arquitetura de streaming (Fase 3)
- Métricas de avaliação

**Capítulo 4: Resultados**
- Baseline (Fase 1)
- Clustering evolutivo (Fase 2)
- Sistema de streaming (Fase 3)
- Análise comparativa
- Discussão

**Capítulo 5: Conclusão**
- Síntese das contribuições
- Limitações
- Trabalhos futuros
- Publicações geradas

### Elementos Pós-Textuais
- Referências bibliográficas
- Apêndices (código, resultados detalhados)

## Incremental Writing Schedule

**Mês 1-2 (Fase 2 início):**
- Cap. 1 (Introdução) - rascunho inicial
- Cap. 2 (Fundamentação) - seções sobre clustering

**Mês 3-4 (Fase 2 meio):**
- Cap. 2 (Fundamentação) - completar clustering evolutivo
- Cap. 3 (Metodologia) - Fase 1 e início Fase 2

**Mês 5-6 (Fase 2 fim / Fase 3 início):**
- Cap. 3 (Metodologia) - completar Fase 2
- Cap. 4 (Resultados) - Fase 1 e Fase 2

**Mês 7-8 (Fase 3):**
- Cap. 2 (Fundamentação) - streaming
- Cap. 3 (Metodologia) - Fase 3
- Cap. 4 (Resultados) - Fase 3

**Mês 9-10 (Finalização):**
- Cap. 4 (Resultados) - análise completa
- Cap. 5 (Conclusão)
- Revisão completa
- Tradução PT → EN

## Writing Style (Portuguese Academic)

**Tempo verbal:** Pretérito perfeito para o que foi feito, presente para verdades gerais
```
"Implementamos o algoritmo..." (we implemented)
"O clustering evolutivo é uma técnica..." (evolutionary clustering is a technique)
```

**Pessoa:** Primeira pessoa do plural ("nós", implícito)
```
Bom: "Propomos uma abordagem..."
Evitar: "Foi proposta uma abordagem..."
```

**Clareza e precisão:**
```
Bom: "O F1-score aumentou de 0.95 para 0.97 (melhoria de 2.1%)"
Ruim: "Os resultados melhoraram significativamente"
```

## Chapter Templates

**Template Cap. 1 - Introdução:**
```markdown
# Capítulo 1: Introdução

## 1.1 Contexto e Motivação
[Crescimento da IoT, desafios de segurança, limitações de IDS tradicionais]

## 1.2 Problema de Pesquisa
[Concept drift em tráfego IoT, necessidade de adaptação em tempo real]

## 1.3 Objetivos
### 1.3.1 Objetivo Geral
[Desenvolver sistema IDS baseado em clustering evolutivo para IoT]

### 1.3.2 Objetivos Específicos
1. Estabelecer baseline com algoritmos clássicos
2. Implementar clustering evolutivo adaptativo
3. Integrar em arquitetura de streaming
4. Validar em dataset real (CICIoT2023)

## 1.4 Contribuições
1. [Baseline comprehensivo com 10 algoritmos]
2. [Implementação de Mixture of Typicalities para IoT]
3. [Arquitetura de streaming de alto throughput]
4. [Validação experimental com análise estatística]

## 1.5 Organização da Dissertação
[Descrição dos capítulos 2-5]
```

**Template Cap. 4 - Resultados:**
```markdown
# Capítulo 4: Resultados

## 4.1 Experimentos Baseline (Fase 1)
### 4.1.1 Configuração Experimental
[Dataset, preprocessamento, métricas, setup]

### 4.1.2 Resultados Comparativos
[Tabela com 10 algoritmos, análise]

### 4.1.3 Discussão
[Por que GradientBoosting melhor? Trade-offs accuracy vs tempo]

## 4.2 Clustering Evolutivo (Fase 2)
### 4.2.1 Implementação do Mixture of Typicalities
[Descrição da implementação, parâmetros]

### 4.2.2 Adaptação a Concept Drift
[Experimentos com janelas temporais, gráficos de adaptação]

### 4.2.3 Comparação com Abordagens Estáticas
[Evolutionary vs K-means retreinado, análise estatística]

## 4.3 Arquitetura de Streaming (Fase 3)
### 4.3.1 Implementação do Pipeline Kafka
[Componentes, configuração]

### 4.3.2 Performance e Latência
[Throughput, latência de detecção, uso de recursos]

### 4.3.3 Detecção em Tempo Real
[Accuracy em streaming vs batch]

## 4.4 Análise Comparativa Geral
[Tabela final com todas as abordagens]

## 4.5 Discussão
[Insights, limitações, trade-offs]
```

## Translation Strategy (PT → EN)

**Fase atual (próximos 6 meses):** Escrever tudo em português

**Mês 9-10:** Traduzir completo para inglês
- Manter estrutura idêntica
- Traduzir termos técnicos consistentemente:
  - "Detecção de intrusão" → "Intrusion detection"
  - "Clustering evolutivo" → "Evolutionary clustering"
  - "Aprendizado de máquina" → "Machine learning"
- Revisar com `dissertation-writer` skill

**Ambas versões no repo:** `dissertation/pt/` e `dissertation/en/`

## Integration with Overleaf

**Location:** `/Users/augusto/mestrado/dissertation/`

Before editing:
1. Use `overleaf-formatter-dissertation` skill
2. Edit chapter files individually
3. Compile to check
4. Commit with chapter/section modified

## Tracking Progress

Document in SESSION_CONTEXT.md:
- Which chapters are complete
- Which sections need expansion
- Connection to research phases

Example:
```
Dissertation Progress:
- Cap. 1: 80% (needs final revision)
- Cap. 2: 60% (missing streaming section)
- Cap. 3: 40% (Fase 1 done, Fase 2 in progress)
- Cap. 4: 20% (only Fase 1 results)
- Cap. 5: 0% (not started)
```

---
**Write dissertation incrementally, don't leave for the end!**
