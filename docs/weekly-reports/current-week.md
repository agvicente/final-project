# Weekly Report - Fase 2A, Semana 2
**Week:** 2025-12-23 to 2025-12-29
**Phase:** Fase 2A - Teoria + Design + Setup (Semana 2 de 24)
**Status:** 🟡 In Progress

---

## 📊 Week Overview

**Goal:** Leitura Angelov (2014) + Setup ambiente remoto + Producer v0.1

**Focus:** Fundamentação teórica + início da implementação

**Planned Hours:** 10-12h

---

## 🎯 Entregáveis Planejados

| # | Entregável | Status | Arquivo |
|---|------------|--------|---------|
| 1 | Fichamento Angelov (2014) | 🟡 85% | `docs/paper-summaries/angelov-2014-teda.md` |
| 2 | Ambiente Kafka rodando | ⏳ | Docker remoto |
| 3 | Producer v0.1 (PCAP reader) | ⏳ | `src/producer/` |
| 4 | Relatório Semanal | ⏳ | Este documento |

---

## 📅 Sprint Plan

### Dias 1-2 (~4h): Leitura Angelov (2014)
- [x] Ler paper completo: "Outside the box: an alternative data analytics framework" ✅
- [x] Criar fichamento seguindo template ✅
- [x] Extrair fórmulas e pseudocódigo ✅
- [ ] Relacionar com MicroTEDAclus (Maia 2020)

### Dias 3-4 (~4h): Setup Ambiente Remoto
- [ ] Conectar via SSH à máquina com PCAPs
- [ ] Instalar Docker + Docker Compose
- [ ] Configurar Kafka (single broker para MVP)
- [ ] Testar producer/consumer básico

### Dias 5-6 (~3h): Producer v0.1
- [ ] Estrutura básica do projeto
- [ ] PCAP reader com dpkt ou scapy
- [ ] Publicar pacotes no Kafka
- [ ] Testar com subset pequeno (~1GB)

### Dia 7 (~1h): Revisão
- [ ] Atualizar relatório semanal
- [ ] Planejar Semana 3

---

## 💻 Sessions Log

### Session 1: 2026-01-03 (~3h)
**Focus:** Leitura e fichamento Angelov (2014)

**Atividades:**
- Leitura completa do paper "Outside the box: an alternative data analytics framework"
- Criação de fichamento detalhado com 14 seções
- Discussão de conceitos: frequentista, belief/possibility theory, first principles
- Documentação de métricas de distância (Euclidean, Manhattan, Mahalanobis, Cosine)
- Explicação de normalização e por que ξ = π normalizado
- Extração de fórmulas: π, ξ, τ com exemplos numéricos

**Arquivos criados/modificados:**
- `docs/paper-summaries/angelov-2014-teda.md` (novo, 85% completo)

**Próxima sessão:**
- Relacionar TEDA com MicroTEDAclus (Maia 2020)
- Iniciar setup Kafka

---

## 📈 Learning Progress

### Leitura: Angelov (2014)
- [x] Motivação do framework TEDA ✅
- [x] Definição formal de eccentricidade ✅
- [x] Definição formal de tipicalidade ✅
- [x] Propriedades estatísticas ✅
- [x] Aplicações demonstradas ✅
- [x] Conceitos adicionais: frequentista, kernels, normalização ✅
- [x] Métricas de distância ✅

### Implementação
- [ ] Kafka basics (topics, producers, consumers)
- [ ] PCAP parsing em Python
- [ ] Estrutura de mensagens

---

## 🧠 Insights & Decisions

*(To be filled during the week)*

---

## 🚧 Blockers & Challenges

*(To be filled during the week)*

---

## 📝 Notes for Advisor Meeting

### Progresso Semana 1 (Recap)
- Fundamentação teórica completa
- Design arquitetura MVP definido
- Plano de leituras estruturado

### Progresso Semana 2
*(To be filled)*

### Questions
*(To be filled)*

---

## 📅 Preview Semana 3

| Tarefa | Leitura |
|--------|---------|
| Consumer 1 (windowing) | Maia (2020) - releitura completa |
| Feature extraction | Survey Concept Drift |
| Testes integração | - |

---

**Week 2 Started.**

*Iniciado em: 2025-12-23*
