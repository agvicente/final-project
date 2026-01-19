# Weekly Report - Fase 2A, Semana 3
**Week:** 2026-01-19 to 2026-01-25
**Phase:** Fase 2A - Teoria + Design + Setup (Semana 3 de 24)
**Status:** 🟡 In Progress

---

## 📊 Week Overview

**Goal:** Implementar TEDA v0.1 (básico) para detecção de anomalias em streaming

**Focus:** Implementação do algoritmo TEDA (Angelov 2014) - eccentricity e typicality

**Planned Hours:** 10-12h

---

## 🎯 Entregáveis Planejados

| # | Entregável | Status | Arquivo |
|---|------------|--------|---------|
| 1 | TEDADetector class | ⏳ | `streaming/src/detector/teda.py` |
| 2 | Integração com Consumer | ⏳ | `streaming/src/detector/streaming_detector.py` |
| 3 | Teste E2E (PCAP → detecção) | ⏳ | Scripts de teste |
| 4 | Documentação TEDA | ⏳ | Atualizar arquitetura |
| 5 | Relatório Semanal | 🟡 | Este documento |

---

## 📅 Sprint Plan

### Dias 1-2 (~4h): Implementar TEDADetector
- [ ] Criar estrutura `streaming/src/detector/`
- [ ] Implementar classe TEDADetector
  - [ ] Atualização recursiva de μ (média)
  - [ ] Atualização recursiva de σ² (variância)
  - [ ] Cálculo de eccentricity: ξ = 1/k + ||x-μ||²/(k×σ²)
  - [ ] Cálculo de typicality: τ = 1 - ξ
  - [ ] Threshold para anomalia (1/k ou Chebyshev)
- [ ] Testes unitários básicos

### Dias 3-4 (~4h): Integração com Pipeline
- [ ] Criar StreamingDetector (Consumer + TEDA)
- [ ] Ler flows do tópico 'flows'
- [ ] Classificar cada flow como normal/anômalo
- [ ] Publicar alertas no tópico 'alerts'
- [ ] Testar pipeline completo

### Dias 5-6 (~3h): Validação com Dados Reais
- [ ] Testar com subset do CICIoT2023
- [ ] Verificar detecção de ataques conhecidos
- [ ] Ajustar threshold se necessário
- [ ] Documentar resultados

### Dia 7 (~1h): Revisão
- [ ] Atualizar documentação de arquitetura
- [ ] Atualizar relatório semanal
- [ ] Planejar Semana 4 (MicroTEDAclus)

---

## 💻 Sessions Log

### Session 1: 2026-01-19
**Focus:** Setup Semana 3, revisão do que foi feito

**Atividades:**
- Revisão completa do projeto (fichamentos, código, arquitetura)
- Confirmação do plano incremental: TEDA básico → MicroTEDAclus
- Atualização do weekly report para Semana 3
- Preparação para implementação do TEDA

**Arquivos revisados:**
- `docs/paper-summaries/angelov-2014-teda.md` (1370 linhas)
- `docs/paper-summaries/maia-2020-microtedaclus.md` (461 linhas)
- `streaming/src/consumer/flow_consumer.py` (792 linhas)
- `docs/architecture/STREAMING_ARCHITECTURE.md`

**Próxima atividade:**
- Implementar TEDADetector

---

## 📈 Learning Progress

### Semana 2 (Recap) ✅
- [x] Fichamento Angelov (2014) - TEDA Framework completo
- [x] Fichamento MicroTEDAclus (Maia 2020) completo
- [x] Setup Kafka local (Docker Compose)
- [x] Producer v0.1 funcionando (2909 pkt/s)
- [x] Consumer v0.1 funcionando (27 features)
- [x] Pipeline E2E testado
- [x] Conceitos Kafka documentados

### Semana 3 (Atual)
- [ ] Implementação TEDA básico
- [ ] Atualização recursiva de estatísticas
- [ ] Threshold de anomalia
- [ ] Integração com streaming

---

## 🧠 Insights & Decisions

### Decision: TEDA básico primeiro
Seguir abordagem incremental:
1. **S3:** TEDA v0.1 - apenas ξ, τ para detecção de anomalias
2. **S4:** MicroTEDAclus - micro/macro clusters para clustering evolutivo

Justificativa: Validar pipeline com detecção simples antes de adicionar complexidade.

### Fórmulas a Implementar (Angelov 2014)

**Atualização recursiva da média:**
```
μ_k = ((k-1)/k) × μ_{k-1} + x_k/k
```

**Atualização recursiva da variância:**
```
σ²_k = ((k-1)/k) × σ²_{k-1} + (1/(k-1)) × ||x_k - μ_k||²
```

**Eccentricity:**
```
ξ(x_k) = 1/k + ||x_k - μ_k||² / (k × σ²_k)
```

**Typicality:**
```
τ(x_k) = 1 - ξ(x_k)
```

**Threshold (Chebyshev):**
```
Anomalia se: ξ > (m² + 1) / (2k)    onde m = 3 (3 desvios padrão)
```

---

## 🚧 Blockers & Challenges

- **Nenhum blocker crítico** - Base teórica e infraestrutura prontas

---

## 📝 Notes for Advisor Meeting

### Progresso Semana 2 (Completo)
- Fichamentos TEDA e MicroTEDAclus 100%
- Pipeline streaming funcionando (Producer + Consumer)
- Documentação Kafka completa

### Plano Semana 3
- Implementar TEDA básico para detecção de anomalias
- Integrar com pipeline streaming
- Validar com dados do CICIoT2023

### Questions
1. **Threshold:** Usar 1/k simples ou Chebyshev (m²+1)/(2k)?
2. **Features:** Quais features do flow usar para TEDA? (todas 27 ou subset?)
3. **Normalização:** Normalizar features antes do TEDA?

---

## 📅 Preview Semana 4

| Tarefa | Foco |
|--------|------|
| MicroTEDAclus v0.1 | Micro-clusters com TEDA |
| Threshold dinâmico | m(k) = 3/(1+e^{-0.007(k-100)}) |
| Merge/split | Critério de interseção |

---

**Week 3 Progress: ~5%**

*Iniciado em: 2026-01-19*
*Última atualização: 2026-01-19*
