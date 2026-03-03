# STATUS — IoT IDS Research
<!-- Atualizar este arquivo ao final de cada sessão significativa -->
<!-- Responde: onde estou, o que foi feito, o que faço agora -->

**Atualizado:** 2026-03-03 | **Branch:** main | **Semana:** 5/24

---

## 🎯 Agora

**Fase 2B — Experimentos Streaming**

O que acabou de ser feito:
- Orquestração de experimentos completa (`run_experiment.py` + métricas prequential)
- Isolamento entre experimentos (group IDs únicos, purge de tópicos)
- Grid 3×2 executado: TEDA vs MicroTEDAclus × r0={0.05, 0.10, 0.20}
- Merge do branch `experiment-orchestration` → `main` (98 testes passando)

**Próxima sessão — 3 ações:**
1. Executar experimento DDoS (validar Recall >= 80% e MTTD <= 500 flows):
   ```bash
   cd streaming && source venv/bin/activate
   # Kafka deve estar rodando: cd docker && docker-compose up -d
   python3 scripts/run_experiment.py \
     --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
     --attack-pcap ../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
     --max-packets 50000 --max-flows 10000 \
     --output ../streaming/results/week5/ddos_detection/
   ```
2. Analisar resultados: Recall, MTTD, comparação TEDA vs MicroTEDAclus
3. Atualizar `docs/weekly-reports/semana5-report.md` e este arquivo

---

## 📊 Critérios de Sucesso (Semana 5)

| Critério | Target | Status |
|----------|--------|--------|
| FPR tráfego benigno | ≤ 5% | ✅ MicroTEDA: ~3-4% |
| Recall ataque DDoS | ≥ 80% | ⏳ Pendente |
| MTTD | ≤ 500 flows | ⏳ Pendente |
| Diferença TEDA vs MicroTEDA | Clara | ⏳ Pendente |

---

## 📁 Código Relevante Agora

| O quê | Onde |
|-------|------|
| Orquestrador de experimentos | `streaming/scripts/run_experiment.py` |
| Comparador de resultados | `streaming/scripts/compare_experiments.py` |
| Detector TEDA + MicroTEDAclus | `streaming/src/detector/` |
| Métricas prequential | `streaming/src/metrics/` |
| Resultados semana 5 | `streaming/results/week5/` |
| Infraestrutura Kafka | `streaming/docker/docker-compose.yml` |

---

## 📅 Roadmap Resumido

| Semana | Status | Foco |
|--------|--------|------|
| S1–S4 | ✅ | Teoria + TEDA + MicroTEDAclus |
| **S5** | 🔄 | **Orquestração + Experimentos E2E** |
| S6 | 📋 | Métricas de avaliação de clustering |
| S7 | 📋 | Experimentos drift sintético |
| S8–S10 | 📋 | TEDA v0.3 + otimização |
| S11–S14 | 📋 | Full dataset + validação estatística |
| S15–S18 | 📋 | Otimização + análise |
| S19–S24 | 📋 | Dissertação + defesa |

**Fase 1:** ✅ Completa — 705 experimentos, 10 algoritmos, F1 > 0.99

---

## 🔗 Referências Rápidas

- Contexto histórico detalhado: `docs/SESSION_CONTEXT.md`
- Arquitetura implementada: `docs/architecture/CURRENT.md`
- Relatório semanal atual: `docs/weekly-reports/semana5-report.md`
- Metodologia de experimentos: `docs/methodology/experiment-methodology.md`
- Teoria TEDA/MicroTEDAclus: `docs/theory/teda-framework.md`
