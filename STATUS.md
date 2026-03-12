# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-12 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~7 semanas)

---

## Agora

**S2 — Cenários A1/A2/A3 completos. DDoS-ICMP não detectado — investigando.**

Sessões 10-12/03 (máquina Linux — execução de experimentos):
- ✅ Bug crítico corrigido: race condition FlowConsumer-Detector (500 → 7500 flows)
- ✅ Cenário A1 completo: FPR ~3.5% (alvo <= 5%) — APROVADO
- ✅ Cenário A2 completo: Recall ~4.5% (alvo >= 80%) — REPROVADO
- ✅ Cenário A3 completo: TEDA Recall ~0.05% — MicroTEDAclus 26x superior
- ✅ Análise completa: `experiments/results/campaign-01/ANALYSIS.md`

**Diagnóstico:** DDoS-ICMP gera flows indistinguíveis do benigno no espaço de
17 features (TCP flags são zero para ICMP). O algoritmo funciona — o problema
é de representação.

**Próxima sessão:**
1. Rodar A2 com **DDoS-SYN_Flood** — validar que TCP flags discriminam
2. Rodar A2 com **DDoS-TCP_Flood** — confirmar com segundo ataque TCP
3. Commitar todos os resultados para a outra máquina escrever

---

## Critérios de Sucesso (Campanha S2)

| Critério | Status |
|----------|--------|
| A1: FPR <= 5% (baseline benigno) | ✅ 3.5% |
| A2-ICMP: Recall >= 80% (DDoS-ICMP) | ❌ 4.5% — features insuficientes |
| A2-SYN: Recall >= 80% (DDoS-SYN) | ⏳ Próximo passo |
| A2: MTTD <= 500 flows | ⚠️ 12-39s (mas Recall invalida) |
| A3: TEDA vs MicroTEDAclus | ✅ MicroTEDAclus 26x melhor |
| r0 ótimo calibrado | ⚠️ Sem impacto em A1/A2-ICMP, testar em A2-SYN |
| Resultados commitados | ⏳ Após A2-SYN |

---

## Código Relevante Agora

| O quê | Onde |
|-------|------|
| Análise de resultados | `experiments/results/campaign-01/ANALYSIS.md` |
| Plano experimental | `experiments/campaign-plan.md` |
| Resultados campanha 01 | `experiments/results/campaign-01/` |
| Orquestrador de experimentos | `experiments/streaming/scripts/run_experiment.py` |
| Sincronização FlowConsumer | `experiments/streaming/docs/flow-consumer-sync.md` |
| Detector TEDA + MicroTEDAclus | `experiments/streaming/src/detector/` |
| Kafka utils (purge + sync) | `experiments/streaming/src/kafka_utils.py` |
| Arquitetura v0.3.0 | `docs/architecture/CURRENT.md` |

---

## Roadmap (8 semanas até defesa)

| Semana | Foco |
|--------|------|
| S1 ✅ | Reorganização repo + campaign-plan |
| S2 🔄 | Validar arquitetura + métricas experimentais (A1 ✅, A2/A3 pendentes) |
| S3-S4 | Campanha experimental (cenários A-E) |
| S5-S6 | Análise de resultados + comparação com literatura |
| S7 | Escrita dissertação (caps 2-5) |
| S8 | Revisão final + preparação defesa |

---

## Referências Rápidas

- Como usar o repositório: `USAGE.md`
- Plano experimental (próxima ação): `experiments/campaign-plan.md`
- Plano da reorganização (concluído): `docs/plans/2026-03-07-repo-reorganization.md`
- Arquitetura do sistema: `docs/architecture/CURRENT.md`
- Leituras e lacunas: `research/reading-log.md`
