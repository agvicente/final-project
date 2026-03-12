# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-12 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~7 semanas)

---

## Agora

**S2 — Campanha 01 completa (17 experimentos). Detector não detecta ataques.**

Sessões 10-12/03 (máquina Linux — execução de experimentos):
- ✅ Bug crítico corrigido: race condition FlowConsumer-Detector (500 → 7500 flows)
- ✅ A1 completo (4 runs): FPR ~3.5% — APROVADO
- ❌ A2 completo (12 runs, 5 tipos de ataque): Recall ~3-4% — REPROVADO
  - DDoS: ICMP (4), SYN (2), TCP (2) — todos ~3.5-4.5% Recall
  - Não-volumétricos: Mirai (2) ~2.7%, Recon-PortScan (2) ~4.0%
- ✅ A3 completo: MicroTEDAclus 26x melhor que TEDA
- ✅ Resultados commitados e análise completa

**Diagnóstico:** O MicroTEDAclus detecta outliers estatísticos (~3.5% do tráfego),
mas NÃO detecta ataques. Flows de ataque são indistinguíveis de flows benignos
no espaço de 17 features. Problema de **representação**, não de algoritmo.

**Próxima sessão:**
1. Analisar distribuição de features (histogramas benign vs attack)
2. Investigar ground truth por IP (labels mais precisos que por fase)
3. Decidir: expandir features, mudar granularidade de detecção, ou documentar limitação

---

## Critérios de Sucesso (Campanha S2)

| Critério | Status |
|----------|--------|
| A1: FPR <= 5% (baseline benigno) | ✅ 3.5% |
| A2-DDoS: Recall >= 80% (ICMP/SYN/TCP) | ❌ ~4% — flows indistinguíveis |
| A2-Mirai: Recall >= 80% | ❌ 2.7% — mesma limitação |
| A2-Recon: Recall >= 80% (PortScan) | ❌ 4.0% — mesma limitação |
| A2: MTTD <= 500 flows | ✅ 6-46s (quando detecta) |
| A3: TEDA vs MicroTEDAclus | ✅ MicroTEDAclus 26x melhor |
| r0 ótimo calibrado | ⚠️ Sem impacto significativo em nenhum cenário |
| Resultados commitados | ✅ 17 experimentos |

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
| S2 ✅ | Campanha 01: 17 exps, A1 ✅, A2 ❌ (todas variantes), A3 ✅ |
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
