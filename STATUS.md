# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-19 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~7 semanas)

---

## Agora

**Campaign-03 S4 concluída — 48 runs (behavioral window features v2).**

Sessão 18-19/03:
- ✅ Commit: behavioral features v2 (19 features = 12 base + 7 comportamentais)
- ✅ Campaign-02 ANALYSIS.md criado (72 runs, S1/S2/S3)
- ✅ Campaign-03 S4 executada: 48 runs, 0 falhas, 66 min
- ✅ 7 plots gerados (v1 vs v2, FPR, F1, r0 sweep, dashboard S3→S4)
- ✅ Campaign-03 ANALYSIS.md criado com resultados e diagnóstico
- ✅ 123 testes passando, architecture v0.5.0

**Resultado S4:** Features v2 **não produzem melhoria consistente**. Desbloqueiam DDoS-ICMP (0%→50%) mas degradam SYN e Mirai. FPR benigno piora em w=10s (2.9%→14.3%). Melhor resultado global: Recon F1=43.7% (v2/w10s/r0=0.05). Problema fundamental: poucos vetores (~210) em 19 dimensões.

**Próxima sessão:**
1. Decidir se implementar S5 (Two-Stage Detection) ou seguir para escrita
2. Se S5: implementar IPAnomalyMonitor + ~24 runs
3. Consolidar best configs por ataque para tabela final da dissertação

---

## Critérios de Sucesso

| Critério | Status |
|----------|--------|
| Campaign-01 completa (17 runs) | ✅ |
| Campaign-02 completa (72 runs) + ANALYSIS.md | ✅ |
| Campaign-03 S4 completa (48 runs) + ANALYSIS.md | ✅ |
| Features v2 melhoram detecção consistentemente | ❌ Parcial (2/5 ataques) |
| FPR benigno <= 5% com v2 | ❌ 14.3% em w=10s |
| Recall >= 80% em algum ataque | ❌ Best: 61.5% (SYN@w30s/r0=0.05, FPR=36%) |
| 123 testes passando | ✅ |
| Decidir S5 (Two-Stage) | ⬜ Pendente |

---

## Código Relevante Agora

| O quê | Onde |
|-------|------|
| Análise Campaign-01 | `experiments/results/campaign-01/ANALYSIS.md` |
| Análise Campaign-02 | `experiments/results/campaign-02/ANALYSIS.md` |
| Análise Campaign-03 S4 | `experiments/results/campaign-03/ANALYSIS.md` |
| Plano experimental | `experiments/campaign-plan.md` |
| Orquestrador | `experiments/streaming/scripts/run_experiment.py` |
| WindowAggregator (v1+v2) | `experiments/streaming/src/detector/window_aggregator.py` |
| Detector | `experiments/streaming/src/detector/streaming_detector.py` |
| Arquitetura v0.5.0 | `docs/architecture/CURRENT.md` |

---

## Roadmap (7 semanas até defesa)

| Semana | Foco |
|--------|------|
| S1 ✅ | Reorganização repo + campaign-plan |
| S2 ✅ | Campaign-01: 17 exps, baseline + diagnóstico |
| S3 ✅ | Campaign-02: 72 exps (IP GT + features + windows) |
| S4 ✅ | Campaign-03 S4: 48 exps (behavioral features) |
| S5 | Decisão S5/S6 + consolidação de resultados |
| S6-S7 | Escrita dissertação (caps 2-5) |
| S8 | Revisão final + preparação defesa |

---

## Referências Rápidas

- Como usar o repositório: `USAGE.md`
- Plano experimental: `experiments/campaign-plan.md`
- Metodologia: `experiments/methodology.md`
- Arquitetura do sistema: `docs/architecture/CURRENT.md`
- Leituras e lacunas: `research/reading-log.md`
