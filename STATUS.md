# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-16 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~7 semanas)

---

## Agora

**S3 — Campaign-02 implementada: 3 melhorias incrementais para resolver Recall ~3-4%.**

Sessão 16/03 — implementação de código (pronto para rodar na Linux):
- ✅ Fix `extract_attack_ips.py`: double-read bug, `--pcap-files`, default 500k, progress logging
- ✅ Feature expansion: v1 (17), v2 (25), v3 (32) feature sets no streaming_detector
- ✅ FlowConsumer: 4 novas features IAT direcionais (fwd/bwd_iat_mean/std)
- ✅ WindowAggregator: novo módulo de detecção por janela temporal (12 features agregadas)
- ✅ Integration: `--features`, `--granularity`, `--window-seconds` no run_experiment.py
- ✅ 110 testes passando (12 novos para WindowAggregator)
- ✅ Documentação atualizada: architecture v0.4.0, campaign-plan, methodology

**Próxima sessão (máquina Linux):**
1. Rodar `extract_attack_ips.py --pcap-files` para os 6 PCAPs da campaign-02
2. Executar Step 1 (6 exps com ground truth IP, features v1) para baseline
3. Executar Step 2 (calibração r0 com features v2, depois 5 ataques)

---

## Critérios de Sucesso (Campaign-02)

| Critério | Status |
|----------|--------|
| Step 0: extract_attack_ips.py fixado | ✅ Implementado |
| Step 1: Ground truth IP confirma Recall | ⬜ Pendente (rodar na Linux) |
| Step 2: Features v2 melhora Recall | ⬜ Pendente |
| Step 3: Window detection melhora Recall | ⬜ Pendente |
| 110 testes passando | ✅ |
| Código commitado e pronto para Linux | ⬜ Pendente commit |

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
