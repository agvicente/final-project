# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-07 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~8 semanas)

---

## Agora

**Reorganização do repositório para projeto científico**

O que foi feito nesta sessão:
- Repositório reorganizado em 3 pilares: `research/`, `experiments/`, `writing/`
- `streaming/` → `experiments/streaming/`, `baseline/` → `experiments/baseline/`
- Fichamentos e teoria movidos para `research/`
- Docs obsoletos arquivados em `docs/_archive/`
- `CLAUDE.md` reescrito com nova estrutura
- `USAGE.md` criado com guia de uso por cenário
- Hooks atualizados: auto-save removido, doc-check + progress archive implementados
- PCAPs removidos do git tracking (~2GB liberados do push)
- Methodology extraída (seções 1-7, 432 linhas) de documento de 2419 linhas
- Reading-log consolidado a partir de reading-plan + knowledge-gaps
- Bibliography.bib em geração (~220 referências do schedule.md)

**Próxima sessão — 3 ações:**
1. Verificar e ajustar `research/bibliography.bib` (gerado por agente)
2. Criar `experiments/campaign-plan.md` — plano experimental de 8 semanas alinhado com defesa
3. Clonar Overleaf da dissertação → `writing/dissertation/`

---

## Critérios de Sucesso (Reorganização)

| Critério | Status |
|----------|--------|
| Estrutura 3 pilares implementada | ✅ |
| Docs obsoletos arquivados | ✅ |
| CLAUDE.md e USAGE.md atualizados | ✅ |
| Hooks funcionando (sem auto-commit) | ✅ |
| PCAPs fora do git | ✅ |
| methodology.md extraída (cap. 4) | ✅ |
| reading-log.md consolidado | ✅ |
| bibliography.bib criado | ⏳ Em andamento |
| campaign-plan.md criado | ⏳ Pendente |
| Dissertação clonada do Overleaf | ⏳ Pendente |
| Testes do streaming passando com novos paths | ⏳ Pendente |

---

## Código Relevante Agora

| O quê | Onde |
|-------|------|
| Guia de uso do repositório | `USAGE.md` |
| Plano de reorganização (checklist) | `docs/plans/2026-03-07-repo-reorganization.md` |
| Metodologia científica (cap. 4) | `experiments/methodology.md` |
| Detector TEDA + MicroTEDAclus | `experiments/streaming/src/detector/` |
| Métricas prequential | `experiments/streaming/src/metrics/` |
| Orquestrador de experimentos | `experiments/streaming/scripts/run_experiment.py` |
| Teoria consolidada | `research/foundations/` |
| Fichamentos | `research/summaries/` |

---

## Roadmap (8 semanas até defesa)

| Semana | Foco |
|--------|------|
| S1 (atual) | Reorganização repo + campaign-plan |
| S2 | Validar arquitetura + métricas experimentais |
| S3-S4 | Campanha experimental (cenários A-E) |
| S5-S6 | Análise de resultados + comparação com literatura |
| S7 | Escrita dissertação (caps 2-5) |
| S8 | Revisão final + preparação defesa |

---

## Referências Rápidas

- Como usar o repositório: `USAGE.md`
- Plano detalhado desta reorganização: `docs/plans/2026-03-07-repo-reorganization.md`
- Arquitetura do sistema: `docs/architecture/CURRENT.md`
- Leituras e lacunas: `research/reading-log.md`
