# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-09 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~8 semanas)

---

## Agora

**S1 concluída — Limpeza feita, pronto para S2**

Sessão 09/03 (máquina Linux):
- ✅ Plano de limpeza executado (`docs/plans/2026-03-09-cleanup-old-paths.md`)
- ✅ `.gitignore` limpo (removidas todas as regras `iot-ids-research/`)
- ✅ `.dvc/config` limpo (removido remote apontando para `iot-ids-research/`)
- ✅ PCAPs migrados: `iot-ids-research/data/raw/PCAP/` → `data/pcaps/` (34 pastas, 547GB)
- ✅ Paths atualizados em: `campaign-plan.md`, `CLAUDE.md`, `SKILL.md`, `config.py`, testes, docs
- ✅ `venv` criado em `experiments/streaming/venv/`
- ✅ 98 testes passando

**Próxima sessão:**
1. Subir Kafka (`cd experiments/streaming/docker && docker-compose up -d`)
2. **Iniciar S2** — Cenário A1 (baseline benigno): seguir `experiments/campaign-plan.md`
3. Substituir `research/bibliography.bib` pelo export do Zotero (baixa prioridade)

**Nota:** `iot-ids-research/` ainda existe na máquina Linux mas está vazio de PCAPs. Pode ser deletado quando conveniente. O `.gitignore` já o ignora.

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
| bibliography.bib criado | ⚠️ Esqueleto — usar Zotero como fonte |
| campaign-plan.md criado | ✅ |
| Paths `iot-ids-research/` limpos do repo | ✅ |
| Paths `data/raw/PCAP/` → `data/pcaps/` | ✅ |
| PCAPs migrados para `data/pcaps/` | ✅ 34 pastas, 547GB |
| Dissertação clonada do Overleaf | ⏳ Pendente |
| Testes do streaming passando com novos paths | ✅ 98 passed |

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
| S1 ✅ | Reorganização repo + campaign-plan |
| S2 | Validar arquitetura + métricas experimentais |
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
