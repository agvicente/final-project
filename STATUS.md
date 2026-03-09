# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-03-09 | **Branch:** main | **Prazo defesa:** ~maio 2026 (~8 semanas)

---

## Agora

**Transição S1→S2: limpeza de paths antigos + migração de PCAPs**

Sessão 09/03:
- `experiments/campaign-plan.md` criado (cenários A/B/E obrigatórios, D se der tempo, C cortado)
- Plano de limpeza criado: `docs/plans/2026-03-09-cleanup-old-paths.md`
- Repo tem vestígios de `iot-ids-research/` e paths `data/raw/PCAP/` que precisam ser limpos

**Próxima sessão (executar na máquina Linux):**

Fase A — Limpeza (pré-requisito, ~30min):
1. **Executar plano de limpeza** (`docs/plans/2026-03-09-cleanup-old-paths.md`): Passo 0 primeiro (inspecionar estrutura real), depois limpar `.gitignore`, `.dvc/config`, padronizar paths, criar script de migração de PCAPs
2. **Migrar PCAPs** de `iot-ids-research/data/raw/PCAP/` → `data/pcaps/` (PCAPs existem na máquina Linux, não nesta)
3. Rodar testes do streaming para confirmar que paths novos funcionam

Fase B — Retomar sequência do projeto:
4. Substituir `research/bibliography.bib` pelo export do Zotero (completar incrementalmente)
5. Fazer symlink `writing/dissertation/` → Overleaf ou copiar (decidir com Augusto)
6. **Iniciar S2** — Cenário A (detecção básica) seguindo `experiments/campaign-plan.md`

**Contexto importante:** Na máquina Linux, `iot-ids-research/` ainda existe com os PCAPs (~548GB). O git não deleta arquivos não-versionados. O plano de limpeza inclui um Passo 0 de inspeção obrigatória antes de qualquer edição. Após a limpeza, os comandos do `campaign-plan.md` estarão com paths corretos e prontos para uso.

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
| Paths `iot-ids-research/` limpos do repo | ⏳ Plano criado — `docs/plans/2026-03-09-cleanup-old-paths.md` |
| Paths `data/raw/PCAP/` → `data/pcaps/` | ⏳ Plano criado |
| PCAPs migrados para `data/pcaps/` | ❌ Bloqueante (PCAPs existem na máquina Linux) |
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
- **Plano de limpeza (próxima ação):** `docs/plans/2026-03-09-cleanup-old-paths.md`
- Plano da reorganização (concluído): `docs/plans/2026-03-07-repo-reorganization.md`
- Arquitetura do sistema: `docs/architecture/CURRENT.md`
- Leituras e lacunas: `research/reading-log.md`
