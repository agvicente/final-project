# Plano: Reorganização do Repositório para Projeto Científico

**Data:** 2026-03-07
**Status:** Em execução
**Prazo da defesa:** ~2 meses (maio 2026)
**Branch:** main

---

## Contexto

O repositório cresceu organicamente como ferramenta de desenvolvimento. Precisa ser transformado
em máquina de produção científica com três pilares: CONHECIMENTO, EVIDÊNCIA, PRODUÇÃO.

Referência visual: `USAGE.md` na raiz.

---

## Estrutura Alvo

```
final-project/
├── STATUS.md
├── CLAUDE.md
├── USAGE.md
├── README.md
│
├── research/                         ← CONHECIMENTO
│   ├── bibliography.bib              ← consolidar refs do schedule.md + Zotero
│   ├── reading-log.md                ← fundir reading-plan.md + KNOWLEDGE_GAPS.md
│   ├── summaries/                    ← mover de docs/paper-summaries/
│   │   ├── angelov-2014-teda.md
│   │   ├── maia-2020-microtedaclus.md
│   │   └── gama-2013-prequential-evaluation.md
│   └── foundations/                  ← mover de docs/theory/
│       ├── teda-framework.md
│       └── concept-drift.md
│
├── experiments/                      ← EVIDÊNCIA
│   ├── methodology.md                ← extrair seções 1-7 do experiment-methodology.md
│   ├── campaign-plan.md              ← CRIAR: plano de 8 semanas (substituir seção 8)
│   ├── streaming/                    ← mover de streaming/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── scripts/
│   │   └── docker/
│   ├── baseline/                     ← mover de baseline/
│   └── results/                      ← resultados das campanhas experimentais
│       └── campaign-01/
│
├── writing/                          ← PRODUÇÃO
│   ├── dissertation/                 ← clone do Overleaf
│   ├── figures/                      ← geradas pelos experimentos
│   └── tables/                       ← geradas pelos experimentos
│
├── docs/                             ← OPERACIONAL (reduzido)
│   ├── progress/                     ← logs automáticos
│   └── architecture/                 ← CURRENT.md
│
└── .claude/                          ← hooks e config
```

---

## Tarefas

### Fase 1: Mover (mecânico, sem risco)

- [x] T1.1: Criar estrutura de diretórios alvo
- [x] T1.2: Mover `docs/paper-summaries/*` → `research/summaries/`
- [x] T1.3: Mover `docs/theory/teda-framework.md` e `concept-drift.md` → `research/foundations/`
- [x] T1.4: Mover `streaming/` → `experiments/streaming/`
- [x] T1.5: Mover `baseline/` → `experiments/baseline/`
- [x] T1.6: Copiar `experiment-methodology.md` → `experiments/experiment-methodology-full.md`
- [x] T1.7: `docs/architecture/` fica onde está
- [x] T1.8: `docs/progress/` fica onde está
- [x] T1.9: Criar `writing/dissertation/`, `writing/figures/`, `writing/tables/`
- [x] T1.10: Criar `experiments/results/campaign-01/`

### Fase 2: Limpar (mecânico, baixo risco)

- [x] T2.1: Mover para `docs/_archive/` (todos arquivados)
- [x] T2.2: Mover `schedule.md` → `docs/_archive/schedule.md`
- [x] T2.3: Diretórios vazios removidos
- [x] T2.4: Diretórios vazios removidos

### Fase 3: Consolidar (requer cuidado)

- [x] T3.1: Extrair seções 1-7 → `experiments/methodology.md` (432 linhas)
- [x] T3.2: Fundir reading-plan + KNOWLEDGE_GAPS → `research/reading-log.md`
- [x] T3.3: Extrair referências → `research/bibliography.bib` (147 entradas BibTeX)

### Fase 4: Criar (requer input do Augusto)

- [ ] T4.1: `experiments/campaign-plan.md` — plano experimental de 8 semanas
  (**PENDENTE** — precisa de brainstorming com Augusto)
- [ ] T4.2: Clonar Overleaf → `writing/dissertation/`
  (**PENDENTE** — Augusto precisa fornecer URL do Overleaf)
- [x] T4.3: Atualizar `CLAUDE.md` com nova estrutura de diretórios
- [ ] T4.4: Atualizar `STATUS.md` com novo estado
- [ ] T4.5: Atualizar hooks (paths não mudaram nos hooks)

### Fase 5: Verificação

- [ ] T5.1: Rodar testes do streaming (`cd experiments/streaming && python -m pytest`)
- [ ] T5.2: Verificar que imports/paths nos scripts ainda funcionam
- [ ] T5.3: Verificar que docker-compose ainda funciona
- [ ] T5.4: Commit final e push

---

## Dependências

```
Fase 1 (mover) → Fase 2 (limpar) → Fase 3 (consolidar) → Fase 4 (criar) → Fase 5 (verificar)
```

Fases 1 e 2 podem rodar em paralelo parcialmente.
Fase 3 pode usar agentes paralelos (T3.1, T3.2, T3.3 são independentes).
Fase 4 requer input do Augusto para T4.1 e T4.2.

---

## Notas

- O `evolving_clustering/` (repo externo do Maia et al.) NÃO é movido agora.
  Será integrado depois de resultados analisáveis com o algoritmo atual.
- Resultados experimentais anteriores (streaming/results/week5/) são descartáveis.
  A campanha experimental começa do zero.
- O `artigo1/` é histórico, publicado. Não é tocado.
