# Como Usar Este Repositório

## Estrutura: Três Pilares

```
research/       → CONHECIMENTO (o que sei)
experiments/    → EVIDÊNCIA (o que comprovei)
writing/        → PRODUÇÃO (o que publico)
```

## Cenários de Uso

### "Onde parei? O que faço agora?"

```
STATUS.md → seções "Agora" e "Próxima sessão"
```

### "Preciso escrever um capítulo da dissertação"

```
1. experiments/methodology.md      → base para Cap. 4 (Metodologia)
2. research/foundations/            → base para Cap. 2 (Fundamentação Teórica)
3. research/summaries/              → fichamentos para citar no texto
4. research/bibliography.bib       → referências prontas para \cite{}
5. experiments/results/campaign-01/ → dados para gerar figuras e tabelas
6. writing/figures/ + tables/       → artefatos prontos para incluir no LaTeX
```

### "Preciso rodar/analisar experimentos"

```
1. experiments/campaign-plan.md     → o que rodar, em que ordem, com que parâmetros
2. experiments/methodology.md       → cenários (A-E), métricas, protocolo
3. experiments/streaming/scripts/   → scripts de execução
4. experiments/results/campaign-01/ → onde salvar resultados
```

### "Preciso preparar algo para o orientador"

```
1. STATUS.md                        → resumo do estado atual
2. docs/progress/                   → histórico do que foi feito (automático)
3. experiments/results/campaign-01/  → resultados para mostrar
4. writing/figures/                  → gráficos prontos
```

### "Preciso estudar um tema / ler um paper"

```
1. research/reading-log.md          → o que ler e por quê
2. research/summaries/              → fichamentos já feitos
3. research/bibliography.bib       → referência completa do paper
4. research/foundations/             → teoria consolidada
```

## Regra Geral

| Pergunta | Onde |
|----------|------|
| Onde estou? | `STATUS.md` |
| O que sei? | `research/` |
| O que comprovei? | `experiments/results/` |
| O que publico? | `writing/` |
| O que aconteceu? | `docs/progress/` |
