# CLAUDE.md

Instruções para o Claude Code neste repositório de pesquisa de mestrado.

## Projeto

Dissertação de Mestrado — UFMG PPGEE
**Tema:** Detecção de intrusão baseada em anomalias em IoT com clustering evolutivo (TEDA/MicroTEDAclus) e streaming (Kafka).
**Prazo:** Defesa em ~maio 2026 (~8 semanas restantes).
**Fase atual:** 2B — Experimentos Streaming. Fase 1 completa (705 exps, F1>0.99, artigo publicado).

## Estrutura do Repositório

```
research/           ← CONHECIMENTO (teoria, fichamentos, referências)
experiments/        ← EVIDÊNCIA (código, testes, resultados)
  streaming/        ← Fase 2: Kafka + TEDA (código ativo)
  baseline/         ← Fase 1: ML clássico (referência, intocado)
  results/          ← resultados das campanhas experimentais
writing/            ← PRODUÇÃO (dissertação, figuras, tabelas)
docs/               ← OPERACIONAL (progress logs, arquitetura)
```

Referência completa de uso: `USAGE.md`

## Documentation Maintenance (OBRIGATÓRIO)

**Leia sempre primeiro:** `STATUS.md` — responde onde estamos e o que fazer agora.

**Ao final de qualquer sessão significativa, SUBSTITUA as seções dinâmicas do `STATUS.md`:**
- Seção "Agora": SUBSTITUIR com o que foi feito (não acumular)
- Seção "Próxima sessão": SUBSTITUIR com os próximos 3 passos concretos
- Seção "Critérios de Sucesso": atualizar status dos itens

**STATUS.md é um snapshot, não um log.** Deve refletir apenas o estado ATUAL.
O histórico fica em `docs/progress/` (gerado automaticamente pelo hook de SessionEnd).

**Nunca deixe STATUS.md desatualizado.** Um hook `Stop` vai lembrar se houver mudanças pendentes.

## Essential Commands

### Streaming Experiments (Fase 2 — foco atual)

```bash
# Ambiente
cd experiments/streaming && source venv/bin/activate

# Kafka
cd docker && docker-compose up -d  # Kafka + Zookeeper + Kafka-UI (localhost:8080)

# Testes
python -m pytest tests/ -v

# Extrair IPs de atacantes dos PCAPs (executar 1x na máquina Linux)
python3 scripts/extract_attack_ips.py --pcap-dir ../../data/pcaps/

# Rodar experimento (ground truth por IP = default, fallback para phase)
python scripts/run_experiment.py \
  --pcap ../../data/pcaps/Benign_Final/BenignTraffic.pcap \
  --attack-pcap ../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap \
  --max-packets 50000 --max-flows 10000 \
  --algorithm micro_teda \
  --output ../../experiments/results/campaign-02/A2-ddos-r0_0.10/

# Comparar resultados
python scripts/compare_experiments.py ../../experiments/results/campaign-01/
```

### Baseline (Fase 1 — referência)

```bash
cd experiments/baseline
dvc repro                              # Pipeline completo
python3 experiments/run_single_algorithm.py <algorithm_name>  # Experimento individual
```

## Key Files

| O quê | Onde |
|-------|------|
| Estado atual | `STATUS.md` |
| Como usar o repo | `USAGE.md` |
| Metodologia científica | `experiments/methodology.md` |
| Plano experimental | `experiments/campaign-plan.md` |
| Detector TEDA + MicroTEDAclus | `experiments/streaming/src/detector/` |
| Métricas prequential | `experiments/streaming/src/metrics/` |
| Orquestrador de experimentos | `experiments/streaming/scripts/run_experiment.py` |
| Teoria TEDA/MicroTEDAclus | `research/foundations/` |
| Fichamentos | `research/summaries/` |
| Referências bibliográficas | `research/bibliography.bib` |
| Arquitetura do sistema | `docs/architecture/CURRENT.md` |
| Dissertação (LaTeX/Overleaf) | `writing/dissertation/` |

## Regras

- Sempre que modificar arquitetura do streaming, atualizar `docs/architecture/CURRENT.md`
- PCAPs e PDFs NÃO são versionados (ver `data/.gitignore`)
- Resultados experimentais antigos (antes de campaign-01) são descartáveis
- O `experiments/baseline/` não deve ser modificado sem necessidade explícita

## Skills Disponíveis

- `/resume` — Carrega contexto atual e próximos passos
- `/start-sprint` — Inicia nova semana de trabalho
- `/finalize-week` — Gera relatório para orientador
- `/paper-summary [nome]` — Resume paper do Zotero

## Hooks

- **SessionStart** — Grava commit de referência para tracking de progresso
- **Stop** — Verifica se STATUS.md foi atualizado (lembrete, não commita)
- **SessionEnd** — Arquiva STATUS.md + git activity em `docs/progress/`
