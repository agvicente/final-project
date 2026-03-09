# Plano: Limpar vestígios do iot-ids-research + resolver PCAPs

**Criado:** 2026-03-09
**Executar na:** máquina Linux (onde `iot-ids-research/` ainda existe)
**Pré-requisito:** `git pull` para ter este plano + todas as mudanças recentes

---

## PASSO 0 — Inspecionar antes de editar (OBRIGATÓRIO)

Antes de qualquer edição, rodar os comandos abaixo na outra máquina para entender o estado real:

```bash
# 0.1 — Estrutura atual do iot-ids-research/
ls -la iot-ids-research/
ls -la iot-ids-research/data/raw/PCAP/
find iot-ids-research/data/raw/PCAP/ -name "*.pcap" -exec ls -lh {} \;

# 0.2 — Estrutura atual do data/ (nova)
ls -la data/
ls -la data/pcaps/ 2>/dev/null || echo "data/pcaps/ não existe"

# 0.3 — Verificar se há outros arquivos importantes em iot-ids-research/
find iot-ids-research/ -not -path "*/data/*" -not -path "*/__pycache__/*" -not -path "*/.git/*" -not -path "*/venv/*" -not -name "*.pyc" | head -50

# 0.4 — Verificar git status
git status

# 0.5 — Listar nomes exatos dos PCAPs (necessário para mapeamento)
find iot-ids-research/data/raw/PCAP/ -name "*.pcap" | sort
```

**Registrar a saída** destes comandos antes de prosseguir. Os nomes exatos dos PCAPs determinam o mapeamento correto no Passo 3.

---

## Passo 1 — Limpar `.gitignore` (remover regras `iot-ids-research/`)

**Arquivo:** `.gitignore` (linhas 47-110 aproximadamente)

Remover TODAS as linhas que contêm `iot-ids-research/`. São regras para um subdiretório que não existe mais no git. O `data/.gitignore` já ignora `*.pcap`.

**Linhas a remover:**
```
iot-ids-research/data/
iot-ids-research/models/
iot-ids-research/.dvc/cache/
iot-ids-research/.dvc/tmp/
iot-ids-research/.dvc/plots/
iot-ids-research/.dvc/logs/
iot-ids-research/experiments/logs/
iot-ids-research/experiments/temp/
iot-ids-research/experiments/artifacts/mlflow/
iot-ids-research/experiments/results/
iot-ids-research/experiments/results_test/
iot-ids-research/experiments/logs_test/
iot-ids-research/experiments/final_results/
iot-ids-research/experiments/final_results_test/
iot-ids-research/experiments/final_plots/
iot-ids-research/experiments/final_plots_test/
iot-ids-research/experiments/final_tables/
iot-ids-research/experiments/final_tables_test/
iot-ids-research/experiments/final_report/
iot-ids-research/experiments/final_report_test/
iot-ids-research/experiments/results/*/individual_analysis/
iot-ids-research/experiments/models/
iot-ids-research/experiments/artifacts/
iot-ids-research/experiments/mlflow-tracking/
iot-ids-research/experiments/.current_run_timestamp
```

---

## Passo 2 — Corrigir `.dvc/config`

**Arquivo:** `.dvc/config`

**Antes:**
```ini
[core]
    remote = local
    no_scm = false
['remote "local"']
    url = ../iot-ids-research/tmp/dvc-remote
```

**Depois:** (DVC só é usado na Fase 1, manter mínimo)
```ini
[core]
    no_scm = false
```

---

## Passo 3 — Padronizar paths de PCAP

**DEPENDE DO PASSO 0** — o mapeamento exato de nomes de pasta depende da inspeção.

Mapeamento esperado (confirmar com saída do Passo 0):
- `data/raw/PCAP/Benign/` → `data/pcaps/benign/`
- `data/raw/PCAP/DDoS/` → `data/pcaps/ddos/`
- `data/raw/PCAP/DoS/` → `data/pcaps/dos/`
- `data/raw/PCAP/Mirai/` → `data/pcaps/mirai/`
- `data/raw/PCAP/Recon/` → `data/pcaps/recon/`
- `data/raw/PCAP/Spoofing/` → `data/pcaps/spoofing/`

### Arquivos a editar (por prioridade):

**Alta prioridade:**
| Arquivo | O que mudar |
|---------|-------------|
| `CLAUDE.md` | Exemplos de `--pcap` e `--attack-pcap` |
| `experiments/campaign-plan.md` | Todos os comandos de experimento |
| `experiments/streaming/src/producer/config.py` | `DEV_PCAP` path |

**Média prioridade:**
| Arquivo | O que mudar |
|---------|-------------|
| `README.md` | Estrutura do repo + Quick Start |
| `STATUS.md` | Referência a `data/raw/PCAP/` |
| `experiments/streaming/tests/test_integration_event_time.py` | `PCAP_PATH` |

**Alta prioridade (contexto permanente — lido em toda sessão):**
| Arquivo | O que mudar |
|---------|-------------|
| `.claude/skills/iot-ids-research-context/SKILL.md` | Paths, exemplos, path absoluto → relativo, status campaign-plan |

**Baixa prioridade:**
| `experiments/streaming/docs/experiment-isolation.md` | Exemplos de uso |
| `docs/plans/2026-02-28-event-time-and-idle-timeout.md` | Paths nos exemplos |

**Nota sobre nomes de arquivos PCAP:** Os nomes dos ficheiros em si (ex: `BenignTraffic.pcap` vs `Benign_Final.pcap`) só podem ser confirmados no Passo 0. Usar os nomes reais encontrados.

---

## Passo 4 — Script de migração de PCAPs

**Arquivo a criar:** `scripts/migrate-pcaps.sh`

Script bash que:
1. Verifica que `iot-ids-research/data/raw/PCAP` existe
2. Para cada subpasta (Benign, DDoS, etc.):
   - Cria `data/pcaps/{benign,ddos,...}` se não existe
   - Move/copia PCAPs para o destino correto (lowercase)
3. Lista os PCAPs migrados
4. Pergunta se pode deletar `iot-ids-research/`

**NOTA:** O script deve ser criado APÓS o Passo 0 confirmar os nomes reais das pastas e arquivos.

---

## Passo 5 — Limpar referências em notebooks (baixa prioridade)

**Arquivos:** `labs/lab04-clustering/{kmeans,teda,concept-drift,dbscan}.ipynb`

Notebooks referenciam `../../iot-ids-research/data/processed/binary/`. São labs de aprendizado, não produção. Atualizar ou ignorar.

---

## Verificação final

```bash
# Após execução completa:
grep -r "iot-ids-research" . --include="*.py" --include="*.md" --include="*.sh" | grep -v _archive | grep -v .claude/plans | grep -v docs/plans
# Deve retornar 0 (ou apenas docs arquivados)

grep -r "data/raw/PCAP" . --include="*.py" --include="*.md" | grep -v _archive | grep -v .claude/plans | grep -v docs/plans
# Deve retornar 0

# Se o script de migração foi executado:
ls data/pcaps/*/
# Deve listar PCAPs em cada subpasta
```

---

## Ordem de execução

```
Passo 0 (inspecionar)  ← OBRIGATÓRIO antes de tudo
    ↓
Passos 1-2 (limpeza .gitignore + .dvc/config)  ← seguros, não dependem do Passo 0
    ↓
Passo 4 (script de migração)  ← depende do Passo 0 para nomes corretos
    ↓
Passo 3 (padronizar paths)  ← depende do Passo 0 para mapeamento correto
    ↓
Passo 5 (notebooks)  ← só se der tempo
```
