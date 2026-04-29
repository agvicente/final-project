# Campaign-05 — Análise de Resultados

**Data:** 2026-04-28
**Objetivo:** Comparar baselines de streaming (HST, LOF) com `micro_teda` e variantes V0/V4 no CICIoT2023, alimentando a Table VIII do paper SoftCom.
**Algoritmos alvo:** `halfspace_trees`, `lof`, `micro_teda`, `variant_V0_original`, `variant_V4_selective_update`
**Algoritmos leftover:** `isolation_forest`, `ocsvm`, `variant_V1_welford_var`, `variant_V3_welford_and_ecc` (de execuções anteriores incompletas)
**Dataset:** CICIoT2023 (Benign_Final + 5 ataques) | **Max flows:** 10.000 | **Seed:** 42 | **Ground truth:** IP-based
**Total de runs:** 37 (30 alvo + 7 leftover) | **Tempo total:** ~49h | **Failures:** 0

---

## 1. Resumo Executivo

| Algoritmo | F1 médio (5 ataques) | FPR benign-only | Throughput (flows/s) | Notas |
|-----------|----------------------|-----------------|----------------------|-------|
| **halfspace_trees** | **0.504** | 47.0% | **134.6** | Melhor F1 absoluto, FPR alto |
| variant_V4_selective_update | 0.453 | 48.0% | 16.9 | Marginalmente melhor que V0 |
| variant_V0_original | 0.448 | 48.7% | 16.4 | Baseline da implementação própria |
| ocsvm (1 run) | 0.282 | 23.7% | 781.5 | Mais rápido, mas só DDoS |
| isolation_forest (1 run) | 0.267 | 13.2% | 29.4 | Só DDoS, recall baixo |
| lof | 0.141 | 19.2% | **1.7** | **Inviável para streaming** |
| **micro_teda** | 0.098 | **3.8%** | 124.9 | **Detector silencioso** (recall=5%) |
| variant_V1_welford_var | 0.001 | 1.0% | 471.6 | **Quebrado** (Welford sozinho) |
| variant_V3_welford_and_ecc | — (só benign) | **72.1%** | — | **Quebrado** (Welford+ecc juntos) |

**Conclusão principal:** `halfspace_trees` é o **único baseline viável** (F1=0.504, throughput >100 flows/s). `micro_teda` e `lof` parecem "bem-comportados" pelo FPR baixo, mas isso é **artefato de undercalibration** — eles quase nunca alarmam (anomaly_rate 4-19% vs ataque real 25-63%).

---

## 2. Tabela Completa: Algoritmo × Ataque

### Runs alvo (30 = 5 algos × 6 cenários)

| Algoritmo | Cenário | Flows | Atk% real | Anom% prev | Prec | Rec | F1 | FPR | Throughput |
|-----------|---------|-------|-----------|------------|------|-----|-----|-----|------------|
| **halfspace_trees** | benign | 3214 | 0.0 | 47.0 | — | — | — | 0.470 | 71.8 |
| | ddos | 10000 | 63.1 | 82.1 | 0.756 | **0.983** | **0.854** | 0.544 | 136.3 |
| | mirai | 5445 | 37.2 | 52.4 | 0.404 | 0.568 | **0.472** | 0.498 | 102.9 |
| | recon | 10000 | 55.8 | 49.2 | 0.448 | 0.395 | **0.420** | 0.615 | 145.0 |
| | syn | 10000 | 63.1 | 19.3 | 0.357 | 0.109 | 0.167 | 0.336 | 145.3 |
| | tcp | 10000 | 37.1 | 79.5 | 0.445 | 0.953 | **0.607** | 0.701 | 143.6 |
| **lof** | benign | 3244 | 0.0 | 19.2 | — | — | — | 0.192 | 3.1 |
| | ddos | 10000 | 62.8 | 11.5 | 0.458 | 0.084 | 0.142 | 0.168 | 1.6 |
| | mirai | 6595 | 43.1 | 15.5 | 0.352 | 0.126 | 0.186 | 0.176 | 1.8 |
| | recon | 10000 | 53.4 | 11.9 | 0.346 | 0.077 | 0.126 | 0.167 | 1.7 |
| | syn | 10000 | 55.0 | 10.7 | 0.388 | 0.075 | 0.126 | 0.145 | 1.7 |
| | tcp | 10000 | 37.5 | 10.7 | 0.280 | 0.080 | 0.124 | 0.123 | 1.6 |
| **micro_teda** | benign | 3241 | 0.0 | 3.8 | — | — | — | **0.038** | 136.3 |
| | ddos | 10000 | 63.1 | 7.0 | **0.842** | 0.093 | 0.168 | **0.030** | 64.9 |
| | mirai | 4659 | 29.4 | 4.7 | 0.484 | 0.078 | 0.135 | 0.035 | 144.9 |
| | recon | 10000 | 55.7 | 4.1 | 0.698 | 0.052 | 0.097 | 0.028 | 103.0 |
| | syn | 10000 | 63.2 | 2.9 | 0.512 | 0.023 | 0.045 | 0.038 | 143.9 |
| | tcp | 10000 | 37.5 | 2.4 | 0.408 | 0.026 | 0.049 | 0.023 | 168.0 |
| **variant_V0** | benign | 3228 | 0.0 | 48.7 | — | — | — | 0.487 | 35.0 |
| | ddos | 10000 | 63.0 | 67.0 | 0.710 | 0.754 | 0.731 | 0.526 | 10.9 |
| | mirai | 4654 | 29.1 | 47.2 | 0.259 | 0.421 | 0.321 | 0.493 | 27.1 |
| | recon | 10000 | 55.7 | 36.1 | 0.428 | 0.277 | 0.336 | 0.467 | 14.7 |
| | syn | 10000 | 63.2 | 29.5 | 0.416 | 0.194 | 0.265 | 0.468 | 17.8 |
| | tcp | 10000 | 37.2 | 62.7 | 0.468 | 0.789 | 0.587 | 0.531 | 11.4 |
| **variant_V4** | benign | 3247 | 0.0 | 48.0 | — | — | — | 0.480 | 36.6 |
| | ddos | 10000 | 63.0 | 66.8 | 0.712 | 0.755 | 0.733 | 0.521 | 11.4 |
| | mirai | 4424 | 25.3 | 48.9 | 0.258 | 0.499 | 0.340 | 0.486 | 28.0 |
| | recon | 10000 | 55.9 | 34.9 | 0.425 | 0.266 | 0.327 | 0.455 | 15.6 |
| | syn | 10000 | 62.9 | 29.9 | 0.427 | 0.203 | **0.276** | 0.463 | 17.6 |
| | tcp | 10000 | 37.5 | 63.5 | 0.469 | 0.796 | 0.590 | 0.539 | 11.9 |

CSV completo: `metrics_summary.csv`

---

## 3. Insights Principais

### 3.1 HST domina F1 mas paga em FPR

`halfspace_trees` é o **único baseline com F1>0.4** em DDoS, Mirai, Recon e TCP. Mas tem FPR de 47-70% — alarma quase metade dos flows benignos como anômalos. Em produção isso é inaceitável; em comparação acadêmica é aceitável quando o ranking por F1 é o critério.

**Implicação:** Se a tese vai vender o detector evolutivo como "alternativa", precisa argumentar que o FPR de V0/V4 (~49%) é comparável ao de HST e que a operação adaptativa é o diferencial. Atualmente F1 de V0/V4 (~0.45) é levemente abaixo de HST (~0.50), mas dentro da mesma ordem de grandeza.

### 3.2 V0 ≈ V4 — adaptação isolada não ajuda em IoT

| Métrica | V0_original | V4_selective | Δ |
|---------|-------------|--------------|---|
| Avg F1 (5 ataques) | 0.448 | 0.453 | +0.005 |
| Avg FPR | 0.497 | 0.493 | -0.004 |
| Avg Throughput | 16.4 | 16.9 | +0.5 |

V4 (apenas `selective_update`) traz **melhora marginal não significativa** sobre V0. Isso **confirma a tese central do paper**: as 5 adaptações técnicas são **acopladas** — uma sozinha não move o ponteiro. Esse achado se conecta diretamente com Exp 2 (ablation V0-V7 no TEDA-HD), onde Friedman p<10⁻⁴⁰ mostra que a ordem/combinação importa.

### 3.3 micro_teda: "low FPR" é miragem (undercalibration)

A leitura ingênua dos dados sugere `micro_teda` como vencedor pelo FPR baixíssimo (3.8% benign-only, 2.3-3.8% em ataques). Mas a calibração revela o problema:

| Cenário | Atk% real | micro_teda anom% | Diff |
|---------|-----------|------------------|------|
| ddos | 63.1 | 7.0 | **-56** |
| mirai | 29.4 | 4.7 | -25 |
| recon | 55.7 | 4.1 | **-52** |
| syn | 63.2 | 2.9 | **-60** |
| tcp | 37.5 | 2.4 | -35 |

`micro_teda` rotula **<7% de qualquer cenário como anômalo**, mesmo quando 63% é ataque. O detector é praticamente um "trivial classifier" que diz "tudo normal". O FPR baixo vem disso, não de melhor discriminação. **Recall=5% é o número honesto.**

**Implicação para o paper:** A Table VIII deve incluir `anomaly_rate_predicted` vs `attack_rate_real` (calibração) — sem isso, o leitor é induzido a pensar que `micro_teda` é melhor que HST por FPR. Com calibração, fica óbvio que `micro_teda` é silencioso.

### 3.4 LOF é inviável em streaming (1.7 flows/s)

LOF processa 1.7 flows/segundo — **80× mais lento** que HST/micro_teda. Para 10k flows, gasta ~6.000s (1h40). Não escala para IoT real. Inclui no paper como referência clássica, mas marca como "infeasible for streaming volumes".

### 3.5 V1 e V3 confirmam acoplamento de adaptações

- **V1** (Welford apenas): F1=0.001 em DDoS (detectou 2 de 6303 ataques). Quebrado.
- **V3** (Welford + eccentricity adaptada): FPR=72% em benign-only. Quebrado em direção oposta.

V1 sozinho silencia o detector; V3 (V1+V2) o torna paranoico. Só V4+ começa a recuperar comportamento útil. **Esse é o achado-âncora do paper:** a fórmula original parece "frágil" porque adaptar uma coisa de cada vez quebra o equilíbrio que Maia 2020 estabilizou.

### 3.6 SYN flood é o ataque mais difícil

| Ataque | Avg F1 (8 algos) | Best F1 | Best algoritmo |
|--------|------------------|---------|----------------|
| ddos | 0.397 | 0.854 | halfspace_trees |
| tcp | 0.391 | 0.607 | halfspace_trees |
| mirai | 0.291 | 0.472 | halfspace_trees |
| recon | 0.261 | 0.420 | halfspace_trees |
| **syn** | **0.176** | **0.276** | variant_V4 |

SYN flood é estatisticamente o pior — **nenhum algoritmo passa de F1=0.28**. Isso faz sentido: SYN packets se confundem com tentativas legítimas de conexão. Vale destacar essa limitação na seção de discussão do paper.

---

## 4. Comparação com Campaign-04 (implementação original Maia 2020)

| Critério | Campaign-04 (original) | Campaign-05 V0_original | Campaign-05 micro_teda |
|----------|------------------------|-------------------------|------------------------|
| FPR benign-only | 54.4% | 48.7% | 3.8% |
| F1 médio em ataques | 19.7% | 44.8% | 9.8% |
| Anomaly rate em ataques | ~50-55% (constante) | 30-67% (variável) | 2-7% (silencioso) |

**Reading:**
- A reimplementação `variant_V0_original` (que tenta espelhar o algoritmo de Maia mais fielmente possível) tem FPR similar (~50%) à original (~55%), mas com F1 mais alto (44% vs 20%). Diferenças de implementação melhoram detection sem mudar o problema do FPR.
- `micro_teda` (implementação própria com 5 adaptações) trocou FPR alto por silêncio — não é mais ruído, é mais surdez.
- O **trade-off central** que o paper precisa nomear: as 5 adaptações de Maia 2020 são uma "calibração entre alarmar demais e calar demais", e isso não generaliza para todos os cenários.

---

## 5. Throughput — Viabilidade de Streaming

Ranking por capacidade de processamento (flows/s, média entre runs):

| Algoritmo | Avg flows/s | 10k flows ≈ |
|-----------|-------------|-------------|
| ocsvm | 781.5 | 13s |
| variant_V1 | 471.6 | 21s (mas quebrado) |
| halfspace_trees | 134.6 | 74s |
| micro_teda | 124.9 | 80s |
| isolation_forest | 29.4 | 5min |
| variant_V4 | 16.9 | 10min |
| variant_V0 | 16.4 | 10min |
| **lof** | **1.7** | **1h40** |

**Insight inesperado:** V0/V4 (implementação Python pura do TEDA evolutivo) é **8× mais lento** que `micro_teda`, embora ambos sejam algoritmos similares. A diferença está provavelmente em:
- micro_teda usa NumPy vetorizado para distância
- V0/V4 usa loops Python no `variants.py` para controlar flags toggle

Vale considerar otimizar V0/V4 antes de submeter, ou registrar essa diferença como "implementation overhead" no paper.

---

## 6. Decisão sobre os Leftovers

Os 7 diretórios de algoritmos não-alvo em `experiments/results/campaign-05/`:
- `isolation_forest-{benign,ddos}-seed42` — IF foi removido em favor de HST (commit `d602990`)
- `ocsvm-{benign,ddos}-seed42` — OC-SVM idem
- `variant_V1_welford_var-{benign,ddos}-seed42` — V1 testado parcialmente
- `variant_V3_welford_and_ecc-benign-seed42` — V3 testado só em benign

**Recomendação:** **Manter onde estão.** Eles não confundem o leitor (nomes auto-descritivos), ocupam ~7M (irrelevante), e fornecem dados de calibração úteis (V1=0.001 e V3=0.72 quantificam quanto cada bug isolado quebra o detector).

Alternativa: mover para `experiments/results/campaign-05-archive/` se a análise final do paper só citar HST/LOF/micro_teda/V0/V4.

---

## 7. Próximos Passos para o Paper SoftCom

1. **Preencher Table VIII** com as colunas: Algorithm | Precision | Recall | F1 | FPR | Throughput | (todas as 5×6=30 células dos algos alvo)
2. **Adicionar coluna "Calibration"** (predicted_anomaly_rate vs real_attack_rate) — diferencial não-trivial sobre baselines da literatura
3. **Gerar 2 figuras adicionais:**
   - Heatmap F1 (8 algos × 5 ataques) — mostra dominância de HST e silêncio de micro_teda
   - Scatter throughput vs F1 — coloca LOF no canto inferior-esquerdo (lento e ruim)
4. **Discussão:** Nomear o trade-off "alarmar demais vs calar demais" como contribuição teórica. As 5 adaptações de Maia são uma calibração entre esses dois extremos.

---

## 8. Arquivos

| Arquivo | Descrição | Status no repo |
|---------|-----------|----------------|
| `ANALYSIS.md` | Este documento | ✅ versionado |
| `metrics_summary.csv` | 37 runs × 23 colunas (input para plots) | ⚠️ gitignored (`*.csv` no `.gitignore` raiz) |
| `<run>/detection_results.json` | Métricas prequential por run | ✅ versionado |
| `<run>/run_meta.json` | Parâmetros + git commit + duração | ✅ versionado |
| `<run>/clusters_state.jsonl` | Estado dos clusters ao longo do tempo (TEDA-based) | ✅ versionado |
| `<run>/metrics_windowed.csv` | Métricas em janelas de tempo | ❌ **só no Linux** (gitignored) |
| `<run>/system_usage.csv` | CPU/memória por intervalo | ❌ **só no Linux** (gitignored) |

### ⚠️ Gap de dados: CSVs de evolução temporal não foram commitados

O `.gitignore` raiz exclui `*.csv` globalmente (linha 47). Isso foi deliberado para evitar commitar dumps de PCAP/dados crus, mas acabou excluindo também os outputs analíticos por-run:
- `metrics_windowed.csv` — métricas em janelas (precision/recall/f1 por janela de N segundos) → **necessário para análise de evolução temporal e plots de série temporal**
- `system_usage.csv` — CPU/memória ao longo do experimento → **necessário para reportar overhead computacional no paper**

**Recomendação:** Adicionar exceção no `.gitignore` para `experiments/results/**/*.csv`, depois re-push do Linux para trazer esses arquivos. Sugestão de patch:

```diff
 # Data files
 *.csv
 *.parquet
 *.pkl
 *.joblib
+
+# Permitir CSVs analíticos dentro de experiments/results/
+!experiments/results/**/*.csv
```

Sem isso, qualquer análise temporal (evolução de F1 ao longo do stream, comportamento adaptativo, picos de CPU) fica indisponível no Mac e tem que ser refeita ou transferida manualmente.
