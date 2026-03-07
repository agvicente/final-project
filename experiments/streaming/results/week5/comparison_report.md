# Relatório de Comparação de Experimentos

## Resumo

Total de experimentos: 8

## Resultados

| Experimento | Algoritmo | r0 | Flows | F1 | Precision | Recall | FPR | MTTD | #Clusters |
|-------------|-----------|-------|-------|--------|-----------|--------|-----|------|----------|
| sanity_quick | micro_teda | 0.10 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.4900 | N/A | 58 |
| consolidation_test | micro_teda | 0.10 | 2000 | 0.0000 | 0.0000 | 0.0000 | 0.0435 | N/A | 96 |
| grid_micro_teda_r0_0.05 | micro_teda | 0.05 | 2808 | 0.0000 | 0.0000 | 0.0000 | 0.0296 | N/A | 92 |
| grid_teda_r0_0.05 | teda | 0.05 | 2807 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | N/A | 0 |
| grid_teda_r0_0.10 | teda | 0.10 | 2807 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | N/A | 0 |
| grid_micro_teda_r0_0.10 | micro_teda | 0.10 | 2809 | 0.0000 | 0.0000 | 0.0000 | 0.0306 | N/A | 95 |
| grid_teda_r0_0.20 | teda | 0.20 | 2805 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | N/A | 0 |
| grid_micro_teda_r0_0.20 | micro_teda | 0.20 | 2809 | 0.0000 | 0.0000 | 0.0000 | 0.0299 | N/A | 93 |

## Observações

- Experimentos ordenados por F1-Score (melhor primeiro)
- FPR = False Positive Rate (menor é melhor)
- MTTD = Mean Time To Detection em segundos (menor é melhor)
