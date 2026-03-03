# Quick Start - Próxima Sessão (2026-02-26)

**Status Atual:** Sistema 100% funcional com isolamento garantido entre experimentos

---

## 🚀 Tarefas da Próxima Sessão

### Pré-requisito: Kafka Rodando

```bash
cd /Users/augusto/mestrado/final-project
docker-compose up -d kafka zookeeper

# Aguardar ~30s
sleep 30

# Verificar
docker ps | grep kafka
```

---

## 1. Grid 3×2: TEDA vs MicroTEDAclus (6 experimentos)

**Tempo estimado:** 60 minutos total (~10 min cada)

```bash
cd /Users/augusto/mestrado/final-project/streaming
source venv/bin/activate

# Loop automático
for algo in teda micro_teda; do
    for r0 in 0.05 0.10 0.20; do
        echo "=========================================="
        echo "Executando: $algo com r0=$r0"
        echo "=========================================="

        python3 scripts/run_experiment.py \
            --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
            --max-packets 50000 \
            --max-flows 5000 \
            --algorithm $algo \
            --r0 $r0 \
            --output ../results/week5/grid_${algo}_r0_${r0}/

        echo "✅ Completo: $algo r0=$r0"
        echo ""
    done
done

echo "=========================================="
echo "✅ GRID COMPLETO - 6 experimentos executados"
echo "=========================================="
```

**Output esperado:**
```
results/week5/
├── grid_teda_r0_0.05/
├── grid_teda_r0_0.10/
├── grid_teda_r0_0.20/
├── grid_micro_teda_r0_0.05/
├── grid_micro_teda_r0_0.10/
└── grid_micro_teda_r0_0.20/
```

---

## 2. Comparação de Resultados

```bash
python3 scripts/compare_experiments.py ../results/week5/
```

**Output:** Tabela terminal + `../results/week5/comparison_report.md`

---

## 3. Experimento com Ataque DDoS

**Tempo estimado:** 15 minutos

```bash
python3 scripts/run_experiment.py \
    --pcap ../data/raw/PCAP/Benign/BenignTraffic.pcap \
    --attack-pcap ../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
    --max-packets 50000 \
    --max-flows 10000 \
    --algorithm micro_teda \
    --r0 0.10 \
    --output ../results/week5/ddos_detection/
```

**Verificar:**
```bash
# Métricas
jq '.prequential_metrics' ../results/week5/ddos_detection/detection_results.json

# Esperado:
# - Precision: > 0.70
# - Recall: > 0.80
# - F1: > 0.75
# - MTTD: < 500 flows
```

---

## 4. Análise Visual (Opcional)

Se tiver tempo, criar gráficos básicos:

```python
# notebooks/week5_analysis.ipynb (criar se necessário)
import json
import pandas as pd
import matplotlib.pyplot as plt

# Carregar resultados do grid
results = []
for algo in ['teda', 'micro_teda']:
    for r0 in [0.05, 0.10, 0.20]:
        path = f'../results/week5/grid_{algo}_r0_{r0}/detection_results.json'
        with open(path) as f:
            data = json.load(f)
            results.append({
                'algorithm': algo,
                'r0': r0,
                'num_clusters': data['detector_stats']['num_clusters'],
                'anomaly_rate': data['detector_stats']['anomaly_count'] / data['detector_stats']['total_samples']
            })

df = pd.DataFrame(results)
print(df)

# Gráfico: Anomaly Rate vs r0
df.pivot(index='r0', columns='algorithm', values='anomaly_rate').plot(kind='bar')
plt.title('Anomaly Rate por Algoritmo e r0')
plt.ylabel('Anomaly Rate')
plt.savefig('../results/week5/anomaly_rate_comparison.png')
```

---

## ✅ Checklist de Conclusão

Ao final da sessão, verificar:

- [ ] 6 experimentos do grid executados com sucesso
- [ ] `comparison_report.md` gerado
- [ ] Experimento DDoS executado
- [ ] Métricas DDoS validadas (Recall > 80%, MTTD < 500)
- [ ] Atualizar `docs/weekly-reports/semana5-report.md` com resultados finais
- [ ] Commit das mudanças

---

## 📊 Critérios de Sucesso

| Métrica | Alvo | Status |
|---------|------|--------|
| FPR (benign) | <= 5% | ✅ 4.3% (validado hoje) |
| Recall (DDoS) | >= 80% | ⏳ Pendente |
| MTTD | <= 500 flows | ⏳ Pendente |
| Diferença TEDA vs MicroTEDAclus | Detectável | ⏳ Pendente |

---

## 🔧 Troubleshooting

### Kafka não conecta
```bash
docker-compose down
docker-compose up -d kafka zookeeper
sleep 30
```

### Experimento trava
- Usar `--max-flows` menor (ex: 2000) para teste rápido
- Verificar logs: `docker logs <container_id>`

### Erro "attack-pcap none"
- Já corrigido! Mas se ocorrer, simplesmente omita `--attack-pcap`

---

## 📁 Documentação Relacionada

- **Relatório Semanal:** `docs/weekly-reports/semana5-report.md`
- **Contexto:** `docs/SESSION_CONTEXT.md`
- **Metodologia:** `docs/methodology/experiment-methodology.md` (Seção 8.1)
- **Isolamento:** `streaming/docs/experiment-isolation.md`

---

**Tempo Total Estimado:** 90 minutos
**Complexidade:** Baixa (comandos prontos, sistema validado)
**Objetivo:** Completar Semana 5 com resultados comparativos sólidos
