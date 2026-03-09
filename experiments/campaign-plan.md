# Plano de Campanha Experimental

**Criado:** 2026-03-08
**Prazo defesa:** ~maio 2026 (~8 semanas)
**Metodologia:** ver `experiments/methodology.md`

---

## Objetivo

Produzir evidencias experimentais suficientes para 3 contribuicoes da dissertacao:

1. **Deteccao funciona** — MicroTEDAclus detecta ataques IoT em streaming
2. **Adaptacao a drift** — sistema se adapta a mudancas abruptas de padrao
3. **Generalizacao** — sistema detecta ataques nunca vistos (zero-day)

---

## Cenarios Priorizados

| Prioridade | Cenario | O que testa | Semana |
|------------|---------|-------------|--------|
| **Obrigatorio** | A - Deteccao Basica | Sistema detecta ataques? | S2 |
| **Obrigatorio** | B - Drift Subito | Adapta a mudanca abrupta? | S3 |
| **Obrigatorio** | E - Zero-Day | Detecta ataques nunca vistos? | S3-S4 |
| Se der tempo | D - Drift Recorrente | "Lembra" padroes anteriores? | S4 |
| **Cortado** | C - Drift Gradual | Complexo, valor marginal vs B | — |

---

## Semana 2: Validacao de Arquitetura + Cenario A

### Objetivo
Confirmar que o pipeline funciona end-to-end e produzir primeiros resultados publicaveis.

### Pre-requisitos
- [ ] Testes do streaming passando (97 passed — OK)
- [ ] Docker Kafka operacional
- [ ] PCAPs disponiveis: Benign, DDoS-ICMP_Flood

### Experimentos

**A1: Baseline benigno (falsos positivos)**
```bash
cd experiments/streaming && source venv/bin/activate
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A1-benign/
```
- Criterio de sucesso: FPR <= 5%
- Repetir com r0 = {0.05, 0.10, 0.15, 0.20} para calibrar

**A2: Deteccao DDoS**
```bash
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --attack-pcap ../../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm micro_teda --r0 0.10 \
  --output ../results/campaign-01/A2-ddos/
```
- Criterio de sucesso: Recall >= 80%, MTTD <= 500 flows
- Repetir com melhor r0 encontrado em A1

**A3: Comparacao TEDA vs MicroTEDAclus**
```bash
# Mesmo PCAP, mesmo protocolo, algoritmo diferente
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --attack-pcap ../../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
  --max-packets 100000 --max-flows 10000 \
  --algorithm teda --r0 0.10 \
  --output ../results/campaign-01/A3-teda-baseline/
```
- Comparar: F1, FPR, MTTD, throughput, memoria

### Entregaveis S2
- [ ] Tabela comparativa TEDA vs MicroTEDAclus (Cenario A)
- [ ] Grafico temporal F1 (prequential) para ambos algoritmos
- [ ] r0 otimo calibrado
- [ ] Decisao: prosseguir com MicroTEDAclus ou ajustar

---

## Semana 3: Cenario B (Drift Subito) + Inicio Cenario E

### Pre-requisitos
- [ ] Cenario A completo com resultados satisfatorios
- [ ] PCAPs disponiveis: Benign, DDoS, Mirai (pelo menos 1 variante)

### Experimentos

**B1: Drift DDoS → Mirai**
```bash
# Fase 1: Treinar com DDoS
# Fase 2: Testar com Mirai (drift subito)
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --attack-pcap ../../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
  --drift-pcap ../../data/raw/PCAP/Mirai/[variante].pcap \
  --max-flows 20000 \
  --algorithm micro_teda --r0 [otimo de A1] \
  --output ../results/campaign-01/B1-drift-ddos-mirai/
```
- Metricas: F1 antes/depois do drift, tempo de adaptacao (flows ate recuperar 95% F1)
- **NOTA:** pode ser necessario implementar `--drift-pcap` no orquestrador

**B2: Drift Mirai → Recon** (se houver PCAPs)

**E1: Zero-day com holdout**
```bash
# Treinar: Benign + DDoS + DoS
# Testar: Mirai + Recon (nunca vistos)
python3 scripts/run_experiment.py \
  --pcap ../../data/raw/PCAP/Benign/BenignTraffic.pcap \
  --attack-pcap ../../data/raw/PCAP/DDoS/DDoS-ICMP_Flood.pcap \
  --holdout-pcap ../../data/raw/PCAP/Mirai/[variante].pcap \
  --max-flows 20000 \
  --algorithm micro_teda --r0 [otimo] \
  --output ../results/campaign-01/E1-zero-day/
```
- Metricas: taxa de deteccao de zero-day, FPR
- **NOTA:** pode ser necessario implementar `--holdout-pcap` no orquestrador

### Entregaveis S3
- [ ] Grafico de adaptacao ao drift (F1 temporal com marcador de drift)
- [ ] Tempo de adaptacao medido
- [ ] Taxa de deteccao zero-day

---

## Semana 4: Finalizar E + Cenario D (se der tempo)

### Experimentos

**E2: Mais combinacoes de holdout**
- Treinar com {Benign + Mirai}, testar com {DDoS} — inverte
- Treinar com {Benign + DoS}, testar com {Spoofing}

**D1: Drift recorrente** (se der tempo)
```
Fase 1: DDoS (aprende)
Fase 2: Benign (periodo calmo)
Fase 3: DDoS (deve reconhecer mais rapido)
Fase 4: Benign
Fase 5: DDoS (deve reconhecer ainda mais rapido)
```
- Metrica: tempo de reconhecimento decresce a cada retorno?

### Entregaveis S4
- [ ] Tabela consolidada de todos os cenarios
- [ ] Analise estatistica (media +- desvio, 5 runs por config)
- [ ] Dados prontos para gerar figuras da dissertacao

---

## Semanas 5-6: Analise e Figuras

### Tarefas
- [ ] Gerar figuras para dissertacao (`writing/figures/`)
  - Grafico temporal F1 (cenario A)
  - Grafico de adaptacao ao drift (cenario B)
  - Tabela comparativa TEDA vs MicroTEDAclus
  - Grafico de deteccao zero-day (cenario E)
- [ ] Gerar tabelas (`writing/tables/`)
  - Tabela de parametros otimos
  - Tabela de resultados por cenario
  - Tabela comparativa com literatura
- [ ] Comparar resultados com papers da area (seção de trabalhos relacionados)
- [ ] Escrever capitulo de Resultados (cap. 5)

---

## Semanas 7-8: Dissertacao + Defesa

### Tarefas
- [ ] Finalizar capitulos 2-5
- [ ] Revisao geral
- [ ] Preparar apresentacao de defesa
- [ ] Rehearsal com orientador

---

## PCAPs Necessarios

| PCAP | Cenarios | Disponivel? |
|------|----------|-------------|
| Benign/BenignTraffic.pcap | Todos | Verificar |
| DDoS/DDoS-ICMP_Flood.pcap | A, B, D, E | Verificar |
| Mirai/[variante].pcap | B, E | Verificar |
| DoS/[variante].pcap | E | Verificar |
| Recon/[variante].pcap | E | Verificar |
| Spoofing/[variante].pcap | E (opcional) | Verificar |

---

## Funcionalidades a Implementar no Orquestrador

O `run_experiment.py` atual suporta `--pcap` e `--attack-pcap`. Para os cenarios B e E:

1. **Multi-fase:** suporte a `--drift-pcap` (fase 3 do experimento)
2. **Holdout:** suporte a `--holdout-pcap` (ataques zero-day)
3. **Metricas de adaptacao:** tempo ate recuperar 95% do F1 baseline

Avaliar se implementar no orquestrador ou como scripts separados.

---

## Criterios de Sucesso Globais

| Metrica | Alvo | Cenario |
|---------|------|---------|
| FPR (trafego benigno) | <= 5% | A |
| Recall (ataques conhecidos) | >= 80% | A |
| MTTD | <= 500 flows | A |
| Throughput | >= 100 flows/s | Todos |
| Tempo de adaptacao (drift) | <= 2000 flows | B |
| Deteccao zero-day | >= 60% | E |
| Execucoes por config | >= 5 | Todos |

---

## Notas

- Resultados anteriores (`streaming/results/week5/`) sao descartaveis
- Campanha comeca do zero com metodologia rigorosa
- Cada experimento deve ter: run_meta.json, detection_results.json, metrics_windowed.csv
- Reprodutibilidade: registrar git commit, seeds, parametros completos
