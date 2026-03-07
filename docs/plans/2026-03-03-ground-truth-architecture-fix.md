# Plano: Correção da Arquitetura de Ground Truth

**Data:** 2026-03-03
**Branch:** fix/ground-truth-architecture
**Status:** Em execução

---

## Problema

O sistema é um detector de anomalias **não-supervisionado**: aprqende padrões de tráfego benigno
(clusters TEDA/MicroTEDAclus) e detecta desvios. Nenhum label externo é necessário para detectar.

O ground truth existe apenas para **avaliar** a qualidade das detecções offline.

### O que está errado

**1. GroundTruthProvider dentro do StreamingDetector (streaming_detector.py:192)**

O detector recebe `ground_truth` como parâmetro e chama `get_flow_label(flow)` durante
o processamento (linha 436). Isso vaza conhecimento supervisionado para dentro do loop de
detecção — conceitualmente errado. O detector deve ser cego a labels.

**2. Um único GroundTruthProvider para experimentos com dois PCAPs (run_experiment.py:257)**

Quando o experimento roda benign + attack sequencialmente:
- `gt_pcap = attack_pcap` → GroundTruthProvider("SqlInjection.pcap") → `is_attack = True`
- Todos os flows (benignos E de ataque) recebem `y_true = True`
- Resultado: Precision = 1.0 por construção (não há FP possível), Recall inválido

**3. Metodologia documenta integração com CSV (experiment-methodology.md seção 8.2.2)**

Menciona "matching flow-by-flow com CSV do CICIoT2023" como melhoria. Isso contradiz
a abordagem não-supervisionada — o sistema não precisa de CSVs rotulados.

---

## Solução

### Princípio

```
StreamingDetector → produz is_anomaly (sem knowledge de labels)
run_experiment.py → aplica y_true por fase e chama metrics.update()
```

O StreamingDetector **não sabe** o que é ataque. Ele apenas diz: "este flow é anômalo
em relação ao que aprendi até agora". O run_experiment.py sabe em qual fase estamos
(benign ou attack) e usa isso para avaliar se a detecção foi correta.

### Ground truth por fase

```
Fase 1 (Warm-up): flows do BenignTraffic.pcap → y_true = False
Fase 2 (Ataque):  flows do SqlInjection.pcap  → y_true = True
```

Isso é suficiente para Precision/Recall/F1/MTTD corretos.

---

## Tarefas

### T1: Remover ground_truth do StreamingDetector
- Arquivo: `streaming/src/detector/streaming_detector.py`
- Remover parâmetros `ground_truth` e `metrics` do `__init__`
- Remover bloco de avaliação dentro do `run()` (linhas 434-438)
- `run()` passa a retornar lista de resultados com `(flow, is_anomaly, timestamp)`
  para o chamador avaliar

### T2: Corrigir run_experiment.py — avaliação por fase
- Arquivo: `streaming/scripts/run_experiment.py`
- `run_detector()` passa a retornar lista de `(is_anomaly, timestamp)`
- `run_experiment.py` sabe quantos flows são da fase benign e quantos do ataque
- Aplica `y_true = False` para flows da fase warm-up, `y_true = True` para fase ataque
- Chama `metrics.update()` externamente, fora do detector

### T3: Atualizar testes
- Remover testes que passam ground_truth para StreamingDetector
- Adicionar testes da nova interface

### T4: Corrigir experiment-methodology.md
- Seção 3.2: corrigir step 3 de `update(x, y)` para `update(x)`
- Seção 8.2.2: remover/marcar "Solução 2 CSV" como fora do escopo

---

## Verificação

Rodar experimento SqlInjection novamente. Resultados esperados corretos:
- FPR durante warm-up (benign): flows anômalos / total flows benignos
- Recall durante ataque: flows anômalos do SqlInjection / total flows SqlInjection
- MTTD: flows desde início da fase ataque até primeiro alerta
