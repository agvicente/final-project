# Concept Drift: Fundamentos para Estudo Aprofundado

**Criado:** 2025-12-09
**Status:** Visão geral - requer estudo aprofundado da literatura
**Contexto:** Fase 2 do mestrado - IoT IDS com Clustering Evolutivo

---

## 1. Definição Formal

**Concept Drift:** Mudança na distribuição conjunta P(X, y) ao longo do tempo.

```
Em t=0:    P₀(X, y)
Em t=100:  P₁(X, y) ≠ P₀(X, y)  ← Drift aconteceu!
```

**Decomposição:**
```
P(X, y) = P(y) × P(X|y)
          ↑       ↑
     Prior    Likelihood
```

| Componente | Nome | O que muda | Exemplo IoT |
|------------|------|------------|-------------|
| P(y) | Prior drift | Proporção das classes | Mais ataques que antes |
| P(X\|y) | Likelihood drift | Características das classes | Ataques com novos padrões |
| P(X) | Virtual drift | Distribuição de X | Novos dispositivos IoT |
| P(y\|X) | Real drift | Fronteira de decisão | Mesmo tráfego, classificação diferente |

---

## 2. Tipos de Concept Drift

### 2.1 Súbito (Sudden/Abrupt)
```
────────────┐
            │████████████
Conceito A  │ Conceito B
            └──────────────────► tempo
```
- **Característica:** Mudança instantânea de um conceito para outro
- **Exemplo IoT:** Novo malware lançado (Mirai aparece de repente)
- **Desafio:** Detectar rapidamente, adaptar sem dados históricos

### 2.2 Gradual
```
──────────────────────────
   ████  ░░████  ░░░░████
   A     A+B     B
──────────────────────────► tempo
```
- **Característica:** Período de transição com mistura dos conceitos
- **Exemplo IoT:** Atacantes adaptando técnicas aos poucos
- **Desafio:** Distinguir de ruído, decidir quando conceito antigo "morreu"

### 2.3 Incremental
```
════════════════════════
A → A' → A'' → A''' → B
════════════════════════► tempo
```
- **Característica:** Mudança lenta e contínua, sem saltos
- **Exemplo IoT:** Tráfego normal evoluindo com novos dispositivos
- **Desafio:** Detectar mudança cumulativa, evitar "esquecimento" gradual

### 2.4 Recorrente (Recurring)
```
████    ░░░░    ████    ░░░░
  A      B       A       B
────────────────────────────► tempo
```
- **Característica:** Conceitos que voltam ciclicamente
- **Exemplo IoT:** Padrões dia/noite, ataques sazonais
- **Desafio:** Reconhecer conceito antigo, evitar reaprender do zero

---

## 3. Mapeamento para IoT IDS

| Cenário Real | Tipo de Drift | Justificativa |
|--------------|---------------|---------------|
| Novos malwares lançados | Súbito | Aparecem "do nada" |
| Atacantes adaptando técnicas | Gradual | Evolução progressiva |
| Novos dispositivos na rede | Incremental | Tráfego "normal" muda aos poucos |
| Padrões de uso dia/noite | Recorrente | Ciclos que se repetem |
| Atualizações de firmware | Súbito | Comportamento muda de uma vez |
| Botnets evoluindo | Gradual/Incremental | Variantes novas surgem |

---

## 4. CICIoT2023: Estrutura para Simulação de Drift

### Categorias de Ataque (7 + Benign)

| Categoria | Nº Ataques | Rows | % Dataset |
|-----------|------------|------|-----------|
| **DDoS** | 12 | 33,984,560 | ~73% |
| **DoS** | 4 | 8,090,738 | ~17% |
| **Mirai** | 3 | 2,634,124 | ~6% |
| **Recon** | 5 | 354,565 | ~0.8% |
| **Spoofing** | 2 | 486,504 | ~1% |
| **Web-Based** | 6 | 24,829 | ~0.05% |
| **Brute Force** | 1 | 13,064 | ~0.03% |
| **Benign** | - | 1,098,195 | ~2.4% |

### 33 Ataques Específicos

**DDoS (12):** ACK Fragmentation, UDP Flood, SlowLoris, ICMP Flood, RSTFIN Flood, PSHACK Flood, HTTP Flood, UDP Fragmentation, ICMP Fragmentation, TCP Flood, SYN Flood, SynonymousIP Flood

**DoS (4):** TCP Flood, HTTP Flood, SYN Flood, UDP Flood

**Mirai (3):** GREIP Flood, Greeth Flood, UDPPlain

**Recon (5):** Ping Sweep, OS Scan, Vulnerability Scan, Port Scan, Host Discovery

**Spoofing (2):** ARP Spoofing, DNS Spoofing

**Web-Based (6):** SQL Injection, Command Injection, Backdoor Malware, Uploading Attack, XSS, Browser Hijacking

**Brute Force (1):** Dictionary Brute Force

### Estratégias de Simulação

| Tipo de Drift | Cenário de Simulação |
|---------------|---------------------|
| **Súbito** | Treinar com DDoS → Testar com Mirai |
| **Gradual** | DDoS → DDoS+Mirai misturados → Só Mirai |
| **Incremental** | Adicionar Recon gradualmente ao stream |
| **Recorrente** | DDoS → Benign → DDoS → Benign (ondas) |

---

## 5. Métodos de Detecção de Drift (ESTUDAR)

### 5.1 Baseados em Performance
- Monitorar accuracy/F1 em janelas deslizantes
- Queda de performance indica drift
- Exemplos: DDM, EDDM, ADWIN

### 5.2 Baseados em Distribuição
- Comparar distribuições de janelas consecutivas
- Testes estatísticos (KS, chi-squared)
- Exemplos: HDDDM, CDBD

### 5.3 Implícitos (como TEDA)
- Não detectam drift explicitamente
- Adaptam continuamente
- Novos padrões → novos clusters

**TODO:** Estudar papers específicos de cada método

---

## 6. Métricas de Avaliação (ESTUDAR)

### Métricas Temporais
| Métrica | O que mede |
|---------|------------|
| Tempo de detecção | Pontos até detectar drift |
| Tempo de recuperação | Pontos até accuracy voltar |
| Falsos alarmes | Drifts detectados erroneamente |

### Métricas de Performance
| Métrica | O que mede |
|---------|------------|
| Prequential accuracy | Accuracy em janelas deslizantes |
| Accuracy pós-drift | Performance após adaptação |
| Kappa temporal | Accuracy ajustada ao longo do tempo |

### Avaliação Prequential
```python
def prequential_evaluation(model, stream, window_size=100):
    """
    Avalia modelo em janelas deslizantes.
    Permite visualizar quando drift afeta performance.
    """
    accuracies = []

    for i in range(0, len(stream) - window_size, window_size):
        window = stream[i:i+window_size]
        X_window, y_window = window[:, :-1], window[:, -1]

        # Testar ANTES de atualizar (prequential)
        predictions = model.predict(X_window)
        acc = (predictions == y_window).mean()
        accuracies.append(acc)

        # Atualizar com novos dados
        model.partial_fit(X_window, y_window)

    return accuracies
```

---

## 7. Como TEDA/MicroTEDAclus Lida com Drift

| Tipo de Drift | Mecanismo de Adaptação |
|---------------|------------------------|
| Súbito | Chebyshev rejeita → novos clusters criados rapidamente |
| Gradual | Coexistência de clusters antigos e novos |
| Incremental | Centroids atualizam gradualmente via média móvel |
| Recorrente | Clusters podem ser "reativados" |

**Vantagem:** Adaptação implícita, sem detectar drift explicitamente
**Limitação:** Não sabe "quando" drift aconteceu (útil para análise)

---

## 8. Referências para Estudo Aprofundado

### Surveys Recomendados

1. **"A systematic review on detection and adaptation of concept drift"** (2024)
   - Wiley WIREs Data Mining
   - [Link](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1536)
   - Revisão abrangente de métodos

2. **"A benchmark and survey of fully unsupervised concept drift detectors"** (2024)
   - International Journal of Data Science and Analytics
   - [Link](https://link.springer.com/article/10.1007/s41060-024-00620-y)
   - 10 algoritmos analisados

3. **"A survey on machine learning for recurring concept drifts"** (2023)
   - Expert Systems with Applications
   - [Link](https://www.sciencedirect.com/science/article/pii/S0957417422019522)
   - Foco em drift recorrente

4. **"Temporal silhouette: validation of stream clustering"** (2023)
   - Machine Learning Journal
   - [Link](https://link.springer.com/article/10.1007/s10994-023-06462-2)
   - Validação robusta a drift

### Papers Clássicos

- **DDM (Drift Detection Method):** Gama et al., 2004
- **EDDM (Early Drift Detection):** Baena-García et al., 2006
- **ADWIN (Adaptive Windowing):** Bifet & Gavaldà, 2007

### Ferramentas/Bibliotecas

- **River:** Biblioteca Python para online learning
  - https://riverml.xyz/
  - Implementa vários detectores de drift

- **scikit-multiflow:** Extensão do sklearn para streams
  - https://scikit-multiflow.github.io/

---

## 9. Exercícios para Estudo Posterior

### Exercício 1: Implementar Detector Simples
```python
# TODO: Implementar detector de drift baseado em janelas
# Comparar accuracy de janelas consecutivas
# Se diferença > threshold, sinalizar drift
```

### Exercício 2: Simular Drift no CICIoT2023
```python
# TODO: Criar função que gera stream com drift
# Parâmetros: tipo de drift, momento do drift, categorias envolvidas
# Retorna: X, y ordenados temporalmente
```

### Exercício 3: Avaliar TEDA com Drift
```python
# TODO: Rodar MicroTEDAclus em stream com drift simulado
# Plotar accuracy ao longo do tempo
# Identificar tempo de recuperação
```

### Exercício 4: Comparar Detectores
```python
# TODO: Comparar DDM vs ADWIN vs TEDA implícito
# Métricas: tempo de detecção, falsos alarmes, accuracy final
```

---

## 10. Conexão com Dissertação

### Capítulo de Fundamentação Teórica
- Definição formal de concept drift
- Taxonomia dos 4 tipos
- Métodos de detecção (DDM, EDDM, ADWIN)
- Justificativa para abordagem evolutiva

### Capítulo de Metodologia
- Estratégias de simulação de drift no CICIoT2023
- Métricas de avaliação escolhidas
- Protocolo experimental

### Capítulo de Resultados
- Performance do MicroTEDAclus sob cada tipo de drift
- Comparação com baselines
- Análise de tempo de adaptação

---

**Este documento serve como guia para estudo aprofundado de concept drift.**
**Requer leitura dos papers referenciados e implementação dos exercícios.**

*Última atualização: 2025-12-09*
