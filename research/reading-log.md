# Reading Log

Status das leituras e lacunas de conhecimento para a dissertacao.

## Leituras Concluidas

| Paper | Fichamento | Status |
|-------|-----------|--------|
| Angelov (2014) - TEDA Framework | [summary](summaries/angelov-2014-teda.md) | Completo |
| Maia et al. (2020) - MicroTEDAclus | [summary](summaries/maia-2020-microtedaclus.md) | Completo |
| Gama et al. (2013) - Prequential Evaluation | [summary](summaries/gama-2013-prequential-evaluation.md) | Completo |

## Leituras Prioritarias (Proximas 8 Semanas)

Reorganizadas por urgencia considerando o prazo de 2 meses (marco-abril 2026).

### Prioridade 1 - Fundamentacao Imediata (Semanas 1-2)

| # | Paper | Area | Justificativa |
|---|-------|------|---------------|
| CS-P1 | Neto et al. (2023) "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment" - Sensors | Cyber | Dataset usado na dissertacao; leitura obrigatoria para fundamentacao |
| CS-P2 | Survey "Machine Learning-Based Intrusion Detection Methods in IoT Systems" (2024) - MDPI Electronics | Cyber | Estado da arte IDS + ML; essencial para trabalhos relacionados |

### Prioridade 2 - Teoria e Metodologia (Semanas 3-4)

| # | Paper | Area | Justificativa |
|---|-------|------|---------------|
| ML-A1 | "A systematic review on detection and adaptation of concept drift" (2024) - Wiley WIREs | ML | Survey de metodos de drift; fundamenta escolha do TEDA |
| ML-A3 | "Temporal silhouette: validation of stream clustering" (2023) - Machine Learning Journal | ML | Metricas de validacao para streaming; impacta metodologia |
| IoT-P1 | Survey recente (2023/2024) sobre IoT Security Challenges | IoT | Ameacas e vulnerabilidades IoT; contexto da dissertacao |

### Prioridade 3 - Arquitetura e Comparacao (Semanas 5-6)

| # | Paper | Area | Justificativa |
|---|-------|------|---------------|
| AS-P2 | Kreps, J. "Kafka: The Definitive Guide" (2017) - Caps 1-6 | Arq | Fundamentos Kafka; justifica escolha arquitetural |
| CS-A1 | "Two-step data clustering for improved intrusion detection system using CICIoT2023" (2024) | Cyber | Mesmo dataset, clustering; trabalho relacionado direto |
| CS-A2 | "Hybrid evolutionary machine learning model for advanced intrusion detection" (2024) - PLOS ONE | Cyber | Algoritmos evolutivos + IDS; trabalho relacionado direto |

### Prioridade 4 - Complementar (Semanas 7-8)

| # | Paper | Area | Justificativa |
|---|-------|------|---------------|
| ML-A2 | "A benchmark and survey of fully unsupervised concept drift detectors" (2024) - Springer | ML | Benchmark de drift; fortalece justificativa |
| AS-A1 | Kreps, J. "Questioning the Lambda Architecture" | Arq | Lambda vs Kappa; justifica decisao arquitetural |
| IoT-A1 | Antonakakis et al. "Understanding the Mirai Botnet" (2017) - USENIX Security | IoT | Contexto de ameacas IoT reais |

### Ainda nao identificados (buscar no Zotero)

- IoT-P2: Edge Computing for IoT IDS (2024)
- IoT-A2: Taxonomia de ataques IoT
- IoT-A3: Caracteristicas de trafego IoT
- AS-P1: Surianarayanan et al. (2024) - confirmar referencia exata
- AS-A2: Real-time ML Pipelines best practices
- AS-A3: Comparacao Kafka vs Flink vs Spark Streaming

## Lacunas de Conhecimento

Extraidas do levantamento de janeiro 2026, organizadas por prioridade.

### Alta Prioridade

**Estatistica Basica** (impacta compreensao do TEDA)

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Variancia e Desvio Padrao | Base da formula recursiva do TEDA | Pendente |
| Media como centro de massa | Propriedade usada na derivacao | Pendente |
| Esperanca E[X] | Formula de Konig: Var = E[X^2] - E[X]^2 | Pendente |

**Teoria de Probabilidade** (impacta fundamentacao teorica)

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Abordagem Frequentista vs Bayesiana | Entender critica do TEDA a probabilidade classica | Pendente |
| Distribuicoes (Gaussiana, etc.) | Por que TEDA nao assume distribuicao previa | Pendente |
| Funcao Densidade de Probabilidade (PDF) | Entender por que tipicalidade "resembles PDF" | Pendente |

**Algebra Linear Basica** (impacta todas as formulas de distancia)

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Norma de vetor | Todas as formulas de distancia usam normas | Pendente |
| Produto interno (dot product) | Expansao de norma ao quadrado | Pendente |
| Matriz de covariancia | Distancia de Mahalanobis | Pendente |

**Identidades Matematicas Classicas** (impacta derivacao recursiva)

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Teorema de Huygens-Steiner | Base da simplificacao O(n^2) para O(n) no TEDA | Pendente |
| Formula de Konig-Huygens | Expressar variancia recursivamente | Pendente |

### Media Prioridade

**Normalizacao e Escalas**

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Tipos de normalizacao (Min-Max, Z-Score) | Escolher corretamente para features de rede | Pendente |
| Por que normalizar | Comparabilidade e estabilidade numerica | Pendente |

**Metricas de Distancia**

| Conceito | Por que e necessario | Status |
|----------|---------------------|--------|
| Quando usar cada metrica | Escolha correta para features de rede IoT | Pendente |
| Propriedades metricas | Simetria, desigualdade triangular, positividade | Pendente |
| Distancia vs Similaridade | Cosseno e similaridade, precisa converter | Pendente |

### Baixa Prioridade

**Teoria da Possibilidade** - Somente se houver tempo. Contexto historico (Dempster-Shafer, Possibilidade vs Necessidade) nao e essencial para implementacao.

## Referencias por Area

### Machine Learning (Clustering Evolutivo)

- **Principais:** Angelov (2014), Maia et al. (2020)
- **Auxiliares:** Survey Concept Drift (Wiley 2024), Benchmark Drift Detectors (Springer 2024), Temporal Silhouette (ML Journal 2023)
- **Livros:** Gama "Knowledge Discovery from Data Streams", Aggarwal "Data Streams: Models and Algorithms"

### Ciberseguranca (IDS / Anomaly Detection)

- **Principais:** CICIoT2023 - Neto et al. (2023), Survey IDS IoT (MDPI 2024)
- **Auxiliares:** Two-step clustering CICIoT2023 (2024), Hybrid evolutionary IDS (PLOS ONE 2024), Deep learning DDoS IoT (2024)
- **Livros:** Northcutt "Network Intrusion Detection", NIST Cybersecurity Framework

### IoT (Internet das Coisas)

- **Principais:** Survey IoT Security (a identificar), Edge Computing IDS (a identificar)
- **Auxiliares:** Antonakakis et al. "Understanding Mirai Botnet" (USENIX 2017), Taxonomia de ataques IoT (a identificar)
- **Livros:** Vasseur "Interconnecting Smart Objects with IP"

### Arquitetura de Software (Streaming)

- **Principais:** Surianarayanan et al. (2024), Kreps "Kafka: The Definitive Guide"
- **Auxiliares:** "Questioning the Lambda Architecture", Real-time ML Pipelines (a identificar)
- **Livros:** Kleppmann "Designing Data-Intensive Applications", Narkhede et al. "Kafka: The Definitive Guide"
