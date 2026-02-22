# Plano de Leituras - Dissertação de Mestrado

**Criado:** 2025-12-17
**Objetivo:** Fundamentação teórica rigorosa para dissertação
**Meta:** Mínimo 1 artigo principal por semana

---

## 1. Visão Geral

### 1.1 Áreas de Conhecimento

| Área | Foco na Dissertação | Capítulos Relacionados |
|------|---------------------|------------------------|
| **Machine Learning** | Clustering evolutivo, concept drift | Fundamentação, Metodologia |
| **Cibersegurança** | IDS, detecção de anomalias | Fundamentação, Trabalhos Relacionados |
| **IoT** | Características de redes IoT, ataques | Fundamentação, Contexto |
| **Arquitetura de Software** | Streaming, Kafka, pipelines | Metodologia, Implementação |

### 1.2 Tipos de Leitura

| Tipo | Descrição | Quantidade |
|------|-----------|------------|
| **Principal** | Leitura completa + resumo detalhado + fichamento | 8 artigos |
| **Auxiliar** | Leitura focada + anotações de conceitos-chave | 12+ artigos |
| **Referência** | Consulta conforme necessidade | Livros, docs |

---

## 2. Leituras por Área

### 2.1 Machine Learning (Clustering Evolutivo)

#### Artigos Principais (Leitura Obrigatória)

| # | Referência | Foco | Status |
|---|------------|------|--------|
| **ML-P1** | Angelov, P. (2014). "Outside the box: an alternative data analytics framework." *Journal of Automation Mobile Robotics and Intelligent Systems*, 8(2), pp.29-35. | Framework TEDA original: eccentricidade, tipicalidade | ✅ Completo |
| **ML-P2** | Maia, J. et al. (2020). "Evolving clustering algorithm based on mixture of typicalities." *Future Generation Computer Systems*, 106, pp.672-684. | MicroTEDAclus: micro-clusters, concept drift | ✅ Completo |

**Links:**
- ML-P1: [DOI: 10.14313/JAMRIS_2-2014/16](https://doi.org/10.14313/JAMRIS_2-2014/16)
- ML-P2: [DOI: 10.1016/j.future.2020.01.017](https://doi.org/10.1016/j.future.2020.01.017)

#### Artigos Auxiliares

| # | Referência | Foco | Status |
|---|------------|------|--------|
| ML-A1 | "A systematic review on detection and adaptation of concept drift" (2024) - Wiley WIREs | Survey de métodos de drift | ❌ Não lido |
| ML-A2 | "A benchmark and survey of fully unsupervised concept drift detectors" (2024) - Springer | 10 algoritmos analisados | ❌ Não lido |
| ML-A3 | "Temporal silhouette: validation of stream clustering" (2023) - Machine Learning Journal | Validação robusta a drift | ❌ Não lido |

**Links:**
- ML-A1: [Wiley](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1536)
- ML-A2: [Springer](https://link.springer.com/article/10.1007/s41060-024-00620-y)
- ML-A3: [Springer](https://link.springer.com/article/10.1007/s10994-023-06462-2)

#### Livros/Capítulos de Referência

| Referência | Capítulos Relevantes |
|------------|---------------------|
| Gama, J. "Knowledge Discovery from Data Streams" | Caps sobre concept drift, stream clustering |
| Aggarwal, C. "Data Streams: Models and Algorithms" | Clustering em streams |

---

### 2.2 Cibersegurança (IDS / Anomaly Detection)

#### Artigos Principais (Leitura Obrigatória)

| # | Referência | Foco | Status |
|---|------------|------|--------|
| **CS-P1** | Neto, E.C.P. et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." *Sensors*, 23(13), 5941. | Dataset, ataques, features | ❌ Não lido |
| **CS-P2** | Survey: "Machine Learning-Based Intrusion Detection Methods in IoT Systems: A Comprehensive Review" (2024) - MDPI Electronics | Estado da arte IDS + ML | ❌ Não lido |

**Links:**
- CS-P1: [MDPI](https://www.mdpi.com/1424-8220/23/13/5941)
- CS-P2: [MDPI Electronics](https://www.mdpi.com/2079-9292/13/18/3601)

#### Artigos Auxiliares

| # | Referência | Foco | Status |
|---|------------|------|--------|
| CS-A1 | "Two-step data clustering for improved intrusion detection system using CICIoT2023" (2024) | Mesmo dataset, clustering | ❌ Não lido |
| CS-A2 | "Hybrid evolutionary machine learning model for advanced intrusion detection" (2024) - PLOS ONE | Algoritmos evolutivos + IDS | ❌ Não lido |
| CS-A3 | "A novel deep learning-based intrusion detection system for IoT DDoS security" (2024) | DDoS em IoT | ❌ Não lido |

**Links:**
- CS-A1: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772671124002535)
- CS-A2: [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308206)
- CS-A3: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2542660524002774)

#### Livros/Capítulos de Referência

| Referência | Capítulos Relevantes |
|------------|---------------------|
| Northcutt, S. "Network Intrusion Detection" | Fundamentos de IDS |
| NIST Cybersecurity Framework | Conceitos e terminologia |

---

### 2.3 IoT (Internet das Coisas)

#### Artigos Principais (Leitura Obrigatória)

| # | Referência | Foco | Status |
|---|------------|------|--------|
| **IoT-P1** | Survey recente (2023/2024) sobre IoT Security Challenges | Ameaças, vulnerabilidades IoT | ❌ Identificar |
| **IoT-P2** | Edge Computing for IoT IDS (2024) | Processamento na borda | ❌ Identificar |

**Ação:** Buscar surveys recentes no Zotero ou bases acadêmicas.

#### Artigos Auxiliares

| # | Referência | Foco | Status |
|---|------------|------|--------|
| IoT-A1 | Antonakakis et al. "Understanding the Mirai Botnet" (2017) - USENIX Security | Análise Mirai | ❌ Não lido |
| IoT-A2 | Taxonomia de ataques IoT | Classificação de ataques | ❌ Identificar |
| IoT-A3 | Características de tráfego IoT | Padrões normais/anômalos | ❌ Identificar |

**Links:**
- IoT-A1: [USENIX](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/antonakakis)

#### Livros/Capítulos de Referência

| Referência | Capítulos Relevantes |
|------------|---------------------|
| Vasseur, J.P. "Interconnecting Smart Objects with IP" | Arquitetura IoT |
| Capítulos de segurança em livros de IoT | Ameaças e contramedidas |

---

### 2.4 Arquitetura de Software (Streaming)

#### Artigos Principais (Leitura Obrigatória)

| # | Referência | Foco | Status |
|---|------------|------|--------|
| **AS-P1** | Surianarayanan et al. (2024). "High-throughput streaming architecture" | Arquitetura de streaming | ❌ Não lido |
| **AS-P2** | Kreps, J. "Kafka: The Definitive Guide" (2017) - O'Reilly | Fundamentos Kafka | ❌ Não lido |

**Ação:** Confirmar referência exata de AS-P1 no Zotero.

#### Artigos Auxiliares

| # | Referência | Foco | Status |
|---|------------|------|--------|
| AS-A1 | Kreps, J. "Questioning the Lambda Architecture" | Lambda vs Kappa | ❌ Não lido |
| AS-A2 | Real-time ML Pipelines best practices | Padrões de arquitetura | ❌ Identificar |
| AS-A3 | Comparação Kafka vs Flink vs Spark Streaming | Trade-offs | ❌ Identificar |

**Links:**
- AS-A1: [O'Reilly](https://www.oreilly.com/radar/questioning-the-lambda-architecture/)

#### Livros/Capítulos de Referência

| Referência | Capítulos Relevantes |
|------------|---------------------|
| Kleppmann, M. "Designing Data-Intensive Applications" | Streaming, processamento distribuído |
| Narkhede et al. "Kafka: The Definitive Guide" | Caps 1-6 para fundamentos |

---

## 3. Cronograma de Leituras

### Meta: 1 artigo principal + 1-2 auxiliares por semana

| Semana | Data Início | Área | Principal | Auxiliar | Entregável |
|--------|-------------|------|-----------|----------|------------|
| **S1** | 2025-12-23 | ML | ML-P1: Angelov (2014) | - | Resumo + fichamento |
| **S2** | 2025-12-30 | ML | ML-P2: Maia (2020) releitura | ML-A1: Survey Drift | Resumo completo |
| **S3** | 2026-01-06 | Cyber | CS-P2: Survey IDS IoT | CS-A1: Two-step | Resumo + mapeamento |
| **S4** | 2026-01-13 | IoT | IoT-P1: Survey IoT Security | IoT-A1: Mirai | Resumo + taxonomia |
| **S5** | 2026-01-20 | Arq | AS-P2: Kafka Guide (1-3) | AS-A1: Lambda/Kappa | Notas técnicas |
| **S6** | 2026-01-27 | ML | ML-A3: Temporal Silhouette | ML-A2: Benchmark | Métricas identificadas |
| **S7** | 2026-02-03 | Cyber | CS-P1: CICIoT2023 releitura | CS-A2: Evolutionary | Resumo completo |
| **S8** | 2026-02-10 | Arq | AS-P2: Kafka Guide (4-6) | AS-A2: ML Pipelines | Design patterns |
| **S9** | 2026-02-17 | IoT | IoT-P2: Edge IDS | IoT-A2/A3 | Integração com arq |
| **S10** | 2026-02-24 | Todas | Revisão e consolidação | - | Capítulo fundamentação |

---

## 4. Template de Fichamento

Para cada artigo principal, criar fichamento em `docs/paper-summaries/`:

```markdown
# Fichamento: [Título do Artigo]

**Referência completa:**
**Data de leitura:**
**Área:** ML / Cyber / IoT / Arq

## 1. Objetivo do Artigo
[O que os autores se propõem a fazer]

## 2. Metodologia
[Como fizeram]

## 3. Principais Contribuições
[Bullet points com contribuições]

## 4. Resultados Chave
[Números, métricas, conclusões]

## 5. Limitações Identificadas
[O que não resolveram ou assumiram]

## 6. Relação com Minha Pesquisa
[Como uso isso na dissertação]

## 7. Citações Importantes
[Trechos para citar diretamente]

## 8. Referências Relevantes
[Outros papers citados que devo ler]
```

---

## 5. Integração com Dissertação

### Mapeamento Leituras → Capítulos

| Capítulo | Leituras Necessárias |
|----------|---------------------|
| **1. Introdução** | CS-P1, IoT-P1 (contexto e motivação) |
| **2. Fundamentação Teórica** | ML-P1, ML-P2, CS-P2, IoT-P1, AS-P2 |
| **3. Trabalhos Relacionados** | CS-A1, CS-A2, ML-A1, ML-A2 |
| **4. Metodologia** | ML-P2, AS-P1, ML-A3 |
| **5. Implementação** | AS-P2, AS-A1 |
| **6. Resultados** | ML-A3 (métricas), CS-P1 (comparação) |
| **7. Conclusão** | Síntese de todas |

---

## 6. Acompanhamento

### Status Geral

| Área | Principais | Lidos | Auxiliares | Lidos |
|------|------------|-------|------------|-------|
| ML | 2 | **2** ✅ | 3 | 0 |
| Cyber | 2 | 0 | 3 | 0 |
| IoT | 2 | 0 | 3 | 0 |
| Arq | 2 | 0 | 3 | 0 |
| **Total** | **8** | **2** | **12** | **0** |

### Próxima Leitura
- **Artigo:** ML-A1: Survey Concept Drift (Wiley 2024)
- **Prazo:** Semana 3 (2026-01-19)
- **Entregável:** Notas sobre tipos de drift e métodos de detecção

### Leituras Completas
- ✅ ML-P1: Angelov (2014) - `docs/paper-summaries/angelov-2014-teda.md`
- ✅ ML-P2: Maia (2020) - `docs/paper-summaries/maia-2020-microtedaclus.md`

---

## 7. Referências Rápidas (Links)

### Bases de Dados
- [Google Scholar](https://scholar.google.com/)
- [IEEE Xplore](https://ieeexplore.ieee.org/)
- [ACM Digital Library](https://dl.acm.org/)
- [ScienceDirect](https://www.sciencedirect.com/)
- [Springer](https://link.springer.com/)
- [arXiv](https://arxiv.org/)

### Zotero
- Biblioteca local exportada para: `/Users/augusto/mestrado/references.bib`

---

**Este documento é atualizado semanalmente conforme progresso das leituras.**

*Última atualização: 2026-01-19*
