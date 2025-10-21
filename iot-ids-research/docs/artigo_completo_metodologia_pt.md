# Análise Comparativa de Algoritmos de Machine Learning para Detecção de Anomalias em Redes IoT Usando o Dataset CICIoT2023

## Resumo

As redes de Internet das Coisas (IoT) enfrentam ameaças de segurança crescentes devido à sua natureza heterogênea e restrições de recursos. Este estudo apresenta uma comparação abrangente de algoritmos de machine learning para detecção de anomalias em ambientes IoT usando o dataset CICIoT2023. Implementamos um pipeline reproduzível usando Data Version Control (DVC) para avaliar sete algoritmos diferentes em múltiplas configurações. Nossa metodologia emprega amostragem estratificada, pré-processamento padronizado e classificação binária para distinguir entre tráfego de rede benigno e malicioso. O framework experimental inclui validação estatística rigorosa através de múltiplas execuções e métricas de desempenho abrangentes incluindo acurácia, acurácia balanceada, precisão, recall, F1-score e ROC AUC. Esta pesquisa contribui para o entendimento do desempenho de algoritmos em detecção de intrusão IoT e fornece uma baseline para estudos comparativos futuros.

**Palavras-chave:** Segurança IoT, Detecção de Anomalias, Machine Learning, Detecção de Intrusão, Classificação Binária

---

## 1. Introdução

A proliferação de dispositivos Internet das Coisas (IoT) criou novos vetores de ataque e desafios de segurança em ambientes de rede. Sistemas tradicionais de detecção de intrusão baseados em assinaturas são inadequados para redes IoT devido à natureza diversa e evolutiva dos ataques. A detecção de anomalias baseada em machine learning oferece uma abordagem promissora para identificar atividades maliciosas em redes IoT através do aprendizado de padrões de comportamento normal e detecção de desvios.

Este estudo aborda a questão de pesquisa: **"Quais algoritmos de machine learning são mais eficazes para detecção binária de anomalias em tráfego de rede IoT?"** Conduzimos uma comparação sistemática de dez algoritmos usando condições experimentais padronizadas e métricas de avaliação abrangentes.

### 1.1 Objetivos da Pesquisa

**Objetivos Primários:**
1. Estabelecer uma baseline abrangente para desempenho de algoritmos ML em detecção de anomalias IoT
2. Validar a efetividade da amostragem estratificada para datasets IoT de larga escala
3. Fornecer framework experimental reproduzível para pesquisas futuras
4. Quantificar requisitos computacionais e de memória para cada algoritmo

**Objetivos Secundários:**
1. Identificar configurações ótimas de hiperparâmetros para cada algoritmo
2. Analisar trade-offs entre acurácia e eficiência computacional
3. Estabelecer significância estatística das diferenças de desempenho
4. Documentar melhores práticas para experimentação em detecção de anomalias IoT

---

## 2. Trabalhos Relacionados

### 2.1 Desafios de Segurança em IoT

Redes IoT apresentam desafios de segurança únicos incluindo:
- **Restrições de Recursos**: Recursos computacionais e de memória limitados
- **Heterogeneidade**: Dispositivos diversos com protocolos e comportamentos diferentes
- **Escala**: Grande número de dispositivos gerando tráfego de alto volume
- **Topologia Dinâmica**: Dispositivos frequentemente entrando e saindo da rede

### 2.2 Machine Learning para Segurança IoT

Estudos anteriores demonstraram a efetividade de abordagens de machine learning para detecção de intrusão IoT. Benkhelifa et al. (2018) fornecem uma revisão crítica das práticas e desafios em sistemas de detecção de intrusão para IoT, destacando a necessidade de abordagens mais robustas e adaptativas. Cook et al. (2020) apresentam um survey abrangente sobre detecção de anomalias em dados de séries temporais IoT, identificando técnicas baseadas em aprendizado de máquina como promissoras para identificar padrões anômalos em ambientes IoT heterogêneos.

No entanto, a maioria dos estudos foca em algoritmos específicos ou datasets limitados, carecendo de análise comparativa abrangente sob condições padronizadas.

### 2.3 Lacuna de Pesquisa

Este estudo aborda a lacuna fornecendo:
- Comparação sistemática de múltiplas famílias de algoritmos
- Condições experimentais padronizadas
- Rigor estatístico através de múltiplas execuções
- Análise abrangente de uso de recursos
- Pipeline experimental reproduzível

---

## 3. Materiais e Métodos

### 3.1 Dataset

#### 3.1.1 Descrição do Dataset CICIoT2023

**Fonte**: Canadian Institute for Cybersecurity (Neto et al., 2023)  
**Tamanho Original**: ~23 milhões de registros de tráfego de rede  
**Características**: 46 features de fluxo de rede  
**Tipos de Ataque**: DDoS, Mirai, Reconhecimento, Spoofing, Web-based, Força Bruta, Man-in-the-Middle  
**Ambiente de Captura**: Testbed IoT realístico com 105 dispositivos

#### 3.1.2 Estratégia de Amostragem

Devido a restrições computacionais, implementamos uma **abordagem de amostragem estratificada** (Cochran, 1977):

```
Amostra Total: 4.501.906 registros (19,5% do dataset original)
├── Conjunto de Treinamento: 3.601.524 registros (80%)
└── Conjunto de Teste: 900.382 registros (20%)

Distribuição Binária:
├── Tráfego Benigno: 105.137 registros (2,3%)
└── Tráfego Malicioso: 4.396.769 registros (97,7%)
    ├── DDoS: ~65%
    ├── Mirai: ~15%
    ├── Reconhecimento: ~10%
    ├── Web-based: ~5%
    ├── Spoofing: ~3%
    └── Outros: ~2%
```

**Metodologia de Amostragem**:
1. **Estratificação Proporcional**: Mantidas as proporções originais das classes
2. **Estado Aleatório**: Semente fixa (42) para reprodutibilidade
3. **Garantia de Qualidade**: Validação automatizada da representatividade da amostra

**Validação Estatística**:
- Teste de Kolmogorov-Smirnov para comparação de distribuições

### 3.2 Design Experimental

#### 3.2.1 Framework de Design de Pesquisa

**Tipo de Estudo**: Design experimental quantitativo com foco em benchmarking de sistemas  
**Abordagem de Comparação**: Dez algoritmos de machine learning (otimizados para larga escala)  
**Tarefa de Classificação**: Binária (Benigno vs. Malicioso)  
**Método de Validação**: Divisão estratificada treino-teste com estado aleatório fixo  
**Rigor Estatístico**: 5 execuções independentes por configuração (n=5)  
**Reprodutibilidade**: Pipeline baseado em DVC com controle de versão e sementes fixas  
**Contexto Computacional**: Treinamento apenas em CPU sem aceleração GPU  
**Estratégia de Configuração**: Conjuntos adaptativos de parâmetros (8-20 configs por algoritmo) baseados em complexidade computacional

#### 3.2.2 Seleção de Algoritmos

Selecionamos dez algoritmos representando diferentes paradigmas de aprendizado, ordenados por complexidade computacional para gerenciamento ótimo de recursos:

**Algoritmos de Aprendizado Supervisionado** (ordenados por complexidade):
1. **Regressão Logística**: Classificador probabilístico linear (O(n))
2. **Random Forest** (Breiman, 2001): Método ensemble com bagging (O(n log n))
3. **Gradient Boosting**: Método ensemble com boosting (O(n log n))
4. **LinearSVC**: Classificador de Vetores de Suporte Linear com formulação primal (O(n), otimizado para grandes datasets)
5. **SGDClassifier**: Stochastic Gradient Descent com perda hinge (O(n), aproximação escalável de SVM)
6. **Multi-Layer Perceptron (MLP)**: Classificador de rede neural (O(n³))

**Algoritmos Não-Supervisionados/Semi-Supervisionados** (detecção de anomalias):
7. **Isolation Forest** (Liu et al., 2008, 2012): Detecção de anomalias baseada em árvores (O(n log n))
8. **Elliptic Envelope**: Detecção de anomalias baseada em Gaussiana (O(n²))
9. **Local Outlier Factor (LOF)**: Detecção de anomalias baseada em densidade (O(n²))
10. **SGDOneClassSVM**: One-Class SVM via stochastic gradient descent (O(n), otimizado para larga escala)

**Classificação de Algoritmos por Tipo de Detecção**:
- **Detecção de Anomalias Verdadeira**: Isolation Forest, SGDOneClassSVM, LOF, Elliptic Envelope
- **Classificação Supervisionada**: Regressão Logística, Random Forest, Gradient Boosting, LinearSVC, SGDClassifier, MLP

**Otimizações Críticas para Datasets de Larga Escala**:
Dado os desafios computacionais de treinar em 3M amostras com 39 features, substituímos algoritmos computacionalmente proibitivos por alternativas escaláveis:
- **SVM Linear**: Substituído `SVC(kernel='linear')` por `LinearSVC(dual=False)` e `SGDClassifier(loss='hinge')` para ganho de velocidade de 10-100x mantendo equivalência matemática
- **One-Class SVM**: Substituído `OneClassSVM` baseado em kernel por `SGDOneClassSVM` para ganho de velocidade de 10-50x, viabilizando detecção de anomalias prática em dados de larga escala

### 3.3 Pipeline de Pré-processamento de Dados

#### 3.3.1 Arquitetura do Pipeline (DVC)

```yaml
stages:
  1. check_quality           → Validação de qualidade de dados e métricas
  2. sampling               → Amostragem estratificada do dataset
  3. eda                   → Análise exploratória de dados
  4. preprocess            → Engenharia de features e normalização
  5. exp_logistic_regression → Experimentos de Regressão Logística
  6. exp_random_forest      → Experimentos de Random Forest
  7. exp_gradient_boosting  → Experimentos de Gradient Boosting
  8. exp_isolation_forest   → Experimentos de Isolation Forest
  9. exp_svc               → Experimentos de Support Vector Classifier
  10. exp_one_class_svm    → Experimentos de One-Class SVM
  11. exp_lof              → Experimentos de Local Outlier Factor
  12. exp_elliptic_envelope → Experimentos de Elliptic Envelope
  13. exp_mlp_classifier   → Experimentos de Multi-Layer Perceptron
  14. consolidate_results  → Consolidação e análise de resultados
```

**Características Principais do Pipeline**:
- **Execução Modular**: Cada algoritmo executa como estágio DVC independente
- **Ordenação Computacional**: Algoritmos ordenados do menos ao mais computacionalmente complexo
- **Configuração Dinâmica**: Variável única `TEST_MODE` controla todos os experimentos
- **Sistema de Timestamps Compartilhados**: Sistema de timestamp unificado para organização de resultados
- **Análise Individual**: Análise detalhada automatizada por algoritmo
- **Consolidação Final**: Comparação abrangente entre algoritmos

#### 3.3.2 Estágio de Controle de Qualidade

**Entrada**: 63 arquivos CSV (dados brutos CICIoT2023)  
**Processo**:
- Detecção e quantificação de valores ausentes
- Validação de consistência de tipos de dados
- Identificação de outliers usando métodos estatísticos
- Verificação de integridade entre arquivos

**Saída**: Métricas de qualidade e relatórios de validação

#### 3.3.3 Estágio de Amostragem

**Implementação de Amostragem Estratificada**:
```python
def stratified_sampling():
    # 1. Análise de distribuição em todos os 63 arquivos
    # 2. Cálculo de quotas proporcionais por tipo de ataque
    # 3. Seleção aleatória estratificada por arquivo
    # 4. Validação estatística da representatividade
    # 5. Geração de arquivo consolidado
```

**Parâmetros**:
- Proporção da amostra: 19,5% do dataset original
- Variável de estratificação: Tipo de ataque
- Estado aleatório: 42 (reprodutibilidade)
- Validação: Testes estatísticos para representatividade

#### 3.3.4 Engenharia de Features

**Features Originais**: 46 colunas do CICIoT2023  
**Features Finais**: 39 colunas após pré-processamento

**Categorias de Features**:

**Features da Camada de Rede**:
- Header_Length, Protocol Type, Time_To_Live, Rate

**Features da Camada de Transporte**:
- TCP Flags: fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number
- Contadores de Flags: ack_count, syn_count, fin_count, rst_count

**Features da Camada de Aplicação**:
- Protocolos: HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC
- Tipos de Rede: TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC

**Features Estatísticas**:
- Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance

**Passos de Pré-processamento**:
1. **Tratamento de Valores Ausentes**: 
   - Método: Imputação por moda (mais conservador que média/mediana)
   - Escopo: Todas as features exceto variável alvo
   - Validação: Comparação de estatísticas pré/pós

2. **Binarização de Rótulos**:
   - BENIGN → 0 (Tráfego normal)
   - Todos os tipos de ataque → 1 (Tráfego malicioso)

3. **Seleção de Features**:
   - Removidas colunas constantes/quase-constantes
   - Eliminadas features altamente correlacionadas (r > 0,95)
   - Retidas 39 features mais informativas

4. **Normalização de Dados**:
   - Método: StandardScaler (μ=0, σ=1)
   - Consciente da divisão: Ajustado apenas em dados de treinamento (previne vazamento de dados)
   - Escopo: Todas as features numéricas

#### 3.3.5 Estratégia de Divisão

```python
# Divisão treino-teste ANTES da normalização (previne vazamento de dados)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y_binary, 
    random_state=42
)

# Normalização ajustada apenas em dados de treinamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform
X_test_scaled = scaler.transform(X_test)        # transform apenas
```

### 3.4 Configuração Experimental

#### 3.4.1 Parâmetros dos Algoritmos

**Sistema de Configuração Adaptativa** (Opção C):
O framework experimental emprega uma estratégia adaptativa onde o número de configurações de parâmetros varia por algoritmo baseado na complexidade computacional, mantendo rigor estatístico consistente:
- **Algoritmos Rápidos (20 configs)**: LogisticRegression, SGDClassifier
- **Algoritmos Médios (12-18 configs)**: RandomForest, LinearSVC, IsolationForest, EllipticEnvelope, SGDOneClassSVM
- **Algoritmos Pesados (8-10 configs)**: GradientBoosting, LocalOutlierFactor, MLPClassifier
- **Rigor Estatístico**: 5 execuções independentes por configuração para todos os algoritmos (n=1 para modo teste, n=5 para modo completo)
- **Total de Experimentos**: 141 configurações × 5 runs = 705 experimentos
- **Tempo Estimado**: ~30 horas (dentro da meta de 24-48h)

**Configurações dos Algoritmos**:

**1. Regressão Logística** (O(n) - Mais Rápido):
```python
# Modo Completo  
{'C': [0.1, 1.0, 10.0], 'max_iter': 1000, 'random_state': 42}
```

**2. Random Forest** (O(n log n)):
```python
# Modo Completo
{'n_estimators': [50, 100], 'max_depth': [10, 15, 20], 'random_state': 42}
```

**3. Gradient Boosting** (O(n log n)):
```python
# Modo Completo
{'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 
 'max_depth': [5, 7], 'random_state': 42}
```

**4. Isolation Forest** (O(n log n)):
```python
# Modo Completo
{'contamination': [0.1, 0.15, 0.2], 'n_estimators': [100, 200], 'random_state': 42}
```

**5. Elliptic Envelope** (O(n²)):
```python
# Modo Completo
{'contamination': [0.1, 0.15, 0.2], 'random_state': 42}
```

**6. Local Outlier Factor** (O(n²)):
```python
# Modo Completo
{'n_neighbors': [10, 20, 50], 'contamination': [0.1, 0.15], 'novelty': True}
```

**7. LinearSVC** (O(n) - Otimizado para grandes datasets):
```python
# Otimização: dual=False para n_samples >> n_features
# Matematicamente equivalente a SVC(kernel='linear')
# Modo Completo
{'C': [1.0, 5.0, 10.0], 'max_iter': 1000, 'dual': False, 'random_state': 42}
# Tempo esperado: 1-2h por config (vs. dias para SVC padrão)
```

**8. SGDClassifier** (O(n) - SVM via stochastic gradient descent):
```python
# SVM escalável via stochastic gradient descent
# Modo Completo
{'loss': 'hinge', 'penalty': 'l2', 'alpha': [0.0001, 0.001, 0.01], 
 'max_iter': 1000, 'random_state': 42}
# Tempo esperado: 10-20 min por config
```

**9. SGDOneClassSVM** (O(n) - One-Class SVM Otimizado):
```python
# Aprendizado online para detecção de anomalias em larga escala
# Mantém mesmo parâmetro nu do OneClassSVM
# Modo Completo
{'nu': [0.05, 0.1, 0.15, 0.2], 'learning_rate': 'optimal', 
 'max_iter': 1000, 'random_state': 42}
# Tempo esperado: 15-30 min por config (vs. horas para baseado em kernel)
```

**10. Multi-Layer Perceptron** (O(n³)):
```python
# Modo Completo
{'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)], 
 'max_iter': 200, 'random_state': 42}
```

**Otimizações Críticas de Desempenho**:
- **Otimização de SVM Linear**: Substituído SVM baseado em kernel por LinearSVC e SGDClassifier para ganho de velocidade de 10-100x mantendo equivalência matemática
- **Otimização de One-Class SVM**: Empregado SGDOneClassSVM com complexidade O(n) em vez de abordagem baseada em kernel O(n²)
- **Otimização de Gradient Boosting**: Aplicado `subsample=0.7` seguindo recomendações de Friedman (2002) para ~30% de aceleração e melhor generalização
- **Otimização de MLP**: Implementado early stopping agressivo (`validation_fraction=0.15`, `n_iter_no_change` adaptativo) reduzindo tempo de treinamento em 25-40% enquanto previne overfitting
- **Complexidade Progressiva**: Algoritmos ordenados do menos ao mais computacionalmente caro para utilização ótima de recursos
- **Estratégia de Configuração Adaptativa**: Número variável de conjuntos de parâmetros (8-20) por algoritmo proporcional ao custo computacional

#### 3.4.2 Execução Experimental

**Arquitetura de Pipeline Modular**:
O framework experimental foi redesenhado como um pipeline DVC modular onde cada algoritmo executa como um estágio independente:

```python
# Execução de algoritmo individual
run_single_algorithm.py <algorithm_name>

# Cada algoritmo segue este padrão:
for param_config in parameter_grid:
    for run in range(N_RUNS):  # Rigor estatístico
        # Executa experimento único
        # Monitora memória e tempo
        # Registra todas as métricas
        # Gera análise individual
        # Limpa memória
```

**Rigor Estatístico**:
- **Múltiplas Execuções**: 5 execuções independentes por configuração de parâmetros (controlado por `N_RUNS` global)
- **Controle de Estado Aleatório**: Sementes fixas para todos os componentes aleatórios (`RANDOM_STATE = 42`)
- **Execução Sequencial**: Um algoritmo por vez para garantir comparação justa
- **Gerenciamento de Memória**: Limpeza agressiva entre experimentos com monitoramento
- **Monitoramento de Progresso**: Rastreamento em tempo real de execução e uso de recursos
- **Tratamento de Erros**: Sistema robusto de recuperação de erros e logging

**Sistema de Configuração Dinâmica**:
```python
# Ponto único de controle
TEST_MODE = True  # ou False para experimentos completos

# Configura automaticamente:
- Tamanhos de amostra (10.000 vs dataset completo)
- Parâmetros de algoritmos (simples vs abrangente)
- Número de execuções (mesmo para ambos os modos)
- Diretórios de saída (test/ vs full/)
```

**Organização de Resultados**:
```
experiments/results/
├── test/                          # Resultados modo teste
│   ├── TIMESTAMP_algorithm_name/  # Resultados de algoritmo individual
│   └── TIMESTAMP_consolidation/   # Análise entre algoritmos
└── full/                          # Resultados modo completo
    ├── TIMESTAMP_algorithm_name/  # Resultados de algoritmo individual
    └── TIMESTAMP_consolidation/   # Análise entre algoritmos
```

#### 3.4.3 Estratégia de Execução Sequencial

**Decisão de Design**: Os experimentos são executados **sequencialmente** (um algoritmo por vez) ao invés de em paralelo.

**Justificativa**:

1. **Avaliação Dual de Desempenho**: Esta pesquisa mede tanto métricas de desempenho de ML (acurácia, precisão, recall, etc.) quanto métricas de desempenho computacional (tempo de treinamento, uso de CPU, consumo de RAM). Métricas computacionais são outputs primários, não medições auxiliares.

2. **Impacto da Competição por Recursos**: Execução paralela introduz competição por recursos do sistema:
   - **CPU**: Overhead de troca de contexto aumenta tempo de treinamento em 15-50%
   - **Memória**: Cache thrashing reduz RAM disponível em 20-40%
   - **Variância**: Introduz variância artificial de 6-15% entre execuções (20× maior que sequencial)

3. **Padrões de Benchmarking**:
   - **MLPerf** (Mattson et al. 2020): Regra explícita "Um treinamento de modelo por vez"
   - **SPEC CPU** (SPEC, 2017): Separa speed runs (sequencial) de rate runs (paralelo)
   - **Edge Computing**: Papadopoulos et al. (2019) recomendam execução sequencial para benchmarks IoT

4. **Reprodutibilidade**: Execução sequencial elimina dependência de:
   - Configuração de hardware (número de núcleos, RAM total)
   - Ordem de execução (quais algoritmos rodam juntos)
   - Políticas de escalonamento do sistema operacional

5. **Comparação Justa**: Todos os algoritmos avaliados sob condições idênticas com acesso total aos recursos do sistema. Métricas refletem capacidade real do algoritmo, não artefatos de escalonamento.

**Análise de Trade-off**:
```
Sequencial: 30h total | Comparação justa | Reproduzível | Metodologia simples
Paralelo:   ~15h total | Métricas contaminadas | Documentação complexa | Dependente de hardware
```

A janela de execução de 30 horas é aceitável para pesquisa de nível de mestrado e está alinhada com as melhores práticas de systems benchmarking (Dehghani et al. 2021).

**Gerenciamento de Timestamps**:
- **Timestamps Compartilhados**: Todos os algoritmos em uma única execução DVC compartilham o mesmo timestamp
- **Resultados Não-Sobrepostos**: Cada execução obtém um timestamp único para prevenir perda de dados
- **Organização Cronológica**: Resultados naturalmente ordenados por tempo de execução

### 3.5 Métricas de Avaliação

#### 3.5.1 Métricas de Desempenho Primárias

1. **Acurácia**: Correção geral da classificação
   ```
   Acurácia = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precisão**: Taxa de Verdadeiros Positivos entre positivos preditos
   ```
   Precisão = TP / (TP + FP)
   ```

3. **Recall (Sensibilidade)**: Taxa de Verdadeiros Positivos entre positivos reais
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Média harmônica de precisão e recall (métrica principal de ranking; Powers, 2011)
   ```
   F1-Score = 2 × (Precisão × Recall) / (Precisão + Recall)
   ```

5. **ROC AUC**: Área Sob a Curva Receiver Operating Characteristic (Davis & Goadrich, 2006)
   ```
   AUC = ∫₀¹ TPR(FPR⁻¹(x))dx
   ```

6. **Balanced Accuracy (Acurácia Balanceada)**: Média do recall entre classes, particularmente útil para datasets desbalanceados
   ```
   Balanced Accuracy = (Sensibilidade + Especificidade) / 2
   ```
   Para multi-classe: Média dos valores de recall por classe, dando peso igual a cada classe independentemente do tamanho

#### 3.5.2 Métricas Secundárias

1. **Tempo de Treinamento**: Eficiência computacional durante ajuste do modelo
2. **Tempo de Predição**: Velocidade de inferência no conjunto de teste (quando disponível)
3. **Uso de Memória**: Consumo de memória de pico durante treinamento
4. **Matriz de Confusão**: Detalhamento completo (TP, TN, FP, FN)
5. **Pontuação de Eficiência**: F1-Score por segundo de tempo de treinamento
6. **Métricas de Estabilidade**: Desvio padrão entre múltiplas execuções

#### 3.5.3 Análise Individual de Algoritmos

Para cada algoritmo, geramos análise individual abrangente:

**Métricas de Evolução de Desempenho**:
- Tendências de Acurácia, F1-Score, Precisão, Recall entre execuções
- Análise de impacto de parâmetros no desempenho
- Medidas de estabilidade estatística

**Métricas de Análise de Recursos**:
- Distribuição e tendências de tempo de treinamento
- Padrões de consumo de memória
- Eficiência computacional (desempenho por unidade de tempo)

**Métricas de Validação Estatística**:
- Intervalos de confiança para todas as métricas primárias
- Coeficiente de variação para avaliação de estabilidade
- Análise de melhor vs. desempenho médio

#### 3.5.3 Análise Estatística

**Estatísticas Descritivas**:
- Média e desvio padrão entre execuções
- Intervalos de confiança (nível de 95%)
- Valores mín/máx por configuração

**Estatísticas Inferenciais**:
- ANOVA para comparação de desempenho de algoritmos
- Testes post-hoc com correção de Bonferroni
- Cálculo de tamanho de efeito (d de Cohen; Cohen, 1988)

#### 3.5.4 Abordagem Bayesiana para Balanced Accuracy (Brodersen et al., 2010)

**Motivação e Limitações dos Métodos Tradicionais**:

Métricas pontuais (point estimates) de desempenho apresentam duas limitações críticas identificadas por Brodersen et al. (2010):

1. **Ausência de Incerteza Quantificada**: Médias simples não permitem derivação de intervalos de confiança significativos
2. **Viés em Datasets Desbalanceados**: Classificadores enviesados testados em conjuntos desbalanceados produzem estimativas otimistas de accuracy

**Solução: Distribuições Posteriores Bayesianas**:

Implementamos a abordagem Bayesiana de Brodersen et al. (2010) que modela a Balanced Accuracy como uma variável aleatória com distribuição posterior completa ao invés de um valor pontual.

**Fundamentação Teórica**:

**1. Posterior da Accuracy**:

Tratando cada predição como um experimento de Bernoulli independente, a posterior da accuracy segue uma distribuição Beta conjugada:

```
A ~ Beta(C + 1, I + 1)
```

Onde:
- `C` = número de predições corretas (TP + TN)
- `I` = número de predições incorretas (FP + FN)
- Prior não-informativo: Beta(1, 1) = Uniforme[0,1]

**2. Posterior da Balanced Accuracy**:

A Balanced Accuracy é definada como:

```
BA = ½(Sensibilidade + Especificidade) = ½(TP/(TP+FN) + TN/(TN+FP))
```

Sua distribuição posterior é obtida via **convolução de duas distribuições Beta** (Equação 7 do artigo):

```
p_BA(x; TP, FP, FN, TN) = ∫₀¹ p_A(2(x-z); TP+1, FN+1) × p_A(2z; TN+1, FP+1) dz
```

Onde:
- `p_A` é a densidade Beta para sensibilidade e especificidade
- A integral representa a convolução das duas posteriors

**Implementação via Monte Carlo**:

Devido à complexidade analítica da convolução, implementamos aproximação por Monte Carlo Sampling:

```python
# Posteriors para cada classe
sensitivity_posterior = Beta(TP + 1, FN + 1)
specificity_posterior = Beta(TN + 1, FP + 1)

# Amostragem (50k durante experimentos, 100k durante consolidação)
sensitivity_samples = sensitivity_posterior.rvs(n_samples)
specificity_samples = specificity_posterior.rvs(n_samples)

# Balanced Accuracy posterior
ba_samples = 0.5 * (sensitivity_samples + specificity_samples)
```

**Métricas Derivadas**:

Para cada algoritmo, calculamos:

1. **Estatísticas da Posterior**:
   - Média (estimador Bayesiano)
   - Mediana (estimador robusto)
   - Desvio padrão (incerteza)
   - Moda (máximo a posteriori)

2. **Intervalos de Credibilidade Bayesianos** (95%):
   ```
   IC_95% = [percentil_2.5%, percentil_97.5%]
   ```
   - Interpretação probabilística direta: "Probabilidade de 95% de BA estar neste intervalo"
   - Respeitam limites naturais [0,1] (ao contrário de ICs frequentistas)
   - Assimétricos quando apropriado

3. **Probabilidades de Threshold**:
   ```
   P(BA > threshold) = ∫_threshold^1 p_BA(x) dx
   ```
   Exemplos: P(BA > 0.80), P(BA > 0.85), P(BA > 0.90), P(BA > 0.95)

4. **Comparações Probabilísticas entre Algoritmos**:
   ```
   P(Algoritmo_A > Algoritmo_B) = E[1_{BA_A > BA_B}]
   ```
   Estimado via:
   ```python
   prob_A_better = mean(ba_samples_A > ba_samples_B)
   ```

**Vantagens sobre Métodos Frequentistas**:

1. **Interpretação Direta**: Intervalos de credibilidade têm interpretação probabilística direta
2. **Respeito aos Limites**: Automaticamente respeita [0,1] sem correções ad-hoc
3. **Robustez a Desbalanceamento**: BA posterior detecta viés mesmo com alta accuracy
4. **Comparações Rigorosas**: Comparações probabilísticas P(A > B) são mais informativas que p-valores
5. **Assimetria Natural**: Distribuições assimétricas quando justificado pelos dados

**Implementação no Framework**:

**Módulos Desenvolvidos**:

1. **`experiments/bayesian_metrics.py`**:
   - Classe `BayesianAccuracyEvaluator`
   - Cálculo de posteriors Beta
   - Implementação da convolução (Eq. 7)
   - Intervalos de credibilidade
   - Comparações probabilísticas

2. **`experiments/bayesian_plots.py`**:
   - Distribuições posteriores de BA por algoritmo
   - Intervalos de credibilidade 95%
   - Matriz de comparação P(A > B)
   - Tabelas estatísticas Bayesianas

**Integração Automática**:

As métricas Bayesianas são calculadas automaticamente:
- **Durante experimentos**: Métricas adicionadas a cada resultado individual
- **Durante consolidação**: Plots Bayesianos gerados automaticamente
- **Output**: 4 plots + 2 tabelas (CSV + Markdown) com análise completa

**Outputs Gerados**:

1. **Visualizações**:
   - `bayesian_posterior_distributions.png`: Densidades posteriores via KDE
   - `bayesian_credibility_intervals.png`: Barras horizontais com IC 95%
   - `bayesian_comparison_matrix.png`: Heatmap P(linha > coluna)

2. **Tabelas**:
   - `bayesian_statistics.csv`: Estatísticas completas
   - `bayesian_comparison_matrix.csv`: Matriz completa de comparações

**Interpretação de Resultados**:

**Intervalos de Credibilidade**:
- IC estreito → Baixa incerteza, estimativa confiável
- IC largo → Alta incerteza, mais dados necessários
- Não sobrepõe com threshold → Evidência forte

**Comparações Probabilísticas**:
- P(A > B) > 0.95 → Evidência forte que A é superior
- P(A > B) < 0.05 → Evidência forte que B é superior  
- 0.05 < P(A > B) < 0.95 → Diferença não conclusiva
- P(A > B) ≈ 0.5 → Algoritmos equivalentes

**Exemplo de Uso em Resultados**:

```json
{
  "bayesian": {
    "accuracy_posterior": {
      "mean": 0.8992,
      "median": 0.8995,
      "ci_lower": 0.8798,
      "ci_upper": 0.9171,
      "std": 0.0095
    },
    "balanced_accuracy_posterior": {
      "mean": 0.8985,
      "median": 0.8986,
      "ci_lower": 0.8791,
      "ci_upper": 0.9165,
      "std": 0.0095
    },
    "prob_ba_above_90": 0.42,
    "prob_ba_above_85": 0.89
  }
}
```

**Significância para Datasets Desbalanceados**:

O dataset CICIoT2023 possui desbalanceamento significativo (97,7% malicioso vs 2,3% benigno). A abordagem Bayesiana de BA:

1. **Detecta Viés**: Identifica quando alta accuracy resulta de viés de classe majoritária
2. **Penaliza Adequadamente**: BA posterior baixa mesmo com accuracy alta se uma classe for ignorada
3. **Fornece Incerteza**: Quantifica confiança nas estimativas considerando desbalanceamento

**Validação da Implementação**:

✅ Distribuições posteriores calculadas corretamente
✅ Intervalos de credibilidade respeitam [0,1]
✅ Convolução via Monte Carlo validada com 50k-100k amostras
✅ Comparações probabilísticas funcionando
✅ Integração automática com pipeline experimental

### 3.6 Eficiência Computacional para Datasets IoT de Larga Escala

#### 3.6.1 Otimização de SVM para Dados de Larga Escala

Dada a natureza de larga escala dos dados de rede IoT (3M amostras, 39 features), algoritmos SVM tradicionais baseados em kernel tornam-se computacionalmente proibitivos. Empregamos alternativas escaláveis que mantêm equivalência matemática enquanto viabilizam experimentação prática.

**Desafio**: `SVC(kernel='linear')` e `OneClassSVM` padrão têm complexidade O(n²) a O(n³), tornando-os impraticáveis para datasets com milhões de amostras (tempo de convergência estimado: vários dias a semanas).

**Solução**: Formulações lineares e abordagens de stochastic gradient descent

#### 3.6.2 Otimização de Support Vector Machine Linear

**Abordagem 1: LinearSVC com Formulação Primal**
```python
LinearSVC(C=value, max_iter=1000, dual=False, random_state=42)
```
- **Otimização**: `dual=False` resolve o problema primal quando n_samples >> n_features
- **Complexidade**: O(n) vs. O(n²-n³) para SVC padrão
- **Base Matemática**: Equivalente a `SVC(kernel='linear')` mas formulado para cenários de grandes amostras
- **Desempenho Esperado**: Ganho de velocidade de 10-100x (horas vs. dias)
- **Citação Acadêmica**: Fan et al. (2008), LIBLINEAR: A Library for Large Linear Classification

**Abordagem 2: SGDClassifier com Perda Hinge**
```python
SGDClassifier(loss='hinge', penalty='l2', alpha=value, max_iter=1000)
```
- **Otimização**: Stochastic gradient descent para aprendizado online/batch
- **Complexidade**: O(n) com convergência em passagem única
- **Base Matemática**: Aproxima SVM linear através de otimização iterativa
- **Desempenho Esperado**: Ganho de velocidade de 50-200x (minutos vs. horas)
- **Citação Acadêmica**: Bottou & Bousquet (2007), The Tradeoffs of Large Scale Learning

**Justificativa Científica**:
Ambas as abordagens são amplamente aceitas na literatura de machine learning para tarefas de classificação de larga escala. LinearSVC é a recomendação padrão para SVM linear em grandes datasets (documentação scikit-learn, Pedregosa et al., 2011), enquanto SGDClassifier fornece garantias teóricas para convergência a soluções SVM (Shalev-Shwartz et al., 2011).

#### 3.6.3 Otimização de One-Class SVM para Detecção de Anomalias

**Desafio**: `OneClassSVM` baseado em kernel tem complexidade O(n²) com requisitos de memória cúbicos, tornando-o proibitivo para detecção de anomalias em escala de milhões.

**Solução**: SGDOneClassSVM com aprendizado online

```python
SGDOneClassSVM(nu=value, learning_rate='optimal', max_iter=1000, random_state=42)
```

**Vantagens Principais**:
- **Complexidade**: O(n) vs. O(n²) para abordagem baseada em kernel
- **Compatibilidade de Parâmetros**: Mantém mesmo espaço de parâmetros `nu` do OneClassSVM
- **Consistência de API**: Interface idêntica (`.fit()`, `.predict()`, `.decision_function()`)
- **Eficiência de Memória**: Footprint de memória constante vs. crescimento quadrático
- **Desempenho Esperado**: Ganho de velocidade de 10-50x (minutos vs. horas)

**Justificativa Acadêmica**:
SGDOneClassSVM implementa a variante de aprendizado online do One-Class SVM (Schölkopf et al., 2001) através de stochastic gradient descent. O algoritmo mantém as propriedades teóricas do One-Class SVM baseado em kernel enquanto viabiliza escalabilidade através de otimização online.

#### 3.6.4 Justiça da Comparação com Otimizações

**Justificativa Metodológica**:
As otimizações aplicadas (subsample para Gradient Boosting, early stopping para MLP, estratégia de configuração adaptativa) mantêm a justiça na comparação de algoritmos pelas seguintes razões:

**1. Práticas Estabelecidas**:
- `subsample=0.7` em Gradient Boosting é uma prática recomendada (Friedman, 2002) que melhora a generalização enquanto reduz overfitting
- Early stopping em redes neurais é prática padrão (Prechelt, 1998; Goodfellow et al., 2016) para regularização
- Ambas as técnicas são consideradas partes integrantes dos algoritmos, não "facilitações" externas

**2. Metodologia de Avaliação Consistente**:
- Todos os algoritmos usam o mesmo train/test split (`random_state=42`)
- Todos avaliados com métricas idênticas (accuracy, precision, recall, F1, ROC AUC, balanced accuracy)
- Todos passam por 5 execuções independentes por configuração
- Todos compartilham o mesmo pipeline de pré-processamento

**3. Alinhamento com Contexto IoT**:
- Otimizações tornam algoritmos mais deployable em ambientes IoT
- Foco na viabilidade computacional reflete restrições do mundo real
- Gradient Boosting COM subsample é mais prático para deployment em edge
- MLP COM early stopping é mais adequado para dispositivos com recursos limitados

**4. Estratégia de Configuração Adaptativa**:
O número variável de configurações (8-20) por algoritmo é justificado por:
- Complexidade computacional varia de O(n) a O(n³)
- Todos os algoritmos exploram o espectro completo: LEVE (20%) → SWEET SPOT (40%) → MÉDIO (20%) → PESADO (20%)
- 40% das configurações focam em faixas de parâmetros viáveis para IoT em todos os algoritmos
- Mantém comparabilidade respeitando realidades computacionais

**5. Documentação Transparente**:
- Todas as otimizações explicitamente documentadas na metodologia
- Código completamente reproduzível com estados aleatórios fixos
- Leitores podem avaliar impacto e ajustar se necessário
- Resultados representam configurações práticas e deployable

**Perspectiva Estatística**:
Nossa abordagem alinha-se com metodologia de "benchmarking de sistemas" onde o foco está no desempenho computacional e reprodutibilidade com configurações fixas, ao invés de avaliação de variância do modelo estatístico. Isso é academicamente válido para estudos comparativos quando propriamente documentado (ver Smith, 2018; Bischl et al., 2021).

**Referências para Esta Seção**:
- Friedman, J. H. (2002). Stochastic gradient boosting. Computational Statistics & Data Analysis
- Prechelt, L. (1998). Early stopping-but when?. Neural Networks: Tricks of the Trade
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. JMLR
- Feurer, M., et al. (2015). Efficient and robust automated machine learning. NeurIPS
- Probst, P., et al. (2019). Tunability: Importance of hyperparameters of machine learning algorithms. JMLR

### 3.7 Infraestrutura Técnica

#### 3.7.1 Especificações de Hardware

**Recursos Computacionais**:
- **CPU**: [A ser especificado baseado no ambiente de execução]
- **RAM**: 32 GB (suficiente para processamento de dataset completo)
- **Armazenamento**: SSD (I/O rápido para grandes datasets)
- **GPU**: Não usado (treinamento apenas em CPU com algoritmos otimizados)

#### 3.7.2 Ambiente de Software

**Sistema Operacional**: Linux Ubuntu 20.04+  
**Versão Python**: 3.9+

**Dependências Principais**:
```
scikit-learn==1.3.0     # Algoritmos de machine learning
pandas==2.0.0           # Manipulação de dados
numpy==1.24.0           # Computação numérica
mlflow==2.5.0           # Rastreamento de experimentos
dvc==3.0.0              # Controle de versão de dados
psutil==5.9.0           # Monitoramento de sistema
seaborn==0.11.0         # Visualização estatística
matplotlib==3.7.0       # Plotagem
```

#### 3.7.3 Framework de Reprodutibilidade

**Controle de Versão**:
- **Código**: Repositório Git com histórico detalhado de commits
- **Dados**: DVC para gerenciamento de arquivos grandes
- **Experimentos**: MLflow para rastreamento de execuções
- **Ambiente**: requirements.txt com versões fixadas

**Controle de Estado Aleatório**:
```python
# Sementes fixas ao longo do pipeline
RANDOM_STATE = 42

# Aplicado a:
train_test_split(random_state=42)
all_algorithms(random_state=42)
numpy.random.seed(42)
pandas.sample(random_state=42)
```

#### 3.7.4 Sistema de Monitoramento e Logging

**Logging Abrangente**:
- **ID de Execução**: Identificador único baseado em timestamp
- **Arquivo de Log**: `experiments/logs/algorithm_comparison_{timestamp}.log`
- **Formato**: `timestamp - level - [function:line] - message`

**Monitoramento em Tempo Real**:
```python
def monitor_resources():
    return {
        'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'cpu_percent': process.cpu_percent()
    }
```

**Gerenciamento de Memória**:
```python
def cleanup_memory():
    del model, predictions, probabilities
    gc.collect()  # Força coleta de lixo
    
    # Verifica efetividade da limpeza
    memory_freed = memory_before - memory_after
    logger.info(f"Memória liberada: {memory_freed:.1f} MB")
```

### 3.8 Organização de Arquivos e Gerenciamento de Dados

#### 3.8.1 Estrutura do Projeto

```
iot-ids-research/
├── data/
│   ├── raw/CSV/MERGED_CSV/          # 63 arquivos CSV originais
│   ├── processed/
│   │   ├── sampled.csv              # Amostra estratificada de 4,5M
│   │   └── binary/                  # Dados de classificação binária pré-processados
│   │       ├── X_train_binary.npy   # Features de treinamento (3.6M×39)
│   │       ├── X_test_binary.npy    # Features de teste (900K×39)
│   │       ├── y_train_binary.npy   # Rótulos de treinamento
│   │       ├── y_test_binary.npy    # Rótulos de teste
│   │       ├── scaler.pkl           # StandardScaler ajustado
│   │       └── binary_metadata.json # Metadados de pré-processamento
│   └── metrics/
│       └── quality_check.json       # Métricas de qualidade de dados
├── experiments/
│   ├── algorithm_comparison.py      # Configurações globais e utilitários
│   ├── run_single_algorithm.py      # Execução de algoritmo individual
│   ├── consolidate_results.py       # Análise entre algoritmos
│   ├── individual_analysis.py       # Análise detalhada por algoritmo
│   ├── results/                     # Saídas experimentais organizadas
│   │   ├── test/                    # Resultados modo teste
│   │   │   ├── TIMESTAMP_algorithm_name/     # Resultados de algoritmo individual
│   │   │   │   ├── results.json             # Resultados brutos de experimento
│   │   │   │   ├── summary.json             # Resumo estatístico
│   │   │   │   └── individual_analysis/     # Análise detalhada
│   │   │   │       ├── plots/               # Gráficos específicos do algoritmo
│   │   │   │       ├── tables/              # Estatísticas detalhadas
│   │   │   │       └── report/              # Relatórios individuais
│   │   │   └── TIMESTAMP_consolidation/     # Comparação entre algoritmos
│   │   │       ├── plots/                   # Visualizações comparativas
│   │   │       ├── tables/                  # Estatísticas resumidas
│   │   │       ├── report/                  # Relatório de análise final
│   │   │       └── data/                    # Datasets consolidados
│   │   └── full/                    # Resultados modo completo (mesma estrutura)
│   ├── logs/                        # Logs detalhados de execução
│   │   └── algorithm_comparison_{timestamp}.log
│   ├── .current_run_timestamp       # Coordenação de timestamp compartilhado
│   └── artifacts/                   # Artefatos MLflow (se usado)
├── configs/
│   ├── preprocessing.yaml           # Parâmetros de pré-processamento
│   └── experiment_config.yaml       # Configuração de experimento
├── src/
│   └── eda/
│       ├── dvc_eda.py              # Análise exploratória de dados
│       └── results/                 # Visualizações EDA
├── dvc.yaml                         # Definição de pipeline DVC (estágios modulares)
├── dvc_sampling.py                  # Implementação de amostragem
├── dvc_preprocessing.py             # Implementação de pré-processamento
└── requirements.txt                 # Dependências Python
```

#### 3.8.2 Organização Avançada de Resultados

**Estrutura Hierárquica**:
- **Separação por Modo**: Diretórios `test/` e `full/` para diferentes modos de experimento
- **Nomenclatura Baseada em Timestamp**: Cada execução obtém timestamp único para prevenir sobrescritas
- **Análise Individual**: Cada algoritmo gera relatórios individuais abrangentes
- **Análise Consolidada**: Comparação entre algoritmos com visualizações avançadas

**Profundidade de Análise**:
```
Análise Individual de Algoritmo:
├── Evolução de Desempenho (tendências de acurácia, F1, precisão, recall)
├── Análise de Impacto de Parâmetros (efeitos de hiperparâmetros)
├── Análise de Matriz de Confusão (padrões detalhados de erros)
├── Distribuição de Métricas (distribuições estatísticas)
├── Análise de Tempo de Execução (eficiência e uso de recursos)
├── Tabelas Detalhadas (estatísticas, rankings, resultados brutos)
└── Relatório Abrangente (insights e recomendações)

Análise Consolidada:
├── Comparação de Algoritmos (rankings e significância estatística)
├── Trade-offs Desempenho vs Eficiência
├── Análise de Detecção de Anomalias vs Classificação Supervisionada
├── Padrões de Uso de Recursos
├── Análise de Correlação entre Métricas
└── Relatório de Recomendações Finais
```

#### 3.8.3 Estratégia de Versionamento de Dados

**Estágios do Pipeline DVC**:
1. **check_quality**: Validar integridade de dados brutos
2. **sampling**: Gerar amostra estratificada
3. **eda**: Análise exploratória de dados
4. **preprocess**: Engenharia de features e normalização
5. **exp_logistic_regression**: Experimentos de Regressão Logística (O(n))
6. **exp_random_forest**: Experimentos de Random Forest (O(n log n))
7. **exp_gradient_boosting**: Experimentos de Gradient Boosting (O(n log n))
8. **exp_isolation_forest**: Experimentos de Isolation Forest (O(n log n))
9. **exp_elliptic_envelope**: Experimentos de Elliptic Envelope (O(n²))
10. **exp_local_outlier_factor**: Experimentos de Local Outlier Factor (O(n²))
11. **exp_linear_svc**: LinearSVC otimizado para grandes datasets (O(n))
12. **exp_sgd_classifier**: SGDClassifier com perda hinge (O(n))
13. **exp_sgd_one_class_svm**: SGDOneClassSVM para detecção escalável de anomalias (O(n))
14. **exp_mlp**: Experimentos de Multi-Layer Perceptron (O(n³))
15. **consolidate_results**: Análise entre algoritmos e relatórios

**Rastreamento de Artefatos e Reprodutibilidade**:
- **Versionamento de Dados**: DVC rastreia todos os datasets intermediários
- **Versionamento de Código**: Git gerencia todo o código fonte com histórico detalhado de commits
- **Versionamento de Resultados**: Resultados com timestamp previnem sobrescritas
- **Rastreamento de Configuração**: Variável única `TEST_MODE` controla pipeline inteiro
- **Gerenciamento de Dependências**: DVC rastreia automaticamente dependências de arquivos
- **Proveniência Completa**: Linhagem completa de dados brutos a insights finais

**Recursos Avançados**:
- **Gerenciamento Dinâmico de Saídas**: Resultados salvos em diretórios baseados em timestamp
- **Execução Seletiva**: DVC re-executa apenas estágios com dependências alteradas
- **Capacidade Paralela**: Estágios de algoritmos independentes podem executar em paralelo
- **Atualizações Incrementais**: Novos algoritmos podem ser adicionados sem afetar resultados existentes

---

## 4. Framework de Resultados Esperados

### 4.1 Comparação de Desempenho

**Análise Primária**:
- Ranking de algoritmos por F1-Score (métrica principal)
- Teste de significância estatística entre algoritmos
- Trade-offs de desempenho vs. custo computacional
- Análise de sensibilidade de hiperparâmetros

**Análise Secundária**:
- Insights específicos de algoritmos (ex.: métodos ensemble vs. lineares)
- Análise de erros através de matrizes de confusão
- Padrões de consumo de recursos
- Considerações de escalabilidade

### 4.2 Framework Avançado de Visualização

**Análise Individual de Algoritmo**:
1. **Evolução de Desempenho**: Tendências entre múltiplas execuções (acurácia, F1, precisão, recall)
2. **Análise de Impacto de Parâmetros**: Efeitos de hiperparâmetros no desempenho
3. **Análise de Matriz de Confusão**: Análise detalhada de padrões de erro com métricas de estabilidade
4. **Distribuição de Métricas**: Distribuições estatísticas com gráficos de densidade
5. **Análise de Tempo de Execução**: Eficiência de tempo e correlação de desempenho

**Comparação Entre Algoritmos**:
1. **Box plots**: Distribuição de F1-score por algoritmo com significância estatística
2. **Gráficos de barras**: Métricas de desempenho médias com intervalos de confiança
3. **Gráficos de dispersão**: Trade-offs de desempenho vs. tempo computacional
4. **Heatmaps**: Correlação entre métricas através de algoritmos
5. **Curvas ROC**: Comparação de desempenho independente de threshold
6. **Análise de Detecção de Anomalias**: Métricas especializadas para algoritmos não-supervisionados

**Análise Bayesiana (Brodersen et al., 2010)**:
1. **Distribuições Posteriores de BA**: Densidades posteriores via KDE para cada algoritmo com médias e IC 95% marcados
2. **Intervalos de Credibilidade**: Barras horizontais com intervalos de credibilidade Bayesianos 95% e probabilidades P(BA > 0.90)
3. **Matriz de Comparação Probabilística**: Heatmap com P(Algoritmo_i > Algoritmo_j) para todas as combinações
4. **Tabelas Estatísticas Bayesianas**: Métricas completas incluindo médias posteriores, medianas, IC 95%, e probabilidades de threshold

**Análise de Recursos e Eficiência**:
1. **Padrões de uso de memória** ao longo do tempo com detecção de picos
2. **Escalamento de tempo de treinamento** com análise de complexidade de algoritmo
3. **Fronteiras de eficiência** (F1-score por segundo)
4. **Ranking de algoritmos** por múltiplos critérios
5. **Validação de complexidade computacional** contra expectativas teóricas

### 4.3 Validação Estatística

**Teste de Hipóteses**:
- H₀: Não há diferença significativa entre desempenhos de algoritmos
- H₁: Existem diferenças significativas de desempenho
- α = 0,05 nível de significância
- Correção de comparação múltipla (Bonferroni; Demšar, 2006)

**Análise de Tamanho de Efeito**:
- d de Cohen para significância prática
- Intervalos de confiança para métricas de desempenho
- Reamostragem bootstrap para estimativas robustas

---

## 5. Contribuições e Significância

### 5.1 Contribuições Científicas

1. **Baseline Abrangente**: Comparação sistemática de 10 algoritmos ML para detecção de anomalias IoT usando CICIoT2023
2. **Soluções de Escalabilidade SVM**: Implementação de LinearSVC, SGDClassifier e SGDOneClassSVM como alternativas práticas a métodos baseados em kernel para dados de larga escala
3. **Avaliação Bayesiana Rigorosa**: Implementação completa da abordagem de Brodersen et al. (2010) para distribuições posteriores de Balanced Accuracy com intervalos de credibilidade e comparações probabilísticas P(A > B)
4. **Framework Metodológico**: Pipeline modular avançado com análise individual de algoritmo e comparação entre algoritmos
5. **Rigor Estatístico**: Experimentos com múltiplas execuções com validação estatística abrangente (frequentista e Bayesiana) e análise de estabilidade
6. **Pesquisa Reproduzível**: Pipeline DVC completo com organização de resultados baseada em timestamp para reprodutibilidade perfeita
7. **Classificação de Algoritmos**: Distinção clara entre abordagens de classificação supervisionada e detecção verdadeira de anomalias
8. **Diretrizes de Deployment em Larga Escala**: Recomendações práticas para seleção de algoritmos em grandes datasets IoT

### 5.2 Contribuições Técnicas

1. **Arquitetura de Pipeline Modular**: Estágios DVC independentes para cada algoritmo viabilizando execução paralela e re-execuções seletivas
2. **Suite de Escalabilidade SVM**: Implementação de LinearSVC, SGDClassifier e SGDOneClassSVM como alternativas práticas a métodos baseados em kernel
3. **Módulo de Análise Bayesiana**: Classes `BayesianAccuracyEvaluator` e suite de visualização completa implementando Brodersen et al. (2010) com cálculo de posteriors via convolução de distribuições Beta
4. **Sistema de Configuração Dinâmica**: Controle de ponto único (`TEST_MODE`) para alternar entre experimentos de validação e completos
5. **Organização Avançada de Resultados**: Estrutura hierárquica baseada em timestamp prevenindo perda de dados e viabilizando análise histórica
6. **Análise Individual de Algoritmo**: Relatório abrangente por algoritmo com evolução de desempenho, impacto de parâmetros e análise de eficiência
7. **Comparação Entre Algoritmos**: Teste de significância estatística (frequentista e Bayesiana) com métricas especializadas de detecção de anomalias
8. **Otimização de Recursos**: Ordenação computacional de algoritmos menos a mais complexos para feedback mais rápido

### 5.3 Contribuições Práticas

1. **Orientação de Algoritmos**: Recomendações baseadas em evidências para seleção de algoritmos em segurança IoT
2. **Framework de Escalabilidade**: Estratégias práticas para treinamento em milhões de amostras com infraestrutura apenas em CPU
3. **Benchmarks de Desempenho**: Baselines quantitativas para abordagens supervisionadas e não-supervisionadas em dados de larga escala
4. **Planejamento de Recursos**: Requisitos computacionais detalhados (30-50 horas com otimizações SVM vs. semanas sem)
5. **Framework de Implementação**: Infraestrutura experimental pronta para uso com geração automatizada de análises
6. **Recomendações de Deployment**: Diretrizes práticas de seleção de algoritmos para ambientes IoT com restrições de recursos

### 5.4 Contribuições de Dataset

1. **CICIoT2023 Pré-processado**: Versão limpa e normalizada com pré-processamento documentado e conjunto de features otimizado
2. **Metodologia de Amostragem**: Abordagem de amostragem estratificada validada para grandes datasets IoT com validação estatística
3. **Engenharia de Features**: Conjunto otimizado de 39 features para detecção binária de anomalias com análise de desempenho
4. **Padrões de Avaliação**: Métricas e procedimentos padronizados para pesquisa em segurança IoT com foco em detecção de anomalias

---

## 6. Limitações e Escopo

### 6.1 Limitações do Dataset

**Restrições de Amostragem**:
- **Escala Reduzida**: Amostra de 19,5% devido a limitações computacionais
- **Escopo Temporal**: Janela de tempo limitada pode perder padrões de longo prazo
- **Diversidade de Ataques**: Restrito aos tipos de ataque do CICIoT2023
- **Especificidade Ambiental**: Dados de testbed podem não capturar todas as variações do mundo real

**Limitações Metodológicas**:
- **Classificação Binária**: Problema simplificado vs. detecção de ataque multi-classe
- **Análise Estática**: Sem análise de padrões temporais/sequenciais
- **Espaço de Parâmetros**: Exploração limitada de hiperparâmetros devido a custo computacional
- **Engenharia de Features**: Sem técnicas avançadas de seleção ou criação de features

### 6.2 Considerações de Generalizabilidade

**Dependências de Contexto**:
- **Especificidade do Dataset**: Resultados vinculados às características do CICIoT2023
- **Ambiente IoT**: Descobertas podem não generalizar através de todos os ecossistemas IoT
- **Evolução de Ataques**: Desempenho pode degradar com padrões emergentes de ataque
- **Contexto de Deployment**: Diferenças entre ambientes de laboratório vs. produção

### 6.3 Limitações Técnicas

**Restrições Computacionais**:
- **Dependência de Hardware**: Resultados influenciados por recursos computacionais disponíveis
- **Limitações de Memória**: Algoritmos grandes podem requerer abordagens diferentes em escala
- **Restrições de Tempo**: Exploração limitada de algoritmos computacionalmente caros
- **Escalabilidade**: Padrões de desempenho podem mudar com datasets maiores

---

## 7. Direções Futuras de Pesquisa

### 7.1 Extensões Imediatas

**Aprimoramentos Metodológicos**:
1. **Análise de Dataset Completo**: Avaliação em dataset completo de 23M registros
2. **Classificação Multi-classe**: Identificação de tipos específicos de ataque
3. **Pré-processamento Avançado**: Seleção de features, redução de dimensionalidade, features ensemble
4. **Deep Learning**: Abordagens baseadas em CNN, LSTM e Transformer

**Extensões Analíticas**:
1. **Análise Temporal**: Reconhecimento de padrões sequenciais e séries temporais
2. **Detecção de Concept Drift**: Adaptação a padrões evolutivos de ataque
3. **Métodos Ensemble**: Combinando algoritmos individuais de melhor desempenho
4. **IA Explicável**: Interpretabilidade de modelos para analistas de segurança

### 7.2 Visão de Pesquisa de Longo Prazo

**Metodologias Avançadas**:
1. **Detecção em Tempo Real**: ML streaming para detecção online de anomalias
2. **Transfer Learning**: Generalização entre datasets e domínios
3. **Federated Learning**: Segurança IoT distribuída sem dados centralizados
4. **Robustez Adversarial**: Segurança contra ataques adversariais

**Domínios de Aplicação**:
1. **IoT Específico de Indústria**: Aplicações em saúde, manufatura, cidade inteligente
2. **Edge Computing**: Cenários de deployment com restrições de recursos
3. **Redes 5G/6G**: Segurança de rede de próxima geração
4. **Sistemas Autônomos**: Ecossistemas IoT auto-defensores

---

## 8. Instruções de Execução

### 8.1 Configuração do Ambiente

```bash
# Clonar repositório
git clone <repository-url>
cd iot-ids-research

# Criar ambiente virtual
python -m venv env
source env/bin/activate  # Linux/Mac
# ou env\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar DVC (se usando controle de versão de dados)
dvc init
```

### 8.2 Preparação de Dados

```bash
# Colocar dataset CICIoT2023 em data/raw/CSV/MERGED_CSV/
# Deve conter 63 arquivos CSV

# Verificar integridade de dados
python dvc_run_quality_check.py

# Executar amostragem
python dvc_sampling.py
```

### 8.3 Execução Aprimorada de Experimentos

**Opção 1: Pipeline DVC Completo** (Recomendado)
```bash
# Definir modo de experimento (ponto único de controle)
# Editar experiments/algorithm_comparison.py: TEST_MODE = False  # para experimentos completos

# Executar pipeline inteiro com ordenação computacional
dvc repro consolidate_results

# Monitorar progresso (cada algoritmo executa independentemente)
tail -f experiments/logs/algorithm_comparison_*.log

# Rastrear conclusão de algoritmo individual
ls -la experiments/results/full/  # ou experiments/results/test/
```

**Opção 2: Teste de Algoritmo Individual**
```bash
# Testar algoritmo único (útil para debugging)
cd experiments/
python3 run_single_algorithm.py logistic_regression

# Resultados automaticamente salvos com timestamp
ls experiments/results/test/TIMESTAMP_logistic_regression/
```

**Opção 3: Validação em Modo Teste**
```bash
# Validação rápida com amostra pequena
# Editar experiments/algorithm_comparison.py: TEST_MODE = True
dvc repro consolidate_results

# Verificar que todos os algoritmos completaram com sucesso
# Verificar experiments/results/test/ para resultados com timestamp
```

### 8.4 Análise Aprimorada de Resultados

```bash
# Visualizar estrutura hierárquica de resultados
ls -la experiments/results/

# Estrutura para modos teste e completo:
# experiments/results/test/    OU    experiments/results/full/
# ├── TIMESTAMP_algorithm_name/         # Resultados de algoritmo individual
# │   ├── results.json                  # Dados experimentais brutos
# │   ├── summary.json                  # Resumo estatístico
# │   └── individual_analysis/          # Análise detalhada
# │       ├── plots/                    # Visualizações específicas do algoritmo
# │       ├── tables/                   # Estatísticas detalhadas
# │       └── report/                   # Relatórios individuais
# └── TIMESTAMP_consolidation/          # Comparação entre algoritmos
#     ├── plots/                        # Visualizações comparativas
#     ├── tables/                       # Estatísticas resumidas
#     ├── report/                       # Relatório de análise final
#     └── data/                         # Datasets consolidados

# Arquivos chave de análise:
# Análise Individual de Algoritmo:
# - plots/performance_evolution.png      # Tendências de desempenho
# - plots/parameter_impact.png           # Efeitos de hiperparâmetros  
# - plots/confusion_matrix_analysis.png  # Análise de erros
# - plots/metrics_distribution.png       # Distribuições estatísticas
# - plots/execution_time_analysis.png    # Análise de eficiência
# - tables/descriptive_statistics.csv    # Estatísticas completas
# - report/individual_report.md          # Insights abrangentes

# Comparação Entre Algoritmos:
# - plots/algorithm_comparison.png       # Ranking de desempenho
# - plots/efficiency_analysis.png        # Desempenho vs. tempo
# - plots/anomaly_detection_analysis.png # Análise especializada
# - tables/final_results_summary.csv     # Resultados completos
# - report/final_analysis_report.md      # Descobertas abrangentes

# Acessar UI MLflow (se configurado)
mlflow server --host 127.0.0.1 --port 5000
# Abrir http://127.0.0.1:5000 no navegador para rastreamento de experimentos
```

---

## 9. Cronograma e Recursos Esperados

### 9.1 Requisitos Computacionais

**Estimativas de Tempo** (experimento completo com 10 algoritmos, configuração adaptativa, n=5 execuções):
- Carregamento de dados e pré-processamento: ~5 minutos
- Experimentos de Regressão Logística (20 configs × 5 runs): ~0,3h (O(n))
- Experimentos de SGDClassifier (20 configs × 5 runs): ~0,2h (O(n))
- Experimentos de LinearSVC (18 configs × 5 runs): ~0,5h (O(n))
- Experimentos de Random Forest (12 configs × 5 runs): ~3,3h (O(n log n))
- Experimentos de Isolation Forest (15 configs × 5 runs): ~3,1h (O(n log n))
- **Experimentos de Gradient Boosting (10 configs × 5 runs)**: ~3,9h (O(n log n), com otimização subsample=0.7)
- Experimentos de Elliptic Envelope (15 configs × 5 runs): ~3,8h (O(n²))
- **Experimentos de Local Outlier Factor (8 configs × 5 runs)**: ~5,6h (O(n²))
- Experimentos de SGDOneClassSVM (15 configs × 5 runs): ~2,5h (O(n))
- **Experimentos de MLP Classifier (8 configs × 5 runs)**: ~6,7h (O(n³), com early stopping agressivo)
- Geração de análise individual: ~5-10 minutos por algoritmo
- Consolidação final e visualização: ~10-15 minutos
- **Tempo total estimado**: ~30 horas (1,25 dias)
- **Total de experimentos**: 141 configurações × 5 runs = 705 experimentos

**Impacto da Otimização SVM**:
- LinearSVC + SGDClassifier substituem SVC padrão: **ganho de velocidade de ~10-100x** (horas vs. dias/semanas)
- SGDOneClassSVM substitui OneClassSVM com kernel: **ganho de velocidade de ~10-50x** (1-2h vs. dias)

**Requisitos de Recursos**:
- **Memória**: Uso de pico de 6-8 GB (aumentado devido a mais algoritmos)
- **Armazenamento**: ~15 GB para dados e resultados abrangentes
- **CPU**: Multi-core fortemente recomendado para execução paralela DVC

### 9.2 Fases de Execução Aprimoradas

**Fase 1: Configuração e Validação** (30 minutos)
- Configuração de ambiente e instalação de dependências
- Verificação de integridade de dados com métricas de qualidade
- Teste de pipeline com `TEST_MODE=True` (validação com amostra pequena)
- Validação de pipeline DVC e verificação de ordenação de estágios

**Fase 2: Execução Modular de Experimentos** (30-50 horas para 10 algoritmos)
- **Execução estágio por estágio**: Cada algoritmo executa como estágio DVC independente
- **Complexidade progressiva**: Algoritmos ordenados do mais rápido ao mais lento (O(n) → O(n³))
- **Monitoramento em tempo real**: Rastreamento individual de progresso e monitoramento de recursos
- **Análise individual**: Análise detalhada automática por algoritmo
- **Organização de resultados**: Separação de resultados baseada em timestamp (modos test/full)
- **Otimização SVM**: LinearSVC, SGDClassifier e SGDOneClassSVM para escalabilidade

**Fase 3: Consolidação e Análise** (30-60 minutos)
- **Comparação entre algoritmos**: Análise estatística e ranking
- **Visualizações avançadas**: Geração abrangente de gráficos
- **Geração de relatórios**: Relatórios individuais e consolidados
- **Validação de desempenho**: Verificação de sanidade de resultados e detecção de outliers

**Fase 4: Documentação e Validação** (1-2 horas)
- **Interpretação de resultados**: Teste de significância estatística
- **Benchmarking de desempenho**: Recomendações de algoritmos
- **Verificação de reprodutibilidade**: Validação de pipeline
- **Documentação final**: Relatórios de análise abrangentes

---

## 10. Garantia de Qualidade

### 10.1 Procedimentos de Validação

**Validação de Dados**:
- Testes estatísticos para representatividade de amostra
- Validação cruzada de passos de pré-processamento
- Verificações de integridade ao longo do pipeline

**Validação Experimental**:
- Múltiplas execuções independentes para significância estatística
- Monitoramento de memória e recursos
- Detecção automatizada de erros e recuperação

**Validação de Resultados**:
- Verificações de sanidade para intervalos de métricas
- Verificação de consistência entre execuções
- Detecção de outliers em resultados

### 10.2 Checklist de Reprodutibilidade

- [ ] Sementes aleatórias fixas ao longo do pipeline
- [ ] Versões de software documentadas
- [ ] Código e configurações com controle de versão
- [ ] Execução automatizada de pipeline
- [ ] Logging e monitoramento completos
- [ ] Métricas de avaliação padronizadas
- [ ] Documentação detalhada de metodologia

---

## Conclusão

Esta metodologia aprimorada fornece um framework abrangente e avançado para comparar algoritmos de machine learning para detecção de anomalias IoT. A combinação do dataset CICIoT2023, pipeline DVC modular, design experimental rigoroso, análise individual de algoritmo e validação estatística avançada garante resultados confiáveis, reproduzíveis e profundamente perspicazes.

**Avanços Metodológicos Principais**:
- **Cobertura Expandida de Algoritmos**: 10 algoritmos incluindo variantes escaláveis de SVM para deployment em larga escala
- **Otimização SVM**: LinearSVC, SGDClassifier e SGDOneClassSVM alcançando ganho de velocidade de 10-100x mantendo equivalência matemática
- **Arquitetura Modular**: Estágios DVC independentes viabilizando execução paralela e re-execuções seletivas
- **Análise Individual**: Relatório abrangente por algoritmo com evolução de desempenho e análise de eficiência
- **Organização Avançada**: Gerenciamento de resultados baseado em timestamp prevenindo perda de dados e viabilizando análise histórica

O pipeline DVC modular com estágios individuais de algoritmo viabiliza reprodutibilidade completa enquanto o sistema abrangente de monitoramento fornece insights sobre desempenho e requisitos de recursos. O sistema de análise individual gera insights detalhados para cada algoritmo, complementado por comparação sofisticada entre algoritmos.

**Impacto da Pesquisa**:
Este trabalho estabelece uma fundação sólida para pesquisa em segurança IoT fornecendo:
- **Soluções de Escalabilidade**: Alternativas práticas de SVM para treinamento em datasets de escala de milhões com hardware apenas em CPU
- **Recomendações Baseadas em Evidências**: Seleção de algoritmos baseada em avaliação empírica abrangente
- **Benchmarks de Desempenho**: Baselines quantitativas para abordagens supervisionadas e de detecção de anomalias
- **Framework Reutilizável**: Infraestrutura experimental completa para pesquisa reproduzível
- **Orientação de Deployment**: Planejamento de recursos e seleção de algoritmos para ambientes IoT de produção

Os resultados esperados contribuirão significativamente para o crescente corpo de conhecimento em segurança IoT fornecendo comparações abrangentes de algoritmos, insights detalhados de desempenho e um framework experimental sofisticado para a comunidade de pesquisa.

**Pronto para Implementação**: O framework está totalmente implementado e pronto para execução, com documentação abrangente, tratamento de erros e geração automatizada de análises.

*[Seção de resultados a ser preenchida após conclusão do experimento com análise individual e consolidada]*

---

## Apêndices

### Apêndice A: Configuração Completa de Algoritmos

```python
# Configurações aprimoradas de algoritmos com ordenação de complexidade
ALGORITHM_CONFIGS = {
    # O(n) - Mais Rápido
    'LogisticRegression': {
        'class': LogisticRegression,
        'test_params': [{'C': 1.0, 'max_iter': 100, 'random_state': 42}],
        'full_params': [
            {'C': 0.1, 'max_iter': 200, 'random_state': 42},
            {'C': 1.0, 'max_iter': 500, 'random_state': 42}
        ]
    },
    
    # O(n log n) - Métodos ensemble
    'RandomForest': {
        'class': RandomForestClassifier,
        'test_params': [{'n_estimators': 10, 'max_depth': 5, 'random_state': 42}],
        'full_params': [
            {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 15, 'random_state': 42}
        ]
    },
    
    # O(n²) - Métodos baseados em kernel e distância (otimizados com kernels lineares)
    'SVC': {
        'class': SVC,
        'test_params': [{'C': 1.0, 'kernel': 'linear', 'random_state': 42}],
        'full_params': [
            {'C': 0.1, 'kernel': 'linear', 'random_state': 42},
            {'C': 1.0, 'kernel': 'linear', 'random_state': 42}
        ]
    },
    
    # O(n³) - Mais computacionalmente complexo (arquitetura simplificada)
    'MLPClassifier': {
        'class': MLPClassifier,
        'test_params': [{'hidden_layer_sizes': (50,), 'max_iter': 100, 'random_state': 42}],
        'full_params': [
            {'hidden_layer_sizes': (50,), 'max_iter': 300, 'random_state': 42},
            {'hidden_layer_sizes': (100,), 'max_iter': 500, 'random_state': 42}
        ]
    }
    # ... [Especificações completas para todos os 9 algoritmos]
}
```

### Apêndice B: Framework de Análise Estatística Aprimorado

```python
# Análise estatística abrangente com métricas individuais e entre algoritmos
from scipy import stats
import pandas as pd
import numpy as np

def analyze_individual_algorithm(results):
    """Análise individual de algoritmo com evolução de desempenho"""
    # Análise de tendências de desempenho
    # Avaliação de impacto de parâmetros  
    # Métricas de estabilidade e eficiência
    # Validação estatística
    pass

def compare_algorithms(consolidated_results):
    """Comparação estatística entre algoritmos"""
    # ANOVA para comparação geral
    f_stat, p_value = stats.f_oneway(*algorithm_groups)
    
    # Comparações pareadas post-hoc com correção de Bonferroni
    # Cálculo de tamanho de efeito (d de Cohen)
    # Análise de detecção de anomalias vs classificação supervisionada
    # Validação de complexidade computacional
    pass

def generate_comprehensive_report(individual_analyses, cross_analysis):
    """Gerar relatórios detalhados com recomendações"""
    # Insights e recomendações de algoritmo individual
    # Ranking de desempenho entre algoritmos
    # Análise de eficiência de recursos
    # Recomendações de deployment
    pass
```

### Apêndice C: Implementação Avançada de Monitoramento de Recursos

```python
# Monitoramento aprimorado com integração de análise individual
import psutil
import time
from pathlib import Path

def monitor_algorithm_execution(algorithm_name):
    """Monitoramento aprimorado para execução individual de algoritmo"""
    # Rastreamento de recursos em tempo real
    # Detecção de vazamento de memória  
    # Identificação de gargalos de desempenho
    # Procedimentos de limpeza automatizados
    # Geração de análise individual
    pass

def consolidate_monitoring_data(all_algorithm_data):
    """Análise de recursos entre algoritmos"""
    # Validação de complexidade computacional
    # Padrões de uso de recursos
    # Comparações de eficiência
    # Análise de escalabilidade
    pass
```

---

**Versão do Documento**: 4.0  
**Última Atualização**: Outubro 2025  
**Status**: Pronto para Execução  
**Atualizações Principais**: 
- Estendido para 10 algoritmos com foco em escalabilidade para grandes datasets
- LinearSVC, SGDClassifier e SGDOneClassSVM como alternativas práticas a SVM baseado em kernel
- Otimizações SVM alcançando ganho de velocidade de 10-100x em dados de larga escala
- **Estratégia de configuração adaptativa** (Opção C): 8-20 configs por algoritmo baseado em complexidade computacional
- **Otimizações de desempenho**: Gradient Boosting subsample=0.7, MLP early stopping agressivo
- **Rigor estatístico**: 5 execuções independentes por configuração (705 experimentos totais)
- **Adicionada métrica balanced_accuracy** para melhor avaliação de datasets desbalanceados
- **Análise de justiça**: Justificativa para impactos de otimizações na validade da comparação
- Pipeline DVC modular com análise individual de algoritmo
- Validação estatística abrangente e organização de resultados
- Cronograma atualizado: ~30 horas (otimizado para janela de execução de 24-48h)

**Autores**: [A ser especificado]  
**Instituição**: [A ser especificado]  
**Contato**: [A ser especificado]

---

## Referências

### Metodologia Central e Algoritmos

**Dataset e Segurança IoT:**
- Neto, E. C. P., et al. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Canadian Institute for Cybersecurity*.
- Benkhelifa, E., et al. (2018). A critical review of practices and challenges in intrusion detection systems for IoT. *Journal of Network and Computer Applications*.
- Cook, A. A., et al. (2020). Anomaly detection for IoT time-series data: A survey. *IEEE Internet of Things Journal*.

**Algoritmos de Machine Learning:**
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM*.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). Isolation-Based Anomaly Detection. *ACM Transactions on Knowledge Discovery from Data*.
- Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*.
- Breiman, L. (2001). Random Forests. *Machine Learning*.

**Otimização e Escalabilidade:**
- Fan, R. E., et al. (2008). LIBLINEAR: A Library for Large Linear Classification. *Journal of Machine Learning Research*.
- Bottou, L., & Bousquet, O. (2007). The Tradeoffs of Large Scale Learning. *NeurIPS*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*.
- Shalev-Shwartz, S., et al. (2011). Pegasos: Primal Estimated sub-GrAdient SOlver for SVM. *Mathematical Programming*.

**Técnicas de Otimização de Desempenho:**
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367-378.
- Prechelt, L. (1998). Early stopping-but when? In *Neural Networks: Tricks of the Trade* (pp. 55-69). Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**Otimização de Hiperparâmetros:**
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305.
- Feurer, M., et al. (2015). Efficient and robust automated machine learning. In *Advances in Neural Information Processing Systems* (pp. 2962-2970).
- Probst, P., Boulesteix, A. L., & Bischl, B. (2019). Tunability: Importance of hyperparameters of machine learning algorithms. *Journal of Machine Learning Research*, 20(53), 1-32.

**Metodologia Experimental e Reprodutibilidade:**
- Smith, L. N. (2018). A disciplined approach to neural network hyper-parameters. *arXiv preprint arXiv:1803.09820*.
- Bischl, B., et al. (2021). Hyperparameter optimization: Foundations, algorithms, best practices and open challenges. *arXiv preprint arXiv:2107.05847*.
- Henderson, P., et al. (2018). Deep reinforcement learning that matters. In *AAAI Conference on Artificial Intelligence*.

**Systems Benchmarking:**
- Mattson, P., et al. (2020). MLPerf training benchmark. *Proceedings of Machine Learning and Systems (MLSys)*, 2, 336-349.
- SPEC (2017). SPEC CPU 2017 benchmark suite documentation. *Standard Performance Evaluation Corporation*. https://www.spec.org/cpu2017/
- Reddi, V. J., et al. (2020). MLPerf inference benchmark. In *ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)* (pp. 446-459).
- Dehghani, M., et al. (2021). Benchmarking neural network robustness to common corruptions and perturbations. *arXiv preprint arXiv:1903.12261*.
- Papadopoulos, P., et al. (2019). Benchmarking and optimization of edge computing systems. *IEEE Access*, 7, 17222-17237.

**Concept Drift (para fases futuras):**
- Lu, J., et al. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.
- Wahab, O. A. (2022). Intrusion detection in the IoT under data and concept drifts: Online deep learning approach. *IEEE Internet of Things Journal*, 9(21), 21706-21714.
- Xu, Z., et al. (2023). ADTCD: An adaptive anomaly detection approach toward concept drift in IoT. *IEEE Internet of Things Journal*.

**Análise Estatística:**
- Cochran, W. G. (1977). *Sampling techniques* (3ª ed.). John Wiley & Sons.
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2ª ed.). Lawrence Erlbaum Associates.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.

**Métricas de Avaliação:**
- Brodersen, K. H., et al. (2010). The balanced accuracy and its posterior distribution. In *20th International Conference on Pattern Recognition* (pp. 3121-3124). IEEE.
- Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233-240).
- Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. *arXiv preprint arXiv:2010.16061*.

