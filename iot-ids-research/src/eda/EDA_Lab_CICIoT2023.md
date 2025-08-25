# Laboratório de Análise Exploratória de Dados (EDA) - Dataset CICIoT2023

**Projeto:** Sistema de Detecção de Intrusão em Tempo Real para Ambientes IoT  
**Dataset:** CICIoT2023 - A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment  
**Autor:** [Augusto Custódio Vicente]  
**Instituição:** [Universidade Federal de Minas Gerais]  
**Data:** 

---

## 1. Introdução Teórica

### 1.1 Análise Exploratória de Dados (EDA)

A Análise Exploratória de Dados (EDA) é uma abordagem fundamental na ciência de dados que envolve a investigação sistemática de conjuntos de dados para descobrir padrões, anomalias, testar hipóteses e verificar suposições através de técnicas estatísticas descritivas e representações visuais (Tukey, 1977).

#### 1.1.1 Definição e Propósito

Segundo Tukey (1977), a EDA é "uma atitude, uma flexibilidade e uma disposição para procurar por aquelas coisas que acreditamos não estar lá, assim como por aquelas que acreditamos estar lá". No contexto de segurança cibernética e detecção de intrusão, a EDA assume papel crítico na compreensão dos padrões de comportamento normal e anômalo em redes de computadores.

#### 1.1.2 Objetivos da EDA em Segurança Cibernética

1. **Caracterização do Tráfego de Rede:** Identificar padrões normais de comunicação
2. **Detecção de Anomalias:** Descobrir comportamentos suspeitos ou maliciosos
3. **Preparação de Dados:** Limpeza e pré-processamento para algoritmos de ML
4. **Seleção de Features:** Identificar características mais relevantes para classificação
5. **Validação de Qualidade:** Verificar integridade e consistência dos dados

### 1.2 Contextualização do Dataset CICIoT2023

#### 1.2.1 Origem e Motivação

O dataset CICIoT2023 foi desenvolvido pelo Canadian Institute for Cybersecurity (CIC) da Universidade de New Brunswick para superar limitações dos datasets existentes em segurança IoT. As principais motivações incluem:

- **Carência de Diversidade:** Datasets anteriores não contemplavam ampla gama de ataques IoT
- **Topologia Limitada:** Poucos dispositivos reais em configurações experimentais
- **Origem dos Ataques:** Ataques executados por sistemas convencionais, não por dispositivos IoT comprometidos
- **Realismo Insuficiente:** Falta de cenários que representem implementações reais de IoT

#### 1.2.2 Características Distintivas

O CICIoT2023 apresenta características inovadoras:

1. **Topologia Extensiva:** 105 dispositivos IoT reais de diferentes marcas e tipos
2. **Diversidade de Ataques:** 33 tipos de ataques organizados em 7 categorias
3. **Realismo:** Ataques executados por dispositivos IoT maliciosos contra outros dispositivos IoT
4. **Completude:** Dados em formatos pcap (brutos) e CSV (features extraídas)

### 1.3 Taxonomia de Ataques no CICIoT2023

#### 1.3.1 Categorias de Ataques

1. **DDoS (Distributed Denial of Service):** 12 variações
2. **DoS (Denial of Service):** 4 tipos
3. **Reconnaissance:** 5 técnicas de reconhecimento
4. **Web-based:** 6 ataques baseados em aplicações web
5. **Brute Force:** 1 ataque de força bruta
6. **Spoofing:** 2 tipos de falsificação
7. **Mirai:** 3 variações do botnet Mirai

#### 1.3.2 Fundamentação Teórica dos Ataques

**Ataques de Negação de Serviço (DoS/DDoS):**
Baseados na teoria de sobrecarga de recursos, onde o atacante visa esgotar recursos computacionais, de rede ou de aplicação do alvo (Mirkovic & Reiher, 2004).

**Ataques de Reconhecimento:**
Fundamentados na metodologia de cyber kill chain (Hutchins et al., 2011), representando a fase inicial de coleta de informações.

**Ataques Web-based:**
Exploram vulnerabilidades em aplicações web seguindo a taxonomia OWASP Top 10 (OWASP, 2021).

---

## 2. Fundamentos Matemáticos e Estatísticos

### 2.1 Estatística Descritiva

#### 2.1.1 Medidas de Tendência Central

Para cada feature $X_i$ no dataset:

- **Média:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Mediana:** Valor central quando dados ordenados
- **Moda:** Valor mais frequente

#### 2.1.2 Medidas de Dispersão

- **Variância:** $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$
- **Desvio Padrão:** $\sigma = \sqrt{\sigma^2}$
- **Coeficiente de Variação:** $CV = \frac{\sigma}{\bar{x}}$

#### 2.1.3 Medidas de Forma

- **Assimetria (Skewness):** $\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$
- **Curtose:** $\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4}$

### 2.2 Análise Multivariada

#### 2.2.1 Matriz de Correlação

Para features $X$ e $Y$:
$$\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$$

#### 2.2.2 Análise de Componentes Principais (PCA)

Transformação linear que projeta dados em espaço de menor dimensão:
$$Y = WX$$

onde $W$ é a matriz de autovetores da matriz de covariância.

### 2.3 Teoria da Informação

#### 2.3.1 Entropia

Para uma variável aleatória $X$:
$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

#### 2.3.2 Ganho de Informação

$$IG(S,A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

---

## 3. Frameworks e Metodologias

### 3.1 Framework CRISP-DM Adaptado

#### 3.1.1 Entendimento do Negócio
- Definição de objetivos de segurança
- Identificação de requisitos de detecção
- Análise de ameaças específicas

#### 3.1.2 Entendimento dos Dados
- Coleta de metadados
- Exploração inicial
- Verificação de qualidade

#### 3.1.3 Preparação dos Dados
- Limpeza e tratamento de valores ausentes
- Normalização e padronização
- Engenharia de features

### 3.2 Framework KDD (Knowledge Discovery in Databases)

1. **Selection:** Seleção de dados relevantes
2. **Preprocessing:** Limpeza e transformação
3. **Transformation:** Redução de dimensionalidade
4. **Data Mining:** Aplicação de algoritmos
5. **Interpretation:** Avaliação e interpretação

### 3.3 Framework SEMMA

- **Sample:** Amostragem estatística
- **Explore:** Exploração visual e estatística
- **Modify:** Transformação de variáveis
- **Model:** Modelagem estatística
- **Assess:** Avaliação de resultados

---

## 4. Features do Dataset CICIoT2023

### 4.1 Categorização das Features

#### 4.1.1 Features Temporais
- `ts`: Timestamp
- `flow_duration`: Duração do fluxo
- `IAT`: Inter-arrival time

#### 4.1.2 Features de Protocolo
- `Protocol Type`: Tipo de protocolo (IP, UDP, TCP, etc.)
- `HTTP`, `HTTPS`, `DNS`, `TCP`, `UDP`: Indicadores binários de protocolo

#### 4.1.3 Features de Flags TCP
- `syn_flag_number`, `ack_flag_number`, `fin_flag_number`
- `rst_flag_number`, `psh_flag_number`

#### 4.1.4 Features Estatísticas
- `Rate`, `Srate`, `Drate`: Taxas de transmissão
- `Min`, `Max`, `AVG`, `Std`: Estatísticas de tamanho de pacote
- `Magnitude`, `Radius`, `Covariance`, `Variance`: Métricas derivadas

### 4.2 Análise Matemática das Features

#### 4.2.1 Features Derivadas

**Magnitude:**
$$Magnitude = \sqrt{(\text{avg\_in\_packets} + \text{avg\_out\_packets})^2}$$

**Radius:**
$$Radius = \sqrt{(\text{var\_in\_packets} + \text{var\_out\_packets})^2}$$

**Weight:**
$$Weight = \text{n\_in\_packets} \times \text{n\_out\_packets}$$

---

## 5. Framework Metodológico para EDA

### 5.1 Fase 1: Preparação e Carregamento dos Dados

#### 5.1.1 Estrutura dos Dados
```python
# Estrutura esperada dos arquivos CSV
# Arquivos: Merged01.csv até Merged63.csv
# Tamanho: ~13MB a ~181MB cada
# Features: 47 colunas + 1 coluna de rótulo
```

#### 5.1.2 Verificação de Integridade
1. **Verificação de Completude:** Identificar valores ausentes
2. **Verificação de Consistência:** Validar tipos de dados
3. **Verificação de Duplicatas:** Identificar registros duplicados

### 5.2 Fase 2: Análise Descritiva Univariada

#### 5.2.1 Análise de Distribuições
Para cada feature numérica:
1. Calcular estatísticas descritivas
2. Gerar histogramas
3. Avaliar normalidade (Teste de Shapiro-Wilk)
4. Identificar outliers (Método IQR)

#### 5.2.2 Análise de Variáveis Categóricas
1. Tabelas de frequência
2. Gráficos de barras
3. Análise de balanceamento de classes

### 5.3 Fase 3: Análise Bivariada

#### 5.3.1 Correlação entre Features
1. Matriz de correlação de Pearson
2. Heatmap de correlações
3. Identificação de multicolinearidade

#### 5.3.2 Análise por Classe de Ataque
1. Distribuições condicionais
2. Boxplots por categoria
3. Testes de hipóteses (ANOVA, Kruskal-Wallis)

### 5.4 Fase 4: Análise Multivariada

#### 5.4.1 Redução de Dimensionalidade
1. Análise de Componentes Principais (PCA)
2. t-SNE para visualização
3. Análise de agrupamento (K-means)

#### 5.4.2 Seleção de Features
1. Análise de importância (Random Forest)
2. Seleção baseada em correlação
3. Análise de ganho de informação

### 5.5 Fase 5: Visualização e Interpretação

#### 5.5.1 Visualizações Principais
1. **Distribuição de Classes:** Gráfico de pizza/barras
2. **Evolução Temporal:** Séries temporais por tipo de ataque
3. **Padrões de Tráfego:** Heatmaps de atividade
4. **Características por Ataque:** Radar charts

#### 5.5.2 Análise de Padrões
1. Identificação de assinaturas de ataques
2. Análise de comportamento anômalo
3. Caracterização de tráfego normal

---

## 6. Metodologia Passo a Passo

### Passo 1: Configuração do Ambiente
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
```

### Passo 2: Carregamento e Consolidação dos Dados
```python
# Função para carregar todos os arquivos CSV
def load_merged_datasets(path_pattern):
    datasets = []
    for i in range(1, 64):  # Merged01 até Merged63
        filename = f"Merged{i:02d}.csv"
        try:
            df = pd.read_csv(os.path.join(path_pattern, filename))
            datasets.append(df)
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado")
    return pd.concat(datasets, ignore_index=True)
```

### Passo 3: Análise Preliminar dos Dados
```python
def preliminary_analysis(df):
    print("=== ANÁLISE PRELIMINAR ===")
    print(f"Dimensões: {df.shape}")
    print(f"Memória utilizada: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Tipos de dados:\n{df.dtypes.value_counts()}")
    print(f"Valores ausentes:\n{df.isnull().sum().sum()}")
    
    return df.describe()
```

### Passo 4: Análise de Distribuição de Classes
```python
def analyze_class_distribution(df, label_column='label'):
    class_counts = df[label_column].value_counts()
    
    # Gráfico de distribuição
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', rot=45)
    plt.title('Distribuição de Classes')
    plt.ylabel('Frequência')
    
    plt.subplot(1, 2, 2)
    class_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Proporção de Classes')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    return class_counts
```

### Passo 5: Análise Estatística Descritiva
```python
def descriptive_statistics(df):
    # Separar features numéricas e categóricas
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Estatísticas descritivas
    desc_stats = df[numeric_features].describe()
    
    # Análise de assimetria e curtose
    skewness = df[numeric_features].skew()
    kurtosis = df[numeric_features].kurtosis()
    
    return desc_stats, skewness, kurtosis
```

### Passo 6: Análise de Correlações
```python
def correlation_analysis(df):
    # Matriz de correlação
    correlation_matrix = df.corr()
    
    # Visualização
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação - Features CICIoT2023')
    plt.tight_layout()
    plt.show()
    
    # Identificar correlações altas
    high_corr = correlation_matrix.abs() > 0.8
    high_corr_pairs = []
    for i in range(len(high_corr.columns)):
        for j in range(i+1, len(high_corr.columns)):
            if high_corr.iloc[i, j]:
                high_corr_pairs.append((
                    high_corr.columns[i], 
                    high_corr.columns[j], 
                    correlation_matrix.iloc[i, j]
                ))
    
    return correlation_matrix, high_corr_pairs
```

### Passo 7: Análise de Outliers
```python
def outlier_analysis(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    
    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outliers_info[feature] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'bounds': (lower_bound, upper_bound)
        }
    
    return outliers_info
```

### Passo 8: Análise por Tipo de Ataque
```python
def attack_type_analysis(df, label_column='label'):
    # Estatísticas por tipo de ataque
    attack_stats = df.groupby(label_column).agg({
        'flow_duration': ['mean', 'std', 'median'],
        'Tot_size': ['mean', 'std', 'median'],
        'Rate': ['mean', 'std', 'median']
    }).round(4)
    
    # Visualização comparativa
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribuição de duração de fluxo por ataque
    df.boxplot(column='flow_duration', by=label_column, ax=axes[0,0])
    axes[0,0].set_title('Duração do Fluxo por Tipo de Ataque')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Distribuição de tamanho total por ataque
    df.boxplot(column='Tot_size', by=label_column, ax=axes[0,1])
    axes[0,1].set_title('Tamanho Total por Tipo de Ataque')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Taxa de transmissão por ataque
    df.boxplot(column='Rate', by=label_column, ax=axes[1,0])
    axes[1,0].set_title('Taxa de Transmissão por Tipo de Ataque')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return attack_stats
```

### Passo 9: Análise de Componentes Principais
```python
def pca_analysis(df, n_components=10):
    # Preparar dados
    numeric_features = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_features].fillna(0)
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualizar variância explicada
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_components+1), pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada')
    plt.title('Variância Explicada por Componente')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 'ro-')
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Variância Explicada Acumulada')
    
    plt.tight_layout()
    plt.show()
    
    return pca, X_pca
```

### Passo 10: Análise de Importância de Features
```python
def feature_importance_analysis(df, label_column='label'):
    # Preparar dados
    numeric_features = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_features].fillna(0)
    y = df[label_column]
    
    # Random Forest para importância
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Importância das features
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualização
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Features Mais Importantes')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()
    
    return feature_importance
```

### Passo 11: Análise Temporal
```python
def temporal_analysis(df):
    # Converter timestamp
    df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
    
    # Análise por hora
    df['hour'] = df['timestamp'].dt.hour
    hourly_attacks = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    
    # Visualização
    plt.figure(figsize=(15, 8))
    hourly_attacks.plot(kind='bar', stacked=True)
    plt.title('Distribuição de Ataques por Hora')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Número de Ataques')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return hourly_attacks
```

### Passo 12: Relatório Final
```python
def generate_eda_report(df):
    report = {
        'dataset_info': {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage().sum() / 1024**2,
            'attack_types': df['label'].nunique(),
            'benign_ratio': (df['label'] == 'BenignTraffic').mean()
        },
        'data_quality': {
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'zero_variance_features': (df.var() == 0).sum()
        },
        'recommendations': []
    }
    
    # Adicionar recomendações baseadas na análise
    if report['data_quality']['missing_values'] > 0:
        report['recommendations'].append("Implementar estratégia de tratamento de valores ausentes")
    
    if report['dataset_info']['benign_ratio'] < 0.1:
        report['recommendations'].append("Dataset desbalanceado - considerar técnicas de balanceamento")
    
    return report
```

---

## 7. Considerações Específicas para Detecção de Intrusão

### 7.1 Métricas de Avaliação

#### 7.1.1 Métricas Clássicas
- **Acurácia:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precisão:** $\frac{TP}{TP + FP}$
- **Recall (Sensibilidade):** $\frac{TP}{TP + FN}$
- **F1-Score:** $\frac{2 \times Precision \times Recall}{Precision + Recall}$

#### 7.1.2 Métricas Específicas para IDS
- **Taxa de Falsos Positivos (FPR):** $\frac{FP}{FP + TN}$
- **Taxa de Detecção (DR):** $\frac{TP}{TP + FN}$
- **Custo de Classificação Incorreta**

### 7.2 Desafios em Datasets de Segurança

#### 7.2.1 Desbalanceamento de Classes
O tráfego benigno é tipicamente mais comum que ataques, criando desafios para algoritmos de ML.

#### 7.2.2 Evolução Temporal
Novos tipos de ataques podem não estar representados nos dados históricos.

#### 7.2.3 Dimensionalidade
Alto número de features pode causar "curse of dimensionality".

---

## 8. Validação e Limitações

### 8.1 Validação Estatística

#### 8.1.1 Testes de Normalidade
- Kolmogorov-Smirnov
- Shapiro-Wilk
- Anderson-Darling

#### 8.1.2 Testes de Homogeneidade
- Levene Test
- Bartlett Test

### 8.2 Limitações do Dataset

#### 8.2.1 Representatividade
- Limitado a ambiente de laboratório
- Dispositivos específicos podem não representar toda diversidade IoT

#### 8.2.2 Evolução Temporal
- Ataques podem evoluir após coleta dos dados
- Novos tipos de dispositivos IoT

---

## 9. Conclusões e Próximos Passos

### 9.1 Síntese da Análise

A EDA do dataset CICIoT2023 deve revelar:
1. Padrões distintivos para cada tipo de ataque
2. Features mais discriminativas
3. Qualidade e completude dos dados
4. Estratégias de pré-processamento necessárias

### 9.2 Recomendações para Modelagem

1. **Balanceamento de Classes:** Aplicar SMOTE ou técnicas similares
2. **Seleção de Features:** Usar análise de importância e correlação
3. **Normalização:** Aplicar StandardScaler ou MinMaxScaler
4. **Validação:** Implementar validação cruzada temporal

### 9.3 Direções Futuras

1. **Ensemble Methods:** Combinar múltiplos algoritmos
2. **Deep Learning:** Explorar redes neurais profundas
3. **Detecção em Tempo Real:** Implementar pipeline de streaming
4. **Explicabilidade:** Usar SHAP ou LIME para interpretabilidade

---

## 10. Referências

### Referências Principais

1. **Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023).** CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. *Sensors*, 23(13), 5941.

2. **Tukey, J. W. (1977).** Exploratory Data Analysis. *Addison-Wesley*.

3. **Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996).** From data mining to knowledge discovery in databases. *AI magazine*, 17(3), 37-54.

### Referências Metodológicas

4. **Wickham, H., & Grolemund, G. (2016).** R for Data Science: Import, Tidy, Transform, Visualize, and Model Data. *O'Reilly Media*.

5. **Géron, A. (2019).** Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. *O'Reilly Media*.

6. **Bruce, P., Bruce, A., & Gedeck, P. (2020).** Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python. *O'Reilly Media*.

### Referências em Segurança Cibernética

7. **Mirkovic, J., & Reiher, P. (2004).** A taxonomy of DDoS attack and DDoS defense mechanisms. *ACM SIGCOMM Computer Communication Review*, 34(2), 39-53.

8. **Hutchins, E. M., Cloppert, M. J., & Amin, R. M. (2011).** Intelligence-driven computer network defense informed by analysis of adversary campaigns and intrusion kill chains. *Leading Issues in Information Warfare & Security Research*, 1(1), 80.

9. **Sommer, R., & Paxson, V. (2010).** Outside the closed world: On using machine learning for network intrusion detection. *2010 IEEE symposium on security and privacy* (pp. 305-316).

### Referências em Machine Learning para Segurança

10. **Buczak, A. L., & Guven, E. (2016).** A survey of data mining and machine learning methods for cyber security intrusion detection. *IEEE Communications surveys & tutorials*, 18(2), 1153-1176.

11. **Khraisat, A., Gondal, I., Vamplew, P., & Kamruzzaman, J. (2019).** Survey of intrusion detection systems: techniques, datasets and challenges. *Cybersecurity*, 2(1), 1-22.

12. **Ahmad, Z., Shahid Khan, A., Wai Shiang, C., Abdullah, J., & Ahmad, F. (2021).** Network intrusion detection system: A systematic study of machine learning and deep learning approaches. *Transactions on Emerging Telecommunications Technologies*, 32(1), e4150.

### Referências em Análise Estatística

13. **Wilcox, R. R. (2016).** Introduction to robust estimation and hypothesis testing. *Academic press*.

14. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** The elements of statistical learning: data mining, inference, and prediction. *Springer Science & Business Media*.

15. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013).** An introduction to statistical learning. *Springer*.

### Referências em IoT Security

16. **Sivanathan, A., Sherratt, D., Gharakheili, H. H., Radford, A., Wijenayake, C., Vishwanath, A., & Sivaraman, V. (2017).** Characterizing and classifying IoT traffic in smart cities and campuses. *2017 IEEE conference on computer communications workshops* (pp. 559-564).

17. **Koroniotis, N., Moustafa, N., Sitnikova, E., & Turnbull, B. (2019).** Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset. *Future Generation Computer Systems*, 100, 779-796.

### Referências em Frameworks de Análise

18. **Shearer, C. (2000).** The CRISP-DM model: the new blueprint for data mining. *Journal of data warehousing*, 5(4), 13-22.

19. **Fayyad, U. M., Piatetsky-Shapiro, G., Smyth, P., & Uthurusamy, R. (Eds.). (1996).** Advances in knowledge discovery and data mining. *MIT press*.

20. **Shmueli, G., & Koppius, O. R. (2011).** Predictive analytics in information systems research. *MIS quarterly*, 553-572.

### Referências Técnicas

21. **OWASP Foundation. (2021).** OWASP Top 10 2021. *Available at: https://owasp.org/www-project-top-ten/*

22. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).** Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12(Oct), 2825-2830.

23. **McKinney, W. (2010).** Data structures for statistical computing in python. *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).

24. **Hunter, J. D. (2007).** Matplotlib: A 2D graphics environment. *Computing in science & engineering*, 9(3), 90-95.

25. **Waskom, M. L. (2021).** Seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.

---

**Nota:** Este laboratório foi desenvolvido com base no dataset CICIoT2023 e deve ser adaptado conforme as especificidades do ambiente de desenvolvimento e objetivos específicos da pesquisa de mestrado.

**Contato:** [email do autor]  
**Repositório:** [link do repositório do projeto]  
**Versão:** 1.0  
**Última atualização:** [data] 