# Framework Prático para EDA - Dataset CICIoT2023

**Objetivo:** Implementar uma análise exploratória de dados sistemática e academicamente fundamentada para o dataset CICIoT2023.

---

## 📋 Checklist de Implementação

### ✅ Pré-requisitos
- [ ] Python 3.8+
- [ ] Bibliotecas: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- [ ] Dados CICIoT2023 na pasta `datasets/CSV/MERGED_CSV/`
- [ ] Notebook Jupyter configurado

### ✅ Estrutura do Projeto
```
src/eda/
├── EDA_Lab_CICIoT2023.md          # Documentação teórica
├── EDA_Framework_Pratico.md       # Este arquivo
├── eda_implementation.ipynb       # Implementação principal
├── utils/
│   ├── data_loader.py            # Carregamento de dados
│   ├── statistical_analysis.py   # Análises estatísticas
│   ├── visualization.py          # Visualizações
│   └── report_generator.py       # Geração de relatórios
└── results/
    ├── figures/                  # Gráficos gerados
    ├── tables/                   # Tabelas estatísticas
    └── reports/                  # Relatórios finais
```

---

## 🚀 Implementação Passo a Passo

### FASE 1: Preparação do Ambiente

#### Passo 1.1: Instalação de Dependências
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install plotly jupyter-dash # Para visualizações interativas
pip install memory-profiler # Para monitoramento de memória
```

#### Passo 1.2: Configuração do Notebook
```python
# Configurações iniciais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import sys
import os

# Configurações de visualização
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# Configurar seed para reprodutibilidade
np.random.seed(42)

# Configurar pandas para exibir mais colunas
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
```

### FASE 2: Carregamento e Consolidação dos Dados

#### Passo 2.1: Função de Carregamento Otimizada
```python
def load_ciciot_datasets(data_path='datasets/CSV/MERGED_CSV/', 
                        sample_size=None, 
                        files_to_load=None):
    """
    Carrega e consolida os datasets CICIoT2023
    
    Args:
        data_path: Caminho para os arquivos CSV
        sample_size: Tamanho da amostra (None para carregar tudo)
        files_to_load: Lista de arquivos específicos (None para todos)
    
    Returns:
        DataFrame consolidado
    """
    data_path = Path(data_path)
    datasets = []
    
    # Listar arquivos disponíveis
    if files_to_load is None:
        csv_files = sorted(data_path.glob('Merged*.csv'))
    else:
        csv_files = [data_path / f for f in files_to_load]
    
    print(f"Carregando {len(csv_files)} arquivos...")
    
    total_rows = 0
    for i, file_path in enumerate(csv_files):
        try:
            print(f"  {i+1}/{len(csv_files)}: {file_path.name}")
            
            # Carregar com chunking para grandes arquivos
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                if sample_size and total_rows >= sample_size:
                    break
                chunks.append(chunk)
                total_rows += len(chunk)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                datasets.append(df)
                
        except Exception as e:
            print(f"    Erro ao carregar {file_path}: {e}")
    
    # Consolidar todos os datasets
    if datasets:
        full_dataset = pd.concat(datasets, ignore_index=True)
        
        # Aplicar amostragem se especificada
        if sample_size and len(full_dataset) > sample_size:
            full_dataset = full_dataset.sample(n=sample_size, random_state=42)
        
        print(f"Dataset consolidado: {full_dataset.shape}")
        return full_dataset
    else:
        raise ValueError("Nenhum dataset foi carregado com sucesso")

# Exemplo de uso
df = load_ciciot_datasets(sample_size=100000)  # Amostra de 100k para teste
```

#### Passo 2.2: Verificação Inicial dos Dados
```python
def initial_data_inspection(df):
    """Inspeção inicial dos dados"""
    print("=" * 50)
    print("INSPEÇÃO INICIAL DOS DADOS")
    print("=" * 50)
    
    # Informações básicas
    print(f"Dimensões: {df.shape}")
    print(f"Uso de memória: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Período dos dados: {pd.to_datetime(df['ts'], unit='s').min()} a {pd.to_datetime(df['ts'], unit='s').max()}")
    
    # Tipos de dados
    print("\nTipos de dados:")
    print(df.dtypes.value_counts())
    
    # Valores ausentes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nValores ausentes: {missing_values.sum()}")
        print(missing_values[missing_values > 0])
    else:
        print("\nNenhum valor ausente encontrado")
    
    # Valores duplicados
    duplicates = df.duplicated().sum()
    print(f"Registros duplicados: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Informações sobre as colunas
    print(f"\nColunas ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    
    return df.info()

# Executar inspeção inicial
initial_data_inspection(df)
```

### FASE 3: Análise Descritiva Univariada

#### Passo 3.1: Análise de Distribuição de Classes
```python
def analyze_class_distribution(df, label_col='label'):
    """Análise detalhada da distribuição de classes"""
    print("=" * 50)
    print("ANÁLISE DE DISTRIBUIÇÃO DE CLASSES")
    print("=" * 50)
    
    # Contagem de classes
    class_counts = df[label_col].value_counts()
    class_percentages = df[label_col].value_counts(normalize=True) * 100
    
    # Criar DataFrame para visualização
    class_summary = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    }).sort_values('Count', ascending=False)
    
    print("Distribuição de classes:")
    print(class_summary)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Gráfico de barras
    class_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Distribuição de Classes - Contagem')
    axes[0,0].set_ylabel('Frequência')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Gráfico de pizza
    class_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Distribuição de Classes - Proporção')
    axes[0,1].set_ylabel('')
    
    # Log scale para melhor visualização
    class_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral', logy=True)
    axes[1,0].set_title('Distribuição de Classes - Escala Log')
    axes[1,0].set_ylabel('Frequência (log)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Gráfico de barras horizontal
    class_counts.plot(kind='barh', ax=axes[1,1], color='lightgreen')
    axes[1,1].set_title('Distribuição de Classes - Horizontal')
    axes[1,1].set_xlabel('Frequência')
    
    plt.tight_layout()
    plt.show()
    
    # Análise de balanceamento
    total_samples = len(df)
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[-1]
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    print(f"\nAnálise de Balanceamento:")
    print(f"Classe majoritária: {majority_class} ({class_counts.max():,} samples)")
    print(f"Classe minoritária: {minority_class} ({class_counts.min():,} samples)")
    print(f"Razão de desbalanceamento: {imbalance_ratio:.2f}:1")
    
    return class_summary

# Executar análise de classes
class_analysis = analyze_class_distribution(df)
```

#### Passo 3.2: Análise Estatística Descritiva
```python
def comprehensive_statistical_analysis(df):
    """Análise estatística completa das features numéricas"""
    print("=" * 50)
    print("ANÁLISE ESTATÍSTICA DESCRITIVA")
    print("=" * 50)
    
    # Separar features numéricas
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remover timestamp se presente
    if 'ts' in numeric_features:
        numeric_features.remove('ts')
    
    print(f"Analisando {len(numeric_features)} features numéricas...")
    
    # Estatísticas descritivas básicas
    desc_stats = df[numeric_features].describe()
    
    # Estatísticas adicionais
    additional_stats = pd.DataFrame({
        'Skewness': df[numeric_features].skew(),
        'Kurtosis': df[numeric_features].kurtosis(),
        'CV': df[numeric_features].std() / df[numeric_features].mean(),
        'IQR': df[numeric_features].quantile(0.75) - df[numeric_features].quantile(0.25)
    })
    
    # Combinar estatísticas
    full_stats = pd.concat([desc_stats.T, additional_stats], axis=1)
    
    # Salvar estatísticas
    full_stats.to_csv('src/eda/results/tables/descriptive_statistics.csv')
    
    # Identificar features com comportamento especial
    print("\nFeatures com alta variabilidade (CV > 1):")
    high_cv = additional_stats[additional_stats['CV'] > 1].sort_values('CV', ascending=False)
    print(high_cv)
    
    print("\nFeatures com assimetria alta (|Skewness| > 2):")
    high_skew = additional_stats[abs(additional_stats['Skewness']) > 2].sort_values('Skewness', ascending=False)
    print(high_skew)
    
    # Visualização das distribuições
    plot_feature_distributions(df, numeric_features[:16])  # Primeiras 16 features
    
    return full_stats

def plot_feature_distributions(df, features):
    """Plota distribuições das features"""
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        df[feature].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue')
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
        
        # Adicionar linha da média
        mean_val = df[feature].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
        axes[i].legend()
    
    # Remover subplots extras
    for i in range(n_features, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()

# Executar análise estatística
statistical_analysis = comprehensive_statistical_analysis(df)
```

### FASE 4: Análise Bivariada

#### Passo 4.1: Análise de Correlações
```python
def correlation_analysis(df):
    """Análise completa de correlações"""
    print("=" * 50)
    print("ANÁLISE DE CORRELAÇÕES")
    print("=" * 50)
    
    # Selecionar features numéricas
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ts' in numeric_features:
        numeric_features.remove('ts')
    
    # Calcular matriz de correlação
    correlation_matrix = df[numeric_features].corr()
    
    # Visualização da matriz de correlação
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Matriz de Correlação - Features CICIoT2023')
    plt.tight_layout()
    plt.show()
    
    # Identificar correlações altas
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold para alta correlação
                high_corr_pairs.append({
                    'Feature1': correlation_matrix.columns[i],
                    'Feature2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                           key=abs, ascending=False)
    
    print(f"\nPares de features com alta correlação (|r| > 0.7):")
    print(high_corr_df)
    
    # Salvar resultados
    correlation_matrix.to_csv('src/eda/results/tables/correlation_matrix.csv')
    high_corr_df.to_csv('src/eda/results/tables/high_correlations.csv', index=False)
    
    return correlation_matrix, high_corr_df

# Executar análise de correlações
corr_matrix, high_corr = correlation_analysis(df)
```

#### Passo 4.2: Análise por Classe de Ataque
```python
def attack_class_analysis(df, label_col='label'):
    """Análise das features por classe de ataque"""
    print("=" * 50)
    print("ANÁLISE POR CLASSE DE ATAQUE")
    print("=" * 50)
    
    # Selecionar features numéricas importantes
    important_features = ['flow_duration', 'Tot_size', 'Rate', 'Srate', 'Drate', 
                         'AVG', 'Min', 'Max', 'Std']
    
    # Estatísticas por classe
    class_stats = df.groupby(label_col)[important_features].agg(['mean', 'std', 'median']).round(4)
    
    # Visualizações comparativas
    n_features = len(important_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(important_features):
        df.boxplot(column=feature, by=label_col, ax=axes[i])
        axes[i].set_title(f'{feature} por Classe de Ataque')
        axes[i].set_xlabel('Classe de Ataque')
        axes[i].set_ylabel(feature)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remover subplots extras
    for i in range(n_features, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()
    
    # Teste ANOVA para verificar diferenças significativas
    from scipy.stats import f_oneway
    
    anova_results = []
    classes = df[label_col].unique()
    
    for feature in important_features:
        groups = [df[df[label_col] == cls][feature].dropna() for cls in classes]
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({
            'Feature': feature,
            'F_statistic': f_stat,
            'p_value': p_value,
            'Significant': p_value < 0.05
        })
    
    anova_df = pd.DataFrame(anova_results)
    
    print("\nResultados do teste ANOVA:")
    print(anova_df)
    
    # Salvar resultados
    class_stats.to_csv('src/eda/results/tables/class_statistics.csv')
    anova_df.to_csv('src/eda/results/tables/anova_results.csv', index=False)
    
    return class_stats, anova_df

# Executar análise por classe
class_stats, anova_results = attack_class_analysis(df)
```

### FASE 5: Análise de Outliers

#### Passo 5.1: Detecção de Outliers
```python
def comprehensive_outlier_analysis(df):
    """Análise completa de outliers"""
    print("=" * 50)
    print("ANÁLISE DE OUTLIERS")
    print("=" * 50)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ts' in numeric_features:
        numeric_features.remove('ts')
    
    outlier_summary = []
    
    for feature in numeric_features:
        # Método IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        
        # Método Z-score
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        outliers_zscore = df[z_scores > 3]
        
        outlier_summary.append({
            'Feature': feature,
            'IQR_Outliers': len(outliers_iqr),
            'IQR_Percentage': (len(outliers_iqr) / len(df)) * 100,
            'ZScore_Outliers': len(outliers_zscore),
            'ZScore_Percentage': (len(outliers_zscore) / len(df)) * 100,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    # Visualização de outliers
    plot_outlier_boxplots(df, numeric_features[:12])
    
    print("\nResumo de outliers:")
    print(outlier_df[['Feature', 'IQR_Percentage', 'ZScore_Percentage']])
    
    # Salvar resultados
    outlier_df.to_csv('src/eda/results/tables/outlier_analysis.csv', index=False)
    
    return outlier_df

def plot_outlier_boxplots(df, features):
    """Plota boxplots para visualizar outliers"""
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        df[feature].plot(kind='box', ax=axes[i])
        axes[i].set_title(f'{feature}')
        axes[i].set_ylabel('Valor')
    
    # Remover subplots extras
    for i in range(n_features, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()

# Executar análise de outliers
outlier_analysis = comprehensive_outlier_analysis(df)
```

### FASE 6: Análise Multivariada

#### Passo 6.1: Análise de Componentes Principais (PCA)
```python
def pca_analysis(df, n_components=10):
    """Análise de Componentes Principais"""
    print("=" * 50)
    print("ANÁLISE DE COMPONENTES PRINCIPAIS")
    print("=" * 50)
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Preparar dados
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ts' in numeric_features:
        numeric_features.remove('ts')
    
    X = df[numeric_features].fillna(0)
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Análise de variância explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Variância explicada por componente
    axes[0,0].bar(range(1, n_components+1), explained_variance, alpha=0.7)
    axes[0,0].set_title('Variância Explicada por Componente')
    axes[0,0].set_xlabel('Componente Principal')
    axes[0,0].set_ylabel('Variância Explicada')
    
    # Variância explicada acumulada
    axes[0,1].plot(range(1, n_components+1), cumulative_variance, 'ro-')
    axes[0,1].set_title('Variância Explicada Acumulada')
    axes[0,1].set_xlabel('Componente Principal')
    axes[0,1].set_ylabel('Variância Explicada Acumulada')
    
    # Projeção 2D das duas primeiras componentes
    scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=df['label'].astype('category').cat.codes, 
                               alpha=0.6, cmap='tab10')
    axes[1,0].set_title('Projeção PCA - PC1 vs PC2')
    axes[1,0].set_xlabel('PC1')
    axes[1,0].set_ylabel('PC2')
    
    # Heatmap dos loadings
    loadings = pca.components_[:5]  # Primeiras 5 componentes
    sns.heatmap(loadings, xticklabels=numeric_features, 
                yticklabels=[f'PC{i+1}' for i in range(5)], 
                cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Loadings das Componentes Principais')
    
    plt.tight_layout()
    plt.show()
    
    # Análise dos loadings
    loadings_df = pd.DataFrame(
        pca.components_[:5].T,
        columns=[f'PC{i+1}' for i in range(5)],
        index=numeric_features
    )
    
    print(f"\nVariância explicada pelas primeiras {n_components} componentes:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print(f"\nVariância total explicada: {cumulative_variance[-1]:.3f} ({cumulative_variance[-1]*100:.1f}%)")
    
    # Salvar resultados
    pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)]).to_csv(
        'src/eda/results/tables/pca_components.csv', index=False
    )
    loadings_df.to_csv('src/eda/results/tables/pca_loadings.csv')
    
    return pca, X_pca, loadings_df

# Executar PCA
pca_results, pca_components, pca_loadings = pca_analysis(df, n_components=15)
```

#### Passo 6.2: Análise de Importância de Features
```python
def feature_importance_analysis(df, label_col='label'):
    """Análise de importância de features usando Random Forest"""
    print("=" * 50)
    print("ANÁLISE DE IMPORTÂNCIA DE FEATURES")
    print("=" * 50)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Preparar dados
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ts' in numeric_features:
        numeric_features.remove('ts')
    
    X = df[numeric_features].fillna(0)
    y = df[label_col]
    
    # Codificar labels se necessário
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Treinar Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    # Calcular importância
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top 20 features mais importantes
    top_20 = feature_importance.head(20)
    sns.barplot(data=top_20, x='Importance', y='Feature', ax=axes[0,0])
    axes[0,0].set_title('Top 20 Features Mais Importantes')
    axes[0,0].set_xlabel('Importância')
    
    # Distribuição das importâncias
    axes[0,1].hist(feature_importance['Importance'], bins=30, alpha=0.7, color='skyblue')
    axes[0,1].set_title('Distribuição das Importâncias')
    axes[0,1].set_xlabel('Importância')
    axes[0,1].set_ylabel('Frequência')
    
    # Importância acumulada
    cumulative_importance = np.cumsum(feature_importance['Importance'])
    axes[1,0].plot(range(len(cumulative_importance)), cumulative_importance, 'b-')
    axes[1,0].set_title('Importância Acumulada')
    axes[1,0].set_xlabel('Número de Features')
    axes[1,0].set_ylabel('Importância Acumulada')
    axes[1,0].grid(True)
    
    # Heatmap das correlações das top features
    top_10_features = feature_importance.head(10)['Feature'].tolist()
    corr_top = df[top_10_features].corr()
    sns.heatmap(corr_top, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Correlações - Top 10 Features')
    
    plt.tight_layout()
    plt.show()
    
    # Análise de redundância
    print("\nTop 10 features mais importantes:")
    print(feature_importance.head(10))
    
    # Encontrar número de features que explicam 90% da importância
    cumsum_importance = np.cumsum(feature_importance['Importance'])
    n_features_90 = np.argmax(cumsum_importance >= 0.9) + 1
    
    print(f"\nNúmero de features que explicam 90% da importância: {n_features_90}")
    
    # Salvar resultados
    feature_importance.to_csv('src/eda/results/tables/feature_importance.csv', index=False)
    
    return feature_importance, rf

# Executar análise de importância
feature_importance, rf_model = feature_importance_analysis(df)
```

### FASE 7: Análise Temporal

#### Passo 7.1: Análise de Padrões Temporais
```python
def temporal_analysis(df):
    """Análise de padrões temporais nos dados"""
    print("=" * 50)
    print("ANÁLISE TEMPORAL")
    print("=" * 50)
    
    # Converter timestamp
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['date'] = df['datetime'].dt.date
    
    # Análise por hora
    hourly_analysis = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    
    # Análise por dia da semana
    daily_analysis = df.groupby(['day_of_week', 'label']).size().unstack(fill_value=0)
    
    # Análise por data
    date_analysis = df.groupby(['date', 'label']).size().unstack(fill_value=0)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Distribuição por hora
    hourly_analysis.plot(kind='bar', stacked=True, ax=axes[0,0])
    axes[0,0].set_title('Distribuição de Ataques por Hora')
    axes[0,0].set_xlabel('Hora do Dia')
    axes[0,0].set_ylabel('Número de Ataques')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Heatmap por hora
    hourly_heatmap = hourly_analysis.T
    sns.heatmap(hourly_heatmap, cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Heatmap de Ataques por Hora')
    axes[0,1].set_xlabel('Hora do Dia')
    axes[0,1].set_ylabel('Tipo de Ataque')
    
    # Distribuição por dia da semana
    daily_analysis.plot(kind='bar', stacked=True, ax=axes[1,0])
    axes[1,0].set_title('Distribuição de Ataques por Dia da Semana')
    axes[1,0].set_xlabel('Dia da Semana (0=Segunda)')
    axes[1,0].set_ylabel('Número de Ataques')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Série temporal
    if len(date_analysis) > 1:
        date_analysis.sum(axis=1).plot(kind='line', ax=axes[1,1])
        axes[1,1].set_title('Série Temporal - Total de Ataques')
        axes[1,1].set_xlabel('Data')
        axes[1,1].set_ylabel('Número de Ataques')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Salvar resultados
    hourly_analysis.to_csv('src/eda/results/tables/hourly_analysis.csv')
    daily_analysis.to_csv('src/eda/results/tables/daily_analysis.csv')
    
    return hourly_analysis, daily_analysis, date_analysis

# Executar análise temporal
hourly_data, daily_data, date_data = temporal_analysis(df)
```

### FASE 8: Geração de Relatório Final

#### Passo 8.1: Relatório Automatizado
```python
def generate_comprehensive_report(df, analysis_results):
    """Gera relatório completo da EDA"""
    print("=" * 50)
    print("GERANDO RELATÓRIO FINAL")
    print("=" * 50)
    
    report = {
        'dataset_overview': {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage().sum() / 1024**2,
            'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}",
            'attack_types': df['label'].nunique(),
            'unique_attacks': df['label'].unique().tolist()
        },
        'data_quality': {
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        },
        'class_distribution': {
            'most_common_attack': df['label'].value_counts().index[0],
            'least_common_attack': df['label'].value_counts().index[-1],
            'imbalance_ratio': df['label'].value_counts().max() / df['label'].value_counts().min()
        },
        'feature_analysis': {
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'high_correlation_pairs': len(analysis_results.get('high_correlations', [])),
            'top_important_feature': analysis_results.get('feature_importance', pd.DataFrame()).iloc[0]['Feature'] if 'feature_importance' in analysis_results else 'N/A'
        },
        'outlier_analysis': {
            'features_with_outliers': len(analysis_results.get('outlier_analysis', [])),
            'avg_outlier_percentage': analysis_results.get('outlier_analysis', pd.DataFrame())['IQR_Percentage'].mean() if 'outlier_analysis' in analysis_results else 0
        },
        'temporal_patterns': {
            'peak_hour': hourly_data.sum(axis=1).idxmax() if 'hourly_data' in locals() else 'N/A',
            'peak_day': daily_data.sum(axis=1).idxmax() if 'daily_data' in locals() else 'N/A'
        },
        'recommendations': []
    }
    
    # Gerar recomendações
    if report['data_quality']['missing_percentage'] > 5:
        report['recommendations'].append("Implementar estratégia robusta de tratamento de valores ausentes")
    
    if report['class_distribution']['imbalance_ratio'] > 10:
        report['recommendations'].append("Aplicar técnicas de balanceamento de classes (SMOTE, undersampling)")
    
    if report['feature_analysis']['high_correlation_pairs'] > 10:
        report['recommendations'].append("Considerar remoção de features altamente correlacionadas")
    
    if report['outlier_analysis']['avg_outlier_percentage'] > 10:
        report['recommendations'].append("Implementar tratamento de outliers")
    
    # Salvar relatório
    import json
    with open('src/eda/results/reports/eda_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Imprimir resumo
    print("RESUMO EXECUTIVO:")
    print(f"- Total de registros: {report['dataset_overview']['total_records']:,}")
    print(f"- Tipos de ataques: {report['dataset_overview']['attack_types']}")
    print(f"- Qualidade dos dados: {100 - report['data_quality']['missing_percentage']:.1f}%")
    print(f"- Razão de desbalanceamento: {report['class_distribution']['imbalance_ratio']:.1f}:1")
    print(f"- Features numéricas: {report['feature_analysis']['numeric_features']}")
    
    if report['recommendations']:
        print("\nRECOMENDAÇÕES:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    return report

# Compilar resultados das análises
analysis_results = {
    'high_correlations': high_corr,
    'feature_importance': feature_importance,
    'outlier_analysis': outlier_analysis
}

# Gerar relatório final
final_report = generate_comprehensive_report(df, analysis_results)
```

---

## 🔧 Utilidades e Funções Auxiliares

### Função para Criar Estrutura de Pastas
```python
def create_eda_structure():
    """Cria estrutura de pastas para o projeto EDA"""
    import os
    
    directories = [
        'src/eda/utils',
        'src/eda/results/figures',
        'src/eda/results/tables',
        'src/eda/results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Criada pasta: {directory}")

# Executar no início do projeto
create_eda_structure()
```

### Função de Monitoramento de Memória
```python
def monitor_memory_usage():
    """Monitora uso de memória durante a análise"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Uso de memória: {memory_info.rss / 1024**2:.2f} MB")
    print(f"Uso de memória virtual: {memory_info.vms / 1024**2:.2f} MB")
    
    return memory_info

# Usar periodicamente durante a análise
monitor_memory_usage()
```

### Função de Backup dos Resultados
```python
def backup_results():
    """Faz backup dos resultados da EDA"""
    import shutil
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"src/eda/results_backup_{timestamp}"
    
    if os.path.exists('src/eda/results'):
        shutil.copytree('src/eda/results', backup_dir)
        print(f"Backup criado em: {backup_dir}")

# Executar ao final da análise
backup_results()
```

---

## 📊 Checklist de Validação

### ✅ Validação da Análise
- [ ] Todos os arquivos CSV foram carregados corretamente
- [ ] Distribuição de classes analisada e documentada
- [ ] Estatísticas descritivas calculadas para todas as features
- [ ] Matriz de correlação gerada e interpretada
- [ ] Outliers identificados e quantificados
- [ ] Análise PCA realizada com interpretação dos componentes
- [ ] Importância das features calculada
- [ ] Padrões temporais analisados
- [ ] Relatório final gerado

### ✅ Validação dos Resultados
- [ ] Gráficos salvos em alta resolução
- [ ] Tabelas exportadas em formato CSV
- [ ] Relatório JSON gerado com métricas principais
- [ ] Recomendações para pré-processamento documentadas
- [ ] Backup dos resultados realizado

### ✅ Documentação
- [ ] Código comentado e documentado
- [ ] Metodologia explicada
- [ ] Limitações identificadas
- [ ] Próximos passos definidos

---

## 🎯 Próximos Passos

### Fase de Pré-processamento
1. **Limpeza de Dados:** Implementar tratamento de valores ausentes e outliers
2. **Normalização:** Aplicar StandardScaler ou MinMaxScaler
3. **Balanceamento:** Implementar SMOTE ou outras técnicas
4. **Seleção de Features:** Usar resultados da análise de importância

### Fase de Modelagem
1. **Baseline Models:** Implementar modelos simples (Logistic Regression, Random Forest)
2. **Advanced Models:** Testar XGBoost, LightGBM, Neural Networks
3. **Ensemble Methods:** Combinar múltiplos modelos
4. **Hyperparameter Tuning:** Otimizar parâmetros dos modelos

### Fase de Validação
1. **Cross-Validation:** Implementar validação cruzada temporal
2. **Metrics Evaluation:** Calcular métricas específicas para IDS
3. **Confusion Matrix Analysis:** Analisar tipos de erros
4. **Feature Importance Validation:** Verificar estabilidade das features importantes

---

**Nota:** Este framework deve ser adaptado conforme as especificidades do ambiente de desenvolvimento e recursos computacionais disponíveis.

**Estimativa de Tempo:** 20-30 horas para implementação completa  
**Recursos Necessários:** 8-16 GB RAM, processador multi-core  
**Dependências:** Python 3.8+, bibliotecas científicas padrão 