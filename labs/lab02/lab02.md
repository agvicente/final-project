# Laboratório Prático: Setup do Ambiente de Pesquisa (Dias 1-2)
## Detecção de Intrusão Baseada em Anomalias em Sistemas IoT com Clustering Evolutivo

### 🎯 Objetivos do Laboratório

Ao final deste laboratório, você será capaz de:

1. **Configurar um ambiente Python reproduzível** com virtual environments e requirements.txt
2. **Implementar tracking de experimentos** com MLflow para pesquisa científica
3. **Gerenciar versionamento de dados** com DVC para projetos de machine learning
4. **Containerizar ambientes** com Docker para máxima reprodutibilidade
5. **Configurar Jupyter Lab** com extensões essenciais para data science

### 📋 Pré-requisitos

- Sistema Ubuntu 20.04+ ou Arch Linux
- Python 3.8+ instalado
- Git configurado
- Conexão com internet estável
- Privilégios de administrador (sudo)

---

## 🧱 Módulo 1: Fundamentos dos Virtual Environments

### 1.1 Teoria: Por que Virtual Environments?

**Problema**: Em pesquisas de machine learning, diferentes projetos frequentemente requerem versões específicas de bibliotecas. Imagine trabalhar em:
- Projeto A: TensorFlow 2.8 + NumPy 1.21
- Projeto B: TensorFlow 2.12 + NumPy 1.24

Instalar globalmente causaria conflitos e resultados não reproduzíveis.

**Solução**: Virtual environments isolam dependências por projeto, garantindo:
- **Reprodutibilidade**: Mesmo ambiente em qualquer máquina
- **Isolamento**: Projetos não interferem entre si
- **Documentação**: Lista exata de dependências (requirements.txt)
- **Colaboração**: Colegas podem replicar seu ambiente exato

### 1.2 Prática: Configurando o Workspace Python

#### Passo 1: Criando a estrutura do projeto

```bash
# Criar diretório principal da pesquisa
mkdir ~/iot-ids-research
cd ~/iot-ids-research

# Estrutura recomendada para pesquisa científica
mkdir -p {data/raw,data/processed,notebooks,src,models,experiments,docs,configs}
```

#### Passo 2: Configurando Virtual Environment

```bash
# Criar virtual environment
python3 -m venv venv

# Ativar (Ubuntu/Arch)
source venv/bin/activate

# Verificar ativação (prompt deve mostrar (venv))
which python
python --version
```

#### Passo 3: Criando requirements.txt científico

Crie `requirements.txt` com dependências essenciais para pesquisa:

```txt
# Core Data Science
numpy==1.24.3
pandas==1.5.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2

# Machine Learning
scikit-learn==1.2.2
imbalanced-learn==0.10.1

# Deep Learning (opcional, descomente se necessário)
# torch==2.0.1
# tensorflow==2.12.0

# Experiment Tracking
mlflow==2.3.1
wandb==0.15.3

# Data Version Control
dvc[all]==2.58.2

# Jupyter
jupyter==1.0.0
jupyterlab==4.0.2
ipywidgets==8.0.6

# Visualization
plotly==5.14.1
bokeh==3.1.1

# Development
pytest==7.3.1
black==23.3.0
flake8==6.0.0
pre-commit==3.3.2

# Data handling
openpyxl==3.1.2
xlrd==2.0.1

# Network/IoT specific
networkx==3.1
pyshark==0.6

# Statistics
statsmodels==0.14.0
```

#### Passo 4: Instalação e validação

```bash
# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# Validar instalação
python -c "import numpy, pandas, sklearn, mlflow, dvc; print('✅ Instalação bem-sucedida!')"
```

#### Passo 5: Script de ativação automatizada

Crie `activate_env.sh` para facilitar ativação:

```bash
#!/bin/bash
# activate_env.sh - Script para ativar ambiente de pesquisa

echo "🔬 Ativando ambiente de pesquisa IoT-IDS..."
source venv/bin/activate

echo "📦 Verificando dependências principais..."
python -c "
import sys
modules = ['numpy', 'pandas', 'sklearn', 'mlflow', 'dvc']
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module} - FALTANDO')
        sys.exit(1)
"

echo "🚀 Ambiente pronto! Use 'deactivate' para sair."
echo "📂 Estrutura do projeto:"
tree -L 2 -I 'venv'
```

```bash
# Tornar executável
chmod +x activate_env.sh

# Usar
./activate_env.sh
```

### 1.3 Exercício Prático

Teste o isolamento do ambiente:

```python
# test_environment.py
import sys
import numpy as np
import pandas as pd
import sklearn

print(f"Python executable: {sys.executable}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Criar dataset sintético para teste
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                         n_redundant=10, n_clusters_per_class=1, random_state=42)

# Testar algoritmo baseline
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X)

print(f"✅ Dataset criado: {X.shape}")
print(f"✅ Anomalias detectadas: {sum(anomalies == -1)}/{len(anomalies)}")
print("🔬 Ambiente funcionando corretamente!")
```

---

## 🔬 Módulo 2: MLflow para Tracking de Experimentos

### 2.1 Teoria: Por que MLflow em Pesquisa Científica?

MLflow é uma plataforma open-source para gerenciar o ciclo de vida completo de machine learning, permitindo rastrear experimentos, armazenar modelos e garantir reprodutibilidade.

**Problemas que MLflow resolve:**
- **Perda de experimentos**: Sem tracking, experimentos são perdidos
- **Falta de reprodutibilidade**: Dificulta replicação de resultados
- **Falta de transparência**: Dificulta compreensão de como modelos foram criados
- **Colaboração**: Facilita compartilhamento de experimentos entre pesquisadores

**Componentes principais:**
1. **Tracking**: Log de parâmetros, métricas e artifacts
2. **Projects**: Empacotamento reproduzível de código ML
3. **Models**: Formato padrão para deployment
4. **Registry**: Armazenamento centralizado de modelos

### 2.2 Prática: Configurando MLflow

#### Passo 1: Inicialização básica

```bash
# Criar diretório para experimentos
mkdir -p experiments/mlflow-tracking
cd experiments/mlflow-tracking

# Inicializar MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 127.0.0.1 \
              --port 5000 &

# Verificar se está rodando
curl http://127.0.0.1:5000/
```

#### Passo 2: Primeiro experimento científico

Crie `baseline_experiment.py`:

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IoT-IDS-Baseline")

def generate_iot_like_data(n_samples=10000, contamination=0.1):
    """Simula dados IoT com anomalias"""
    np.random.seed(42)
    
    # Features normais
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0], 
        cov=np.eye(4), 
        size=int(n_samples * (1 - contamination))
    )
    
    # Anomalias (ataques)
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3, 3], 
        cov=np.eye(4) * 2, 
        size=int(n_samples * contamination)
    )
    
    # Combinar dados
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([
        np.ones(len(normal_data)),  # 1 = normal
        -np.ones(len(anomaly_data))  # -1 = anomalia
    ])
    
    # Embaralhar
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def run_isolation_forest_experiment(contamination=0.1, n_estimators=100):
    """Executa experimento com Isolation Forest"""
    
    with mlflow.start_run(run_name=f"IsolationForest_cont_{contamination}"):
        # Log parâmetros
        mlflow.log_param("algorithm", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("n_samples", 10000)
        
        # Gerar dados
        X, y_true = generate_iot_like_data(contamination=contamination)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42
        )
        
        # Treinar modelo
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train)
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_accuracy = sum(y_pred_train == y_train) / len(y_train)
        test_accuracy = sum(y_pred_test == y_test) / len(y_test)
        
        # AUC (convertendo predições para scores)
        scores_test = model.decision_function(X_test)
        auc_score = roc_auc_score(y_test == -1, -scores_test)  # Inverter scores
        
        # Log métricas
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("auc_score", auc_score)
        
        # Visualização
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Distribuição de scores
        plt.subplot(1, 3, 1)
        plt.hist(scores_test[y_test == 1], alpha=0.7, label='Normal', bins=30)
        plt.hist(scores_test[y_test == -1], alpha=0.7, label='Anomaly', bins=30)
        plt.xlabel('Isolation Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        
        # Plot 2: Confusion Matrix visual
        plt.subplot(1, 3, 2)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 3: Features scatter (primeiras 2 dimensões)
        plt.subplot(1, 3, 3)
        normal_mask = y_test == 1
        anomaly_mask = y_test == -1
        plt.scatter(X_test[normal_mask, 0], X_test[normal_mask, 1], 
                   alpha=0.6, label='Normal', s=20)
        plt.scatter(X_test[anomaly_mask, 0], X_test[anomaly_mask, 1], 
                   alpha=0.8, label='Anomaly', s=20, c='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Feature Space')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('experiment_visualization.png', dpi=150, bbox_inches='tight')
        
        # Log artifacts
        mlflow.log_artifact('experiment_visualization.png')
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        
        # Log dataset info
        dataset_info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "contamination_real": sum(y_true == -1) / len(y_true),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        for key, value in dataset_info.items():
            mlflow.log_param(f"data_{key}", value)
        
        print(f"✅ Experimento concluído!")
        print(f"📊 Acurácia de teste: {test_accuracy:.3f}")
        print(f"📊 AUC Score: {auc_score:.3f}")
        
        return model, test_accuracy, auc_score

if __name__ == "__main__":
    # Executar experimentos com diferentes parâmetros
    results = []
    
    contaminations = [0.05, 0.1, 0.15, 0.2]
    n_estimators_list = [50, 100, 200]
    
    for cont in contaminations:
        for n_est in n_estimators_list:
            print(f"\n🔬 Executando: contamination={cont}, n_estimators={n_est}")
            model, acc, auc = run_isolation_forest_experiment(cont, n_est)
            results.append({
                'contamination': cont,
                'n_estimators': n_est,
                'accuracy': acc,
                'auc': auc
            })
    
    # Resumo dos resultados
    results_df = pd.DataFrame(results)
    print("\n📈 Resumo dos experimentos:")
    print(results_df.round(3))
    print(f"\n🏆 Melhor configuração: {results_df.loc[results_df['auc'].idxmax()]}")
```

#### Passo 3: Executar e analisar

```bash
# Executar experimento
python baseline_experiment.py

# Acessar UI do MLflow
# Abrir http://127.0.0.1:5000 no navegador
```

### 2.3 Exercício Avançado: Comparação de Algoritmos

Crie `algorithm_comparison.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

mlflow.set_experiment("Algorithm-Comparison")

def compare_algorithms():
    """Compara diferentes algoritmos de detecção de anomalias"""
    
    algorithms = {
        'IsolationForest': IsolationForest(contamination=0.1, random_state=42),
        'OneClassSVM': OneClassSVM(nu=0.1, kernel='rbf'),
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
        'EllipticEnvelope': EllipticEnvelope(contamination=0.1, random_state=42)
    }
    
    # Dados sintéticos
    from baseline_experiment import generate_iot_like_data
    X, y_true = generate_iot_like_data(n_samples=5000)
    
    results = []
    
    for name, model in algorithms.items():
        with mlflow.start_run(run_name=f"{name}_comparison"):
            start_time = time.time()
            
            # Treinar
            if name == 'LocalOutlierFactor':
                y_pred = model.fit_predict(X)
            else:
                model.fit(X)
                y_pred = model.predict(X)
            
            training_time = time.time() - start_time
            
            # Métricas
            accuracy = sum(y_pred == y_true) / len(y_true)
            
            # Log
            mlflow.log_param("algorithm", name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)
            
            if name != 'LocalOutlierFactor':
                mlflow.sklearn.log_model(model, f"{name}_model")
            
            results.append({
                'algorithm': name,
                'accuracy': accuracy,
                'time': training_time
            })
            
            print(f"✅ {name}: Acc={accuracy:.3f}, Time={training_time:.2f}s")
    
    return results

if __name__ == "__main__":
    compare_algorithms()
```

---

## 📊 Módulo 3: DVC para Versionamento de Dados

### 3.1 Teoria: Por que DVC?

DVC (Data Version Control) é uma ferramenta de controle de versão especificamente projetada para projetos de ciência de dados e machine learning, que permite versionar grandes datasets e pipelines de ML.

**Problemas que DVC resolve:**
- **Git não suporta arquivos grandes**: Datasets de GB/TB quebram repositórios
- **Reprodutibilidade**: Difícil rastrear versões exatas de dados usados
- **Colaboração**: Compartilhar datasets grandes entre equipes
- **Pipelines**: Automatizar fluxos de dados → modelo → avaliação

**Conceitos fundamentais:**
- **.dvc files**: Metadados sobre arquivos de dados (como .gitignore)
- **Remote storage**: Dados ficam em nuvem (S3, GCS, etc.)
- **Pipelines**: Dependências entre estágios de processamento
- **Experiments**: Comparação de diferentes configurações

### 3.2 Prática: Configurando DVC

#### Passo 1: Inicialização

```bash
# Voltar ao diretório principal
cd ~/iot-ids-research

# Inicializar Git (se ainda não feito)
git init

# Inicializar DVC
dvc init

# Verificar estrutura criada
ls -la .dvc/
cat .dvcignore
```

#### Passo 2: Simulando dataset IoT

Crie `generate_dataset.py`:

```python
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_iot_traffic_dataset(n_samples=100000):
    """Gera dataset sintético de tráfego IoT"""
    np.random.seed(42)
    
    # Tipos de dispositivos IoT
    device_types = ['camera', 'sensor', 'thermostat', 'smart_light', 'door_lock']
    protocols = ['TCP', 'UDP', 'HTTP', 'MQTT', 'CoAP']
    
    # Gerar timestamps
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(seconds=i*10) for i in range(n_samples)]
    
    # Features de tráfego normal
    normal_samples = int(n_samples * 0.9)
    
    # Tráfego normal
    normal_data = {
        'timestamp': timestamps[:normal_samples],
        'device_type': np.random.choice(device_types, normal_samples),
        'protocol': np.random.choice(protocols, normal_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'packet_size': np.random.normal(512, 128, normal_samples).astype(int),
        'duration': np.random.exponential(2.0, normal_samples),
        'src_port': np.random.randint(1024, 65535, normal_samples),
        'dst_port': np.random.choice([80, 443, 1883, 8080, 5683], normal_samples),
        'packet_count': np.random.poisson(10, normal_samples),
        'label': ['normal'] * normal_samples
    }
    
    # Tráfego anômalo (ataques)
    anomaly_samples = n_samples - normal_samples
    attack_types = ['ddos', 'mirai', 'mitm', 'recon', 'spoofing']
    
    anomaly_data = {
        'timestamp': timestamps[normal_samples:],
        'device_type': np.random.choice(device_types, anomaly_samples),
        'protocol': np.random.choice(protocols, anomaly_samples),
        'packet_size': np.random.normal(1024, 512, anomaly_samples).astype(int),  # Maior
        'duration': np.random.exponential(5.0, anomaly_samples),  # Maior duração
        'src_port': np.random.randint(1, 1024, anomaly_samples),  # Portas suspeitas
        'dst_port': np.random.randint(1, 65535, anomaly_samples),
        'packet_count': np.random.poisson(50, anomaly_samples),  # Muito mais pacotes
        'label': np.random.choice(attack_types, anomaly_samples)
    }
    
    # Combinar dados
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = list(normal_data[key]) + list(anomaly_data[key])
    
    df = pd.DataFrame(all_data)
    
    # Adicionar features derivadas
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['bytes_per_packet'] = df['packet_size'] * df['packet_count']
    df['packets_per_second'] = df['packet_count'] / (df['duration'] + 0.001)
    
    # Embaralhar
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def create_multiple_datasets():
    """Cria múltiplas versões do dataset para simular evolução"""
    
    datasets = {
        'v1_small': 10000,
        'v2_medium': 50000,
        'v3_large': 100000
    }
    
    os.makedirs('data/raw', exist_ok=True)
    
    for version, size in datasets.items():
        print(f"Gerando {version} com {size} amostras...")
        df = generate_iot_traffic_dataset(size)
        
        filepath = f'data/raw/iot_traffic_{version}.csv'
        df.to_csv(filepath, index=False)
        
        print(f"✅ Salvo: {filepath} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        print(f"   Distribuição de labels: {df['label'].value_counts().to_dict()}")
        print()

if __name__ == "__main__":
    create_multiple_datasets()
```

```bash
# Gerar datasets
python generate_dataset.py

# Verificar arquivos criados
ls -lh data/raw/
```

#### Passo 3: Versionando dados com DVC

```bash
# Adicionar primeiro dataset ao DVC
dvc add data/raw/iot_traffic_v1_small.csv

# Verificar o que foi criado
cat data/raw/iot_traffic_v1_small.csv.dvc
cat .gitignore

# Fazer commit no Git (apenas metadados)
git add data/raw/iot_traffic_v1_small.csv.dvc .gitignore
git commit -m "Add v1 small dataset"

# Adicionar outros datasets
dvc add data/raw/iot_traffic_v2_medium.csv
dvc add data/raw/iot_traffic_v3_large.csv

git add data/raw/*.dvc
git commit -m "Add medium and large datasets"
```

#### Passo 4: Configurando remote storage (simulado)

```bash
# Criar "remote" local para teste (em produção seria S3/GCS)
mkdir -p /tmp/dvc-remote

# Configurar remote
dvc remote add -d local /tmp/dvc-remote

# Fazer push dos dados
dvc push

# Verificar remote
ls -la /tmp/dvc-remote/
```

#### Passo 5: Simulando colaboração

```bash
# Simular git clone em outra máquina
cd /tmp
git clone ~/iot-ids-research iot-ids-research-clone
cd iot-ids-research-clone

# Tentar acessar dados
ls data/raw/  # Arquivos não estão lá (apenas .dvc)

# Baixar dados
dvc pull

# Agora os dados estão disponíveis
ls -lh data/raw/
head data/raw/iot_traffic_v1_small.csv
```

### 3.3 Criando Pipeline de Processamento

Crie `dvc_pipeline.py`:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import json
import os

def load_config():
    """Carrega configuração do pipeline"""
    config = {
        'input_file': 'data/raw/iot_traffic_v2_medium.csv',
        'output_dir': 'data/processed',
        'test_size': 0.2,
        'random_state': 42,
        'features_to_encode': ['device_type', 'protocol'],
        'features_to_scale': ['packet_size', 'duration', 'packet_count', 
                             'bytes_per_packet', 'packets_per_second']
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/preprocessing.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return config

def preprocess_data(config_file='configs/preprocessing.yaml'):
    """Pipeline de pré-processamento"""
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    print("📊 Carregando dados...")
    df = pd.read_csv(config['input_file'])
    
    print(f"   Shape original: {df.shape}")
    print(f"   Distribuição de labels: {df['label'].value_counts().to_dict()}")
    
    # Separar features e target
    feature_cols = ['device_type', 'protocol', 'packet_size', 'duration', 
                   'src_port', 'dst_port', 'packet_count', 'hour', 
                   'day_of_week', 'bytes_per_packet', 'packets_per_second']
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Encoding categórico
    print("🔄 Aplicando encoding categórico...")
    encoders = {}
    for col in config['features_to_encode']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Normalização
    print("📏 Aplicando normalização...")
    scaler = StandardScaler()
    X[config['features_to_scale']] = scaler.fit_transform(X[config['features_to_scale']])
    
    # Encoding do target (binário: normal vs anomalia)
    y_binary = (y != 'normal').astype(int)
    y_multiclass = LabelEncoder().fit_transform(y)
    
    # Split train/test
    print("✂️  Dividindo train/test...")
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multiclass, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y_binary
    )
    
    # Salvar dados processados
    os.makedirs(config['output_dir'], exist_ok=True)
    
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train_bin,
        'y_test_binary': y_test_bin,
        'y_train_multiclass': y_train_multi,
        'y_test_multiclass': y_test_multi
    }
    
    for name, data in datasets.items():
        filepath = f"{config['output_dir']}/{name}.csv"
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.Series(data).to_csv(filepath, index=False, header=[name])
    
    # Salvar transformers
    joblib.dump(encoders, f"{config['output_dir']}/encoders.pkl")
    joblib.dump(scaler, f"{config['output_dir']}/scaler.pkl")
    
    # Estatísticas do dataset
    stats = {
        'original_shape': df.shape,
        'processed_shape': X.shape,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'anomaly_rate': float(y_binary.mean()),
        'feature_names': list(X.columns),
        'target_distribution': {
            'normal': int((y_binary == 0).sum()),
            'anomaly': int((y_binary == 1).sum())
        }
    }
    
    with open(f"{config['output_dir']}/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✅ Pré-processamento concluído!")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Taxa de anomalias: {stats['anomaly_rate']:.3f}")
    
    return stats

if __name__ == "__main__":
    config = load_config()
    stats = preprocess_data()
```

#### Configurando pipeline DVC

```bash
# Voltar ao diretório principal
cd ~/iot-ids-research

# Criar arquivo dvc.yaml para pipeline
cat > dvc.yaml << 'EOF'
stages:
  preprocess:
    cmd: python dvc_pipeline.py
    deps:
    - data/raw/iot_traffic_v2_medium.csv
    - dvc_pipeline.py
    - configs/preprocessing.yaml
    outs:
    - data/processed/
    metrics:
    - data/processed/dataset_stats.json
EOF

# Executar pipeline
dvc repro

# Verificar resultados
ls data/processed/
cat data/processed/dataset_stats.json
```

---

## 🐳 Módulo 4: Docker para Reprodutibilidade

### 4.1 Teoria: Por que Docker em Pesquisa?

Docker permite empacotar aplicações e suas dependências em containers leves e portáveis, garantindo que o código execute de forma consistente independente do ambiente de implantação.

**Problemas que Docker resolve:**
- **"Funciona na minha máquina"**: Eliminação de inconsistências de ambiente
- **Reprodutibilidade**: Mesmo ambiente em qualquer sistema
- **Colaboração**: Pesquisadores podem usar ambiente idêntico
- **Deployment**: Transição suave de desenvolvimento para produção

**Conceitos fundamentais:**
- **Image**: Template imutável com SO + dependências
- **Container**: Instância executável de uma image
- **Dockerfile**: Receita para construir uma image
- **Volumes**: Persistência de dados fora do container

### 4.2 Prática: Containerizando o Ambiente

#### Passo 1: Criando Dockerfile otimizado

Crie `Dockerfile`:

```dockerfile
# Dockerfile para pesquisa IoT-IDS
FROM python:3.10-slim-bullseye

# Metadados
LABEL maintainer="pesquisador@universidade.edu"
LABEL description="Ambiente para pesquisa de detecção de intrusão IoT"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Criar usuário não-root para segurança
RUN groupadd -r researcher && useradd -r -g researcher researcher

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /workspace

# Copiar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed models experiments logs && \
    chown -R researcher:researcher /workspace

# Mudar para usuário não-root
USER researcher

# Configurar MLflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expor portas
EXPOSE 8888 5000

# Script de inicialização
COPY docker-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER researcher

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### Passo 2: Script de inicialização

Crie `docker-entrypoint.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Iniciando ambiente de pesquisa IoT-IDS..."

# Verificar se é primeira execução
if [ ! -f "/workspace/.initialized" ]; then
    echo "📦 Configuração inicial..."
    
    # Inicializar DVC se não existir
    if [ ! -d "/workspace/.dvc" ]; then
        echo "   Inicializando DVC..."
        cd /workspace && dvc init --no-scm
    fi
    
    # Inicializar Git se não existir
    if [ ! -d "/workspace/.git" ]; then
        echo "   Inicializando Git..."
        cd /workspace && git init
    fi
    
    # Gerar dados sintéticos se não existirem
    if [ ! -f "/workspace/data/raw/iot_traffic_v1_small.csv" ]; then
        echo "   Gerando dados sintéticos..."
        cd /workspace && python generate_dataset.py
    fi
    
    touch /workspace/.initialized
    echo "✅ Configuração inicial concluída!"
fi

# Verificar componentes
echo "🔍 Verificando componentes..."
python -c "
import numpy, pandas, sklearn, mlflow, dvc
print('✅ Dependências principais OK')
"

# Iniciar MLflow server em background se solicitado
if [ "$1" = "with-mlflow" ]; then
    echo "🔬 Iniciando MLflow server..."
    mlflow server --backend-store-uri sqlite:///mlflow.db \
                  --default-artifact-root ./mlruns \
                  --host 0.0.0.0 \
                  --port 5000 &
    
    # Aguardar MLflow iniciar
    sleep 5
    shift  # Remove 'with-mlflow' dos argumentos
fi

# Executar comando solicitado
exec "$@"
```

#### Passo 3: Docker Compose para orquestração

Crie `docker-compose.yml`:

```yaml
version: '3.8'

services:
  research-env:
    build: .
    container_name: iot-ids-research
    ports:
      - "8888:8888"  # Jupyter Lab
      - "5000:5000"  # MLflow
    volumes:
      - .:/workspace
      - /tmp/dvc-remote:/tmp/dvc-remote  # DVC remote
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - MLFLOW_TRACKING_URI=http://localhost:5000
    command: ["with-mlflow", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
    
  # Opcional: banco de dados para MLflow em produção
  mlflow-db:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

#### Passo 4: Construindo e testando

```bash
# Construir imagem
docker build -t iot-ids-research:latest .

# Executar com docker-compose
docker-compose up -d

# Verificar logs
docker-compose logs -f research-env

# Acessar container
docker exec -it iot-ids-research bash

# Testar Jupyter Lab
# http://localhost:8888

# Testar MLflow
# http://localhost:5000
```

#### Passo 5: Script de desenvolvimento

Crie `dev-setup.sh`:

```bash
#!/bin/bash
# dev-setup.sh - Setup completo do ambiente de desenvolvimento

set -e

echo "🔬 Setup do ambiente de pesquisa IoT-IDS"
echo "========================================"

# Verificar dependências
command -v docker >/dev/null 2>&1 || { echo "❌ Docker não encontrado. Instale primeiro."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose não encontrado. Instale primeiro."; exit 1; }

# Perguntar sobre rebuild
read -p "🤔 Fazer rebuild da imagem Docker? (y/N): " rebuild
if [[ $rebuild =~ ^[Yy]$ ]]; then
    echo "🔨 Fazendo rebuild..."
    docker-compose build --no-cache
fi

# Iniciar serviços
echo "🚀 Iniciando serviços..."
docker-compose up -d

# Aguardar serviços iniciarem
echo "⏳ Aguardando serviços iniciarem..."
sleep 10

# Verificar status
echo "📊 Status dos serviços:"
docker-compose ps

# Obter token do Jupyter
echo "🔑 Token do Jupyter Lab:"
docker exec iot-ids-research jupyter lab list 2>/dev/null | grep token || echo "Verifique logs: docker-compose logs research-env"

echo ""
echo "✅ Ambiente pronto!"
echo "📓 Jupyter Lab: http://localhost:8888"
echo "🔬 MLflow: http://localhost:5000"
echo "🛑 Para parar: docker-compose down"
echo "🔍 Logs: docker-compose logs -f"
```

```bash
chmod +x dev-setup.sh
./dev-setup.sh
```

---

## 🪐 Módulo 5: Jupyter Lab com Extensões

### 5.1 Teoria: Por que Jupyter Lab para Pesquisa?

Jupyter Lab é a interface de notebook de próxima geração que oferece um ambiente de desenvolvimento web flexível para computação interativa, especialmente adequado para ciência de dados e pesquisa.

**Vantagens para pesquisa científica:**
- **Narrativa científica**: Combina código, texto e visualizações
- **Experimentação interativa**: Teste rápido de hipóteses
- **Reprodutibilidade**: Notebooks como "laboratório digital"
- **Colaboração**: Fácil compartilhamento de análises
- **Extensibilidade**: Ecosystem robusto de extensões

### 5.2 Prática: Configurando Jupyter Lab

#### Passo 1: Extensões essenciais para pesquisa

Crie `jupyter_setup.py`:

```python
import subprocess
import sys

def install_extension(extension_name, pip_package=None):
    """Instala extensão do Jupyter Lab"""
    try:
        if pip_package:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pip_package], check=True)
        
        subprocess.run(['jupyter', 'labextension', 'install', extension_name], check=True)
        print(f"✅ {extension_name} instalado com sucesso")
    except subprocess.CalledProcessError:
        print(f"❌ Falha ao instalar {extension_name}")

def setup_jupyter_lab():
    """Configura Jupyter Lab com extensões para pesquisa"""
    
    print("🔧 Configurando Jupyter Lab para pesquisa científica...")
    
    # Extensões essenciais
    extensions = [
        # Produtividade
        ('jupyterlab-git', 'jupyterlab-git'),  # Integração Git
        ('jupyterlab_code_formatter', 'jupyterlab_code_formatter'),  # Formatação de código
        ('@jupyterlab/debugger', None),  # Debugger visual
        
        # Visualização
        ('jupyterlab-plotly', 'jupyterlab-plotly'),  # Plotly interativo
        ('@jupyter-widgets/jupyterlab-manager', None),  # Widgets
        ('jupyterlab-matplotlib', 'ipympl'),  # Matplotlib interativo
        
        # Data Science
        ('@lckr/jupyterlab_variableinspector', None),  # Inspetor de variáveis
        ('jupyterlab-spreadsheet', None),  # Visualizar Excel
        
        # Sistema
        ('jupyterlab-system-monitor', 'jupyterlab-system-monitor'),  # Monitor CPU/RAM
        ('jupyterlab-topbar-extension', 'jupyterlab-topbar'),  # Barra superior
        
        # Documentação
        ('@jupyterlab/latex', 'jupyterlab-latex'),  # LaTeX
        ('@jupyterlab/toc', None),  # Índice automático
        
        # ML/AI específico
        ('jupyterlab-tensorboard', 'jupyterlab_tensorboard'),  # TensorBoard
    ]
    
    for ext_name, pip_pkg in extensions:
        install_extension(ext_name, pip_pkg)
    
    # Configurações personalizadas
    config = {
        "CodeCell": {
            "cm_config": {
                "lineNumbers": True,
                "foldCode": True,
                "highlightSelectionMatches": True
            }
        },
        "NotebookApp": {
            "nbserver_extensions": {
                "jupyterlab": True,
                "jupyterlab_git": True
            }
        }
    }
    
    print("\n🎨 Aplicando configurações personalizadas...")
    
    # Configurar formatação automática
    formatter_config = """
c.JupyterLabCodeFormatter.black_config = {
    'line_length': 88,
    'target_versions': ['py38'],
    'include': '\\.pyi?$',
    'exclude': '''
    /(
        \\.eggs
        | \\.git
        | \\.mypy_cache
        | \\.tox
        | \\.venv
        | _build
        | buck-out
        | build
        | dist
    )/
    '''
}
"""
    
    with open('jupyter_lab_config.py', 'w') as f:
        f.write(formatter_config)
    
    print("✅ Configuração do Jupyter Lab concluída!")
    print("\n📝 Para usar:")
    print("   1. Jupyter Lab: jupyter lab")
    print("   2. Formatação: Ctrl+Shift+I (Black)")
    print("   3. Git: Aba lateral esquerda")
    print("   4. Debugger: Aba lateral direita")
    print("   5. Variáveis: View > Inspector")

if __name__ == "__main__":
    setup_jupyter_lab()
```

#### Passo 2: Notebook template para pesquisa

Crie `research_template.ipynb`:

```python
# Salvar como JSON do notebook
notebook_template = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {
                "tags": ["documentation"]
            },
            "source": [
                "# Experimento: [Nome do Experimento]\n",
                "\n",
                "**Pesquisador**: [Seu Nome]  \n",
                "**Data**: [Data]  \n",
                "**Objetivo**: [Descrição do objetivo]  \n",
                "**Hipótese**: [Hipótese a ser testada]  \n",
                "\n",
                "## 📋 Resumo Executivo\n",
                "- **Dataset**: [Descrição]\n",
                "- **Método**: [Algoritmo/abordagem]\n",
                "- **Métricas**: [Principais métricas]\n",
                "- **Resultado**: [A ser preenchido]\n",
                "\n",
                "## 🔗 Links Relevantes\n",
                "- MLflow Run: [URL]\n",
                "- DVC Commit: [Hash]\n",
                "- Paper/Referência: [DOI/URL]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "tags": ["setup"]
            },
            "outputs": [],
            "source": [
                "# Setup padrão para experimentos\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Machine Learning\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                "\n",
                "# Experiment tracking\n",
                "import mlflow\n",
                "import mlflow.sklearn\n",
                "\n",
                "# Configuração de plots\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette('husl')\n",
                "%matplotlib widget\n",
                "\n",
                "# Configuração pandas\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)\n",
                "\n",
                "# Seed para reprodutibilidade\n",
                "RANDOM_STATE = 42\n",
                "np.random.seed(RANDOM_STATE)\n",
                "\n",
                "print('✅ Setup concluído!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {
                "tags": ["data"]
            },
            "source": [
                "## 📊 Carregamento e Exploração de Dados"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "tags": ["data-loading"]
            },
            "outputs": [],
            "source": [
                "# Carregar dados\n",
                "# df = pd.read_csv('data/processed/dataset.csv')\n",
                "# print(f\"Shape: {df.shape}\")\n",
                "# print(f\"Colunas: {list(df.columns)}\")\n",
                "# df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {
                "tags": ["analysis"]
            },
            "source": [
                "## 🔍 Análise Exploratória"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "tags": ["eda"]
            },
            "outputs": [],
            "source": [
                "# EDA aqui\n",
                "pass"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {
                "tags": ["modeling"]
            },
            "source": [
                "## 🤖 Modelagem e Experimentos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "tags": ["experiment"]
            },
            "outputs": [],
            "source": [
                "# Configurar MLflow\n",
                "mlflow.set_experiment(\"[Nome do Experimento]\")\n",
                "\n",
                "with mlflow.start_run():\n",
                "    # Parâmetros\n",
                "    mlflow.log_param(\"dataset\", \"[nome]\")\n",
                "    mlflow.log_param(\"algorithm\", \"[algoritmo]\")\n",
                "    \n",
                "    # Modelo aqui\n",
                "    \n",
                "    # Métricas\n",
                "    # mlflow.log_metric(\"accuracy\", accuracy)\n",
                "    \n",
                "    # Artifacts\n",
                "    # mlflow.log_artifact(\"plot.png\")\n",
                "    \n",
                "    print(\"Experimento logado no MLflow!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {
                "tags": ["results"]
            },
            "source": [
                "## 📈 Resultados e Discussão\n",
                "\n",
                "### Principais Achados\n",
                "- [Resultado 1]\n",
                "- [Resultado 2]\n",
                "\n",
                "### Limitações\n",
                "- [Limitação 1]\n",
                "- [Limitação 2]\n",
                "\n",
                "### Próximos Passos\n",
                "- [Próximo passo 1]\n",
                "- [Próximo passo 2]"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

import json
with open('research_template.ipynb', 'w') as f:
    json.dump(notebook_template, f, indent=2)

print("✅ Template de pesquisa criado: research_template.ipynb")
```

#### Passo 3: Configuração avançada

Crie `jupyter_config.py`:

```python
# Configuração personalizada do Jupyter Lab
c = get_config()

# Configurações de segurança
c.ServerApp.token = ''  # Apenas para desenvolvimento local
c.ServerApp.password = ''
c.ServerApp.open_browser = False
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888

# Configurações de funcionalidade
c.ServerApp.allow_root = True
c.ServerApp.notebook_dir = '/workspace'

# Extensões habilitadas
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True,
    'jupyterlab_git': True,
    'jupyterlab_code_formatter': True
}

# Configurações do formatador de código
c.JupyterLabCodeFormatter.black_config = {
    'line_length': 88,
    'target_versions': ['py38'],
    'skip_string_normalization': True
}

# Configurações de logging
c.Application.log_level = 'INFO'

# Configurações de salvamento automático
c.FileContentsManager.checkpoints_kwargs = {'root_dir': '.ipynb_checkpoints'}
```

---

## 🧪 Exercício Integrador: Workflow Completo

### Objetivo
Implementar um workflow completo de pesquisa usando todas as ferramentas configuradas.

### Passos

#### 1. Inicializar projeto de pesquisa

```bash
# Ativar ambiente
source venv/bin/activate

# Inicializar Git e DVC (se não feito)
git init
dvc init

# Configurar experimento
mlflow ui --host 0.0.0.0 --port 5000 &
```

#### 2. Criar notebook de experimento

Copie o template e implemente:

```python
# experimento_completo.py
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

def experimento_completo():
    """Experimento completo usando todas as ferramentas"""
    
    # MLflow
    mlflow.set_experiment("Workflow-Completo-Demo")
    
    with mlflow.start_run(run_name="baseline-complete"):
        # 1. Log do ambiente
        mlflow.log_param("python_version", "3.10")
        mlflow.log_param("sklearn_version", sklearn.__version__)
        
        # 2. Carregar dados (DVC)
        df = pd.read_csv('data/processed/X_train.csv')
        y = pd.read_csv('data/processed/y_train_binary.csv').values.ravel()
        
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", df.shape[1])
        
        # 3. Modelo
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(df)
        
        # 4. Predições
        y_pred = model.predict(df)
        accuracy = accuracy_score(y, y_pred)
        
        # 5. Log métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("contamination", 0.1)
        
        # 6. Visualização
        plt.figure(figsize=(8, 6))
        scores = model.decision_function(df)
        plt.hist(scores, bins=50, alpha=0.7)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.savefig('anomaly_scores.png')
        
        # 7. Log artifacts
        mlflow.log_artifact('anomaly_scores.png')
        mlflow.sklearn.log_model(model, "isolation_forest")
        
        # 8. Salvar modelo (DVC)
        joblib.dump(model, 'models/baseline_model.pkl')
        
        print(f"✅ Experimento concluído! Accuracy: {accuracy:.3f}")
        
        return accuracy

if __name__ == "__main__":
    acc = experimento_completo()
    
    # Versionamento com DVC
    !dvc add models/baseline_model.pkl
    !git add models/baseline_model.pkl.dvc
    !git commit -m "Add baseline model with accuracy {:.3f}".format(acc)
```

#### 3. Verificar integração

```bash
# Verificar tracking MLflow
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Verificar DVC
dvc status

# Verificar Git
git log --oneline

# Testar Docker
docker build -t complete-workflow .
docker run -p 8888:8888 -p 5000:5000 complete-workflow
```

---

## 🎯 Critérios de Sucesso

Ao final do laboratório, você deve ter:

### ✅ Checklist de Validação

- [ ] **Virtual Environment funcional** com todas as dependências
- [ ] **MLflow rodando** e logando experimentos corretamente
- [ ] **DVC configurado** e versionando dados
- [ ] **Docker image** construída e funcional
- [ ] **Jupyter Lab** com extensões instaladas
- [ ] **Pipeline completo** funcionando end-to-end
- [ ] **Documentação** clara de todos os passos

### 🔬 Teste Final

Execute o script de validação:

```python
# validacao_final.py
import subprocess
import sys
import os

def validar_ambiente():
    """Valida se todo o ambiente está configurado corretamente"""
    
    testes = []
    
    # 1. Python e dependências
    try:
        import numpy, pandas, sklearn, mlflow, dvc
        testes.append(("✅", "Dependências Python"))
    except ImportError as e:
        testes.append(("❌", f"Dependências Python: {e}"))
    
    # 2. MLflow
    try:
        import requests
        resp = requests.get("http://localhost:5000", timeout=5)
        if resp.status_code == 200:
            testes.append(("✅", "MLflow server"))
        else:
            testes.append(("❌", "MLflow server não responde"))
    except:
        testes.append(("❌", "MLflow server não acessível"))
    
    # 3. DVC
    if os.path.exists('.dvc'):
        testes.append(("✅", "DVC inicializado"))
    else:
        testes.append(("❌", "DVC não inicializado"))
    
    # 4. Git
    if os.path.exists('.git'):
        testes.append(("✅", "Git inicializado"))
    else:
        testes.append(("❌", "Git não inicializado"))
    
    # 5. Docker
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            testes.append(("✅", "Docker disponível"))
        else:
            testes.append(("❌", "Docker não disponível"))
    except:
        testes.append(("❌", "Docker não encontrado"))
    
    # 6. Estrutura de diretórios
    dirs_necessarios = ['data/raw', 'data/processed', 'models', 'experiments']
    dirs_ok = all(os.path.exists(d) for d in dirs_necessarios)
    if dirs_ok:
        testes.append(("✅", "Estrutura de diretórios"))
    else:
        testes.append(("❌", "Estrutura de diretórios incompleta"))
    
    # Relatório
    print("🔍 Validação do Ambiente de Pesquisa")
    print("=" * 40)
    for status, descricao in testes:
        print(f"{status} {descricao}")
    
    sucessos = sum(1 for status, _ in testes if status == "✅")
    total = len(testes)
    
    print(f"\n📊 Score: {sucessos}/{total} ({sucessos/total*100:.1f}%)")
    
    if sucessos == total:
        print("\n🎉 Parabéns! Ambiente completamente configurado!")
        print("🚀 Você está pronto para começar a pesquisa!")
        return True
    else:
        print(f"\n⚠️  {total-sucessos} itens precisam de atenção.")
        print("📝 Revise a configuração dos itens marcados com ❌")
        return False

if __name__ == "__main__":
    validar_ambiente()
```

---

## 🚀 Próximos Passos

Com o ambiente configurado, você está pronto para:

1. **Implementar os experimentos da Fase 1** (Experimentos 1.1 e 1.2)
2. **Começar a coleta e processamento** do dataset CICIoT2023
3. **Desenvolver os algoritmos baseline** de detecção de anomalias
4. **Configurar pipelines de CI/CD** para automação

---

## 📚 Materiais para Aprofundamento

### 📖 Documentação Oficial

#### Python Virtual Environments
- Python Virtual Environments Documentation - Guia oficial do Python sobre criação e uso de ambientes virtuais
- Python venv Module Reference - Documentação completa do módulo venv
- The Hitchhiker's Guide to Python: Pipenv & Virtual Environments - Guia comunitário sobre melhores práticas

#### MLflow
- MLflow Official Documentation - Documentação completa sobre tracking, projetos e modelos
- MLflow Tracking Quickstart - Tutorial rápido para começar com tracking
- MLflow Tracking APIs - Referência completa das APIs de tracking

#### DVC (Data Version Control)
- DVC Get Started Guide - Tutorial oficial para iniciantes
- DVC User Guide - Documentação completa do usuário
- DVC Tutorial: Data and Model Versioning - Tutorial prático com exemplos

#### Docker
- Docker for AI/ML - Guia oficial do Docker para aplicações de ML
- Getting Started with Docker for AI/ML - Tutorial para iniciantes
- Docker and Reproducibility - Uso do Docker para pesquisa reproduzível

#### Jupyter Lab
- Getting Started with JupyterLab - Tutorial oficial de instalação e uso
- JupyterLab Extensions List - Lista abrangente de extensões disponíveis
- JupyterLab Extension Examples - Exemplos oficiais de desenvolvimento de extensões

### 📝 Tutoriais Práticos

#### MLflow para Pesquisa
- Machine Learning Experiment Tracking Using MLflow - Analytics Vidhya
- How to Track ML Experiments and Manage Models with MLflow - Artiba
- Elevate Your Machine Learning Workflow: MLflow for Experiment Tracking - Medium
- Learn to Streamline Your Machine Learning Workflow with MLFlow - DataCamp

#### DVC na Prática
- The Complete Guide to Data Version Control With DVC - DataCamp
- Data Version Control (DVC) Tutorial for Machine Learning - RidgeRun
- Understanding DVC: A practical guide to Data Version Control - Medium
- Data Version Control With Python and DVC - Real Python

#### Docker para ML/AI
- Best Practices When Working With Docker for Machine Learning - Neptune.ai
- Step-by-Step Guide to Containerizing ML Models with Docker - DEV Community
- Deploying Machine Learning Models in Docker Containers - Tutoriais práticos
- Top 12 Docker Container Images for Machine Learning and AI - DataCamp

#### Jupyter Lab Extensions
- Jupyter Lab extensions for Data Scientist - Medium por Alexander Osipenko
- 10 Jupyter Lab Extensions to Boost Your Productivity - Towards Data Science
- Awesome JupyterLab Extensions - Towards Data Science
- Top 9 JupyterLab extensions and how to pick yours - Tabnine
- Awesome JupyterLab - Lista curada de extensões e recursos

### 🎓 Cursos e Especializações

#### Ambientes Python
- Python Virtual Environment Tutorial - GeeksforGeeks
- Setting Your Python Working Environment, the Right Way - Python GUIs
- Python venv: Complete Guide - Python Land
- Complete Guide to Python Virtual Environments - Dataquest

#### MLOps e Ferramentas
- MLFlow Mastery: Complete Guide to Experiment Tracking - KDnuggets
- How to Use DVC for Machine Learning Model Management - Medium
- Getting Started with Data Version Control (DVC) - Analytics Vidhya

### 🔗 Recursos Avançados

#### Repositórios e Projetos
- **GitHub**: Procure por "mlflow examples", "dvc tutorial", "docker ml" para projetos práticos
- **Kaggle**: Notebooks que usam essas ferramentas em competições reais
- **Papers with Code**: Implementações que seguem essas práticas

#### Comunidades
- **MLflow Community**: Fóruns oficiais e discussões
- **DVC Discord**: Comunidade ativa para dúvidas
- **Reddit r/MachineLearning**: Discussões sobre ferramentas e práticas
- **Stack Overflow**: Para problemas técnicos específicos

#### Blogs e Newsletters
- **Towards Data Science**: Artigos regulares sobre MLOps
- **Neptune.ai Blog**: Focado em experiment tracking
- **The Batch (DeepLearning.AI)**: Newsletter semanal sobre ML
- **MLOps Community**: Práticas e ferramentas do setor

### 📊 Datasets para Prática

#### IoT Security Datasets
- **CICIoT2023**: Dataset principal da sua pesquisa
- **Bot-IoT**: Dataset alternativo para validação cruzada
- **ToN_IoT**: Dataset de telemetria para IoT
- **UNSW-NB15**: Dataset de rede para comparação

#### Exemplos de Uso
Use esses datasets para praticar o workflow completo:
1. Download e versionamento com DVC
2. EDA e preprocessing com Jupyter Lab
3. Experimentos com MLflow
4. Containerização com Docker

---

## 🎯 Resumo do Laboratório

Você completou com sucesso a configuração de um ambiente de pesquisa profissional que inclui:

### ✅ Ferramentas Configuradas
1. **Python Virtual Environment** - Isolamento de dependências
2. **MLflow** - Tracking de experimentos e modelos
3. **DVC** - Versionamento de dados e pipelines
4. **Docker** - Containerização para reprodutibilidade
5. **Jupyter Lab** - Interface de desenvolvimento com extensões

### 🔬 Competências Desenvolvidas
- Configuração de ambientes reproduzíveis
- Rastreamento sistemático de experimentos
- Versionamento de dados grandes
- Containerização de ambientes de pesquisa
- Uso avançado de notebooks científicos

### 🚀 Próximas Etapas
Com este ambiente, você está preparado para:
- Implementar os experimentos da Fase 1 do cronograma
- Trabalhar com o dataset CICIoT2023
- Desenvolver algoritmos de detecção de anomalias
- Manter reprodutibilidade e documentação científica adequada

**Lembre-se**: Este ambiente é a base fundamental para toda sua pesquisa. Mantenha-o bem documentado e atualizado conforme necessário.