# 📊 Guia Completo de Preprocessing - Separação Treino/Teste Correta

## 🎯 **Visão Geral**

Este preprocessing implementa a **abordagem CORRETA** para evitar data leakage:
1. ✅ **Separa** treino/teste **ANTES** da normalização
2. ✅ **Normaliza** usando parâmetros aprendidos **APENAS** do treino
3. ✅ **Aplica** esses parâmetros ao teste (simula produção)

## 📁 **Arquivos Gerados**

O preprocessing gera os seguintes arquivos em `data/processed/`:

### 📋 **1. Dados CSV (Legíveis)**
```
train_normalized.csv    # Dados de treino normalizados + labels
test_normalized.csv     # Dados de teste normalizados + labels
```

### 🔧 **2. Objetos Python**
```
scaler.pkl             # StandardScaler treinado (CRUCIAL!)
```

### ⚡ **3. Arrays NumPy (ML Otimizado)**
```
X_train.npy            # Features de treino (matriz NumPy)
X_test.npy             # Features de teste (matriz NumPy)
y_train.npy            # Labels de treino (array NumPy)
y_test.npy             # Labels de teste (array NumPy)
```

### 📊 **4. Metadados**
```
preprocessing_metadata.json  # Informações sobre o processo
```

---

## 🚀 **Como Usar nos Modelos de ML**

### **Método 1: Carregar Arrays NumPy (Recomendado para ML)**
```python
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar dados pré-processados
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

# Treinar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### **Método 2: Carregar CSV (Para análise)**
```python
import pandas as pd

# Carregar dados
train_df = pd.read_csv('data/processed/train_normalized.csv')
test_df = pd.read_csv('data/processed/test_normalized.csv')

# Separar features e labels
X_train = train_df.drop('Label', axis=1)
y_train = train_df['Label']
X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']

# Usar normalmente...
```

### **Método 3: Usar Scaler em Novos Dados**
```python
import joblib
import pandas as pd

# Carregar scaler treinado
scaler = joblib.load('data/processed/scaler.pkl')

# Aplicar em novos dados (produção)
new_data = pd.read_csv('new_iot_data.csv')
new_data_clean = handle_missing_values(new_data)

# Remover coluna Label se existir
features = new_data_clean.drop('Label', axis=1, errors='ignore')

# Normalizar usando MESMO scaler do treino
new_data_normalized = scaler.transform(features)

# Fazer predições
predictions = model.predict(new_data_normalized)
```

---

## ⚙️ **Configuração (80/20 Configurável)**

Edite `configs/preprocessing.yaml`:

```yaml
input_file: data/processed/sampled.csv
output_dir: data/processed
test_size: 0.2        # 20% para teste (configurável)
random_state: 42      # Para reprodutibilidade
stratify: true        # Manter proporção de classes
```

### **Exemplos de Configuração:**
```yaml
# 70/30 split
test_size: 0.3

# 90/10 split  
test_size: 0.1

# Sem stratify (não recomendado para datasets desbalanceados)
stratify: false
```

---

## 🔍 **Verificação de Data Leakage**

### **✅ Abordagem CORRETA (implementada):**
```python
# 1. Separar primeiro
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Normalizar depois (scaler vê só treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform
X_test_scaled = scaler.transform(X_test)        # transform apenas

# ✅ Resultado: sem data leakage
```

### **❌ Abordagem INCORRETA (evitada):**
```python
# 1. Normalizar primeiro (❌ ERRADO)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # Vê TUDO!

# 2. Separar depois
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y)

# ❌ Resultado: data leakage!
```

---

## 📈 **Verificação das Estatísticas**

### **Esperado após normalização:**
```
Treino - Média: ~0.000000, Std: ~1.000000  ✅
Teste  - Média: ~0.023456, Std: ~0.987654  ✅ (NORMAL!)
```

### **⚠️ IMPORTANTE:**
- Treino tem média ≈ 0 e std ≈ 1 (esperado)
- Teste pode ter valores ligeiramente diferentes (NORMAL!)
- Isso simula o cenário real de produção

---

## 🛠️ **Pipeline Completo de ML**

```python
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import json

# 1. Carregar dados pré-processados
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# 2. Carregar metadados
with open('data/processed/preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Features: {len(metadata['feature_columns'])}")
print(f"Config usado: {metadata['config']}")

# 3. Treinar múltiplos modelos
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n🔄 Treinando {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"Acurácia: {results[name]['accuracy']:.4f}")

# 4. Salvar melhor modelo
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Melhor modelo: {best_model[0]}")

# Retreinar e salvar
final_model = models[best_model[0]]
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'data/models/best_model.pkl')
```

---

## 🔄 **Para Novos Dados em Produção**

```python
# Função para processar novos dados
def predict_new_data(new_csv_path):
    # 1. Carregar scaler e modelo treinados
    scaler = joblib.load('data/processed/scaler.pkl')
    model = joblib.load('data/models/best_model.pkl')
    
    # 2. Carregar e limpar novos dados
    new_data = pd.read_csv(new_csv_path)
    new_data_clean = handle_missing_values(new_data)
    
    # 3. Remover coluna label se existir
    features = new_data_clean.drop('Label', axis=1, errors='ignore')
    
    # 4. Normalizar usando scaler treinado
    features_normalized = scaler.transform(features)
    
    # 5. Fazer predições
    predictions = model.predict(features_normalized)
    probabilities = model.predict_proba(features_normalized)
    
    return predictions, probabilities

# Usar função
preds, probs = predict_new_data('novos_dados.csv')
```

---

## 📊 **Resumo dos Benefícios**

1. **✅ Sem Data Leakage**: Scaler nunca vê dados de teste
2. **✅ Reprodutível**: Random state fixo
3. **✅ Configurável**: Split 80/20 ajustável
4. **✅ Completo**: CSV + NumPy + Pickle + Metadados
5. **✅ Produção**: Scaler reutilizável para novos dados
6. **✅ Eficiente**: Arrays NumPy para ML
7. **✅ Auditável**: Metadados completos
