# Aplicação ao Projeto IoT-IDS

> **Contexto:** Pesquisa de Mestrado  
> **Dataset:** CICIoT2023  
> **Objetivo:** Detecção de Intrusão em Redes IoT

---

## 🎯 Contexto do Projeto

Você está desenvolvendo um **Sistema de Detecção de Intrusão (IDS)** para ambientes IoT usando o dataset **CICIoT2023**. Este documento conecta todos os conceitos teóricos à sua aplicação prática!

**Desafio central:** Avaliar corretamente o desempenho de algoritmos de machine learning em cenários com **classes desbalanceadas** (ataques são raros).

---

## 📊 Características do Problema

### Dataset CICIoT2023

**Natureza:**
- Tráfego de rede IoT real
- 33+ tipos de ataques
- Extremamente **desbalanceado** (tráfego normal >> ataques)

**Exemplo típico:**
```
Tráfego Normal: 95%
Ataques:         5%
```

### Por que [[Acurácia]] Tradicional Falha?

**Cenário realista:**
```
10,000 conexões de teste:
- 9,500 normais
- 500 ataques

Modelo "preguiçoso" prevê tudo como "normal":
```

Acurácia = 9500/10000 = 95%  ← Parece ótimo! ✅  
**MAS:** 0 ataques detectados! ❌ IDS completamente inútil!

**Solução:** [[Acurácia_Balanceada]]! 🎯

---

## 🔬 Aplicando o [[The_Balanced_Accuracy_and_Its_Posterior_Distribution|Artigo Brodersen]]

### Framework Completo

**Passo 1:** Treinar modelo (ex: Random Forest, SVM, MoT)

**Passo 2:** Avaliar com cross-validation

**Passo 3:** **Não** calcular apenas média! Use [[Distribuição_Beta]]:

```python
from scipy import stats
import numpy as np

# Agregar resultados de todos os folds
TP_total = 450  # Ataques detectados corretamente
FN_total = 50   # Ataques perdidos
TN_total = 9300 # Tráfego normal correto
FP_total = 200  # Falsos alarmes

# Posterior para cada classe
pos_posterior = stats.beta(TP_total + 1, FN_total + 1)
neg_posterior = stats.beta(TN_total + 1, FP_total + 1)

# Estatísticas individuais
print("CLASSE ATAQUE:")
print(f"  Sensibilidade: {pos_posterior.mean():.3f}")
print(f"  IC 95%: {pos_posterior.interval(0.95)}")

print("\nCLASSE NORMAL:")
print(f"  Especificidade: {neg_posterior.mean():.3f}")
print(f"  IC 95%: {neg_posterior.interval(0.95)}")

# Balanced Accuracy (via amostragem)
pos_samples = pos_posterior.rvs(100000)
neg_samples = neg_posterior.rvs(100000)
ba_samples = 0.5 * (pos_samples + neg_samples)

print("\nBALANCED ACCURACY:")
print(f"  Média: {np.mean(ba_samples):.3f}")
print(f"  IC 95%: {np.percentile(ba_samples, [2.5, 97.5])}")
```

### Exemplo de Saída

```
CLASSE ATAQUE:
  Sensibilidade: 0.900
  IC 95%: (0.877, 0.920)

CLASSE NORMAL:
  Especificidade: 0.979
  IC 95%: (0.976, 0.982)

BALANCED ACCURACY:
  Média: 0.939
  IC 95%: [0.928, 0.951]
```

**Interpretação:**
- ✅ O modelo detecta 90% dos ataques
- ✅ Gera poucos falsos alarmes (97.9% especificidade)
- ✅ Performance balanceada: 93.9%
- ✅ Com 95% de credibilidade, BA está entre 92.8% e 95.1%

---

## 🧮 Integrando ao Pipeline Experimental

### Estrutura do Seu Código

Baseado no seu `algorithm_comparison.py`:

```python
# experiments/balanced_accuracy_posterior.py

from scipy import stats
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class BayesianMetrics:
    """
    Calcula métricas com distribuições posteriores
    conforme Brodersen et al. (2010).
    """
    
    def __init__(self, y_true, y_pred):
        """
        Args:
            y_true: labels verdadeiros
            y_pred: labels preditos
        """
        self.cm = confusion_matrix(y_true, y_pred)
        
        # Para binário
        if self.cm.shape == (2, 2):
            self.TN, self.FP, self.FN, self.TP = self.cm.ravel()
        
    def accuracy_posterior(self):
        """Distribuição posterior da acurácia."""
        C = self.TP + self.TN
        I = self.FP + self.FN
        return stats.beta(C + 1, I + 1)
    
    def balanced_accuracy_posterior(self, n_samples=100000):
        """
        Distribuição posterior da balanced accuracy
        via convolução (Eq. 7 do artigo).
        """
        # Posteriors por classe
        pos_post = stats.beta(self.TP + 1, self.FN + 1)
        neg_post = stats.beta(self.TN + 1, self.FP + 1)
        
        # Amostragem Monte Carlo da convolução
        pos_samples = pos_post.rvs(n_samples)
        neg_samples = neg_post.rvs(n_samples)
        ba_samples = 0.5 * (pos_samples + neg_samples)
        
        return ba_samples
    
    def report(self, confidence=0.95):
        """Relatório completo estilo artigo."""
        acc_post = self.accuracy_posterior()
        ba_samples = self.balanced_accuracy_posterior()
        
        # Acurácia
        acc_mean = acc_post.mean()
        acc_median = acc_post.median()
        acc_mode = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        acc_ci = acc_post.interval(confidence)
        
        # Balanced Accuracy
        ba_mean = np.mean(ba_samples)
        ba_median = np.median(ba_samples)
        ba_ci = np.percentile(ba_samples, [(1-confidence)*50, (1+confidence)*50])
        
        return {
            'accuracy': {
                'mean': acc_mean,
                'median': acc_median,
                'mode': acc_mode,
                'ci': acc_ci,
                'distribution': acc_post
            },
            'balanced_accuracy': {
                'mean': ba_mean,
                'median': ba_median,
                'ci': ba_ci,
                'samples': ba_samples
            },
            'confusion_matrix': self.cm
        }

# USO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Treinar e prever
clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, X, y, cv=10)

# Métricas Bayesianas
metrics = BayesianMetrics(y, y_pred)
report = metrics.report()

print(f"Accuracy: {report['accuracy']['mean']:.3f} "
      f"{report['accuracy']['ci']}")
print(f"Balanced Accuracy: {report['balanced_accuracy']['mean']:.3f} "
      f"{report['balanced_accuracy']['ci']}")
```

---

## 📈 Comparação de Algoritmos

### Cenário: MoT vs Random Forest vs XGBoost

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Resultados de cada algoritmo (exemplo)
algorithms = {
    'MoT': {'TP': 440, 'FN': 60, 'TN': 9200, 'FP': 300},
    'RF':  {'TP': 450, 'FN': 50, 'TN': 9300, 'FP': 200},
    'XGB': {'TP': 460, 'FN': 40, 'TN': 9250, 'FP': 250}
}

# Calcular BA posterior para cada um
ba_distributions = {}

for name, cm in algorithms.items():
    pos_post = stats.beta(cm['TP'] + 1, cm['FN'] + 1)
    neg_post = stats.beta(cm['TN'] + 1, cm['FP'] + 1)
    
    pos_samples = pos_post.rvs(100000)
    neg_samples = neg_post.rvs(100000)
    ba_samples = 0.5 * (pos_samples + neg_samples)
    
    ba_distributions[name] = ba_samples

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribuições
for name, samples in ba_distributions.items():
    axes[0].hist(samples, bins=50, alpha=0.5, label=name, density=True)

axes[0].set_xlabel('Balanced Accuracy')
axes[0].set_ylabel('Densidade')
axes[0].set_title('Distribuições Posteriores da BA')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Comparação com ICs
names = list(ba_distributions.keys())
means = [np.mean(ba_distributions[n]) for n in names]
cis = [np.percentile(ba_distributions[n], [2.5, 97.5]) for n in names]

y_pos = np.arange(len(names))
axes[1].barh(y_pos, means, xerr=[[m-c[0] for m,c in zip(means, cis)],
                                   [c[1]-m for m,c in zip(means, cis)]],
             capsize=5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(names)
axes[1].set_xlabel('Balanced Accuracy')
axes[1].set_title('Comparação com ICs 95%')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('algorithm_comparison_bayesian.png', dpi=300)
plt.show()

# Comparação probabilística
print("\nCOMPARAÇÕES PROBABILÍSTICAS:")
for alg1 in names:
    for alg2 in names:
        if alg1 < alg2:  # Evitar duplicatas
            prob = np.mean(ba_distributions[alg1] > ba_distributions[alg2])
            print(f"P({alg1} > {alg2}) = {prob:.3f}")
            
            if prob > 0.95:
                print(f"  → {alg1} é SIGNIFICATIVAMENTE melhor!")
            elif prob < 0.05:
                print(f"  → {alg2} é SIGNIFICATIVAMENTE melhor!")
            else:
                print(f"  → Não há diferença significativa")
```

---

## 🎯 Integrando com DVC

### Salvar Métricas Bayesianas

```python
# dvc_bayesian_metrics.py

import json
from scipy import stats
import numpy as np

def compute_and_save_bayesian_metrics(y_true, y_pred, output_path):
    """Computa métricas Bayesianas e salva para DVC."""
    
    metrics_obj = BayesianMetrics(y_true, y_pred)
    report = metrics_obj.report()
    
    # Converter para formato serializável
    metrics_json = {
        'accuracy': {
            'mean': float(report['accuracy']['mean']),
            'median': float(report['accuracy']['median']),
            'mode': float(report['accuracy']['mode']),
            'ci_lower': float(report['accuracy']['ci'][0]),
            'ci_upper': float(report['accuracy']['ci'][1])
        },
        'balanced_accuracy': {
            'mean': float(report['balanced_accuracy']['mean']),
            'median': float(report['balanced_accuracy']['median']),
            'ci_lower': float(report['balanced_accuracy']['ci'][0]),
            'ci_upper': float(report['balanced_accuracy']['ci'][1])
        },
        'confusion_matrix': report['confusion_matrix'].tolist()
    }
    
    # Salvar
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Métricas Bayesianas salvas em: {output_path}")
    return metrics_json
```

### Atualizar dvc.yaml

```yaml
stages:
  evaluate_bayesian:
    cmd: >
      python experiments/evaluate_with_bayesian_metrics.py
      --model-path models/random_forest.pkl
      --test-data data/processed/test.parquet
      --output metrics/bayesian_metrics.json
    deps:
      - models/random_forest.pkl
      - data/processed/test.parquet
      - experiments/evaluate_with_bayesian_metrics.py
    metrics:
      - metrics/bayesian_metrics.json:
          cache: false
```

---

## 📝 Reportando Resultados Acadêmicos

### Para Papers/Dissertação

**Formato recomendado (baseado no artigo):**

```
"O algoritmo MoT obteve balanced accuracy média de 0.939
(intervalo de credibilidade 95%: [0.928, 0.951]), superando
significativamente o Random Forest (BA = 0.912, IC 95%: 
[0.899, 0.925], P(MoT > RF) = 0.997)."
```

### Tabela de Resultados

```markdown
| Algoritmo | BA Média | IC 95% | Sensibilidade | Especificidade |
|-----------|----------|--------|---------------|----------------|
| MoT       | 0.939    | [0.928, 0.951] | 0.900 | 0.979 |
| RF        | 0.912    | [0.899, 0.925] | 0.875 | 0.949 |
| XGBoost   | 0.945    | [0.934, 0.956] | 0.920 | 0.970 |
```

### Visualização para Paper

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo para publicação
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Criar figura
fig, ax = plt.subplots()

# Para cada algoritmo
for name, samples in ba_distributions.items():
    # Densidade posterior
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples)
    x_range = np.linspace(0.85, 0.98, 500)
    density = kde(x_range)
    
    ax.plot(x_range, density, label=name, linewidth=2)
    
    # Marcar média
    mean = np.mean(samples)
    ax.axvline(mean, linestyle='--', alpha=0.5)

ax.set_xlabel('Balanced Accuracy', fontsize=14)
ax.set_ylabel('Posterior Density', fontsize=14)
ax.set_title('Posterior Distributions of Balanced Accuracy\n'
             'for IDS Algorithms on CICIoT2023', fontsize=16)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figure_posterior_ba_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_posterior_ba_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔍 Diagnóstico de Problemas

### Checklist de Avaliação

```python
def diagnostic_report(y_true, y_pred):
    """Relatório de diagnóstico completo."""
    
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    
    # Básico
    cm = confusion_matrix(y_true, y_pred)
    print("MATRIZ DE CONFUSÃO:")
    print(pd.DataFrame(cm, 
                       index=['Real: Normal', 'Real: Ataque'],
                       columns=['Pred: Normal', 'Pred: Ataque']))
    
    # Sklearn report
    print("\n" + "="*50)
    print("RELATÓRIO SCIKIT-LEARN:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Ataque']))
    
    # Bayesiano
    metrics = BayesianMetrics(y_true, y_pred)
    report = metrics.report()
    
    print("\n" + "="*50)
    print("MÉTRICAS BAYESIANAS:")
    print(f"Accuracy: {report['accuracy']['mean']:.4f} "
          f"{report['accuracy']['ci']}")
    print(f"Balanced Accuracy: {report['balanced_accuracy']['mean']:.4f} "
          f"{report['balanced_accuracy']['ci']}")
    
    # Alertas
    print("\n" + "="*50)
    print("DIAGNÓSTICO:")
    
    TN, FP, FN, TP = cm.ravel()
    
    # Taxa de desbalanceamento
    ratio = max(TN+FP, TP+FN) / min(TN+FP, TP+FN)
    if ratio > 10:
        print(f"⚠️  Dataset muito desbalanceado (ratio: {ratio:.1f}:1)")
        print("   → Use Balanced Accuracy como métrica principal!")
    
    # Viés do modelo
    pred_ratio = (TP+FP) / (TN+FN) if (TN+FN) > 0 else float('inf')
    true_ratio = (TP+FN) / (TN+FP) if (TN+FP) > 0 else float('inf')
    
    if abs(pred_ratio - true_ratio) / true_ratio > 0.2:
        print(f"⚠️  Modelo pode estar enviesado")
        print(f"   Proporção real: {true_ratio:.2f}")
        print(f"   Proporção predita: {pred_ratio:.2f}")
    
    # Accuracy vs BA
    acc = (TP + TN) / (TP + TN + FP + FN)
    ba = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
    
    if abs(acc - ba) > 0.05:
        print(f"⚠️  Grande diferença entre Accuracy ({acc:.3f}) "
              f"e BA ({ba:.3f})")
        print("   → BA é mais confiável para este dataset!")
    
    print("\n✅ Diagnóstico completo!")
```

---

## 📚 Citações para Dissertação

### Justificando a Escolha de Métricas

```latex
\section{Métricas de Avaliação}

Devido ao desbalanceamento inerente ao dataset CICIoT2023,
onde ataques representam menos de 5\% do tráfego total,
a acurácia tradicional pode fornecer estimativas enganosas
do desempenho do modelo \cite{brodersen2010balanced}.

Seguindo \citet{brodersen2010balanced}, utilizamos a 
\textit{balanced accuracy}, definida como a média aritmética
das taxas de acerto em cada classe:

\begin{equation}
BA = \frac{1}{2}\left(\frac{TP}{P} + \frac{TN}{N}\right)
\end{equation}

Além disso, adotamos a abordagem Bayesiana proposta pelos
autores para quantificar a incerteza sobre as métricas,
modelando a balanced accuracy através da convolução de
distribuições Beta:

\begin{equation}
\theta \mid \text{dados} \sim \text{Beta}(\alpha + k, \beta + n - k)
\end{equation}

Esta abordagem permite reportar intervalos de credibilidade
que respeitam os limites naturais [0,1] e fornecem
interpretação probabilística direta.
```

---

## 🔗 Conceitos Relacionados

### Teóricos
- [[Acurácia]] - Métrica básica
- [[Acurácia_Balanceada]] - Métrica principal
- [[Distribuição_Beta]] - Modelagem probabilística
- [[Inferência_Bayesiana]] - Framework usado
- [[Intervalos_de_Confiança]] - Quantificação de incerteza
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]] - Artigo base

### Práticos
- [[Exercícios_Práticos]] - Praticar implementação
- [[Referências_Bibliográficas]] - Fontes para citar

---

## 📌 Resumo para Ação Imediata

**1. Substituir em seu código:**
```python
# ANTES
accuracy = accuracy_score(y_true, y_pred)

# DEPOIS
metrics = BayesianMetrics(y_true, y_pred)
report = metrics.report()
ba_mean = report['balanced_accuracy']['mean']
ba_ci = report['balanced_accuracy']['ci']
```

**2. Reportar resultados:**
```
BA = 0.939 [IC 95%: 0.928, 0.951]
```

**3. Comparar algoritmos:**
```python
prob_A_better = np.mean(ba_samples_A > ba_samples_B)
```

**4. Visualizar para papers:**
```python
plot_posterior_distributions(algorithms, 'balanced_accuracy')
```

---

**Tags:** #iot #ids #ciciot2023 #application #pratical #research #mestrado

**Voltar para:** [[INDEX]]  
**Teoria:** [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]  
**Implementar:** [[Exercícios_Práticos]]

