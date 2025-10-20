#!/usr/bin/env python3
"""
Implementação da abordagem Bayesiana de Brodersen et al. (2010)
para Balanced Accuracy com distribuição Beta posterior

Referência:
Brodersen, K.H., et al. (2010). "The balanced accuracy and its posterior distribution".
2010 20th International Conference on Pattern Recognition (ICPR), pp. 3121-3124. IEEE.
"""

from scipy import stats
import numpy as np
from sklearn.metrics import confusion_matrix


class BayesianAccuracyEvaluator:
    """
    Avaliador Bayesiano conforme Brodersen et al. (2010).
    
    Modela a distribuição posterior da Balanced Accuracy usando
    distribuições Beta conjugadas e convolução.
    """
    
    def __init__(self, y_true, y_pred):
        """
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Calcular confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Garantir que temos matriz 2x2 (binária)
        if cm.shape != (2, 2):
            raise ValueError(f"Esperado confusion matrix 2x2, recebido {cm.shape}")
        
        self.TN, self.FP, self.FN, self.TP = cm.ravel()
        
        # Total corretos e incorretos
        self.C = self.TP + self.TN  # Corretos
        self.I = self.FP + self.FN  # Incorretos
        
    def accuracy_posterior(self, prior_alpha=1, prior_beta=1):
        """
        Distribuição posterior da acurácia usando Beta.
        
        Prior não-informativo: Beta(1, 1) = Uniforme[0,1]
        Posterior: Beta(C+1, I+1)
        
        Args:
            prior_alpha: Parâmetro α do prior Beta (default: 1)
            prior_beta: Parâmetro β do prior Beta (default: 1)
        
        Returns:
            scipy.stats.beta: Distribuição posterior
        """
        return stats.beta(self.C + prior_alpha, self.I + prior_beta)
    
    def balanced_accuracy_posterior(self, n_samples=100000, prior_alpha=1, prior_beta=1):
        """
        Distribuição posterior da Balanced Accuracy via convolução.
        
        Conforme Equação (7) do artigo Brodersen et al. (2010):
        p_BA(x) = ∫ p_A(2(x-z); TP+1, FN+1) × p_A(2z; TN+1, FP+1) dz
        
        Implementação via Monte Carlo sampling da convolução de duas distribuições Beta.
        
        Args:
            n_samples: Número de amostras para aproximação Monte Carlo
            prior_alpha: Parâmetro α do prior Beta
            prior_beta: Parâmetro β do prior Beta
            
        Returns:
            np.ndarray: Amostras da distribuição posterior de BA
        """
        # Posteriors para cada classe (sensibilidade e especificidade)
        sensitivity_post = stats.beta(self.TP + prior_alpha, self.FN + prior_beta)
        specificity_post = stats.beta(self.TN + prior_alpha, self.FP + prior_beta)
        
        # Amostrar das posteriors
        sensitivity_samples = sensitivity_post.rvs(n_samples)
        specificity_samples = specificity_post.rvs(n_samples)
        
        # Balanced Accuracy = ½(Sensitivity + Specificity)
        ba_samples = 0.5 * (sensitivity_samples + specificity_samples)
        
        return ba_samples
    
    def compute_metrics(self, confidence=0.95, n_samples=100000):
        """
        Computa todas as métricas Bayesianas.
        
        Args:
            confidence: Nível de confiança para intervalos (default: 0.95)
            n_samples: Amostras para BA posterior
            
        Returns:
            dict: Métricas completas com posteriors
        """
        # Posterior da acurácia
        acc_post = self.accuracy_posterior()
        
        # Posterior da balanced accuracy (via sampling)
        ba_samples = self.balanced_accuracy_posterior(n_samples=n_samples)
        
        # Calcular estatísticas
        alpha_level = (1 - confidence) / 2
        
        results = {
            'accuracy': {
                'mean': float(acc_post.mean()),
                'median': float(acc_post.median()),
                'mode': float(self.C / (self.C + self.I)) if (self.C + self.I) > 0 else 0.0,
                'std': float(acc_post.std()),
                'ci': tuple(float(x) for x in acc_post.interval(confidence)),
                'distribution': 'Beta',
                'params': {'alpha': int(self.C + 1), 'beta': int(self.I + 1)}
            },
            'balanced_accuracy': {
                'mean': float(np.mean(ba_samples)),
                'median': float(np.median(ba_samples)),
                'std': float(np.std(ba_samples)),
                'ci': tuple(float(x) for x in np.percentile(ba_samples, [alpha_level*100, (1-alpha_level)*100])),
                'samples': ba_samples,  # Para análises posteriores
                'distribution': 'Convolution of 2 Beta distributions'
            },
            'sensitivity': {
                'mean': float(self.TP / (self.TP + self.FN)) if (self.TP + self.FN) > 0 else 0.0,
                'posterior_params': {'alpha': int(self.TP + 1), 'beta': int(self.FN + 1)}
            },
            'specificity': {
                'mean': float(self.TN / (self.TN + self.FP)) if (self.TN + self.FP) > 0 else 0.0,
                'posterior_params': {'alpha': int(self.TN + 1), 'beta': int(self.FP + 1)}
            },
            'confusion_matrix': {
                'TP': int(self.TP),
                'TN': int(self.TN),
                'FP': int(self.FP),
                'FN': int(self.FN)
            }
        }
        
        return results
    
    def probability_above_threshold(self, threshold, metric='balanced_accuracy', n_samples=100000):
        """
        Calcula P(métrica > threshold).
        
        Args:
            threshold: Valor limiar
            metric: 'accuracy' ou 'balanced_accuracy'
            n_samples: Amostras para BA
            
        Returns:
            float: Probabilidade
        """
        if metric == 'accuracy':
            posterior = self.accuracy_posterior()
            return float(1 - posterior.cdf(threshold))
        elif metric == 'balanced_accuracy':
            ba_samples = self.balanced_accuracy_posterior(n_samples)
            return float(np.mean(ba_samples > threshold))
        else:
            raise ValueError(f"Métrica desconhecida: {metric}")
    
    def compare_with(self, other_evaluator, metric='balanced_accuracy', n_samples=100000):
        """
        Compara dois modelos: P(este modelo > outro modelo).
        
        Args:
            other_evaluator: Outro BayesianAccuracyEvaluator
            metric: Métrica para comparar
            n_samples: Amostras para estimativa
            
        Returns:
            float: P(self > other)
        """
        if metric == 'accuracy':
            samples_self = self.accuracy_posterior().rvs(n_samples)
            samples_other = other_evaluator.accuracy_posterior().rvs(n_samples)
        elif metric == 'balanced_accuracy':
            samples_self = self.balanced_accuracy_posterior(n_samples)
            samples_other = other_evaluator.balanced_accuracy_posterior(n_samples)
        else:
            raise ValueError(f"Métrica desconhecida: {metric}")
        
        return float(np.mean(samples_self > samples_other))
    
    def summary_report(self, confidence=0.95):
        """
        Gera relatório formatado estilo artigo Brodersen.
        
        Args:
            confidence: Nível de confiança
            
        Returns:
            str: Relatório formatado
        """
        metrics = self.compute_metrics(confidence=confidence)
        
        acc = metrics['accuracy']
        ba = metrics['balanced_accuracy']
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║  BAYESIAN EVALUATION REPORT (Brodersen et al., 2010)      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  ACCURACY:                                                 ║
║    Mean:     {acc['mean']:.4f}                            ║
║    Median:   {acc['median']:.4f}                          ║
║    Mode:     {acc['mode']:.4f}                            ║
║    CI {int(confidence*100)}%:  [{acc['ci'][0]:.4f}, {acc['ci'][1]:.4f}]                  ║
║    Posterior: Beta({acc['params']['alpha']}, {acc['params']['beta']})                            ║
║                                                            ║
║  BALANCED ACCURACY:                                        ║
║    Mean:     {ba['mean']:.4f}                            ║
║    Median:   {ba['median']:.4f}                          ║
║    Std:      {ba['std']:.4f}                            ║
║    CI {int(confidence*100)}%:  [{ba['ci'][0]:.4f}, {ba['ci'][1]:.4f}]                  ║
║    Distribution: Convolution of 2 Betas                   ║
║                                                            ║
║  CONFUSION MATRIX:                                         ║
║    TN: {metrics['confusion_matrix']['TN']:7,}    FP: {metrics['confusion_matrix']['FP']:7,}                    ║
║    FN: {metrics['confusion_matrix']['FN']:7,}    TP: {metrics['confusion_matrix']['TP']:7,}                    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """
        
        return report


def evaluate_with_bayesian_metrics(y_true, y_pred, confidence=0.95, verbose=False):
    """
    Wrapper para avaliação Bayesiana rápida.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        confidence: Nível de confiança
        verbose: Imprimir relatório
        
    Returns:
        dict: Métricas Bayesianas
    """
    evaluator = BayesianAccuracyEvaluator(y_true, y_pred)
    metrics = evaluator.compute_metrics(confidence=confidence)
    
    if verbose:
        print(evaluator.summary_report(confidence=confidence))
    
    return metrics


def compare_algorithms_bayesian(results_dict, n_samples=100000):
    """
    Compara múltiplos algoritmos com análise Bayesiana.
    
    Args:
        results_dict: {nome_algoritmo: {'y_true': ..., 'y_pred': ...}}
        n_samples: Amostras para comparações
        
    Returns:
        pd.DataFrame: Matriz de comparações P(A > B)
    """
    import pandas as pd
    
    # Criar evaluators
    evaluators = {}
    for name, data in results_dict.items():
        evaluators[name] = BayesianAccuracyEvaluator(
            data['y_true'], data['y_pred']
        )
    
    # Matriz de comparações
    algorithms = list(evaluators.keys())
    comparison_matrix = pd.DataFrame(
        index=algorithms,
        columns=algorithms,
        dtype=float
    )
    
    for alg1 in algorithms:
        for alg2 in algorithms:
            if alg1 == alg2:
                comparison_matrix.loc[alg1, alg2] = 0.5
            else:
                prob = evaluators[alg1].compare_with(
                    evaluators[alg2],
                    metric='balanced_accuracy',
                    n_samples=n_samples
                )
                comparison_matrix.loc[alg1, alg2] = prob
    
    return comparison_matrix

