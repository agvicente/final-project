"""
Métricas prequential para avaliação de algoritmos streaming.

Baseado em:
    Gama, J., Sebastião, R., & Rodrigues, P. P. (2013).
    "On evaluating stream learning algorithms."
    Machine Learning, 90(3), 317-346.
    https://doi.org/10.1007/s10994-012-5320-9

Implementação:
    - Estimador prequential com sliding window
    - Estimador prequential com fading factor (recomendado para concept drift)
    - Métricas de classificação (Precision/Recall/F1/FPR)
    - MTTD (Mean Time To Detection) corrigido

Versão: 0.2 (2026-02-22)
"""

from collections import deque
from typing import Dict, Optional
import numpy as np


class PrequentialMetrics:
    """
    Calcula métricas de detecção usando avaliação prequential.

    Implementa estimadores de erro prequential conforme Gama et al. (2013):
    - Erro acumulado (histórico completo)
    - Erro em janela deslizante (sliding window)
    - Erro com fading factor (esquecimento exponencial)

    Metodologia Test-Then-Train:
        1. Testar: y_pred = modelo.predict(x)
        2. Avaliar: comparar y_pred com y_true
        3. Treinar: modelo.update(x, y_true)

    Limitações da v0.2:
        - MTTD considera apenas primeiro ataque → primeiro alerta APÓS ataque
        - Page-Hinkley detector (drift detection) não implementado ainda
        - Não suporta avaliação multi-classe (apenas binário)
        - Fading factor fixo (não adaptativo)

    Exemplo:
        >>> metrics = PrequentialMetrics(window_size=1000, alpha=0.01)
        >>> for i in range(100):
        ...     y_true = i >= 50  # Ataque começa no flow 50
        ...     y_pred = i >= 55  # Detector demora 5 flows
        ...     metrics.update(y_pred, y_true, float(i))
        >>> metrics.get_mttd()
        5
        >>> metrics.get_prequential_error_fading()
        0.05  # 5% de erro com esquecimento

    Args:
        window_size: Tamanho da janela para estimador sliding window
        alpha: Fading factor para esquecimento exponencial (0 < alpha <= 1)
               alpha=1 → sem esquecimento (igual ao acumulado)
               alpha=0.01 → esquecimento rápido (recomendado paper)
    """

    def __init__(self, window_size: int = 1000, alpha: float = 0.01):
        """
        Inicializa métricas prequential.

        Args:
            window_size: Tamanho da janela para métricas (default: 1000)
            alpha: Fading factor para esquecimento exponencial (default: 0.01)
                   Valor recomendado no paper: 0.01 (esquecimento rápido)
        """
        self.window_size = window_size
        self.alpha = alpha

        # Sliding windows (para estimador de janela)
        self.predictions = deque(maxlen=window_size)
        self.ground_truth = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Erro prequential (0 = correto, 1 = erro)
        self.errors = deque(maxlen=window_size)  # Para janela

        # Estimadores prequential acumulados
        self.total_samples = 0
        self.cumulative_error = 0.0  # Soma de erros
        self.fading_error = 0.0      # Erro com fading factor

        # Matriz de confusão global
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0

        # MTTD corrigido (Mean Time To Detection)
        self.first_attack_idx: Optional[int] = None
        self.first_alert_after_attack_idx: Optional[int] = None

    def update(self, y_pred: bool, y_true: bool, timestamp: float):
        """
        Adiciona nova predição e atualiza estimadores prequential.

        Segue protocolo Test-Then-Train:
        1. Modelo já fez predição (y_pred)
        2. Avaliamos comparando com y_true
        3. (Modelo será atualizado pelo chamador)

        Args:
            y_pred: Predição do modelo (True = ataque, False = benign)
            y_true: Ground truth (True = ataque, False = benign)
            timestamp: Timestamp da amostra
        """
        # Adicionar às janelas
        self.predictions.append(y_pred)
        self.ground_truth.append(y_true)
        self.timestamps.append(timestamp)

        # Calcular erro (0 = correto, 1 = erro)
        error = 1 if y_pred != y_true else 0
        self.errors.append(error)

        # Atualizar contadores
        self.total_samples += 1

        # Atualizar estimadores prequential
        self.cumulative_error += error

        # Fading factor: P_α(n) = α * e_n + (1-α) * P_α(n-1)
        # Implementação: soma ponderada exponencialmente decrescente
        self.fading_error = self.alpha * error + (1 - self.alpha) * self.fading_error

        # Atualizar matriz de confusão global
        if y_true and y_pred:
            self.total_tp += 1
        elif not y_true and y_pred:
            self.total_fp += 1
        elif y_true and not y_pred:
            self.total_fn += 1
        else:
            self.total_tn += 1

        # Rastrear MTTD (corrigido: primeiro alerta APÓS primeiro ataque)
        if y_true and self.first_attack_idx is None:
            self.first_attack_idx = self.total_samples

        # Só considera alerta se já houve ataque
        if y_pred and self.first_attack_idx is not None and self.first_alert_after_attack_idx is None:
            self.first_alert_after_attack_idx = self.total_samples

    def get_prequential_error_cumulative(self) -> float:
        """
        Retorna erro prequential acumulado (histórico completo).

        Fórmula: P(n) = (1/n) * Σ e_i

        Returns:
            Taxa de erro acumulada [0, 1]
        """
        if self.total_samples == 0:
            return 0.0
        return self.cumulative_error / self.total_samples

    def get_prequential_error_window(self) -> float:
        """
        Retorna erro prequential em sliding window.

        Fórmula: P_w(n) = (1/w) * Σ e_i (últimos w samples)

        Returns:
            Taxa de erro na janela [0, 1]
        """
        if len(self.errors) == 0:
            return 0.0
        return sum(self.errors) / len(self.errors)

    def get_prequential_error_fading(self) -> float:
        """
        Retorna erro prequential com fading factor (RECOMENDADO para drift).

        Fórmula recursiva: P_α(n) = α * e_n + (1-α) * P_α(n-1)

        Características:
        - Dá mais peso às amostras recentes
        - Esquece gradualmente amostras antigas
        - Adapta-se melhor a concept drift

        Returns:
            Taxa de erro com esquecimento [0, 1]
        """
        return self.fading_error

    def get_current_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de classificação na janela atual.

        Métricas IDS relevantes:
        - Precision: De todos os alertas, quantos são verdadeiros?
        - Recall: De todos os ataques, quantos foram detectados?
        - F1: Média harmônica de Precision e Recall
        - FPR: Taxa de falsos positivos (crítico para IDS)

        Returns:
            Dict com métricas de classificação
        """
        if len(self.predictions) < 10:
            # Janela muito pequena, retornar zeros
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'fpr': 0.0,
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0,
                'error_rate': 0.0,
                'window_size': len(self.predictions),
                'total_samples': self.total_samples
            }

        y_true = np.array(list(self.ground_truth))
        y_pred = np.array(list(self.predictions))

        # Matriz de confusão na janela
        tp = int(np.sum((y_true == True) & (y_pred == True)))
        fp = int(np.sum((y_true == False) & (y_pred == True)))
        tn = int(np.sum((y_true == False) & (y_pred == False)))
        fn = int(np.sum((y_true == True) & (y_pred == False)))

        # Métricas (com proteção divisão por zero)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        error_rate = float((fp + fn) / len(y_true)) if len(y_true) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'error_rate': error_rate,
            'window_size': len(self.predictions),
            'total_samples': self.total_samples
        }

    def get_global_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de classificação globais (todas as amostras).

        Returns:
            Dict com métricas acumuladas
        """
        if self.total_samples == 0:
            return {}

        total_pos = self.total_tp + self.total_fn
        total_neg = self.total_tn + self.total_fp

        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / total_pos if total_pos > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (self.total_tp + self.total_tn) / self.total_samples
        fpr = self.total_fp / total_neg if total_neg > 0 else 0.0
        error_rate = (self.total_fp + self.total_fn) / self.total_samples

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'fpr': fpr,
            'tp': self.total_tp,
            'fp': self.total_fp,
            'tn': self.total_tn,
            'fn': self.total_fn,
            'error_rate': error_rate,
            'total_samples': self.total_samples
        }

    def get_mttd(self) -> Optional[int]:
        """
        Calcula MTTD (Mean Time To Detection) em número de flows.

        MTTD = (índice do primeiro alerta após ataque) - (índice do primeiro ataque)

        IMPORTANTE: Esta implementação considera apenas o primeiro ataque
        e o primeiro alerta APÓS esse ataque. Para múltiplos ataques,
        esta métrica deve ser calculada por período.

        Limitação: Se houve falso positivo ANTES do primeiro ataque,
        ele é ignorado corretamente (versão corrigida).

        Returns:
            Número de flows desde primeiro ataque até primeiro alerta,
            ou None se não houve ataque ou ainda não foi detectado
        """
        if self.first_attack_idx is None:
            # Ainda não houve ataque
            return None

        if self.first_alert_after_attack_idx is None:
            # Houve ataque mas ainda não foi detectado
            return None

        mttd = self.first_alert_after_attack_idx - self.first_attack_idx
        return max(0, mttd)  # MTTD >= 0 (0 = detecção imediata)

    def get_all_prequential_errors(self) -> Dict[str, float]:
        """
        Retorna todos os estimadores de erro prequential.

        Útil para análise comparativa dos estimadores.

        Returns:
            Dict com os 3 estimadores de erro
        """
        return {
            'cumulative': self.get_prequential_error_cumulative(),
            'window': self.get_prequential_error_window(),
            'fading': self.get_prequential_error_fading()
        }

    def reset(self):
        """Reseta todas as métricas para estado inicial."""
        self.predictions.clear()
        self.ground_truth.clear()
        self.timestamps.clear()
        self.errors.clear()

        self.total_samples = 0
        self.cumulative_error = 0.0
        self.fading_error = 0.0

        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0

        self.first_attack_idx = None
        self.first_alert_after_attack_idx = None


# ============================================================
# TESTES (executar com: python3 -m src.metrics.prequential_metrics)
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TESTES - Prequential Metrics v0.2")
    print("Baseado em: Gama et al. (2013)")
    print("=" * 70)

    # Teste 1: Detecção perfeita
    print("\n[Teste 1] Detecção perfeita (sem erros)")
    metrics = PrequentialMetrics(window_size=10, alpha=0.01)

    for i in range(100):
        y_true = i >= 50
        y_pred = i >= 50  # Perfeito
        metrics.update(y_pred, y_true, float(i))

    errs = metrics.get_all_prequential_errors()
    print(f"  Erro cumulativo: {errs['cumulative']:.3f} (esperado: 0.0)")
    print(f"  Erro window: {errs['window']:.3f} (esperado: 0.0)")
    print(f"  Erro fading: {errs['fading']:.3f} (esperado: 0.0)")
    print(f"  MTTD: {metrics.get_mttd()} flows (esperado: 0)")

    assert errs['cumulative'] == 0.0
    assert errs['window'] == 0.0
    assert abs(errs['fading']) < 0.01  # Fading pode ter resíduo
    assert metrics.get_mttd() == 0
    print("  ✅ Passou!")

    # Teste 2: MTTD corrigido (com FP antes do ataque)
    print("\n[Teste 2] MTTD corrigido (ignora FP antes de ataque)")
    metrics2 = PrequentialMetrics(window_size=10, alpha=0.01)

    for i in range(100):
        y_true = i >= 50  # Ataque começa em 50
        # Falso positivo em i=10, detecção correta em i=55
        y_pred = (i == 10) or (i >= 55)
        metrics2.update(y_pred, y_true, float(i))

    mttd = metrics2.get_mttd()
    print(f"  MTTD: {mttd} flows (esperado: 5, não 0)")
    print(f"  Primeiro ataque: idx {metrics2.first_attack_idx}")
    print(f"  Primeiro alerta após ataque: idx {metrics2.first_alert_after_attack_idx}")

    assert mttd == 5, f"MTTD deveria ser 5 (não {mttd}) - FP antes de ataque deve ser ignorado"
    print("  ✅ Passou!")

    # Teste 3: Estimadores prequential com erro constante
    print("\n[Teste 3] Estimadores com erro 10%")
    metrics3 = PrequentialMetrics(window_size=100, alpha=0.1)

    for i in range(1000):
        y_true = True
        y_pred = (i % 10) != 0  # Erro a cada 10 samples (10% erro)
        metrics3.update(y_pred, y_true, float(i))

    errs3 = metrics3.get_all_prequential_errors()
    print(f"  Erro cumulativo: {errs3['cumulative']:.3f} (esperado: ~0.10)")
    print(f"  Erro window: {errs3['window']:.3f} (esperado: ~0.10)")
    print(f"  Erro fading: {errs3['fading']:.3f} (esperado: ~0.10)")

    assert abs(errs3['cumulative'] - 0.10) < 0.01
    assert abs(errs3['window'] - 0.10) < 0.01
    assert abs(errs3['fading'] - 0.10) < 0.05  # Fading pode ter mais variância
    print("  ✅ Passou!")

    # Teste 4: Fading factor adapta a drift
    print("\n[Teste 4] Fading factor adapta a concept drift")
    metrics4 = PrequentialMetrics(window_size=50, alpha=0.01)

    # Fase 1: sem erro (samples 0-499)
    for i in range(500):
        metrics4.update(True, True, float(i))

    # Fase 2: COM erro (samples 500-999) - drift súbito
    for i in range(500, 1000):
        metrics4.update(False, True, float(i))  # Erro 100%

    errs4 = metrics4.get_all_prequential_errors()
    print(f"  Erro cumulativo: {errs4['cumulative']:.3f} (média histórica: ~0.50)")
    print(f"  Erro window (w=50): {errs4['window']:.3f} (últimos 50: 1.0)")
    print(f"  Erro fading (α=0.01): {errs4['fading']:.3f} (adaptado: ~1.0)")

    # Fading deve estar próximo de 1.0 (esqueceu fase 1)
    assert errs4['fading'] > 0.90, "Fading deveria ter esquecido fase sem erro"
    # Window deve ser exatamente 1.0
    assert abs(errs4['window'] - 1.0) < 0.01, "Window deveria ser 1.0"
    # Cumulativo deve ser ~0.5 (média de tudo)
    assert abs(errs4['cumulative'] - 0.5) < 0.01, "Cumulativo deveria ser ~0.5"

    print("  ✅ Passou! (fading adapta melhor que window/cumulative)")

    # Teste 5: Métricas de classificação
    print("\n[Teste 5] Métricas de classificação (P/R/F1)")
    metrics5 = PrequentialMetrics(window_size=100, alpha=0.01)

    for i in range(100):
        y_true = i >= 50  # 50 benign, 50 attack
        y_pred = i >= 55  # 5 FN, resto correto
        metrics5.update(y_pred, y_true, float(i))

    current5 = metrics5.get_current_metrics()
    print(f"  Precision: {current5['precision']:.3f}")
    print(f"  Recall: {current5['recall']:.3f} (esperado: 45/50 = 0.90)")
    print(f"  F1: {current5['f1']:.3f}")
    print(f"  FPR: {current5['fpr']:.3f} (esperado: 0.0)")

    assert abs(current5['recall'] - 0.90) < 0.01, "Recall deveria ser 0.90"
    assert current5['fpr'] == 0.0, "FPR deveria ser 0.0 (sem FP)"
    print("  ✅ Passou!")

    print("\n" + "=" * 70)
    print("✅ Todos os 5 testes passaram!")
    print("=" * 70)
    print("\nNOTA: Esta implementação v0.2 inclui:")
    print("  - Estimadores prequential corretos (cumulative, window, fading)")
    print("  - MTTD corrigido (ignora FP antes de ataque)")
    print("  - Documentação de limitações")
    print("  - Próximos passos: Page-Hinkley detector (opcional)")
