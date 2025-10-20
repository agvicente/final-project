import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import psutil
import traceback
from datetime import datetime
from enhanced_metrics_collector import (
    get_system_info,
    collect_enhanced_metrics,
    monitor_resource_usage_detailed
)
from bayesian_metrics import BayesianAccuracyEvaluator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC, LinearSVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDOneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
import gc  # Para limpeza de memória
warnings.filterwarnings('ignore')

# Configurações globais
TEST_MODE = True # Mudar para False para execução completa
SAMPLE_SIZE = 1000 if TEST_MODE else None  # Tamanho da amostra para teste
N_RUNS = 1 if TEST_MODE else 5  # Número de execuções para rigor estatístico (estratégia adaptativa por algoritmo)

# ID único da execução baseado em timestamp
EXECUTION_ID = int(time.time())
logger = None  # Será inicializado na função setup_logging

def setup_logging():
    """
    Configura sistema de logging detalhado com ID único de execução
    """
    global logger
    
    # Criar diretório de logs
    os.makedirs('experiments/logs', exist_ok=True)
    
    # Nome do arquivo de log com ID único
    log_filename = f"experiments/logs/algorithm_comparison_{EXECUTION_ID}.log"
    
    # Configurar logging
    logger = logging.getLogger(f'algorithm_comparison_{EXECUTION_ID}')
    logger.setLevel(logging.INFO)
    
    # Limpar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formato detalhado
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info(f"🔬 INICIANDO EXPERIMENTOS DE COMPARAÇÃO DE ALGORITMOS")
    logger.info(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🆔 Execution ID: {EXECUTION_ID}")
    logger.info(f"📁 Log file: {log_filename}")
    logger.info(f"🧪 Modo: {'TESTE' if TEST_MODE else 'COMPLETO'}")
    logger.info("=" * 80)
    
    return logger, log_filename

def monitor_memory():
    """
    Monitora uso de memória do processo atual
    
    Returns:
        dict: Informações de memória (MB)
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except Exception as e:
        logger.warning(f"⚠️ Erro ao monitorar memória: {e}")
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0,
            'available_mb': 0
        }

def log_memory_status(context=""):
    """
    Log detalhado do status de memória
    """
    memory = monitor_memory()
    context_str = f" ({context})" if context else ""
    
    logger.info(f"💾 Memória{context_str}:")
    logger.info(f"   RSS: {memory['rss_mb']:.1f} MB")
    logger.info(f"   VMS: {memory['vms_mb']:.1f} MB")
    logger.info(f"   Uso: {memory['percent']:.1f}%")
    logger.info(f"   Disponível: {memory['available_mb']:.1f} MB")
    
    return memory

def load_binary_data(test_mode=None):
    """
    Carrega os dados binários preprocessados com monitoramento detalhado
    
    Args:
        test_mode (bool): Se None, usa TEST_MODE global; se True, carrega apenas uma amostra pequena
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Se test_mode não especificado, usar TEST_MODE global
    if test_mode is None:
        test_mode = TEST_MODE
        
    logger.info(f"📂 Carregando dados binários (test_mode={test_mode})...")
    
    # Monitorar memória antes do carregamento
    memory_before = log_memory_status("antes do carregamento")
    
    try:
        # Carregar dados
        logger.info("   📄 Carregando X_train_binary.npy...")
        X_train = np.load('data/processed/binary/X_train_binary.npy')
        
        logger.info("   📄 Carregando X_test_binary.npy...")
        X_test = np.load('data/processed/binary/X_test_binary.npy')
        
        logger.info("   📄 Carregando y_train_binary.npy...")
        y_train = np.load('data/processed/binary/y_train_binary.npy')
        
        logger.info("   📄 Carregando y_test_binary.npy...")
        y_test = np.load('data/processed/binary/y_test_binary.npy')
        
        memory_after_load = log_memory_status("após carregamento")
        data_memory = memory_after_load['rss_mb'] - memory_before['rss_mb']
        
        logger.info(f"📊 Dados originais carregados:")
        logger.info(f"   Treino: {X_train.shape} ({X_train.nbytes / 1024 / 1024:.1f} MB)")
        logger.info(f"   Teste: {X_test.shape} ({X_test.nbytes / 1024 / 1024:.1f} MB)")
        logger.info(f"   Memória usada pelos dados: +{data_memory:.1f} MB")
        
        if test_mode and SAMPLE_SIZE:
            logger.info(f"🔄 Aplicando amostragem para modo teste (sample_size={SAMPLE_SIZE})...")
            
            # Amostragem para teste rápido
            train_sample = min(SAMPLE_SIZE, len(X_train))
            test_sample = min(SAMPLE_SIZE // 4, len(X_test))
            
            logger.info(f"   Amostras selecionadas - Treino: {train_sample}, Teste: {test_sample}")
            
            # Amostragem estratificada
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=train_sample, 
                stratify=y_train, random_state=42
            )
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test, train_size=test_sample,
                stratify=y_test, random_state=42
            )
            
            memory_after_sample = log_memory_status("após amostragem")
            
            logger.info(f"📊 Dados após amostragem:")
            logger.info(f"   Treino: {X_train.shape}")
            logger.info(f"   Teste: {X_test.shape}")
        
        # Estatísticas de distribuição
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)
        
        logger.info(f"📈 Distribuição de classes:")
        logger.info(f"   Treino - Benigno: {train_dist[0]:,}, Malicioso: {train_dist[1]:,}")
        logger.info(f"   Teste  - Benigno: {test_dist[0]:,}, Malicioso: {test_dist[1]:,}")
        logger.info(f"   Proporção treino: {train_dist[1]/(train_dist[0]+train_dist[1])*100:.1f}% malicioso")
        logger.info(f"   Proporção teste:  {test_dist[1]/(test_dist[0]+test_dist[1])*100:.1f}% malicioso")
        
        logger.info("✅ Dados carregados com sucesso!")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_algorithm_configs(test_mode=None):
    """
    Define configurações de algoritmos para testar
    
    Args:
        test_mode: Override do modo teste (None usa a global TEST_MODE)
    
    Returns:
        dict: Dicionário com algoritmos e suas configurações de parâmetros
    """
    
    use_test_mode = test_mode if test_mode is not None else TEST_MODE
    
    if use_test_mode:
        # Configurações simples para teste - ORDENADOS POR COMPLEXIDADE (GradientBoosting por último)
        return {
            # 1. MAIS RÁPIDO - O(n)
            'LogisticRegression': {
                'class': LogisticRegression,
                'param_combinations': [
                    {'C': 1.0, 'max_iter': 100, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 2. RÁPIDO - O(n log n)
            'RandomForest': {
                'class': RandomForestClassifier,
                'param_combinations': [
                    {'n_estimators': 10, 'max_depth': 5, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 3. RÁPIDO PARA ANOMALIAS - O(n log n)
            'IsolationForest': {
                'class': IsolationForest,
                'param_combinations': [
                    {'contamination': 0.1, 'n_estimators': 30, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 4. MODERADO PARA ANOMALIAS - O(n²)
            'EllipticEnvelope': {
                'class': EllipticEnvelope,
                'param_combinations': [
                    {'contamination': 0.1, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 5. Local Outlier Factor
            'LocalOutlierFactor': {
                'class': LocalOutlierFactor,
                'param_combinations': [
                    {'n_neighbors': 5, 'contamination': 0.1, 'novelty': True}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 6. PESADO - LinearSVC otimizado para datasets grandes
            'LinearSVC': {
                'class': LinearSVC,
                'param_combinations': [
                    {'C': 1.0, 'max_iter': 100, 'dual': False, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 7. RÁPIDO - SVM via gradiente estocástico
            'SGDClassifier': {
                'class': SGDClassifier,
                'param_combinations': [
                    {'loss': 'hinge', 'alpha': 0.0001, 'max_iter': 100, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 8. OTIMIZADO - OneClassSVM via gradiente estocástico
            'SGDOneClassSVM': {
                'class': SGDOneClassSVM,
                'param_combinations': [
                    {'nu': 0.1, 'max_iter': 100, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 9. MLP Classifier - Otimizado para teste rápido
            'MLPClassifier': {
                'class': MLPClassifier,
                'param_combinations': [
                    {'hidden_layer_sizes': (20,), 'max_iter': 50, 'early_stopping': True, 
                     'validation_fraction': 0.1, 'n_iter_no_change': 5, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 10. ÚLTIMO - Gradient Boosting (mais pesado)
            'GradientBoostingClassifier': {
                'class': GradientBoostingClassifier,
                'param_combinations': [
                    {'n_estimators': 5, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            }
        }
    else:
        # Configurações completas - ESTRATÉGIA ADAPTATIVA (Opção C)
        # Algoritmos rápidos: 20 configs | Médios: 15 configs | Pesados: 10-12 configs
        # Organizadas por complexidade crescente (simples → sweet spot IoT → limite)
        return {
            # 1. Logistic Regression - 20 configurações (ALGORITMO RÁPIDO)
            # Escala logarítmica em C, concentrada no sweet spot IoT (0.01-100)
            'LogisticRegression': {
                'class': LogisticRegression,
                'param_combinations': [
                    # LEVES (1-4): Muito regularizado, convergência rápida - 20%
                    {'C': 0.0001, 'max_iter': 200, 'random_state': 42},
                    {'C': 0.0005, 'max_iter': 200, 'random_state': 42},
                    {'C': 0.001, 'max_iter': 300, 'random_state': 42},
                    {'C': 0.005, 'max_iter': 300, 'random_state': 42},
                    # SWEET SPOT IoT (5-12): Range mais deployable - 40%
                    {'C': 0.01, 'max_iter': 500, 'random_state': 42},
                    {'C': 0.05, 'max_iter': 500, 'random_state': 42},
                    {'C': 0.1, 'max_iter': 500, 'random_state': 42},
                    {'C': 0.5, 'max_iter': 700, 'random_state': 42},
                    {'C': 1.0, 'max_iter': 1000, 'random_state': 42},    # Baseline
                    {'C': 2.0, 'max_iter': 1000, 'random_state': 42},
                    {'C': 5.0, 'max_iter': 1000, 'random_state': 42},
                    {'C': 10.0, 'max_iter': 1000, 'random_state': 42},
                    # MÉDIAS (13-16): Menos regularização - 20%
                    {'C': 20.0, 'max_iter': 1200, 'random_state': 42},
                    {'C': 50.0, 'max_iter': 1500, 'random_state': 42},
                    {'C': 100.0, 'max_iter': 1500, 'random_state': 42},
                    {'C': 200.0, 'max_iter': 1500, 'random_state': 42},
                    # PESADAS (17-20): Pouca regularização, limite IoT - 20%
                    {'C': 500.0, 'max_iter': 2000, 'random_state': 42},
                    {'C': 1000.0, 'max_iter': 2000, 'random_state': 42},
                    {'C': 5000.0, 'max_iter': 2000, 'random_state': 42},
                    {'C': 10000.0, 'max_iter': 2000, 'random_state': 42}  # Quase sem regularização
                ],
                'n_runs': N_RUNS
            },
            
            # 2. Random Forest - 12 configurações (ALGORITMO MÉDIO)
            # Crescimento gradual de trees e depth, focado no sweet spot IoT
            'RandomForest': {
                'class': RandomForestClassifier,
                'param_combinations': [
                    # LEVES (1-2): Poucos trees, depth baixo - ~17%
                    {'n_estimators': 20, 'max_depth': 5, 'random_state': 42},
                    {'n_estimators': 30, 'max_depth': 7, 'random_state': 42},
                    # SWEET SPOT (3-7): Range IoT-viável - ~42%
                    {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
                    {'n_estimators': 70, 'max_depth': 12, 'random_state': 42},
                    {'n_estimators': 100, 'max_depth': 15, 'random_state': 42},  # Baseline
                    {'n_estimators': 100, 'max_depth': 18, 'random_state': 42},
                    {'n_estimators': 150, 'max_depth': 20, 'random_state': 42},
                    # MÉDIAS (8-10): Moderadas - 25%
                    {'n_estimators': 200, 'max_depth': 20, 'random_state': 42},
                    {'n_estimators': 200, 'max_depth': 25, 'random_state': 42},
                    {'n_estimators': 250, 'max_depth': 25, 'random_state': 42},
                    # PESADAS (11-12): Limite IoT - ~17%
                    {'n_estimators': 300, 'max_depth': 25, 'random_state': 42},
                    {'n_estimators': 350, 'max_depth': 25, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 3. Isolation Forest - 15 configurações (ALGORITMO MÉDIO)
            # Contamination crescente + n_estimators
            'IsolationForest': {
                'class': IsolationForest,
                'param_combinations': [
                    # LEVES (1-3): Poucos trees, contam baixo
                    {'contamination': 0.05, 'n_estimators': 50, 'random_state': 42},
                    {'contamination': 0.07, 'n_estimators': 50, 'random_state': 42},
                    {'contamination': 0.1, 'n_estimators': 50, 'random_state': 42},
                    # SWEET SPOT (4-9): Range IoT
                    {'contamination': 0.1, 'n_estimators': 100, 'random_state': 42},
                    {'contamination': 0.12, 'n_estimators': 100, 'random_state': 42},
                    {'contamination': 0.15, 'n_estimators': 100, 'random_state': 42},
                    {'contamination': 0.15, 'n_estimators': 150, 'random_state': 42},
                    {'contamination': 0.18, 'n_estimators': 150, 'random_state': 42},
                    {'contamination': 0.2, 'n_estimators': 150, 'random_state': 42},
                    # MÉDIAS (10-12)
                    {'contamination': 0.2, 'n_estimators': 200, 'random_state': 42},
                    {'contamination': 0.22, 'n_estimators': 200, 'random_state': 42},
                    {'contamination': 0.25, 'n_estimators': 200, 'random_state': 42},
                    # PESADAS (13-15)
                    {'contamination': 0.25, 'n_estimators': 250, 'random_state': 42},
                    {'contamination': 0.3, 'n_estimators': 250, 'random_state': 42},
                    {'contamination': 0.3, 'n_estimators': 300, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 4. Elliptic Envelope - 15 configurações (ALGORITMO MÉDIO)
            # Apenas contamination varia (algoritmo simples, mas O(n³) em covariance)
            'EllipticEnvelope': {
                'class': EllipticEnvelope,
                'param_combinations': [
                    # LEVES (1-3): Contamination muito baixo
                    {'contamination': 0.01, 'random_state': 42},
                    {'contamination': 0.03, 'random_state': 42},
                    {'contamination': 0.05, 'random_state': 42},
                    # SWEET SPOT (4-9): Range IoT
                    {'contamination': 0.07, 'random_state': 42},
                    {'contamination': 0.1, 'random_state': 42},
                    {'contamination': 0.12, 'random_state': 42},
                    {'contamination': 0.15, 'random_state': 42},
                    {'contamination': 0.18, 'random_state': 42},
                    {'contamination': 0.2, 'random_state': 42},
                    # MÉDIAS (10-12)
                    {'contamination': 0.22, 'random_state': 42},
                    {'contamination': 0.25, 'random_state': 42},
                    {'contamination': 0.27, 'random_state': 42},
                    # PESADAS (13-15)
                    {'contamination': 0.3, 'random_state': 42},
                    {'contamination': 0.35, 'random_state': 42},
                    {'contamination': 0.4, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 5. Local Outlier Factor - 8 configurações (ALGORITMO MUITO PESADO)
            # n_neighbors crescente (complexidade O(n²)), reduzido para tempo < 24h
            'LocalOutlierFactor': {
                'class': LocalOutlierFactor,
                'param_combinations': [
                    # LEVES (1-2): Poucos vizinhos - 25%
                    {'n_neighbors': 5, 'contamination': 0.1, 'novelty': True},
                    {'n_neighbors': 10, 'contamination': 0.1, 'novelty': True},
                    # SWEET SPOT (3-5): Range IoT - 37.5%
                    {'n_neighbors': 15, 'contamination': 0.1, 'novelty': True},
                    {'n_neighbors': 20, 'contamination': 0.1, 'novelty': True},
                    {'n_neighbors': 20, 'contamination': 0.15, 'novelty': True},
                    # MÉDIAS (6-7) - 25%
                    {'n_neighbors': 30, 'contamination': 0.15, 'novelty': True},
                    {'n_neighbors': 40, 'contamination': 0.15, 'novelty': True},
                    # PESADAS (8) - 12.5%
                    {'n_neighbors': 50, 'contamination': 0.2, 'novelty': True}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 6. LinearSVC - 18 configurações (ALGORITMO RÁPIDO/MÉDIO)
            # Escala logarítmica, dual=False para n_samples > n_features
            'LinearSVC': {
                'class': LinearSVC,
                'param_combinations': [
                    # LEVES (1-3): Muito regularizado
                    {'C': 0.0001, 'max_iter': 300, 'dual': False, 'random_state': 42},
                    {'C': 0.0005, 'max_iter': 300, 'dual': False, 'random_state': 42},
                    {'C': 0.001, 'max_iter': 400, 'dual': False, 'random_state': 42},
                    # SWEET SPOT (4-11): Range IoT - 44%
                    {'C': 0.005, 'max_iter': 500, 'dual': False, 'random_state': 42},
                    {'C': 0.01, 'max_iter': 500, 'dual': False, 'random_state': 42},
                    {'C': 0.05, 'max_iter': 700, 'dual': False, 'random_state': 42},
                    {'C': 0.1, 'max_iter': 700, 'dual': False, 'random_state': 42},
                    {'C': 0.5, 'max_iter': 1000, 'dual': False, 'random_state': 42},
                    {'C': 1.0, 'max_iter': 1000, 'dual': False, 'random_state': 42},
                    {'C': 2.0, 'max_iter': 1000, 'dual': False, 'random_state': 42},
                    {'C': 5.0, 'max_iter': 1000, 'dual': False, 'random_state': 42},
                    # MÉDIAS (12-15)
                    {'C': 10.0, 'max_iter': 1200, 'dual': False, 'random_state': 42},
                    {'C': 20.0, 'max_iter': 1200, 'dual': False, 'random_state': 42},
                    {'C': 50.0, 'max_iter': 1500, 'dual': False, 'random_state': 42},
                    {'C': 100.0, 'max_iter': 1500, 'dual': False, 'random_state': 42},
                    # PESADAS (16-18)
                    {'C': 500.0, 'max_iter': 2000, 'dual': False, 'random_state': 42},
                    {'C': 1000.0, 'max_iter': 2000, 'dual': False, 'random_state': 42},
                    {'C': 5000.0, 'max_iter': 2000, 'dual': False, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 7. SGDClassifier - 20 configurações (ALGORITMO MUITO RÁPIDO)
            # Online learning, ideal para IoT streaming. Escala logarítmica em alpha
            'SGDClassifier': {
                'class': SGDClassifier,
                'param_combinations': [
                    # LEVES (1-4): Muito regularizado, convergência rápida - 20%
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.1, 'max_iter': 300, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.01, 'max_iter': 300, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.001, 'max_iter': 400, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.0005, 'max_iter': 400, 'random_state': 42},
                    # SWEET SPOT (5-12): Range IoT - 40%
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 500, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.00005, 'max_iter': 700, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.00001, 'max_iter': 1000, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.000005, 'max_iter': 1000, 'random_state': 42},
                    {'loss': 'log_loss', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': 42},
                    {'loss': 'modified_huber', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'elasticnet', 'alpha': 0.0001, 'max_iter': 1000, 'l1_ratio': 0.15, 'random_state': 42},
                    # MÉDIAS (13-16): Menos regularização - 20%
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.000001, 'max_iter': 1500, 'random_state': 42},
                    {'loss': 'log_loss', 'penalty': 'l2', 'alpha': 0.00001, 'max_iter': 1500, 'random_state': 42},
                    {'loss': 'modified_huber', 'penalty': 'l2', 'alpha': 0.00001, 'max_iter': 1500, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'elasticnet', 'alpha': 0.00001, 'max_iter': 1500, 'l1_ratio': 0.3, 'random_state': 42},
                    # PESADAS (17-20): Pouca regularização - 20%
                    {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.0000001, 'max_iter': 2000, 'random_state': 42},
                    {'loss': 'log_loss', 'penalty': 'l2', 'alpha': 0.000001, 'max_iter': 2000, 'random_state': 42},
                    {'loss': 'modified_huber', 'penalty': 'l2', 'alpha': 0.000001, 'max_iter': 2000, 'random_state': 42},
                    {'loss': 'hinge', 'penalty': 'elasticnet', 'alpha': 0.000001, 'max_iter': 2000, 'l1_ratio': 0.5, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 8. SGDOneClassSVM - 15 configurações (ALGORITMO MÉDIO)
            # Nu crescente (sensibilidade a outliers), online learning para anomalias
            'SGDOneClassSVM': {
                'class': SGDOneClassSVM,
                'param_combinations': [
                    # LEVES (1-3): Nu baixo (menos sensível)
                    {'nu': 0.01, 'learning_rate': 'optimal', 'max_iter': 300, 'random_state': 42},
                    {'nu': 0.03, 'learning_rate': 'optimal', 'max_iter': 400, 'random_state': 42},
                    {'nu': 0.05, 'learning_rate': 'optimal', 'max_iter': 500, 'random_state': 42},
                    # SWEET SPOT (4-9): Range IoT
                    {'nu': 0.07, 'learning_rate': 'optimal', 'max_iter': 700, 'random_state': 42},
                    {'nu': 0.1, 'learning_rate': 'optimal', 'max_iter': 1000, 'random_state': 42},
                    {'nu': 0.12, 'learning_rate': 'optimal', 'max_iter': 1000, 'random_state': 42},
                    {'nu': 0.15, 'learning_rate': 'optimal', 'max_iter': 1000, 'random_state': 42},
                    {'nu': 0.18, 'learning_rate': 'optimal', 'max_iter': 1000, 'random_state': 42},
                    {'nu': 0.2, 'learning_rate': 'optimal', 'max_iter': 1000, 'random_state': 42},
                    # MÉDIAS (10-12)
                    {'nu': 0.25, 'learning_rate': 'optimal', 'max_iter': 1200, 'random_state': 42},
                    {'nu': 0.3, 'learning_rate': 'optimal', 'max_iter': 1500, 'random_state': 42},
                    {'nu': 0.35, 'learning_rate': 'optimal', 'max_iter': 1500, 'random_state': 42},
                    # PESADAS (13-15): Nu alto (muito sensível)
                    {'nu': 0.4, 'learning_rate': 'optimal', 'max_iter': 1500, 'random_state': 42},
                    {'nu': 0.45, 'learning_rate': 'optimal', 'max_iter': 2000, 'random_state': 42},
                    {'nu': 0.5, 'learning_rate': 'optimal', 'max_iter': 2000, 'random_state': 42}
                ],
                'anomaly_detection': True,
                'n_runs': N_RUNS
            },
            
            # 9. MLP Classifier - 8 configurações (ALGORITMO MUITO PESADO)
            # Early stopping agressivo para manter tempo < 24h
            'MLPClassifier': {
                'class': MLPClassifier,
                'param_combinations': [
                    # LEVES (1-2): Redes muito pequenas - 25%
                    {'hidden_layer_sizes': (10,), 'max_iter': 50, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 5, 'random_state': 42},
                    
                    {'hidden_layer_sizes': (20,), 'max_iter': 50, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 5, 'random_state': 42},
                    
                    # SWEET SPOT (3-5): Redes pequenas - 37.5%
                    {'hidden_layer_sizes': (30,), 'max_iter': 50, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 5, 'random_state': 42},
                    
                    {'hidden_layer_sizes': (20, 10), 'max_iter': 60, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 6, 'random_state': 42},
                    
                    {'hidden_layer_sizes': (30, 15), 'max_iter': 60, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 6, 'random_state': 42},
                    
                    # MÉDIAS (6-7): Redes médias - 25%
                    {'hidden_layer_sizes': (40, 20), 'max_iter': 70, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 7, 'learning_rate': 'adaptive',
                     'random_state': 42},
                    
                    {'hidden_layer_sizes': (50, 25), 'max_iter': 80, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 8, 'learning_rate': 'adaptive',
                     'random_state': 42},
                    
                    # PESADAS (8): Rede máxima - 12.5%
                    {'hidden_layer_sizes': (50, 25, 10), 'max_iter': 80, 'early_stopping': True, 
                     'validation_fraction': 0.15, 'n_iter_no_change': 8, 'learning_rate': 'adaptive',
                     'alpha': 0.01, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            },
            
            # 10. ÚLTIMO - Gradient Boosting - 10 configurações (ALGORITMO PESADO)
            # LR × n_estimators trade-off, subsample agressivo para velocidade
            'GradientBoostingClassifier': {
                'class': GradientBoostingClassifier,
                'param_combinations': [
                    # LEVES (1-2): Convergência rápida - 20%
                    {'n_estimators': 30, 'learning_rate': 0.2, 'max_depth': 3, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 50, 'learning_rate': 0.15, 'max_depth': 3, 'subsample': 0.7, 'random_state': 42},
                    # SWEET SPOT (3-6): IoT-viável - 40%
                    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.7, 'random_state': 42},
                    # MÉDIAS (7-8) - 20%
                    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7, 'random_state': 42},
                    # PESADAS (9-10): Limite IoT - 20%
                    {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.7, 'random_state': 42},
                    {'n_estimators': 250, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.7, 'random_state': 42}
                ],
                'n_runs': N_RUNS
            }
        }

def run_single_experiment(algorithm_name, algorithm_class, params, X_train, X_test, y_train, y_test, is_anomaly_detection=False, run_id=0, param_id=0):
    """
    Executa um único experimento com um algoritmo e conjunto de parâmetros
    Com monitoramento completo e logging detalhado
    
    Returns:
        dict: Resultados do experimento
    """
    experiment_start = time.time()
    
    logger.info(f"      🔬 Iniciando experimento: {algorithm_name} (run {run_id+1}, param_set {param_id+1})")
    logger.info(f"         Parâmetros: {params}")
    logger.info(f"         Detecção de anomalia: {is_anomaly_detection}")
    
    # Monitorar memória antes do experimento
    memory_before = monitor_memory()
    logger.info(f"         💾 Memória antes: {memory_before['rss_mb']:.1f} MB")
    
    try:
        # Inicializar modelo
        logger.info(f"         ⚙️ Inicializando {algorithm_name}...")
        model_init_start = time.time()
        model = algorithm_class(**params)
        model_init_time = time.time() - model_init_start
        
        # Monitorar memória após inicialização
        memory_after_init = monitor_memory()
        init_memory = memory_after_init['rss_mb'] - memory_before['rss_mb']
        logger.info(f"         💾 Memória após init: +{init_memory:.1f} MB, tempo: {model_init_time:.3f}s")
        
        # Treinar modelo
        logger.info(f"         🔄 Treinando modelo...")
        training_start = time.time()
        
        if is_anomaly_detection:
            # Para algoritmos de detecção de anomalia, treinar apenas com dados normais
            normal_indices = np.where(y_train == 0)[0]
            X_train_normal = X_train[normal_indices]
            
            logger.info(f"         📊 Dados normais para treino: {X_train_normal.shape[0]:,} amostras")
            model.fit(X_train_normal)
            
            # Predição
            logger.info(f"         🎯 Realizando predições...")
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - predict_start
            
            # Converter predições de anomalia detection (-1, 1) para (1, 0)
            y_pred = np.where(y_pred == -1, 1, 0)
            logger.info(f"         🔄 Predições convertidas (-1,1) → (1,0)")
            
        else:
            # Algoritmos supervisionados normais
            logger.info(f"         📊 Dados para treino: {X_train.shape[0]:,} amostras")
            model.fit(X_train, y_train)
            
            # Predição
            logger.info(f"         🎯 Realizando predições...")
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - predict_start
        
        training_time = time.time() - training_start
        total_time = time.time() - experiment_start
        
        # Monitorar memória após treinamento
        memory_after_training = monitor_memory()
        training_memory = memory_after_training['rss_mb'] - memory_after_init['rss_mb']
        
        logger.info(f"         ⏱️ Tempo de treino: {training_time:.3f}s")
        logger.info(f"         ⏱️ Tempo de predição: {predict_time:.3f}s")
        logger.info(f"         💾 Memória do treino: +{training_memory:.1f} MB")
        
        # Calcular métricas
        logger.info(f"         📊 Calculando métricas...")
        metrics_start = time.time()
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC AUC se possível
        roc_auc = None
        try:
            if hasattr(model, 'predict_proba') and not is_anomaly_detection:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
                logger.info(f"         📈 ROC AUC calculado via predict_proba")
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_scores)
                logger.info(f"         📈 ROC AUC calculado via decision_function")
            else:
                logger.info(f"         ⚠️ ROC AUC não disponível para este modelo")
        except Exception as e:
            logger.warning(f"         ⚠️ Erro ao calcular ROC AUC: {str(e)}")
        
        metrics_time = time.time() - metrics_start
        
        # Log detalhado dos resultados
        logger.info(f"         📊 RESULTADOS:")
        logger.info(f"            Accuracy:          {accuracy:.4f}")
        logger.info(f"            Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"            Precision:         {precision:.4f}")
        logger.info(f"            Recall:            {recall:.4f}")
        logger.info(f"            F1-Score:          {f1:.4f}")
        if roc_auc is not None:
            logger.info(f"            ROC AUC:   {roc_auc:.4f}")
        logger.info(f"         📊 MATRIZ DE CONFUSÃO:")
        logger.info(f"            TN: {tn:,}, FP: {fp:,}")
        logger.info(f"            FN: {fn:,}, TP: {tp:,}")
        logger.info(f"         ⏱️ Tempo métricas: {metrics_time:.3f}s")
        logger.info(f"         ⏱️ Tempo total: {total_time:.3f}s")
        
        # 🔬 MÉTRICAS BAYESIANAS (Brodersen et al., 2010)
        bayesian_metrics = None
        try:
            logger.info(f"         🔬 Calculando métricas Bayesianas (Brodersen et al., 2010)...")
            bayesian_eval = BayesianAccuracyEvaluator(y_test, y_pred)
            bayesian_metrics = bayesian_eval.compute_metrics(confidence=0.95, n_samples=50000)
            
            ba_bayes = bayesian_metrics['balanced_accuracy']
            logger.info(f"         📊 MÉTRICAS BAYESIANAS:")
            logger.info(f"            BA Média: {ba_bayes['mean']:.4f}")
            logger.info(f"            BA IC 95%: [{ba_bayes['ci'][0]:.4f}, {ba_bayes['ci'][1]:.4f}]")
            logger.info(f"            BA Mediana: {ba_bayes['median']:.4f}")
        except Exception as e:
            logger.warning(f"         ⚠️  Erro ao calcular métricas Bayesianas: {e}")
        
        result = {
            'algorithm': algorithm_name,
            'params': params,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'prediction_time': predict_time,
            'total_time': total_time,
            'memory_usage_mb': training_memory,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'success': True,
            'error': None,
            'run_id': run_id,
            'param_id': param_id
        }
        
        # Adicionar métricas Bayesianas se disponíveis
        if bayesian_metrics is not None:
            result['bayesian'] = {
                'accuracy_posterior': {
                    'mean': bayesian_metrics['accuracy']['mean'],
                    'median': bayesian_metrics['accuracy']['median'],
                    'ci_lower': bayesian_metrics['accuracy']['ci'][0],
                    'ci_upper': bayesian_metrics['accuracy']['ci'][1],
                    'std': bayesian_metrics['accuracy']['std']
                },
                'balanced_accuracy_posterior': {
                    'mean': bayesian_metrics['balanced_accuracy']['mean'],
                    'median': bayesian_metrics['balanced_accuracy']['median'],
                    'std': bayesian_metrics['balanced_accuracy']['std'],
                    'ci_lower': bayesian_metrics['balanced_accuracy']['ci'][0],
                    'ci_upper': bayesian_metrics['balanced_accuracy']['ci'][1]
                },
                'sensitivity': bayesian_metrics['sensitivity']['mean'],
                'specificity': bayesian_metrics['specificity']['mean']
            }
        
        # 📊 COLETAR MÉTRICAS APRIMORADAS PARA IoT
        try:
            logger.info(f"         📊 Coletando métricas aprimoradas IoT...")
            
            # Determinar dados de treino corretos (normal ou completo)
            X_train_used = X_train_normal if is_anomaly_detection else X_train
            y_train_used = y_train[normal_indices] if is_anomaly_detection else y_train
            
            enhanced_result = collect_enhanced_metrics(
                model=model,
                X_train=X_train_used,
                X_test=X_test,
                y_train=y_train_used,
                y_test=y_test,
                y_pred=y_pred,
                training_time=training_time,
                prediction_time=predict_time,
                memory_usage_mb=training_memory,
                basic_results=result,
                batch_sizes=[1, 10, 100] if TEST_MODE else [1, 10, 100, 1000]
            )
            result = enhanced_result
            logger.info(f"         ✅ Métricas IoT coletadas com sucesso")
        except Exception as e:
            logger.warning(f"         ⚠️  Erro ao coletar métricas aprimoradas: {e}")
            # Continuar com resultado básico em caso de erro
        
        # 🧹 LIMPEZA CRÍTICA DE MEMÓRIA
        logger.info(f"         🧹 Limpando memória do modelo...")
        cleanup_start = time.time()
        
        del model
        del y_pred
        if 'y_proba' in locals():
            del y_proba
        if 'y_scores' in locals():
            del y_scores
        
        gc.collect()
        cleanup_time = time.time() - cleanup_start
        
        # Verificar efetividade da limpeza
        memory_after_cleanup = monitor_memory()
        memory_freed = memory_after_training['rss_mb'] - memory_after_cleanup['rss_mb']
        
        logger.info(f"         🧹 Limpeza: {memory_freed:.1f} MB liberados em {cleanup_time:.3f}s")
        logger.info(f"         ✅ Experimento concluído com sucesso!")
        
        return result
        
    except Exception as e:
        error_time = time.time() - experiment_start
        
        logger.error(f"         ❌ ERRO no experimento {algorithm_name}:")
        logger.error(f"            Erro: {str(e)}")
        logger.error(f"            Tempo até erro: {error_time:.3f}s")
        logger.error(f"            Traceback: {traceback.format_exc()}")
        
        # 🧹 LIMPEZA DE MEMÓRIA MESMO EM CASO DE ERRO
        logger.info(f"         🧹 Limpando memória após erro...")
        try:
            if 'model' in locals():
                del model
            if 'y_pred' in locals():
                del y_pred
            if 'y_proba' in locals():
                del y_proba
            if 'y_scores' in locals():
                del y_scores
        except:
            pass
        
        gc.collect()
        
        return {
            'algorithm': algorithm_name,
            'params': params,
            'accuracy': 0,
            'balanced_accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'roc_auc': None,
            'training_time': error_time,
            'prediction_time': 0,
            'total_time': error_time,
            'memory_usage_mb': 0,
            'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            'success': False,
            'error': str(e),
            'run_id': run_id,
            'param_id': param_id
        }

def compare_algorithms():
    """
    Função principal para comparar algoritmos com monitoramento completo
    """
    # Configurar logger se ainda não foi configurado
    global logger
    if logger is None:
        logger, _ = setup_logging()

    logger.info(f"🚀 Iniciando experimentos de comparação de algoritmos")
    logger.info(f"   Modo: {'TESTE' if TEST_MODE else 'COMPLETO'}")
    logger.info(f"   Sample size: {SAMPLE_SIZE}")
    logger.info(f"   Runs por configuração: {N_RUNS}")
    
    # Memória inicial
    initial_memory = log_memory_status("inicial")
    
    # Carregar dados
    logger.info("\n📂 CARREGANDO DADOS")
    logger.info("-" * 50)
    X_train, X_test, y_train, y_test = load_binary_data(test_mode=TEST_MODE)
    
    # Obter configurações de algoritmos
    logger.info("\n⚙️ CONFIGURAÇÕES DE ALGORITMOS")
    logger.info("-" * 50)
    algorithms_config = get_algorithm_configs()
    
    # Calcular estatísticas de progresso
    total_algorithms = len(algorithms_config)
    total_param_configs = sum(len(config['params']) for config in algorithms_config.values())
    total_experiments = total_param_configs * N_RUNS
    
    logger.info(f"📊 Estatísticas do experimento:")
    logger.info(f"   Algoritmos: {total_algorithms}")
    logger.info(f"   Configurações de parâmetros: {total_param_configs}")
    logger.info(f"   Total de experimentos: {total_experiments}")
    logger.info(f"   Tempo estimado: {total_experiments * 30:.0f}s ({total_experiments * 30 / 60:.1f} min)")
    
    # Resultados de todos os experimentos
    all_results = []
    current_experiment = 0
    
    experiment_start_time = time.time()
    
    # Executar experimentos
    logger.info(f"\n🔬 EXECUTANDO EXPERIMENTOS")
    logger.info("=" * 80)
    
    for algo_idx, (algorithm_name, config) in enumerate(algorithms_config.items()):
        algorithm_class = config['class']
        params_list = config['params']
        is_anomaly_detection = config.get('anomaly_detection', False)
        
        algo_progress = (algo_idx + 1) / total_algorithms * 100
        logger.info(f"\n🔬 ALGORITMO {algo_idx + 1}/{total_algorithms}: {algorithm_name} ({algo_progress:.1f}%)")
        logger.info(f"   Tipo: {'Detecção de Anomalia' if is_anomaly_detection else 'Supervisionado'}")
        logger.info(f"   Configurações: {len(params_list)}")
        
        # Memória antes do algoritmo
        memory_before_algo = log_memory_status(f"antes {algorithm_name}")
        
        for param_idx, params in enumerate(params_list):
            param_progress = (param_idx + 1) / len(params_list) * 100
            logger.info(f"\n   📋 CONFIGURAÇÃO {param_idx + 1}/{len(params_list)} ({param_progress:.1f}%)")
            logger.info(f"      Parâmetros: {params}")
            
            # Executar múltiplas vezes para rigor estatístico
            run_results = []
            config_start_time = time.time()
            
            for run in range(N_RUNS):
                current_experiment += 1
                overall_progress = current_experiment / total_experiments * 100
                
                logger.info(f"\n      🔄 EXECUÇÃO {run + 1}/{N_RUNS} (Progresso geral: {overall_progress:.1f}%)")
                
                result = run_single_experiment(
                    algorithm_name, algorithm_class, params,
                    X_train, X_test, y_train, y_test, is_anomaly_detection,
                    run_id=run, param_id=param_idx
                )
                
                run_results.append(result)
                
                if result['success']:
                    logger.info(f"      ✅ F1: {result['f1_score']:.4f}, Acc: {result['accuracy']:.4f}")
                else:
                    logger.error(f"      ❌ FALHA: {result['error']}")
            
            # Estatísticas da configuração
            config_time = time.time() - config_start_time
            successful_runs = [r for r in run_results if r['success']]
            
            if successful_runs:
                avg_f1 = np.mean([r['f1_score'] for r in successful_runs])
                std_f1 = np.std([r['f1_score'] for r in successful_runs])
                avg_time = np.mean([r['total_time'] for r in successful_runs])
                
                logger.info(f"   📊 RESUMO DA CONFIGURAÇÃO:")
                logger.info(f"      Execuções bem-sucedidas: {len(successful_runs)}/{N_RUNS}")
                logger.info(f"      F1-Score médio: {avg_f1:.4f} ± {std_f1:.4f}")
                logger.info(f"      Tempo médio por execução: {avg_time:.3f}s")
                logger.info(f"      Tempo total da configuração: {config_time:.3f}s")
            else:
                logger.error(f"   ❌ CONFIGURAÇÃO FALHADA: 0/{N_RUNS} execuções bem-sucedidas")
            
            all_results.extend(run_results)
            
            # 🧹 LIMPEZA DE MEMÓRIA ENTRE CONFIGURAÇÕES DE PARÂMETROS
            logger.info(f"      🧹 Limpeza entre configurações...")
            gc.collect()
        
        # 🧹 LIMPEZA DE MEMÓRIA ENTRE ALGORITMOS
        logger.info(f"\n   🧹 LIMPEZA FINAL DO ALGORITMO {algorithm_name}")
        memory_after_algo = log_memory_status(f"após {algorithm_name}")
        algo_memory_delta = memory_after_algo['rss_mb'] - memory_before_algo['rss_mb']
        
        logger.info(f"   💾 Crescimento de memória no algoritmo: {algo_memory_delta:.1f} MB")
        
        gc.collect()
        
        # Tempo estimado restante
        elapsed_time = time.time() - experiment_start_time
        if current_experiment > 0:
            avg_time_per_experiment = elapsed_time / current_experiment
            remaining_experiments = total_experiments - current_experiment
            estimated_remaining = remaining_experiments * avg_time_per_experiment
            
            logger.info(f"   ⏱️ Tempo decorrido: {elapsed_time:.1f}s")
            logger.info(f"   ⏱️ Tempo estimado restante: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} min)")
    
    # Estatísticas finais
    total_time = time.time() - experiment_start_time
    final_memory = log_memory_status("final")
    total_memory_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"🏁 EXPERIMENTOS CONCLUÍDOS")
    logger.info(f"=" * 80)
    logger.info(f"⏱️ Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"📊 Experimentos totais: {len(all_results)}")
    logger.info(f"✅ Bem-sucedidos: {len(successful_results)}")
    logger.info(f"❌ Falharam: {len(failed_results)}")
    logger.info(f"📈 Taxa de sucesso: {len(successful_results)/len(all_results)*100:.1f}%")
    logger.info(f"💾 Crescimento total de memória: {total_memory_growth:.1f} MB")
    logger.info(f"⚡ Média por experimento: {total_time/len(all_results):.2f}s")
    
    if failed_results:
        logger.warning(f"\n⚠️ EXPERIMENTOS QUE FALHARAM:")
        for result in failed_results:
            logger.warning(f"   {result['algorithm']} (param_set {result['param_id']}, run {result['run_id']}): {result['error']}")
    
    return all_results

def save_results_and_plots(results, output_dir='experiments/results'):
    """
    Salva resultados e gera gráficos
    """
    print(f"\n📊 Salvando resultados em {output_dir}...")
    
    # Criar diretório de resultados
    os.makedirs(output_dir, exist_ok=True)
    
    # Converter resultados para DataFrame
    df = pd.DataFrame(results)
    
    # Salvar resultados brutos
    results_file = os.path.join(output_dir, f'experiment_results_{"test" if TEST_MODE else "full"}.csv')
    df.to_csv(results_file, index=False)
    print(f"   ✅ Resultados salvos: {results_file}")
    
    # Calcular estatísticas agregadas
    successful_results = df[df['success'] == True]
    
    if len(successful_results) > 0:
        # Agrupar por algoritmo e parâmetros
        agg_results = successful_results.groupby(['algorithm', 'param_id']).agg({
            'accuracy': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        # Salvar estatísticas agregadas
        stats_file = os.path.join(output_dir, f'aggregated_stats_{"test" if TEST_MODE else "full"}.csv')
        agg_results.to_csv(stats_file)
        print(f"   ✅ Estatísticas agregadas: {stats_file}")
        
        # Gerar gráficos
        generate_plots(successful_results, output_dir)
        
        # Gerar tabela resumo
        generate_summary_table(successful_results, output_dir)
    
    return results_file

def generate_plots(df, output_dir):
    """
    Gera gráficos individuais para análise
    """
    print("   📈 Gerando gráficos...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Boxplot de F1-Score por algoritmo
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='algorithm', y='f1_score')
    plt.title('Distribuição do F1-Score por Algoritmo')
    plt.xlabel('Algoritmo')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_by_algorithm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de barras com médias das métricas
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
    mean_metrics = df.groupby('algorithm')[metrics].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        mean_metrics[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Média de {metric.title()}')
        axes[i].set_ylabel(metric.title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Tempo de treinamento vs Performance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='training_time', y='f1_score', hue='algorithm', s=100, alpha=0.7)
    plt.title('Tempo de Treinamento vs F1-Score')
    plt.xlabel('Tempo de Treinamento (s)')
    plt.ylabel('F1-Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap de correlação entre métricas
    correlation_metrics = df[['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'training_time']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_metrics, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlação entre Métricas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Gráficos gerados")

def generate_summary_table(df, output_dir):
    """
    Gera tabela resumo dos melhores resultados
    """
    print("   📋 Gerando tabela resumo...")
    
    # Encontrar melhor resultado para cada algoritmo
    best_results = df.loc[df.groupby('algorithm')['f1_score'].idxmax()]
    
    # Criar tabela resumo
    summary = best_results[['algorithm', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'training_time']].round(4)
    summary = summary.sort_values('f1_score', ascending=False)
    
    # Salvar como CSV
    summary_file = os.path.join(output_dir, f'best_results_summary_{"test" if TEST_MODE else "full"}.csv')
    summary.to_csv(summary_file, index=False)
    
    # Criar versão formatada para visualização
    summary_formatted = summary.copy()
    for col in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']:
        summary_formatted[col] = summary_formatted[col].apply(lambda x: f"{x:.3f}")
    summary_formatted['training_time'] = summary_formatted['training_time'].apply(lambda x: f"{x:.2f}s")
    
    print("\n🏆 RESUMO DOS MELHORES RESULTADOS:")
    print(summary_formatted.to_string(index=False))
    
    return summary_file

if __name__ == "__main__":
    try:
        # 1. Configurar logging detalhado
        logger, log_filename = setup_logging()
        
        # 2. Configurar MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment_name = f"Algorithm-Comparison-Binary-{'Test' if TEST_MODE else 'Full'}-{EXECUTION_ID}"
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"🔬 MLflow configurado:")
        logger.info(f"   Tracking URI: http://127.0.0.1:5000")
        logger.info(f"   Experiment: {experiment_name}")
        
        # 3. Executar experimentos
        with mlflow.start_run(run_name=f"Experiment-{EXECUTION_ID}"):
            logger.info(f"\n🚀 INICIANDO EXPERIMENTO MLflow")
            logger.info(f"   Run name: Experiment-{EXECUTION_ID}")
            
            # Log parâmetros do experimento no MLflow
            mlflow.log_param("execution_id", EXECUTION_ID)
            mlflow.log_param("test_mode", TEST_MODE)
            mlflow.log_param("sample_size", SAMPLE_SIZE)
            mlflow.log_param("n_runs", N_RUNS)
            mlflow.log_param("log_file", log_filename)
            mlflow.log_param("start_time", datetime.now().isoformat())
            
            logger.info(f"📝 Parâmetros logados no MLflow")
            
            # Executar comparação
            logger.info(f"\n" + "=" * 80)
            logger.info(f"🔬 EXECUTANDO COMPARAÇÃO DE ALGORITMOS")
            logger.info(f"=" * 80)
            
            experiment_start = time.time()
            results = compare_algorithms()
            experiment_duration = time.time() - experiment_start
            
            # Salvar resultados e gerar plots
            logger.info(f"\n📊 SALVANDO RESULTADOS E GRÁFICOS")
            logger.info("-" * 50)
            
            results_file = save_results_and_plots(results)
            
            # Log resultados agregados no MLflow
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            if successful_results:
                # Calcular estatísticas gerais
                avg_f1 = np.mean([r['f1_score'] for r in successful_results])
                std_f1 = np.std([r['f1_score'] for r in successful_results])
                avg_accuracy = np.mean([r['accuracy'] for r in successful_results])
                std_accuracy = np.std([r['accuracy'] for r in successful_results])
                avg_balanced_acc = np.mean([r['balanced_accuracy'] for r in successful_results])
                std_balanced_acc = np.std([r['balanced_accuracy'] for r in successful_results])
                avg_precision = np.mean([r['precision'] for r in successful_results])
                avg_recall = np.mean([r['recall'] for r in successful_results])
                avg_training_time = np.mean([r['training_time'] for r in successful_results])
                total_memory_used = sum([r['memory_usage_mb'] for r in successful_results])
                
                # Log no MLflow
                mlflow.log_metric("avg_f1_score", avg_f1)
                mlflow.log_metric("std_f1_score", std_f1)
                mlflow.log_metric("avg_accuracy", avg_accuracy)
                mlflow.log_metric("std_accuracy", std_accuracy)
                mlflow.log_metric("avg_balanced_accuracy", avg_balanced_acc)
                mlflow.log_metric("std_balanced_accuracy", std_balanced_acc)
                mlflow.log_metric("avg_precision", avg_precision)
                mlflow.log_metric("avg_recall", avg_recall)
                mlflow.log_metric("avg_training_time", avg_training_time)
                mlflow.log_metric("total_memory_mb", total_memory_used)
                mlflow.log_metric("experiment_duration", experiment_duration)
                mlflow.log_metric("total_experiments", len(results))
                mlflow.log_metric("successful_experiments", len(successful_results))
                mlflow.log_metric("failed_experiments", len(failed_results))
                mlflow.log_metric("success_rate", len(successful_results) / len(results) * 100)
                
                logger.info(f"📈 Estatísticas logadas no MLflow:")
                logger.info(f"   F1-Score médio: {avg_f1:.4f} ± {std_f1:.4f}")
                logger.info(f"   Accuracy médio: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
                logger.info(f"   Balanced Accuracy médio: {avg_balanced_acc:.4f} ± {std_balanced_acc:.4f}")
                logger.info(f"   Taxa de sucesso: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
            
            # Log artifacts (resultados e logs)
            mlflow.log_artifacts("experiments/results")
            mlflow.log_artifact(log_filename)
            
            logger.info(f"📁 Artifacts logados no MLflow")
        
        # 4. Resumo final
        logger.info(f"\n" + "=" * 80)
        logger.info(f"🎉 EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        logger.info(f"=" * 80)
        logger.info(f"🆔 Execution ID: {EXECUTION_ID}")
        logger.info(f"⏱️ Duração total: {experiment_duration:.1f}s ({experiment_duration/60:.1f} min)")
        logger.info(f"📁 Resultados: {results_file}")
        logger.info(f"📄 Log detalhado: {log_filename}")
        logger.info(f"🔬 MLflow experiment: {experiment_name}")
        logger.info(f"🌐 MLflow UI: http://127.0.0.1:5000")
        logger.info(f"=" * 80)
        
        # Imprimir resumo também no console tradicional
        print(f"\n✅ Experimento concluído! Execution ID: {EXECUTION_ID}")
        print(f"📁 Resultados salvos em: {results_file}")
        print(f"📄 Log detalhado: {log_filename}")
        print(f"🔬 MLflow experiment: {experiment_name}")
        print(f"🌐 Acesse: http://127.0.0.1:5000")
        
    except Exception as e:
        error_msg = f"💥 ERRO CRÍTICO: {str(e)}"
        
        if logger:
            logger.error(error_msg)
            logger.error(f"Traceback completo:")
            logger.error(traceback.format_exc())
        else:
            print(error_msg)
            print(traceback.format_exc())
        
        sys.exit(1)
