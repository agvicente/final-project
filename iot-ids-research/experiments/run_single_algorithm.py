#!/usr/bin/env python3
"""
Executor de algoritmo individual para pipeline DVC modular
"""

import sys
import os
import time
import json
import gc
from pathlib import Path
import logging

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison import (
    load_binary_data, get_algorithm_configs, run_single_experiment,
    monitor_memory, log_memory_status, TEST_MODE
)

def setup_algorithm_logging(algorithm_name, test_mode=False):
    """Configura logging específico para o algoritmo"""
    execution_id = int(time.time())
    
    # Escolher diretório baseado no modo
    base_log_dir = 'experiments/logs_test' if test_mode else 'experiments/logs'
    log_dir = Path(base_log_dir) / algorithm_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{algorithm_name}_{execution_id}.log"
    
    logger = logging.getLogger(f'{algorithm_name}_{execution_id}')
    logger.setLevel(logging.INFO)
    
    # Limpar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_algorithm_experiments(algorithm_name, test_mode=None):
    """Executa todos os experimentos para um algoritmo específico"""
    # Se test_mode não foi especificado, usar o TEST_MODE global
    if test_mode is None:
        test_mode = TEST_MODE
    
    logger = setup_algorithm_logging(algorithm_name, test_mode)
    
    logger.info(f"🔬 INICIANDO EXPERIMENTOS: {algorithm_name}")
    logger.info(f"🧪 Modo: {'TESTE' if test_mode else 'COMPLETO'}")
    logger.info(f"📁 Pasta: {'experiments/results_test/' if test_mode else 'experiments/results/'}")
    start_time = time.time()
    
    try:
        # Configurar logger global temporariamente
        import algorithm_comparison
        algorithm_comparison.logger = logger
        
        # Carregar dados
        logger.info("📊 Carregando dados binários...")
        log_memory_status("antes do carregamento")
        
        X_train, X_test, y_train, y_test = load_binary_data(test_mode=test_mode)
        
        log_memory_status("após carregamento")
        
        # Obter configurações do algoritmo
        all_configs = get_algorithm_configs(test_mode=test_mode)
        
        if algorithm_name not in all_configs:
            available = list(all_configs.keys())
            raise ValueError(f"Algoritmo '{algorithm_name}' não encontrado. Disponíveis: {available}")
        
        algorithm_config = all_configs[algorithm_name]
        
        # Compatibilidade: verificar se usa 'params' ou 'param_combinations'
        if 'param_combinations' in algorithm_config:
            param_combinations = algorithm_config['param_combinations']
        elif 'params' in algorithm_config:
            param_combinations = algorithm_config['params']
        else:
            raise ValueError(f"Algoritmo {algorithm_name} não tem configurações de parâmetros definidas")
            
        n_runs = algorithm_config.get('n_runs', 2 if test_mode else 5)
        is_anomaly_detection = algorithm_config.get('anomaly_detection', False)
        
        logger.info(f"🎯 Algoritmo: {algorithm_name}")
        logger.info(f"📋 Configurações: {len(param_combinations)}")
        logger.info(f"🔄 Execuções: {n_runs}")
        logger.info(f"🔍 Detecção de anomalia: {is_anomaly_detection}")
        
        # Preparar diretórios de saída baseados no modo
        base_results_dir = 'experiments/results_test' if test_mode else 'experiments/results'
        results_dir = Path(base_results_dir) / algorithm_name.lower().replace(' ', '_')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Salvando resultados em: {results_dir}")
        
        # Executar experimentos
        all_results = []
        total_experiments = len(param_combinations) * n_runs
        
        for config_idx, params in enumerate(param_combinations):
            logger.info(f"🔧 Config {config_idx + 1}/{len(param_combinations)}: {params}")
            
            for run_idx in range(n_runs):
                experiment_num = config_idx * n_runs + run_idx + 1
                logger.info(f"   📋 Run {run_idx + 1}/{n_runs} ({experiment_num}/{total_experiments})")
                
                result = run_single_experiment(
                    algorithm_name=algorithm_name,
                    algorithm_class=algorithm_config['class'],
                    params=params,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    is_anomaly_detection=is_anomaly_detection,
                    run_id=run_idx,
                    param_id=config_idx
                )
                
                if result:
                    result['config_index'] = config_idx
                    result['run_index'] = run_idx
                    result['algorithm'] = algorithm_name
                    result['test_mode'] = test_mode  # Adicionar flag do modo
                    all_results.append(result)
                
                gc.collect()
        
        # Salvar resultados
        logger.info(f"💾 Salvando {len(all_results)} resultados...")
        
        results_file = results_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Salvar sumário
        if all_results:
            import pandas as pd
            df_results = pd.DataFrame(all_results)
            summary = {
                'algorithm': algorithm_name,
                'total_experiments': len(all_results),
                'configurations': len(param_combinations),
                'runs_per_config': n_runs,
                'execution_time': time.time() - start_time,
                'best_accuracy': float(df_results['accuracy'].max()),
                'mean_accuracy': float(df_results['accuracy'].mean()),
                'best_f1': float(df_results['f1_score'].max()),
                'mean_f1': float(df_results['f1_score'].mean()),
                'test_mode': test_mode,  # Adicionar flag do modo
                'results_path': str(results_dir),
                'timestamp': time.time()
            }
            
            summary_file = results_dir / 'summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ Resultados salvos: {results_dir}")
            logger.info(f"🎯 Melhor Accuracy: {summary['best_accuracy']:.4f}")
            logger.info(f"🎯 Melhor F1: {summary['best_f1']:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"🏁 CONCLUÍDO EM {total_time:.2f}s")
        
        return len(all_results)
        
    except Exception as e:
        logger.error(f"❌ ERRO: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        gc.collect()

def main():
    """Função principal"""
    # Mapeamento de nomes
    algorithm_map = {
        'isolation_forest': 'IsolationForest',
        'one_class_svm': 'OneClassSVM',
        'local_outlier_factor': 'LocalOutlierFactor',
        'elliptic_envelope': 'EllipticEnvelope',
        'random_forest': 'RandomForestClassifier',
        'logistic_regression': 'LogisticRegression',
        'gradient_boosting': 'GradientBoostingClassifier',
        'svc': 'SVC',
        'mlp': 'MLPClassifier'
    }
    
    if len(sys.argv) != 2:
        print("❌ Uso: python3 run_single_algorithm.py <algorithm_key>")
        print("🎯 Algoritmos:", list(algorithm_map.keys()))
        sys.exit(1)
    
    algorithm_key = sys.argv[1]
    
    if algorithm_key not in algorithm_map:
        print(f"❌ Algoritmo '{algorithm_key}' não encontrado!")
        print("🎯 Disponíveis:", list(algorithm_map.keys()))
        sys.exit(1)
    
    algorithm_name = algorithm_map[algorithm_key]
    
    # Usar variável de ambiente se definida, senão usar TEST_MODE global
    env_test_mode = os.getenv('DVC_TEST_MODE', '').lower()
    if env_test_mode in ['true', '1', 'yes']:
        test_mode = True
    elif env_test_mode in ['false', '0', 'no']:
        test_mode = False
    else:
        # Se variável não definida ou inválida, usar TEST_MODE global
        test_mode = TEST_MODE
    
    mode_str = 'TESTE' if test_mode else 'COMPLETO'
    results_folder = 'results_test' if test_mode else 'results'
    logs_folder = 'logs_test' if test_mode else 'logs'
    
    print(f"🧪 Modo de execução: {mode_str}")
    print(f"📁 Resultados em: experiments/{results_folder}/")
    print(f"📄 Logs em: experiments/{logs_folder}/")
    
    try:
        results_count = run_algorithm_experiments(algorithm_name, test_mode=test_mode)
        print(f"✅ Sucesso: {results_count} experimentos para {algorithm_name}")
        print(f"💾 Resultados salvos em: experiments/{results_folder}/{algorithm_name.lower().replace(' ', '_')}/")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()