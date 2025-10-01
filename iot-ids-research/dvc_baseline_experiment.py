#!/usr/bin/env python3
"""
Arquivo DVC para executar experimentos de baseline com comparação de algoritmos de ML

Este script é usado no pipeline DVC para executar experimentos sistemáticos
de comparação entre diferentes algoritmos de Machine Learning para detecção
de anomalias em dados de IoT.

Data: 2025
"""

import os
import sys
import logging
import yaml
import time
from pathlib import Path

# Configurar logging
# Criar diretório de logs se não existir
os.makedirs('experiments/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/dvc_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_experiment_environment():
    """
    Configura o ambiente para execução dos experimentos
    """
    logger.info("🔧 Configurando ambiente de experimentos...")
    
    # Criar diretórios necessários
    dirs_to_create = [
        'experiments/results',
        'experiments/logs',
        'experiments/models',
        'experiments/artifacts'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"   ✅ Diretório criado/verificado: {dir_path}")
    
    # Verificar se os dados preprocessados existem
    required_files = [
        'data/processed/binary/X_train_binary.npy',
        'data/processed/binary/X_test_binary.npy', 
        'data/processed/binary/y_train_binary.npy',
        'data/processed/binary/y_test_binary.npy',
        'data/processed/binary/binary_metadata.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Arquivos de dados necessários não encontrados:")
        for file_path in missing_files:
            logger.error(f"   - {file_path}")
        logger.error("Execute primeiro o pipeline de preprocessing (dvc repro preprocess)")
        sys.exit(1)
    
    logger.info("✅ Ambiente configurado com sucesso!")
    return True

def load_experiment_config():
    """
    Carrega configurações de experimento
    """
    logger.info("📄 Carregando configurações de experimento...")
    
    # Configurações padrão
    config = {
        'experiment': {
            'name': 'baseline_comparison',
            'version': '1.0',
            'test_mode': False,  # Mudar para False para experimento completo
            'description': 'Comparação de algoritmos de ML para detecção de anomalias em IoT'
        },
        'data': {
            'binary_data_dir': 'data/processed/binary/',
            'sample_size_test': 1000,
            'sample_size_full': None
        },
        'algorithms': {
            'n_runs': 1,  # Para teste rápido
            'n_runs_full': 5,  # Para experimento completo
            'random_state': 42
        },
        'mlflow': {
            'tracking_uri': 'http://127.0.0.1:5000',
            'experiment_prefix': 'IoT-IDS-Baseline'
        },
        'output': {
            'results_dir': 'experiments/results',
            'models_dir': 'experiments/models',
            'artifacts_dir': 'experiments/artifacts'
        }
    }
    
    # Verificar se existe arquivo de configuração personalizado
    config_file = 'configs/experiment_config.yaml'
    if os.path.exists(config_file):
        logger.info(f"   📁 Carregando configuração personalizada: {config_file}")
        try:
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
            
            # Merge das configurações
            def deep_merge(base, custom):
                for key, value in custom.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
            
            deep_merge(config, custom_config)
            logger.info("   ✅ Configuração personalizada carregada")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao carregar config personalizada: {e}")
            logger.info("   🔄 Usando configurações padrão")
    else:
        logger.info("   🔄 Usando configurações padrão")
        
        # Criar arquivo de configuração padrão
        os.makedirs('configs', exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"   💾 Arquivo de configuração padrão criado: {config_file}")
    
    return config

def run_experiments(config):
    """
    Executa os experimentos de comparação de algoritmos
    """
    logger.info("🚀 Iniciando execução dos experimentos...")
    
    # Importar o módulo de comparação de algoritmos
    sys.path.insert(0, 'experiments')
    
    try:
        from algorithm_comparison import (
            compare_algorithms, 
            save_results_and_plots,
            TEST_MODE,
            SAMPLE_SIZE,
            N_RUNS
        )
        
        logger.info("   ✅ Módulo de comparação importado com sucesso")
        
        # Configurar modo de execução baseado na configuração
        test_mode = config['experiment']['test_mode']
        logger.info(f"   🔧 Modo de execução: {'TESTE' if test_mode else 'COMPLETO'}")
        
        # Atualizar variáveis globais se necessário
        if not test_mode:
            # Para modo completo, sobrescrever as configurações
            import algorithm_comparison
            algorithm_comparison.TEST_MODE = True
            algorithm_comparison.SAMPLE_SIZE = config['data']['sample_size_full']
            algorithm_comparison.N_RUNS = config['algorithms']['n_runs_full']
            logger.info("   🔄 Configurações atualizadas para modo teste")
        
        # Executar experimentos
        logger.info("   🔬 Executando comparação de algoritmos...")
        start_time = time.time()
        
        results = compare_algorithms()
        
        execution_time = time.time() - start_time
        logger.info(f"   ⏱️ Tempo total de execução: {execution_time:.2f}s")
        
        # Salvar resultados e gerar visualizações
        logger.info("   📊 Salvando resultados e gerando visualizações...")
        results_file = save_results_and_plots(results, config['output']['results_dir'])
        
        # Log estatísticas finais
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        logger.info(f"   ✅ Experimentos bem-sucedidos: {len(successful_results)}")
        logger.info(f"   ❌ Experimentos com falha: {len(failed_results)}")
        
        if failed_results:
            logger.warning("   ⚠️ Experimentos que falharam:")
            for result in failed_results:
                logger.warning(f"      - {result['algorithm']}: {result['error']}")
        
        logger.info(f"   📁 Resultados salvos em: {results_file}")
        
        return {
            'success': True,
            'results_file': results_file,
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results),
            'execution_time': execution_time
        }
        
    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulo de comparação: {e}")
        logger.error("Verifique se o arquivo experiments/algorithm_comparison.py existe e está correto")
        return {'success': False, 'error': str(e)}
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução dos experimentos: {e}")
        return {'success': False, 'error': str(e)}

def save_experiment_metadata(config, results):
    """
    Salva metadados do experimento
    """
    logger.info("💾 Salvando metadados do experimento...")
    
    metadata = {
        'experiment_info': {
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': config['experiment']['version'],
            'mode': 'TEST' if config['experiment']['test_mode'] else 'FULL'
        },
        'config': config,
        'results': results,
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        }
    }
    
    metadata_file = os.path.join(
        config['output']['artifacts_dir'], 
        f"experiment_metadata_{int(time.time())}.yaml"
    )
    
    try:
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)
        
        logger.info(f"   ✅ Metadados salvos: {metadata_file}")
        return metadata_file
        
    except Exception as e:
        logger.error(f"   ❌ Erro ao salvar metadados: {e}")
        return None

def main():
    """
    Função principal do script DVC
    """
    logger.info("="*80)
    logger.info("🧪 INICIANDO EXPERIMENTOS DE BASELINE - IoT IDS")
    logger.info("="*80)
    
    try:
        # 1. Configurar ambiente
        setup_experiment_environment()
        
        # 2. Carregar configurações
        config = load_experiment_config()
        
        # 3. Executar experimentos
        results = run_experiments(config)
        
        # 4. Salvar metadados
        metadata_file = save_experiment_metadata(config, results)
        
        # 5. Resultado final
        if results['success']:
            logger.info("="*80)
            logger.info("✅ EXPERIMENTOS CONCLUÍDOS COM SUCESSO!")
            logger.info(f"📊 Total de experimentos: {results['total_experiments']}")
            logger.info(f"✅ Bem-sucedidos: {results['successful_experiments']}")
            logger.info(f"❌ Com falha: {results['failed_experiments']}")
            logger.info(f"⏱️ Tempo total: {results['execution_time']:.2f}s")
            logger.info(f"📁 Resultados: {results['results_file']}")
            if metadata_file:
                logger.info(f"📄 Metadados: {metadata_file}")
            logger.info("="*80)
            return 0
        else:
            logger.error("="*80)
            logger.error("❌ EXPERIMENTOS FALHARAM!")
            logger.error(f"Erro: {results.get('error', 'Erro desconhecido')}")
            logger.error("="*80)
            return 1
            
    except Exception as e:
        logger.error("="*80)
        logger.error("💥 ERRO CRÍTICO NO SCRIPT DVC!")
        logger.error(f"Erro: {str(e)}")
        logger.error("="*80)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
