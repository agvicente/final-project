#!/usr/bin/env python3
"""
Script simples para executar verificação de qualidade do dataset CiCIoT
"""

import sys
import os

# Adiciona o diretório atual ao path
sys.path.append('.')

from check_dataset_quality import DatasetQualityChecker

def main(config):
    print("🔍 Verificação de Qualidade do Dataset CiCIoT")
    print("=" * 50)
    
    # Diretório com os arquivos CSV
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    
    # Verifica se o diretório existe
    if not os.path.exists(data_dir):
        print(f"❌ Erro: Diretório {data_dir} não encontrado!")
        print("Execute este script a partir do diretório raiz do projeto iot-ids-research")
        return 1
    
    try:
        # Cria o verificador e executa a análise
        checker = DatasetQualityChecker(data_dir, output_dir)
        checker.run_analysis(output_file=f'{output_dir}/quality_check.json')
        print("\n✅ Análise concluída com sucesso!")
        print("📄 Relatório detalhado salvo em: quality_check.json")
        return 0
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        return 1

def load_config():
    config = {
        'data_dir': 'data/raw/CSV/MERGED_CSV',
        'output_dir': 'data/metrics',
    }
    return config


if __name__ == "__main__":
    config = load_config()
    exit_code = main(config)
    sys.exit(exit_code)