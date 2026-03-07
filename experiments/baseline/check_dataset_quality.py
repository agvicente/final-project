#!/usr/bin/env python3
"""
Script para verificar qualidade do dataset CiCIoT
Verifica: missing values, infinite values e duplicated rows
Processa 63 arquivos CSV um de cada vez para economizar memória
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import yaml

class DatasetQualityChecker:
    def __init__(self, data_dir, output_dir):
        """
        Inicializa o verificador de qualidade do dataset
        
        Args:
            data_dir (str): Diretório contendo os arquivos CSV
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.results = {
            'summary': {},
            'per_file': {},
            'global_duplicates': set(),
            'total_rows': 0,
            'columns': None,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_csv_files(self):
        """Retorna lista de arquivos CSV ordenados"""
        pattern = os.path.join(self.data_dir, "Merged*.csv")
        files = sorted(glob.glob(pattern))
        return files
    
    def check_missing_values(self, df, filename):
        """Verifica valores ausentes no dataframe"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'missing_by_column': {
                col: {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }
                for col in df.columns if missing_counts[col] > 0
            }
        }
    
    def check_infinite_values(self, df, filename):
        """Verifica valores infinitos no dataframe"""
        infinite_counts = {}
        total_infinite = 0
        
        # Verifica apenas colunas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = {
                    'count': int(inf_count),
                    'percentage': float((inf_count / len(df)) * 100)
                }
                total_infinite += inf_count
        
        return {
            'total_infinite': int(total_infinite),
            'infinite_by_column': infinite_counts
        }
    
    def check_duplicated_rows(self, df, filename):
        """Verifica linhas duplicadas no dataframe"""
        duplicated_mask = df.duplicated()
        duplicated_count = duplicated_mask.sum()
        
        # Para verificação global de duplicatas, criamos hashes das linhas
        # Isso permite verificar duplicatas entre arquivos diferentes
        if duplicated_count > 0:
            # Converte linhas duplicadas em hashes para comparação global
            duplicated_rows = df[duplicated_mask]
            for idx, row in duplicated_rows.iterrows():
                row_hash = hash(tuple(row.values))
                self.results['global_duplicates'].add(row_hash)
        
        return {
            'duplicated_count': int(duplicated_count),
            'duplicated_percentage': float((duplicated_count / len(df)) * 100)
        }
    
    def process_file(self, filepath):
        """Processa um único arquivo CSV"""
        filename = os.path.basename(filepath)
        print(f"Processando {filename}...")
        
        try:
            # Carrega o arquivo
            df = pd.read_csv(filepath, low_memory=False)
            
            # Primeira verificação das colunas
            if self.results['columns'] is None:
                self.results['columns'] = list(df.columns)
            
            # Atualiza contagem total de linhas
            self.results['total_rows'] += len(df)
            
            # Executa as verificações
            file_results = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': self.check_missing_values(df, filename),
                'infinite_values': self.check_infinite_values(df, filename),
                'duplicated_rows': self.check_duplicated_rows(df, filename)
            }
            
            self.results['per_file'][filename] = file_results
            
            # Limpa o dataframe da memória
            del df
            
            return True
            
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
            self.results['per_file'][filename] = {'error': str(e)}
            return False
    
    def consolidate_results(self):
        """Consolida os resultados de todos os arquivos"""
        total_missing = 0
        total_infinite = 0
        total_duplicated = 0
        missing_by_column = {}
        infinite_by_column = {}
        
        # Processa resultados de cada arquivo
        for filename, file_results in self.results['per_file'].items():
            if 'error' in file_results:
                continue
                
            # Soma totais
            total_missing += file_results['missing_values']['total_missing']
            total_infinite += file_results['infinite_values']['total_infinite']
            total_duplicated += file_results['duplicated_rows']['duplicated_count']
            
            # Consolida por coluna
            for col, info in file_results['missing_values']['missing_by_column'].items():
                if col not in missing_by_column:
                    missing_by_column[col] = {'count': 0, 'files': []}
                missing_by_column[col]['count'] += info['count']
                missing_by_column[col]['files'].append(filename)
            
            for col, info in file_results['infinite_values']['infinite_by_column'].items():
                if col not in infinite_by_column:
                    infinite_by_column[col] = {'count': 0, 'files': []}
                infinite_by_column[col]['count'] += info['count']
                infinite_by_column[col]['files'].append(filename)
        
        # Calcula percentuais globais
        self.results['summary'] = {
            'total_files_processed': len([f for f in self.results['per_file'].values() if 'error' not in f]),
            'total_files_with_errors': len([f for f in self.results['per_file'].values() if 'error' in f]),
            'total_rows': self.results['total_rows'],
            'total_columns': len(self.results['columns']) if self.results['columns'] else 0,
            'missing_values': {
                'total_missing': total_missing,
                'percentage_of_total': (total_missing / self.results['total_rows'] * 100) if self.results['total_rows'] > 0 else 0,
                'columns_with_missing': len(missing_by_column),
                'missing_by_column': {
                    col: {
                        'count': info['count'],
                        'percentage': (info['count'] / self.results['total_rows'] * 100),
                        'files_affected': len(info['files'])
                    }
                    for col, info in missing_by_column.items()
                }
            },
            'infinite_values': {
                'total_infinite': total_infinite,
                'percentage_of_total': (total_infinite / self.results['total_rows'] * 100) if self.results['total_rows'] > 0 else 0,
                'columns_with_infinite': len(infinite_by_column),
                'infinite_by_column': {
                    col: {
                        'count': info['count'],
                        'percentage': (info['count'] / self.results['total_rows'] * 100),
                        'files_affected': len(info['files'])
                    }
                    for col, info in infinite_by_column.items()
                }
            },
            'duplicated_rows': {
                'total_duplicated_within_files': total_duplicated,
                'percentage_within_files': (total_duplicated / self.results['total_rows'] * 100) if self.results['total_rows'] > 0 else 0,
                'potential_global_duplicates': len(self.results['global_duplicates'])
            }
        }
    
    def save_results(self, output_file):
        """Salva os resultados em arquivo JSON"""
        # Converte set para list para serialização JSON
        results_copy = self.results.copy()
        results_copy['global_duplicates'] = list(results_copy['global_duplicates'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Imprime um resumo dos resultados"""
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("RELATÓRIO DE QUALIDADE DO DATASET CICIOT")
        print("="*60)
        
        print(f"\n📊 INFORMAÇÕES GERAIS:")
        print(f"  • Total de arquivos processados: {summary['total_files_processed']}")
        print(f"  • Arquivos com erro: {summary['total_files_with_errors']}")
        print(f"  • Total de linhas: {summary['total_rows']:,}")
        print(f"  • Total de colunas: {summary['total_columns']}")
        
        print(f"\n❌ VALORES AUSENTES (MISSING VALUES):")
        mv = summary['missing_values']
        print(f"  • Total de valores ausentes: {mv['total_missing']:,}")
        print(f"  • Percentual do dataset: {mv['percentage_of_total']:.2f}%")
        print(f"  • Colunas afetadas: {mv['columns_with_missing']}")
        
        if mv['columns_with_missing'] > 0:
            print(f"  • Top 5 colunas com mais missing values:")
            sorted_cols = sorted(mv['missing_by_column'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)
            for col, info in sorted_cols[:5]:
                print(f"    - {col}: {info['count']:,} ({info['percentage']:.2f}%)")
        
        print(f"\n♾️  VALORES INFINITOS (INFINITE VALUES):")
        iv = summary['infinite_values']
        print(f"  • Total de valores infinitos: {iv['total_infinite']:,}")
        print(f"  • Percentual do dataset: {iv['percentage_of_total']:.2f}%")
        print(f"  • Colunas afetadas: {iv['columns_with_infinite']}")
        
        if iv['columns_with_infinite'] > 0:
            print(f"  • Colunas com valores infinitos:")
            for col, info in iv['infinite_by_column'].items():
                print(f"    - {col}: {info['count']:,} ({info['percentage']:.2f}%)")
        
        print(f"\n🔄 LINHAS DUPLICADAS (DUPLICATED ROWS):")
        dr = summary['duplicated_rows']
        print(f"  • Total de duplicadas dentro dos arquivos: {dr['total_duplicated_within_files']:,}")
        print(f"  • Percentual do dataset: {dr['percentage_within_files']:.2f}%")
        print(f"  • Possíveis duplicadas globais: {dr['potential_global_duplicates']:,}")
        
        print("\n" + "="*60)
    
    def run_analysis(self, output_file):
        """Executa a análise completa do dataset"""
        print("Iniciando análise de qualidade do dataset CiCIoT...")
        
        # Obtém lista de arquivos
        csv_files = self.get_csv_files()
        
        if not csv_files:
            print(f"Nenhum arquivo CSV encontrado em {self.data_dir}")
            return
        
        print(f"Encontrados {len(csv_files)} arquivos CSV para processar")
        
        # Processa cada arquivo
        for filepath in tqdm(csv_files, desc="Processando arquivos"):
            self.process_file(filepath)
        
        # Consolida resultados
        print("\nConsolidando resultados...")
        self.consolidate_results()
        
        # Salva resultados
        self.save_results(output_file)
        print(f"Resultados salvos em: {output_file}")
        
        # Imprime resumo
        self.print_summary()