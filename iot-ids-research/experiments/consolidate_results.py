#!/usr/bin/env python3
"""
Consolidador de resultados de todos os experimentos
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os

def load_all_results():
    """Carrega resultados de todos os algoritmos"""
    results_base = Path("experiments/results")
    all_results = []
    algorithms = []
    
    for algo_dir in results_base.iterdir():
        if algo_dir.is_dir():
            summary_file = algo_dir / "summary.json"
            results_file = algo_dir / "results.json"
            
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    algorithms.append(summary['algorithm'])
                    all_results.append(summary)
            
            # Carregar resultados detalhados se disponível
            if results_file.exists():
                with open(results_file) as f:
                    detailed_results = json.load(f)
                    if detailed_results:
                        # Adicionar dados detalhados ao último sumário
                        if all_results:
                            all_results[-1]['detailed_results'] = detailed_results
    
    return all_results, algorithms

def generate_comparison_plots(df, plots_dir):
    """Gera gráficos comparativos"""
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='algorithm', y='best_accuracy')
    plt.title('Comparação de Accuracy entre Algoritmos', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1-Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='algorithm', y='best_f1')
    plt.title('Comparação de F1-Score entre Algoritmos', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel('Best F1-Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy vs F1 Scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['best_accuracy'], df['best_f1'], 
                         s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    
    # Adicionar labels para cada ponto
    for i, row in df.iterrows():
        plt.annotate(row['algorithm'], 
                    (row['best_accuracy'], row['best_f1']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('Best Accuracy', fontsize=12)
    plt.ylabel('Best F1-Score', fontsize=12)
    plt.title('Accuracy vs F1-Score por Algoritmo', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_vs_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Execution Time Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='algorithm', y='execution_time')
    plt.title('Tempo de Execução por Algoritmo', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel('Tempo de Execução (s)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'execution_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(df, tables_dir):
    """Gera tabelas resumo"""
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Tabela principal
    summary_table = df[['algorithm', 'best_accuracy', 'mean_accuracy', 
                       'best_f1', 'mean_f1', 'execution_time', 'total_experiments']].copy()
    
    # Formatar números
    summary_table['best_accuracy'] = summary_table['best_accuracy'].round(4)
    summary_table['mean_accuracy'] = summary_table['mean_accuracy'].round(4)
    summary_table['best_f1'] = summary_table['best_f1'].round(4)
    summary_table['mean_f1'] = summary_table['mean_f1'].round(4)
    summary_table['execution_time'] = summary_table['execution_time'].round(2)
    
    # Salvar CSV
    summary_table.to_csv(tables_dir / 'summary_table.csv', index=False)
    
    # Gerar tabela formatada em markdown
    markdown_table = summary_table.to_markdown(index=False, floatfmt='.4f')
    
    with open(tables_dir / 'summary_table.md', 'w') as f:
        f.write("# 📊 Tabela Resumo dos Experimentos\n\n")
        f.write(markdown_table)
        f.write("\n")

def generate_final_report(df, algorithms, report_dir):
    """Gera relatório final"""
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar melhores algoritmos
    best_accuracy_idx = df['best_accuracy'].idxmax()
    best_f1_idx = df['best_f1'].idxmax()
    fastest_idx = df['execution_time'].idxmin()
    
    best_accuracy_algo = df.loc[best_accuracy_idx]
    best_f1_algo = df.loc[best_f1_idx]
    fastest_algo = df.loc[fastest_idx]
    
    report_content = f"""# 📊 Relatório Final de Experimentos - IoT Anomaly Detection

## 🎯 Resumo Executivo

- **Total de Algoritmos Testados**: {len(algorithms)}
- **Total de Experimentos**: {df['total_experiments'].sum()}
- **Tempo Total de Execução**: {df['execution_time'].sum():.2f} segundos

## 🏆 Melhores Resultados

### 🎯 Melhor Accuracy
- **Algoritmo**: {best_accuracy_algo['algorithm']}
- **Accuracy**: {best_accuracy_algo['best_accuracy']:.4f}
- **F1-Score**: {best_accuracy_algo['best_f1']:.4f}
- **Tempo**: {best_accuracy_algo['execution_time']:.2f}s

### 🎯 Melhor F1-Score
- **Algoritmo**: {best_f1_algo['algorithm']}
- **F1-Score**: {best_f1_algo['best_f1']:.4f}  
- **Accuracy**: {best_f1_algo['best_accuracy']:.4f}
- **Tempo**: {best_f1_algo['execution_time']:.2f}s

### ⚡ Mais Rápido
- **Algoritmo**: {fastest_algo['algorithm']}
- **Tempo**: {fastest_algo['execution_time']:.2f}s
- **Accuracy**: {fastest_algo['best_accuracy']:.4f}
- **F1-Score**: {fastest_algo['best_f1']:.4f}

## 📋 Resultados Detalhados

| Algoritmo | Best Accuracy | Mean Accuracy | Best F1 | Mean F1 | Tempo (s) | Experimentos |
|-----------|---------------|---------------|---------|---------|-----------|--------------|
"""
    
    for _, row in df.iterrows():
        report_content += f"| {row['algorithm']} | {row['best_accuracy']:.4f} | {row['mean_accuracy']:.4f} | {row['best_f1']:.4f} | {row['mean_f1']:.4f} | {row['execution_time']:.1f} | {row['total_experiments']} |\n"
    
    report_content += f"""
## 📊 Análise Estatística

- **Média de Accuracy**: {df['best_accuracy'].mean():.4f}
- **Desvio Padrão Accuracy**: {df['best_accuracy'].std():.4f}
- **Média de F1-Score**: {df['best_f1'].mean():.4f}
- **Desvio Padrão F1-Score**: {df['best_f1'].std():.4f}

## 🔧 Configuração dos Experimentos

- **Configurações por Algoritmo**: {df['configurations'].mean():.1f} (média)
- **Execuções por Configuração**: {df['runs_per_config'].mean():.1f} (média)
- **Tempo Médio por Algoritmo**: {df['execution_time'].mean():.2f}s

---
*Relatório gerado automaticamente pelo pipeline DVC de experimentos de detecção de anomalias em IoT*
"""
    
    with open(report_dir / 'final_report.md', 'w') as f:
        f.write(report_content)
    
    # Salvar também em JSON para uso programático
    report_json = {
        'summary': {
            'total_algorithms': len(algorithms),
            'total_experiments': int(df['total_experiments'].sum()),
            'total_execution_time': float(df['execution_time'].sum()),
        },
        'best_results': {
            'best_accuracy': {
                'algorithm': best_accuracy_algo['algorithm'],
                'value': float(best_accuracy_algo['best_accuracy'])
            },
            'best_f1': {
                'algorithm': best_f1_algo['algorithm'],
                'value': float(best_f1_algo['best_f1'])
            },
            'fastest': {
                'algorithm': fastest_algo['algorithm'],
                'time': float(fastest_algo['execution_time'])
            }
        },
        'statistics': {
            'mean_accuracy': float(df['best_accuracy'].mean()),
            'std_accuracy': float(df['best_accuracy'].std()),
            'mean_f1': float(df['best_f1'].mean()),
            'std_f1': float(df['best_f1'].std())
        }
    }
    
    with open(report_dir / 'final_report.json', 'w') as f:
        json.dump(report_json, f, indent=2)

def consolidate_all_results():
    """Função principal de consolidação"""
    print("📊 Iniciando consolidação de resultados...")
    
    # Carregar todos os resultados
    all_results, algorithms = load_all_results()
    
    if not all_results:
        print("❌ Nenhum resultado encontrado!")
        return False
    
    print(f"✅ Carregados resultados de {len(algorithms)} algoritmos")
    
    # Criar DataFrame
    df = pd.DataFrame(all_results)
    
    # Criar diretórios de saída
    final_plots_dir = Path("experiments/final_plots")
    final_tables_dir = Path("experiments/final_tables")
    final_report_dir = Path("experiments/final_report")
    final_results_dir = Path("experiments/final_results")
    
    for dir_path in [final_plots_dir, final_tables_dir, final_report_dir, final_results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Gerar outputs
    print("📈 Gerando gráficos...")
    generate_comparison_plots(df, final_plots_dir)
    
    print("📋 Gerando tabelas...")
    generate_summary_table(df, final_tables_dir)
    
    print("📄 Gerando relatório...")
    generate_final_report(df, algorithms, final_report_dir)
    
    # Salvar DataFrame consolidado
    df.to_csv(final_results_dir / 'consolidated_results.csv', index=False)
    df.to_json(final_results_dir / 'consolidated_results.json', orient='records', indent=2)
    
    print(f"✅ Consolidação completa!")
    print(f"   - Algoritmos processados: {len(algorithms)}")
    print(f"   - Total de experimentos: {df['total_experiments'].sum()}")
    print(f"   - Melhor Accuracy: {df['best_accuracy'].max():.4f} ({df.loc[df['best_accuracy'].idxmax(), 'algorithm']})")
    print(f"   - Melhor F1-Score: {df['best_f1'].max():.4f} ({df.loc[df['best_f1'].idxmax(), 'algorithm']})")
    
    return True

if __name__ == "__main__":
    try:
        success = consolidate_all_results()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Erro na consolidação: {str(e)}")
        exit(1)
