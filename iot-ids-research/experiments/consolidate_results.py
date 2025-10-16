#!/usr/bin/env python3
"""
Consolidador de resultados com análises detalhadas e avançadas
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from iot_advanced_plots import generate_all_iot_advanced_plots

def get_shared_timestamp():
    """
    Obtém o timestamp compartilhado da rodada atual.
    
    Returns:
        str: Timestamp Unix ou None se não existir
    """
    timestamp_file = Path('experiments/.current_run_timestamp')
    
    if timestamp_file.exists():
        try:
            with open(timestamp_file, 'r') as f:
                timestamp = f.read().strip()
                return timestamp if timestamp else None
        except:
            return None
    else:
        return None

def cleanup_shared_timestamp():
    """Remove o arquivo de timestamp compartilhado após consolidação"""
    timestamp_file = Path('experiments/.current_run_timestamp')
    if timestamp_file.exists():
        timestamp_file.unlink()

def load_all_results(test_mode=None):
    """Carrega resultados de todos os algoritmos com dados detalhados"""
    
    # Se test_mode não especificado, importar TEST_MODE do algorithm_comparison
    if test_mode is None:
        try:
            import sys
            sys.path.append('.')
            from experiments.algorithm_comparison import TEST_MODE
            test_mode = TEST_MODE
            print(f"📋 Usando TEST_MODE do algorithm_comparison.py: {test_mode}")
        except ImportError:
            # Fallback: Auto-detectar baseado em qual subpasta tem mais resultados
            test_results_count = 0
            full_results_count = 0
            
            test_base = Path("experiments/results/test")
            full_base = Path("experiments/results/full")
            
            if test_base.exists():
                test_results_count = len([d for d in test_base.iterdir() if d.is_dir()])
            if full_base.exists():
                full_results_count = len([d for d in full_base.iterdir() if d.is_dir()])
            
            if test_results_count > full_results_count:
                test_mode = True
                print(f"🧪 Auto-detectado: MODO TESTE ({test_results_count} algoritmos)")
            else:
                test_mode = False
                print(f"🚀 Auto-detectado: MODO COMPLETO ({full_results_count} algoritmos)")
    
    # Usar subpasta baseada no modo
    mode_folder = 'test' if test_mode else 'full'
    results_base = Path("experiments/results") / mode_folder
    
    mode_str = "TESTE" if test_mode else "COMPLETO"
    print(f"📁 Carregando resultados do modo: {mode_str}")
    print(f"📂 Diretório: {results_base}")
    
    if not results_base.exists():
        print(f"❌ Diretório de resultados não encontrado: {results_base}")
        return [], [], []
    
    all_summaries = []
    all_detailed_results = []
    algorithms = []
    execution_history = []
    
    # Listar todas as execuções com timestamp (ordenadas por timestamp)
    timestamped_dirs = []
    for algo_dir in results_base.iterdir():
        if algo_dir.is_dir() and '_' in algo_dir.name:
            try:
                # Extrair timestamp do nome da pasta (formato: timestamp_algorithm)
                timestamp_str = algo_dir.name.split('_')[0]
                timestamp = int(timestamp_str)
                timestamped_dirs.append((timestamp, algo_dir))
            except (ValueError, IndexError):
                # Se não conseguir extrair timestamp, usar como está
                timestamped_dirs.append((0, algo_dir))
    
    # Ordenar por timestamp (mais recente primeiro)
    timestamped_dirs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📋 Encontradas {len(timestamped_dirs)} execuções com timestamp")
    
    for timestamp, algo_dir in timestamped_dirs:
        if algo_dir.is_dir():
            summary_file = algo_dir / "summary.json"
            results_file = algo_dir / "results.json"
            
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    # Adicionar informações de timestamp
                    summary['execution_timestamp'] = timestamp
                    summary['execution_folder'] = algo_dir.name
                    
                    algorithms.append(summary['algorithm'])
                    all_summaries.append(summary)
                    
                    # Registrar no histórico
                    execution_history.append({
                        'timestamp': timestamp,
                        'algorithm': summary['algorithm'],
                        'folder': algo_dir.name,
                        'mode': mode_str
                    })
            
            # Carregar resultados detalhados
            if results_file.exists():
                with open(results_file) as f:
                    detailed_results = json.load(f)
                    if detailed_results:
                        # Adicionar timestamp aos resultados detalhados
                        for result in detailed_results:
                            result['execution_timestamp'] = timestamp
                            result['execution_folder'] = algo_dir.name
                        all_detailed_results.extend(detailed_results)
    
    # Mostrar histórico de execuções
    if execution_history:
        print(f"\n📅 HISTÓRICO DE EXECUÇÕES ({mode_str}):")
        for i, exec_info in enumerate(execution_history[:10]):  # Mostrar últimas 10
            from datetime import datetime
            dt = datetime.fromtimestamp(exec_info['timestamp'])
            print(f"   {i+1:2d}. {dt.strftime('%Y-%m-%d %H:%M:%S')} - {exec_info['algorithm']} ({exec_info['folder']})")
        
        if len(execution_history) > 10:
            print(f"   ... e mais {len(execution_history) - 10} execuções anteriores")
    
    print(f"✅ Carregados: {len(set(algorithms))} algoritmos únicos, {len(all_detailed_results)} experimentos, {len(execution_history)} execuções")
    return all_summaries, all_detailed_results, algorithms, execution_history

def generate_confusion_matrices(detailed_df, plots_dir):
    """Gera matrizes de confusão para cada algoritmo"""
    print("   📊 Gerando matrizes de confusão...")
    
    algorithms = detailed_df['algorithm'].unique()
    
    # Criar subplots para todas as matrizes
    n_algos = len(algorithms)
    n_cols = min(3, n_algos)
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_algos == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, algorithm in enumerate(algorithms):
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        # Agregar matriz de confusão (somar todas as execuções)
        total_tn = algo_data['tn'].sum()
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        total_tp = algo_data['tp'].sum()
        
        # Matriz agregada
        cm = np.array([[total_tn, total_fp], 
                       [total_fn, total_tp]])
        
        # Plot individual
        ax = axes[i] if i < len(axes) else plt.subplot(n_rows, n_cols, i+1)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomalia'],
                   yticklabels=['Normal', 'Anomalia'],
                   ax=ax)
        
        ax.set_title(f'{algorithm}\nTotal: {cm.sum()} amostras', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predição')
        ax.set_ylabel('Real')
    
    # Remover subplots vazios
    for i in range(len(algorithms), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_boxplots(detailed_df, plots_dir):
    """Gera boxplots de distribuições das métricas"""
    print("   📦 Gerando boxplots de distribuições...")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'balanced_accuracy' in detailed_df.columns:
        metrics.insert(1, 'balanced_accuracy')
    
    n_rows, n_cols = (2, 3) if len(metrics) == 5 else (2, 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12) if len(metrics) == 5 else (15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=detailed_df, x='algorithm', y=metric, ax=axes[i])
        metric_title = 'Balanced Accuracy' if metric == 'balanced_accuracy' else metric.title()
        axes[i].set_title(f'Distribuição de {metric_title}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Algoritmo', fontsize=10)
        axes[i].set_ylabel(metric_title, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
    
    # Remover subplot extra se tiver 5 métricas em grid 2x3
    if len(metrics) == 5:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_correlation_heatmap(detailed_df, plots_dir):
    """Gera heatmap de correlação entre métricas"""
    print("   🔥 Gerando heatmap de correlação...")
    
    # Selecionar métricas numéricas
    correlation_cols = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 
                       'training_time', 'prediction_time', 'memory_usage_mb']
    
    # Filtrar colunas que existem
    available_cols = [col for col in correlation_cols if col in detailed_df.columns]
    correlation_data = detailed_df[available_cols]
    
    # Calcular matriz de correlação
    corr_matrix = correlation_data.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
    
    plt.title('Correlação entre Métricas de Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_analysis(detailed_df, plots_dir):
    """Gera análises de performance detalhadas"""
    print("   ⚡ Gerando análises de performance...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Tempo de treino vs Accuracy
    axes[0,0].scatter(detailed_df['training_time'], detailed_df['accuracy'], 
                     alpha=0.6, s=50)
    axes[0,0].set_xlabel('Tempo de Treinamento (s)')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('Tempo de Treinamento vs Accuracy')
    axes[0,0].grid(alpha=0.3)
    
    # 2. Memória vs F1-Score
    axes[0,1].scatter(detailed_df['memory_usage_mb'], detailed_df['f1_score'], 
                     alpha=0.6, s=50, c=detailed_df['accuracy'], cmap='viridis')
    axes[0,1].set_xlabel('Uso de Memória (MB)')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].set_title('Uso de Memória vs F1-Score')
    axes[0,1].grid(alpha=0.3)
    
    # 3. Precision vs Recall
    for algorithm in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        axes[1,0].scatter(algo_data['recall'], algo_data['precision'], 
                         label=algorithm, alpha=0.7, s=50)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision vs Recall por Algoritmo')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].grid(alpha=0.3)
    
    # 4. Tempo total por algoritmo
    algo_times = detailed_df.groupby('algorithm')['total_time'].agg(['mean', 'std'])
    axes[1,1].bar(range(len(algo_times)), algo_times['mean'], 
                 yerr=algo_times['std'], capsize=5, alpha=0.7)
    axes[1,1].set_xticks(range(len(algo_times)))
    axes[1,1].set_xticklabels(algo_times.index, rotation=45, ha='right')
    axes[1,1].set_ylabel('Tempo Total (s)')
    axes[1,1].set_title('Tempo Total Médio por Algoritmo')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_parameter_analysis(detailed_df, plots_dir):
    """Gera análises dos parâmetros e seu impacto"""
    print("   🔧 Gerando análises de parâmetros...")
    
    algorithms = detailed_df['algorithm'].unique()
    
    # Análise por algoritmo dos parâmetros
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(12, 4*len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]
    
    for i, algorithm in enumerate(algorithms):
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        # Agrupar por configuração de parâmetros
        param_groups = algo_data.groupby('config_index').agg({
            'f1_score': ['mean', 'std', 'count'],
            'accuracy': 'mean',
            'total_time': 'mean'
        }).round(4)
        
        param_groups.columns = ['f1_mean', 'f1_std', 'n_runs', 'acc_mean', 'time_mean']
        
        # Plot F1-Score por configuração
        x_pos = range(len(param_groups))
        axes[i].bar(x_pos, param_groups['f1_mean'], 
                   yerr=param_groups['f1_std'], capsize=5, alpha=0.7)
        
        axes[i].set_title(f'{algorithm} - F1-Score por Configuração de Parâmetros')
        axes[i].set_xlabel('Configuração de Parâmetros')
        axes[i].set_ylabel('F1-Score Médio')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels([f'Config {j}' for j in range(len(param_groups))])
        axes[i].grid(axis='y', alpha=0.3)
        
        # Adicionar valores no topo das barras
        for j, (mean_val, std_val) in enumerate(zip(param_groups['f1_mean'], param_groups['f1_std'])):
            axes[i].text(j, mean_val + std_val + 0.01, f'{mean_val:.3f}', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_anomaly_detection_analysis(detailed_df, plots_dir):
    """Análise específica para algoritmos de detecção de anomalias"""
    print("   🔍 Gerando análise de detecção de anomalias...")
    
    # Identificar algoritmos de detecção de anomalias
    anomaly_algorithms = []
    supervised_algorithms = []
    
    for algorithm in detailed_df['algorithm'].unique():
        # Assumir que algoritmos com ROC AUC muito baixo são de detecção de anomalias
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        avg_roc = algo_data['roc_auc'].mean() if 'roc_auc' in algo_data.columns else 0.5
        
        if avg_roc < 0.3 or 'Isolation' in algorithm or 'OneClass' in algorithm or 'Outlier' in algorithm or 'Elliptic' in algorithm:
            anomaly_algorithms.append(algorithm)
        else:
            supervised_algorithms.append(algorithm)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Comparação Anomaly vs Supervised
    categories = []
    f1_scores = []
    accuracy_scores = []
    
    if anomaly_algorithms:
        anomaly_data = detailed_df[detailed_df['algorithm'].isin(anomaly_algorithms)]
        categories.append('Detecção de\nAnomalias')
        f1_scores.append(anomaly_data['f1_score'].mean())
        accuracy_scores.append(anomaly_data['accuracy'].mean())
    
    if supervised_algorithms:
        supervised_data = detailed_df[detailed_df['algorithm'].isin(supervised_algorithms)]
        categories.append('Supervisionados')
        f1_scores.append(supervised_data['f1_score'].mean())
        accuracy_scores.append(supervised_data['accuracy'].mean())
    
    x_pos = range(len(categories))
    width = 0.35
    
    axes[0].bar([x - width/2 for x in x_pos], f1_scores, width, label='F1-Score', alpha=0.7)
    axes[0].bar([x + width/2 for x in x_pos], accuracy_scores, width, label='Accuracy', alpha=0.7)
    axes[0].set_xlabel('Categoria de Algoritmo')
    axes[0].set_ylabel('Score Médio')
    axes[0].set_title('Comparação: Detecção de Anomalias vs Supervisionados')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Análise de False Positives vs False Negatives
    algorithms_sample = detailed_df['algorithm'].unique()[:6]  # Limitar para visualização
    fp_rates = []
    fn_rates = []
    algo_names = []
    
    for algorithm in algorithms_sample:
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        total_tn = algo_data['tn'].sum()
        total_tp = algo_data['tp'].sum()
        
        fp_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        fn_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
        
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
        algo_names.append(algorithm)
    
    x_pos = range(len(algo_names))
    axes[1].bar([x - width/2 for x in x_pos], fp_rates, width, label='Taxa de Falsos Positivos', alpha=0.7)
    axes[1].bar([x + width/2 for x in x_pos], fn_rates, width, label='Taxa de Falsos Negativos', alpha=0.7)
    axes[1].set_xlabel('Algoritmo')
    axes[1].set_ylabel('Taxa de Erro')
    axes[1].set_title('Análise de Falsos Positivos vs Falsos Negativos')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(algo_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'anomaly_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_plots(df, plots_dir):
    """Gera gráficos comparativos básicos (mantidos do original)"""
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
    
    # 2. Balanced Accuracy Comparison (se disponível)
    if 'best_balanced_accuracy' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='algorithm', y='best_balanced_accuracy')
        plt.title('Comparação de Balanced Accuracy entre Algoritmos', fontsize=14, fontweight='bold')
        plt.xlabel('Algoritmo', fontsize=12)
        plt.ylabel('Best Balanced Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'balanced_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. F1-Score Comparison
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
    
    # 4. Accuracy vs F1 Scatter
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
    
    # 5. Accuracy vs Balanced Accuracy Scatter (se disponível)
    if 'best_balanced_accuracy' in df.columns:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['best_accuracy'], df['best_balanced_accuracy'], 
                             s=200, alpha=0.7, c=range(len(df)), cmap='viridis',
                             edgecolors='black', linewidth=1.5)
        
        # Adicionar linha diagonal y=x (perfeito balanceamento)
        min_val = min(df['best_accuracy'].min(), df['best_balanced_accuracy'].min())
        max_val = max(df['best_accuracy'].max(), df['best_balanced_accuracy'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='y=x (perfeito)')
        
        # Adicionar labels para cada ponto
        for i, row in df.iterrows():
            plt.annotate(row['algorithm'], 
                        (row['best_accuracy'], row['best_balanced_accuracy']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Best Accuracy', fontsize=12)
        plt.ylabel('Best Balanced Accuracy', fontsize=12)
        plt.title('Accuracy vs Balanced Accuracy por Algoritmo', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_vs_balanced_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Execution Time Comparison
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

def generate_detailed_statistics_table(detailed_df, tables_dir):
    """Gera tabela com estatísticas detalhadas"""
    print("   📋 Gerando tabelas estatísticas detalhadas...")
    
    # Estatísticas por algoritmo
    stats = detailed_df.groupby('algorithm').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'], 
        'recall': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
        'training_time': ['mean', 'std', 'min', 'max'],
        'memory_usage_mb': ['mean', 'std', 'min', 'max'],
        'total_time': 'count'
    }).round(4)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={'total_time_count': 'total_experiments'})
    
    # Salvar estatísticas detalhadas
    stats.to_csv(tables_dir / 'detailed_statistics.csv')
    
    # Gerar tabela markdown com estatísticas principais
    main_stats = detailed_df.groupby('algorithm').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'], 
        'training_time': 'mean',
        'total_time': 'count'
    }).round(4)
    
    main_stats.columns = ['Accuracy_Mean', 'Accuracy_Std', 'F1_Mean', 'F1_Std', 'Training_Time', 'Experiments']
    
    with open(tables_dir / 'detailed_statistics.md', 'w') as f:
        f.write("# 📊 Estatísticas Detalhadas dos Experimentos\n\n")
        f.write("## Resumo por Algoritmo\n\n")
        f.write(main_stats.to_markdown(floatfmt='.4f'))
        f.write("\n\n## Legenda\n")
        f.write("- **Accuracy/F1 Mean**: Valor médio across todas as execuções\n")
        f.write("- **Accuracy/F1 Std**: Desvio padrão (estabilidade)\n")
        f.write("- **Training_Time**: Tempo médio de treinamento (s)\n")
        f.write("- **Experiments**: Total de experimentos executados\n")

def generate_summary_table(df, tables_dir):
    """Gera tabelas resumo (mantida do original)"""
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Tabela principal
    columns = ['algorithm', 'best_accuracy', 'mean_accuracy']
    if 'best_balanced_accuracy' in df.columns:
        columns.extend(['best_balanced_accuracy', 'mean_balanced_accuracy'])
    columns.extend(['best_f1', 'mean_f1', 'execution_time', 'total_experiments'])
    
    summary_table = df[columns].copy()
    
    # Formatar números
    summary_table['best_accuracy'] = summary_table['best_accuracy'].round(4)
    summary_table['mean_accuracy'] = summary_table['mean_accuracy'].round(4)
    if 'best_balanced_accuracy' in summary_table.columns:
        summary_table['best_balanced_accuracy'] = summary_table['best_balanced_accuracy'].round(4)
        summary_table['mean_balanced_accuracy'] = summary_table['mean_balanced_accuracy'].round(4)
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

def generate_final_report(df, algorithms, report_dir, test_mode=False):
    """Gera relatório final melhorado"""
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar melhores algoritmos
    best_accuracy_idx = df['best_accuracy'].idxmax()
    best_f1_idx = df['best_f1'].idxmax()
    fastest_idx = df['execution_time'].idxmin()
    
    best_accuracy_algo = df.loc[best_accuracy_idx]
    best_f1_algo = df.loc[best_f1_idx]
    fastest_algo = df.loc[fastest_idx]
    
    # Análise estatística avançada
    accuracy_cv = df['best_accuracy'].std() / max(df['best_accuracy'].mean(), 0.0001)
    f1_cv = df['best_f1'].std() / max(df['best_f1'].mean(), 0.0001)
    
    report_content = f"""# 📊 Relatório Final de Experimentos - IoT Anomaly Detection {"(MODO TESTE)" if test_mode else "(MODO COMPLETO)"}

## 🎯 Resumo Executivo

- **Modo de Execução**: {"🧪 TESTE (dados reduzidos)" if test_mode else "🚀 COMPLETO (dataset completo)"}
- **Total de Algoritmos Testados**: {len(algorithms)}
- **Total de Experimentos**: {df['total_experiments'].sum()}
- **Tempo Total de Execução**: {df['execution_time'].sum():.2f} segundos ({df['execution_time'].sum()/60:.1f} minutos)
- **Coeficiente de Variação Accuracy**: {accuracy_cv:.3f} ({"baixa" if accuracy_cv < 0.1 else "média" if accuracy_cv < 0.3 else "alta"} variabilidade)
- **Coeficiente de Variação F1-Score**: {f1_cv:.3f} ({"baixa" if f1_cv < 0.1 else "média" if f1_cv < 0.3 else "alta"} variabilidade)

## 🏆 Melhores Resultados

### 🎯 Melhor Accuracy
- **Algoritmo**: {best_accuracy_algo['algorithm']}
- **Accuracy**: {best_accuracy_algo['best_accuracy']:.4f} (±{(best_accuracy_algo['best_accuracy'] - best_accuracy_algo['mean_accuracy']):.4f})
- **F1-Score**: {best_accuracy_algo['best_f1']:.4f}
- **Tempo**: {best_accuracy_algo['execution_time']:.2f}s

### 🎯 Melhor F1-Score
- **Algoritmo**: {best_f1_algo['algorithm']}
- **F1-Score**: {best_f1_algo['best_f1']:.4f} (±{(best_f1_algo['best_f1'] - best_f1_algo['mean_f1']):.4f})
- **Accuracy**: {best_f1_algo['best_accuracy']:.4f}
- **Tempo**: {best_f1_algo['execution_time']:.2f}s

### ⚡ Mais Rápido
- **Algoritmo**: {fastest_algo['algorithm']}
- **Tempo**: {fastest_algo['execution_time']:.2f}s
- **Accuracy**: {fastest_algo['best_accuracy']:.4f}
- **F1-Score**: {fastest_algo['best_f1']:.4f}
- **Eficiência**: {fastest_algo['best_f1']/max(fastest_algo['execution_time'], 0.001):.4f} F1/segundo

## 📋 Resultados Detalhados

| Algoritmo | Best Accuracy | Mean Accuracy | Best F1 | Mean F1 | Tempo (s) | Experimentos | Eficiência |
|-----------|---------------|---------------|---------|---------|-----------|--------------|------------|
"""
    
    for _, row in df.iterrows():
        efficiency = row['best_f1'] / max(row['execution_time'], 0.001)
        report_content += f"| {row['algorithm']} | {row['best_accuracy']:.4f} | {row['mean_accuracy']:.4f} | {row['best_f1']:.4f} | {row['mean_f1']:.4f} | {row['execution_time']:.1f} | {row['total_experiments']} | {efficiency:.4f} |\n"
    
    report_content += f"""
## 📊 Análise Estatística Avançada

### Métricas de Performance
- **Accuracy Média Geral**: {df['best_accuracy'].mean():.4f} ± {df['best_accuracy'].std():.4f}
- **F1-Score Médio Geral**: {df['best_f1'].mean():.4f} ± {df['best_f1'].std():.4f}
- **Algoritmo mais Consistente (menor CV)**: {df.loc[df['best_f1'].index, 'algorithm'].iloc[np.argmin([abs(row['best_f1'] - row['mean_f1'])/max(row['mean_f1'], 0.0001) for _, row in df.iterrows()])]}

### Métricas de Eficiência
- **Tempo Médio por Algoritmo**: {df['execution_time'].mean():.2f}s ± {df['execution_time'].std():.2f}s
- **Total de Experimentos Executados**: {df['total_experiments'].sum()}
- **Experimentos por Minuto**: {df['total_experiments'].sum() / max(df['execution_time'].sum()/60, 0.001):.1f}

### Rankings
1. **Por Performance (F1)**: {', '.join(df.nlargest(3, 'best_f1')['algorithm'].tolist())}
2. **Por Velocidade**: {', '.join(df.nsmallest(3, 'execution_time')['algorithm'].tolist())}
3. **Por Eficiência (F1/tempo)**: {', '.join(df.assign(efficiency=df['best_f1']/df['execution_time'].replace(0, 0.001)).nlargest(3, 'efficiency')['algorithm'].tolist())}

## 🔧 Configuração dos Experimentos

- **Configurações por Algoritmo**: {df['configurations'].mean():.1f} (média)
- **Execuções por Configuração**: {df['runs_per_config'].mean():.1f} (média)
- **Rigor Estatístico**: ✅ Múltiplas execuções (5 runs) para cada configuração
- **Validação**: ✅ Holdout test set independente

### 🎛️ Estratégia Adaptativa de Configurações (Opção C)

**Racional**: O número de configurações varia por algoritmo conforme sua complexidade computacional,
mantendo o tempo total de execução em ~24h e garantindo cobertura abrangente do espaço de hiperparâmetros.

**Distribuição por Complexidade**:
- ⚡ **Algoritmos Rápidos (20 configs)**: LogisticRegression, SGDClassifier
- 🔄 **Algoritmos Médios (12-18 configs)**: RandomForest(12), LinearSVC(18), IsolationForest(15), EllipticEnvelope(15), SGDOneClassSVM(15)
- 🐢 **Algoritmos Pesados (8-10 configs)**: GradientBoosting(10), LocalOutlierFactor(8), MLPClassifier(8)

**Totais**: 141 configurações × 5 runs = 705 experimentos | Tempo estimado: ~30h

**Estratégia de Amostragem**: Cada algoritmo possui configurações organizadas em 4 faixas:
1. **LEVES (20%)**: Modelos muito simples, deployable em edge devices
2. **SWEET SPOT (40%)**: Range ideal para IoT, balanceando performance e recursos
3. **MÉDIAS (20%)**: Configurações moderadas, para edge servers
4. **PESADAS (20%)**: Limite da capacidade IoT, para gateways e fog nodes

**Comparabilidade**: Apesar do número variável, todos os algoritmos são comparáveis pois:
- Utilizam 5 runs cada para rigor estatístico
- Compartilham o mesmo train/test split (random_state=42)
- Incluem configurações leves e pesadas para análise de trade-offs
- Focam no sweet spot IoT (40% das configurações)

## 📈 Gráficos e Análises Geradas

1. **Gráficos Básicos**: Comparações de accuracy, F1-score, tempo de execução
2. **Análises Avançadas**: 
   - 📊 Matrizes de confusão agregadas
   - 📦 Boxplots de distribuições
   - 🔥 Heatmap de correlações
   - ⚡ Análises de performance detalhadas
   - 🔧 Impacto de parâmetros
   - 🔍 Análise específica de detecção de anomalias

## 💡 Recomendações

### Para Produção
- **Melhor Performance**: Use **{best_f1_algo['algorithm']}** (F1: {best_f1_algo['best_f1']:.4f})
- **Melhor Velocidade**: Use **{fastest_algo['algorithm']}** ({fastest_algo['execution_time']:.2f}s)
- **Balanceado**: {'Considere trade-off entre performance e velocidade' if best_f1_algo['algorithm'] != fastest_algo['algorithm'] else f"Use **{best_f1_algo['algorithm']}** (melhor em ambos)"}

### Para Pesquisa
- Investigar parâmetros que causaram maior variabilidade
- Comparar com outros datasets de IoT
- Analisar interpretabilidade dos modelos

---
*Relatório gerado automaticamente pelo pipeline DVC avançado de experimentos de detecção de anomalias em IoT*
*Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_dir / 'final_report.md', 'w') as f:
        f.write(report_content)
    
    # Salvar JSON melhorado
    report_json = {
        'summary': {
            'total_algorithms': len(algorithms),
            'total_experiments': int(df['total_experiments'].sum()),
            'total_execution_time': float(df['execution_time'].sum()),
            'accuracy_cv': float(accuracy_cv),
            'f1_cv': float(f1_cv)
        },
        'best_results': {
            'best_accuracy': {
                'algorithm': best_accuracy_algo['algorithm'],
                'value': float(best_accuracy_algo['best_accuracy']),
                'stability': float(abs(best_accuracy_algo['best_accuracy'] - best_accuracy_algo['mean_accuracy']))
            },
            'best_f1': {
                'algorithm': best_f1_algo['algorithm'],
                'value': float(best_f1_algo['best_f1']),
                'stability': float(abs(best_f1_algo['best_f1'] - best_f1_algo['mean_f1']))
            },
            'fastest': {
                'algorithm': fastest_algo['algorithm'],
                'time': float(fastest_algo['execution_time']),
                'efficiency': float(fastest_algo['best_f1']/max(fastest_algo['execution_time'], 0.001))
            }
        },
        'statistics': {
            'mean_accuracy': float(df['best_accuracy'].mean()),
            'std_accuracy': float(df['best_accuracy'].std()),
            'mean_f1': float(df['best_f1'].mean()),
            'std_f1': float(df['best_f1'].std()),
            'mean_execution_time': float(df['execution_time'].mean()),
            'total_experiments_per_minute': float(df['total_experiments'].sum() / max(df['execution_time'].sum()/60, 0.001))
        },
        'rankings': {
            'by_f1': df.nlargest(5, 'best_f1')['algorithm'].tolist(),
            'by_speed': df.nsmallest(5, 'execution_time')['algorithm'].tolist(),
            'by_efficiency': df.assign(efficiency=df['best_f1']/df['execution_time'].replace(0, 0.001)).nlargest(5, 'efficiency')['algorithm'].tolist()
        }
    }
    
    with open(report_dir / 'final_report.json', 'w') as f:
        json.dump(report_json, f, indent=2)

def consolidate_all_results(test_mode=None):
    """Função principal de consolidação com análises avançadas"""
    print("📊 Iniciando consolidação avançada de resultados...")
    
    # Carregar todos os resultados (detecta automaticamente o modo)
    all_summaries, all_detailed_results, algorithms, execution_history = load_all_results(test_mode)
    
    if not all_summaries:
        print("❌ Nenhum resultado encontrado!")
        return False
    
    # Detectar modo baseado nos resultados
    detected_test_mode = False
    if all_summaries and 'test_mode' in all_summaries[0]:
        detected_test_mode = all_summaries[0]['test_mode']
    elif test_mode is not None:
        detected_test_mode = test_mode
    
    mode_str = "TESTE" if detected_test_mode else "COMPLETO"
    
    print(f"✅ Carregados resultados de {len(algorithms)} algoritmos")
    print(f"📋 Total de {len(all_detailed_results)} experimentos individuais")
    print(f"🧪 Modo: {mode_str}")
    
    # Criar DataFrames
    df_summary = pd.DataFrame(all_summaries)
    df_detailed = pd.DataFrame(all_detailed_results)
    
    # Expandir confusion_matrix para colunas separadas
    if not df_detailed.empty and 'confusion_matrix' in df_detailed.columns:
        cm_df = pd.json_normalize(df_detailed['confusion_matrix'])
        df_detailed = pd.concat([df_detailed.drop('confusion_matrix', axis=1), cm_df], axis=1)
    
    # Salvar consolidação em experiments/results/ com timestamp compartilhado
    mode_folder = 'test' if detected_test_mode else 'full'
    
    # Usar timestamp compartilhado da rodada ou criar novo se não existir
    shared_timestamp = get_shared_timestamp()
    if shared_timestamp is None:
        shared_timestamp = str(int(time.time()))
    
    consolidation_folder = f"{shared_timestamp}_consolidation"
    
    # Criar estrutura: experiments/results/test|full/timestamp_consolidation/
    base_consolidation_dir = Path("experiments/results") / mode_folder / consolidation_folder
    
    final_plots_dir = base_consolidation_dir / "plots"
    final_tables_dir = base_consolidation_dir / "tables"
    final_report_dir = base_consolidation_dir / "report"
    final_results_dir = base_consolidation_dir / "data"
    
    for dir_path in [final_plots_dir, final_tables_dir, final_report_dir, final_results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Salvando consolidação em: {base_consolidation_dir}/ (modo: {mode_str})")
    print(f"🕐 Timestamp da rodada: {shared_timestamp}")
    
    # Gerar análises avançadas
    print("📈 Gerando análises avançadas...")
    
    # Gráficos básicos (mantidos)
    generate_comparison_plots(df_summary, final_plots_dir)
    
    # Análises detalhadas (NOVAS)
    if not df_detailed.empty:
        generate_confusion_matrices(df_detailed, final_plots_dir)
        generate_boxplots(df_detailed, final_plots_dir)
        generate_correlation_heatmap(df_detailed, final_plots_dir)
        generate_performance_analysis(df_detailed, final_plots_dir)
        generate_parameter_analysis(df_detailed, final_plots_dir)
        generate_anomaly_detection_analysis(df_detailed, final_plots_dir)
        
        # 📊 ANÁLISES AVANÇADAS IoT-IDS
        print("\n📈 Gerando análises avançadas IoT-IDS...")
        try:
            generate_all_iot_advanced_plots(df_detailed, final_plots_dir)
        except Exception as e:
            print(f"⚠️  Erro nas análises IoT avançadas: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n📋 Gerando tabelas avançadas...")
    generate_summary_table(df_summary, final_tables_dir)
    if not df_detailed.empty:
        generate_detailed_statistics_table(df_detailed, final_tables_dir)
    
    print("📄 Gerando relatório avançado...")
    generate_final_report(df_summary, algorithms, final_report_dir, detected_test_mode)
    
    # Salvar DataFrames consolidados
    df_summary.to_csv(final_results_dir / 'consolidated_results.csv', index=False)
    df_summary.to_json(final_results_dir / 'consolidated_results.json', orient='records', indent=2)
    
    if not df_detailed.empty:
        df_detailed.to_csv(final_results_dir / 'detailed_results.csv', index=False)
        df_detailed.to_json(final_results_dir / 'detailed_results.json', orient='records', indent=2)
    
    # Salvar histórico de execuções
    if execution_history:
        with open(final_results_dir / 'execution_history.json', 'w') as f:
            json.dump(execution_history, f, indent=2)
        
        # Criar arquivo de histórico legível
        with open(final_results_dir / 'execution_history.md', 'w') as f:
            f.write(f"# 📅 Histórico de Execuções - {mode_str}\n\n")
            f.write(f"Total de execuções: {len(execution_history)}\n\n")
            f.write("| # | Data/Hora | Algoritmo | Pasta | Timestamp |\n")
            f.write("|---|-----------|-----------|-------|----------|\n")
            
            for i, exec_info in enumerate(execution_history, 1):
                from datetime import datetime
                dt = datetime.fromtimestamp(exec_info['timestamp'])
                f.write(f"| {i} | {dt.strftime('%Y-%m-%d %H:%M:%S')} | {exec_info['algorithm']} | `{exec_info['folder']}` | {exec_info['timestamp']} |\n")
    
    print(f"\n🎉 CONSOLIDAÇÃO AVANÇADA COMPLETA! ({mode_str})")
    print(f"   📊 Algoritmos únicos: {len(set(algorithms))}")
    print(f"   🔬 Experimentos analisados: {len(all_detailed_results)}")
    print(f"   📅 Execuções históricas: {len(execution_history)}")
    print(f"   📈 Gráficos gerados: 10+ análises avançadas")
    print(f"   📋 Tabelas: Resumo + estatísticas detalhadas")
    print(f"   📄 Relatório: Análise completa com recomendações")
    print(f"   🏆 Melhor F1-Score: {df_summary['best_f1'].max():.4f} ({df_summary.loc[df_summary['best_f1'].idxmax(), 'algorithm']})")
    print(f"   ⚡ Mais rápido: {df_summary.loc[df_summary['execution_time'].idxmin(), 'algorithm']} ({df_summary['execution_time'].min():.2f}s)")
    print(f"   💾 Resultados em: {base_consolidation_dir}/")
    
    # Limpar timestamp compartilhado após consolidação bem-sucedida
    cleanup_shared_timestamp()
    print(f"🧹 Timestamp da rodada limpo - rodada completa!")
    
    return True

if __name__ == "__main__":
    try:
        success = consolidate_all_results()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Erro na consolidação: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)