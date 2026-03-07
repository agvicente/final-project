#!/usr/bin/env python3
"""
Visualizações avançadas para análise IoT-IDS
Baseado em literatura acadêmica de IoT security e ML performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi
import warnings
warnings.filterwarnings('ignore')

def generate_pareto_frontier_analysis(detailed_df, plots_dir):
    """
    Análise de Pareto Frontier: Trade-offs entre performance e recursos
    
    Referências:
    - Nguyen et al. (2020) - "A Survey of ML Approaches for IoT Security"
    - Ahmad et al. (2021) - "Performance-Energy Trade-offs in ML"
    """
    print("   📊 Gerando análise de Pareto Frontier...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Pareto Frontier Analysis: Performance vs Resources', 
                 fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(detailed_df['algorithm'].unique()))
    algo_colors = dict(zip(detailed_df['algorithm'].unique(), colors))
    
    # 1. F1-Score vs Training Time
    ax = axes[0, 0]
    for algorithm in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        ax.scatter(algo_data['training_time'], algo_data['f1_score'], 
                  label=algorithm, alpha=0.7, s=100, color=algo_colors[algorithm])
    
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Performance vs Training Speed', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target: 0.8')
    
    # 2. F1-Score vs Memory Usage
    ax = axes[0, 1]
    for algorithm in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        ax.scatter(algo_data['memory_usage_mb'], algo_data['f1_score'],
                  label=algorithm, alpha=0.7, s=100, color=algo_colors[algorithm])
    
    ax.set_xlabel('Memory Usage (MB)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Performance vs Memory', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axvline(x=512, color='orange', linestyle='--', alpha=0.5, label='512MB limit')
    ax.axvline(x=1024, color='r', linestyle='--', alpha=0.5, label='1GB limit')
    
    # 3. Efficiency Score (F1 / (Time * Memory))
    ax = axes[0, 2]
    if 'iot_metrics' in detailed_df.columns:
        # Usar score calculado
        efficiency_data = []
        for algorithm in detailed_df['algorithm'].unique():
            algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
            # Tentar extrair do iot_metrics se disponível
            try:
                scores = [d.get('resource_efficiency_score', 0) for d in algo_data['iot_metrics'] if isinstance(d, dict)]
                if scores:
                    efficiency_data.append({'algorithm': algorithm, 'efficiency': np.mean(scores)})
            except:
                pass
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data).sort_values('efficiency')
            eff_df.plot(x='algorithm', y='efficiency', kind='barh', ax=ax, 
                       legend=False, color='skyblue')
    else:
        # Calcular manualmente
        detailed_df['efficiency_score'] = (
            detailed_df['f1_score'] / 
            (detailed_df['training_time'] * detailed_df['memory_usage_mb']).replace(0, 0.001)
        )
        algo_efficiency = detailed_df.groupby('algorithm')['efficiency_score'].mean().sort_values()
        algo_efficiency.plot(kind='barh', ax=ax, color='skyblue')
    
    ax.set_xlabel('Efficiency Score (F1/Resource)', fontsize=12)
    ax.set_title('Overall Resource Efficiency', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. F1-Score vs Prediction Time (Real-time capability)
    ax = axes[1, 0]
    for algorithm in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        ax.scatter(algo_data['prediction_time'], algo_data['f1_score'],
                  label=algorithm, alpha=0.7, s=100, color=algo_colors[algorithm])
    
    ax.set_xlabel('Prediction Time (s)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Performance vs Inference Speed', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which='both')
    ax.axvline(x=0.01, color='g', linestyle='--', alpha=0.5, label='10ms (strict RT)')
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='100ms (moderate RT)')
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1s (relaxed RT)')
    
    # 5. 3D Trade-off: Time vs Memory (sized by F1)
    ax = axes[1, 1]
    for algorithm in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        sizes = algo_data['f1_score'] * 500  # Scale for visibility
        ax.scatter(algo_data['training_time'], algo_data['memory_usage_mb'],
                  s=sizes, alpha=0.6, label=algorithm, color=algo_colors[algorithm])
    
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Time-Memory Trade-off (bubble size = F1)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # 6. IoT Suitability Ranking
    ax = axes[1, 2]
    if 'iot_metrics' in detailed_df.columns:
        try:
            iot_scores = []
            for algorithm in detailed_df['algorithm'].unique():
                algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
                scores = [d.get('iot_suitability_score', 0) for d in algo_data['iot_metrics'] if isinstance(d, dict)]
                if scores:
                    iot_scores.append({'algorithm': algorithm, 'score': np.mean(scores)})
            
            if iot_scores:
                score_df = pd.DataFrame(iot_scores).sort_values('score')
                score_df.plot(x='algorithm', y='score', kind='barh', ax=ax, 
                             legend=False, color='coral')
                ax.set_xlabel('IoT Suitability Score', fontsize=12)
                ax.set_title('IoT Device Suitability Ranking', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'IoT metrics not available\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        # Calcular score simples
        detailed_df['simple_iot_score'] = (
            detailed_df['f1_score'] / 
            ((detailed_df['training_time'] + detailed_df['prediction_time']) * 
             detailed_df['memory_usage_mb']).replace(0, 0.001)
        )
        score_df = detailed_df.groupby('algorithm')['simple_iot_score'].mean().sort_values()
        score_df.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Simple IoT Score', fontsize=12)
        ax.set_title('Basic IoT Suitability Ranking', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'pareto_frontier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Pareto Frontier gerado")

def generate_resource_radar_charts(detailed_df, plots_dir):
    """
    Radar charts mostrando perfil multidimensional de recursos
    
    Referências:
    - Zolanvari et al. (2019) - "ML-Based Network Vulnerability Analysis"
    - Thakkar & Lohiya (2021) - "Performance Evaluation of ML for IoT-IDS"
    """
    print("   📊 Gerando Resource Radar Charts...")
    
    # Métricas para normalizar
    metrics = ['f1_score', 'accuracy', 'training_time', 'memory_usage_mb', 'prediction_time']
    
    # Agregar por algoritmo
    algo_stats = detailed_df.groupby('algorithm')[metrics].mean()
    
    # Normalizar métricas (0-1, onde 1 é melhor)
    algo_stats_norm = algo_stats.copy()
    algo_stats_norm['f1_score'] = algo_stats['f1_score']  # Já 0-1
    algo_stats_norm['accuracy'] = algo_stats['accuracy']  # Já 0-1
    # Inverter tempo e memória (menor é melhor)
    algo_stats_norm['training_time'] = 1 - (algo_stats['training_time'] / algo_stats['training_time'].max())
    algo_stats_norm['memory_usage_mb'] = 1 - (algo_stats['memory_usage_mb'] / algo_stats['memory_usage_mb'].max())
    algo_stats_norm['prediction_time'] = 1 - (algo_stats['prediction_time'] / algo_stats['prediction_time'].max())
    
    # Criar subplots
    n_algos = len(algo_stats_norm)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows), 
                             subplot_kw=dict(projection='polar'))
    if n_algos == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('Resource Profile Radar Charts - Normalized Metrics', 
                 fontsize=16, fontweight='bold')
    
    categories = ['F1-Score', 'Accuracy', 'Speed\n(Training)', 'Memory\nEfficiency', 'Speed\n(Prediction)']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors = sns.color_palette("husl", n_algos)
    
    for idx, ((algorithm, row), color) in enumerate(zip(algo_stats_norm.iterrows(), colors)):
        ax = axes[idx]
        values = row.values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=algorithm)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(algorithm, size=12, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Adicionar scores nos pontos
        for angle, value, label in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=8, color=color)
    
    # Remover subplots extras
    for idx in range(n_algos, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'resource_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Radar Charts gerados")

def generate_latency_throughput_analysis(detailed_df, plots_dir):
    """
    Análise detalhada de latência e throughput - CRÍTICO para IoT
    
    Referências:
    - Diro & Chilamkurti (2018) - "Distributed Attack Detection Scheme"
    - Khraisat et al. (2019) - "Survey of IDS: Performance Metrics"
    """
    print("   📊 Gerando análise de Latência e Throughput...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Latency & Throughput Analysis - Real-time Performance', 
                 fontsize=16, fontweight='bold')
    
    # 1. Prediction Time Distribution (Box + Violin)
    ax = axes[0, 0]
    detailed_df.boxplot(column='prediction_time', by='algorithm', ax=ax, rot=45)
    ax.set_title('Prediction Latency Distribution', fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=10)
    ax.set_ylabel('Latency (seconds)', fontsize=10)
    ax.axhline(y=0.1, color='r', linestyle='--', label='100ms', linewidth=2)
    ax.axhline(y=0.01, color='orange', linestyle='--', label='10ms', linewidth=2)
    ax.legend()
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    # 2. Percentile Analysis (P50, P95, P99)
    ax = axes[0, 1]
    percentiles_data = []
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]['prediction_time']
        percentiles_data.append({
            'algorithm': algo,
            'p50': algo_data.quantile(0.5),
            'p95': algo_data.quantile(0.95),
            'p99': algo_data.quantile(0.99)
        })
    
    perc_df = pd.DataFrame(percentiles_data).set_index('algorithm')
    perc_df.plot(kind='bar', ax=ax)
    ax.set_title('Latency Percentiles (P50, P95, P99)', fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=10)
    ax.set_xlabel('Algorithm', fontsize=10)
    ax.legend(['P50 (Median)', 'P95', 'P99'])
    ax.grid(axis='y', alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    # 3. Cumulative Distribution Function (CDF)
    ax = axes[0, 2]
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]['prediction_time'].sort_values()
        cdf = np.arange(1, len(algo_data) + 1) / len(algo_data)
        ax.plot(algo_data, cdf, label=algo, linewidth=2)
    
    ax.set_xlabel('Prediction Time (seconds)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title('Latency CDF', fontweight='bold')
    ax.axvline(x=0.01, color='g', linestyle='--', alpha=0.5, label='10ms')
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='100ms')
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1s')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # 4. Throughput (samples/second)
    ax = axes[1, 0]
    detailed_df['throughput'] = 1 / detailed_df['prediction_time'].replace(0, 0.001)
    throughput_stats = detailed_df.groupby('algorithm')['throughput'].agg(['mean', 'std'])
    throughput_stats['mean'].plot(kind='barh', xerr=throughput_stats['std'], 
                                   ax=ax, color='lightgreen', capsize=5)
    ax.set_xlabel('Throughput (samples/second)', fontsize=10)
    ax.set_title('Processing Throughput', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 5. Memory vs Latency Trade-off
    ax = axes[1, 1]
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        ax.scatter(algo_data['memory_usage_mb'], algo_data['prediction_time'],
                  label=algo, alpha=0.6, s=80)
    ax.set_xlabel('Memory Usage (MB)', fontsize=10)
    ax.set_ylabel('Prediction Latency (seconds)', fontsize=10)
    ax.set_title('Memory-Latency Trade-off', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    
    # 6. Real-time Capability Classification
    ax = axes[1, 2]
    rt_classification = []
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        median_latency = algo_data['prediction_time'].median()
        
        strict = (algo_data['prediction_time'] < 0.01).mean() * 100
        moderate = (algo_data['prediction_time'] < 0.1).mean() * 100
        relaxed = (algo_data['prediction_time'] < 1.0).mean() * 100
        
        rt_classification.append({
            'algorithm': algo,
            '<10ms': strict,
            '<100ms': moderate,
            '<1s': relaxed
        })
    
    rt_df = pd.DataFrame(rt_classification).set_index('algorithm')
    rt_df.plot(kind='bar', ax=ax, stacked=False)
    ax.set_ylabel('% of Predictions Meeting Threshold', fontsize=10)
    ax.set_title('Real-time Capability Classification', fontweight='bold')
    ax.legend(['Strict (<10ms)', 'Moderate (<100ms)', 'Relaxed (<1s)'])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'latency_throughput_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Análise de Latência/Throughput gerada")

def generate_scalability_analysis(detailed_df, plots_dir):
    """
    Análise de escalabilidade para diferentes tamanhos de dataset
    
    Referências:
    - Ferrag et al. (2020) - "Deep Learning for Cyber Security"
    - Hodo et al. (2016) - "Threat Analysis of IoT Networks"
    """
    print("   📊 Gerando análise de Escalabilidade...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Scalability Analysis - Performance Projections', 
                 fontsize=16, fontweight='bold')
    
    # Dataset sizes para projeção
    dataset_sizes = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    
    # Assumir tamanho atual do dataset
    if 'complexity_metrics' in detailed_df.columns:
        try:
            current_size = detailed_df['complexity_metrics'].iloc[0]['n_train_samples']
        except:
            current_size = 10000
    else:
        current_size = 10000
    
    # 1. Training Time Projection
    ax = axes[0, 0]
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        avg_time = algo_data['training_time'].mean()
        
        # Complexidade estimada baseada no nome do algoritmo
        if any(x in algo for x in ['Logistic', 'SGD', 'Linear']):
            complexity = dataset_sizes  # O(n)
        elif any(x in algo for x in ['Forest', 'Isolation', 'Gradient', 'Tree']):
            complexity = dataset_sizes * np.log(dataset_sizes)  # O(n log n)
        elif any(x in algo for x in ['SVC', 'Elliptic']):
            complexity = dataset_sizes ** 1.5  # O(n^1.5)
        elif any(x in algo for x in ['Outlier', 'Neighbor']):
            complexity = dataset_sizes ** 2  # O(n²)
        else:
            complexity = dataset_sizes * np.log(dataset_sizes)  # Default
        
        projected_time = avg_time * (complexity / current_size)
        ax.plot(dataset_sizes, projected_time, marker='o', label=algo, linewidth=2, markersize=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size (samples)', fontsize=11)
    ax.set_ylabel('Projected Training Time (seconds)', fontsize=11)
    ax.set_title('Training Time Scalability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.axhline(y=3600, color='r', linestyle='--', alpha=0.5, label='1 hour')
    
    # 2. Memory Scalability
    ax = axes[0, 1]
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        avg_memory = algo_data['memory_usage_mb'].mean()
        
        # Projeção linear/log para memória
        if any(x in algo for x in ['Logistic', 'SGD', 'Linear']):
            mem_growth = dataset_sizes / current_size  # Linear
        elif any(x in algo for x in ['Forest', 'Tree']):
            mem_growth = (dataset_sizes / current_size) * np.log(dataset_sizes / current_size)
        else:
            mem_growth = dataset_sizes / current_size
        
        projected_memory = avg_memory * mem_growth
        ax.plot(dataset_sizes, projected_memory, marker='s', label=algo, linewidth=2, markersize=4)
    
    ax.axhline(y=512, color='orange', linestyle='--', label='512MB (IoT limit)', linewidth=2)
    ax.axhline(y=1024, color='r', linestyle='--', label='1GB (Edge limit)', linewidth=2)
    ax.axhline(y=4096, color='purple', linestyle='--', label='4GB (Server)', linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size (samples)', fontsize=11)
    ax.set_ylabel('Projected Memory Usage (MB)', fontsize=11)
    ax.set_title('Memory Scalability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    
    # 3. Performance Degradation Estimate
    ax = axes[1, 0]
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        base_f1 = algo_data['f1_score'].mean()
        
        # Degradação estimada (modelos mais complexos degradam menos)
        if any(x in algo for x in ['Forest', 'Gradient', 'MLP']):
            degradation = np.array([1.0, 0.99, 0.98, 0.96, 0.93, 0.90])
        elif any(x in algo for x in ['Logistic', 'Linear', 'SGD']):
            degradation = np.array([1.0, 0.98, 0.95, 0.90, 0.85, 0.80])
        else:
            degradation = np.array([1.0, 0.98, 0.96, 0.93, 0.89, 0.85])
        
        projected_f1 = base_f1 * degradation
        ax.plot(dataset_sizes, projected_f1, marker='o', label=algo, linewidth=2, markersize=4)
    
    ax.set_xscale('log')
    ax.set_xlabel('Dataset Size (samples)', fontsize=11)
    ax.set_ylabel('Projected F1-Score', fontsize=11)
    ax.set_title('Performance Scalability (estimated)', fontweight='bold')
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Acceptable (0.8)')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Good (0.9)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 4. Scalability Score Summary
    ax = axes[1, 1]
    scalability_scores = []
    for algo in detailed_df['algorithm'].unique():
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        
        # Score = Performance / (log(time_complexity) * log(memory_complexity))
        avg_f1 = algo_data['f1_score'].mean()
        avg_time = algo_data['training_time'].mean()
        avg_memory = algo_data['memory_usage_mb'].mean()
        
        # Penalizar alto tempo e memória
        time_penalty = np.log10(max(avg_time, 0.1))
        memory_penalty = np.log10(max(avg_memory, 1))
        
        score = avg_f1 / max(time_penalty * memory_penalty, 0.1)
        scalability_scores.append({'algorithm': algo, 'score': score})
    
    scale_df = pd.DataFrame(scalability_scores).sort_values('score')
    scale_df.plot(x='algorithm', y='score', kind='barh', ax=ax, legend=False, color='teal')
    ax.set_xlabel('Scalability Score', fontsize=11)
    ax.set_title('Overall Scalability Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Análise de Escalabilidade gerada")

def generate_enhanced_confusion_analysis(detailed_df, plots_dir):
    """
    Análise avançada de confusion matrix com foco em custos de erro
    
    Referências:
    - Error cost analysis em IDS systems
    - False alarm vs miss rate trade-offs
    """
    print("   📊 Gerando análise avançada de Confusion Matrix...")
    
    algorithms = detailed_df['algorithm'].unique()[:9]  # Top 9 para grid 3x3
    n_algos = len(algorithms)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_algos == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('Enhanced Confusion Matrix Analysis - Error Costs', 
                 fontsize=16, fontweight='bold')
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        algo_data = detailed_df[detailed_df['algorithm'] == algo]
        
        # Agregar matriz de confusão
        cm = np.array([[algo_data['tn'].sum(), algo_data['fp'].sum()],
                       [algo_data['fn'].sum(), algo_data['tp'].sum()]])
        
        # Normalizar para percentuais por classe verdadeira
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plotar com cores baseadas em custo (vermelho = ruim)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn_r',
                   xticklabels=['Predicted Normal', 'Predicted Attack'],
                   yticklabels=['True Normal', 'True Attack'],
                   ax=ax, cbar=True, vmin=0, vmax=1)
        
        # Calcular métricas
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
        fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        title = f'{algo}\n'
        title += f'F1:{f1:.3f} | Prec:{precision:.3f} | Rec:{recall:.3f}\n'
        title += f'FPR:{fpr:.3f} (False Alarm) | FNR:{fnr:.3f} (Miss)'
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)
    
    # Remover subplots extras
    for idx in range(n_algos, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'enhanced_confusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Confusion Matrix Avançada gerada")

def generate_all_iot_advanced_plots(detailed_df, plots_dir):
    """
    Gera todos os gráficos avançados IoT
    
    Args:
        detailed_df: DataFrame com resultados detalhados
        plots_dir: Diretório para salvar plots
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📈 Gerando análises avançadas IoT-IDS...")
    
    try:
        generate_pareto_frontier_analysis(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro em Pareto Frontier: {e}")
    
    try:
        generate_resource_radar_charts(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro em Radar Charts: {e}")
    
    try:
        generate_latency_throughput_analysis(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro em Latency Analysis: {e}")
    
    try:
        generate_scalability_analysis(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro em Scalability Analysis: {e}")
    
    try:
        generate_enhanced_confusion_analysis(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro em Enhanced Confusion: {e}")
    
    print("✅ Análises avançadas IoT concluídas!\n")

# Export
__all__ = [
    'generate_pareto_frontier_analysis',
    'generate_resource_radar_charts',
    'generate_latency_throughput_analysis',
    'generate_scalability_analysis',
    'generate_enhanced_confusion_analysis',
    'generate_all_iot_advanced_plots'
]

