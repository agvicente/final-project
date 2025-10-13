#!/usr/bin/env python3
"""
Análise individual detalhada por algoritmo
Gera gráficos, tabelas e relatórios específicos para cada algoritmo
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def analyze_single_algorithm(results_dir):
    """
    Analisa os resultados de um único algoritmo e gera relatórios detalhados
    
    Args:
        results_dir (Path): Diretório com os resultados do algoritmo
    """
    results_dir = Path(results_dir)
    
    # Verificar se existem resultados
    results_file = results_dir / 'results.json'
    summary_file = results_dir / 'summary.json'
    
    if not results_file.exists() or not summary_file.exists():
        print(f"⚠️  Arquivos de resultados não encontrados em {results_dir}")
        return False
    
    # Carregar dados
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    if not results:
        print(f"⚠️  Nenhum resultado encontrado para análise")
        return False
    
    # Criar estrutura de diretórios para análise individual
    analysis_dir = results_dir / 'individual_analysis'
    plots_dir = analysis_dir / 'plots'
    tables_dir = analysis_dir / 'tables'
    report_dir = analysis_dir / 'report'
    
    for dir_path in [plots_dir, tables_dir, report_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    algorithm_name = summary.get('algorithm', 'Unknown')
    test_mode = summary.get('test_mode', False)
    mode_str = "TESTE" if test_mode else "COMPLETO"
    
    print(f"📊 Gerando análise individual para {algorithm_name} (modo: {mode_str})")
    
    # Converter resultados para DataFrame
    df = pd.DataFrame(results)
    
    # Gerar análises
    generate_performance_evolution(df, plots_dir, algorithm_name)
    generate_parameter_impact(df, plots_dir, algorithm_name)
    generate_confusion_matrix_analysis(df, plots_dir, algorithm_name)
    generate_metrics_distribution(df, plots_dir, algorithm_name)
    generate_execution_time_analysis(df, plots_dir, algorithm_name)
    
    # Gerar tabelas detalhadas
    generate_detailed_tables(df, summary, tables_dir, algorithm_name)
    
    # Gerar relatório individual
    generate_individual_report(df, summary, report_dir, algorithm_name, test_mode)
    
    print(f"✅ Análise individual concluída: {analysis_dir}/")
    return True

def generate_performance_evolution(df, plots_dir, algorithm_name):
    """Gera gráfico de evolução da performance ao longo das execuções"""
    if len(df) < 2:
        return
    
    # Verificar se balanced_accuracy existe
    has_balanced_acc = 'balanced_accuracy' in df.columns
    n_rows, n_cols = (2, 3) if has_balanced_acc else (2, 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12) if has_balanced_acc else (15, 10))
    fig.suptitle(f'{algorithm_name} - Evolução da Performance', fontsize=16, fontweight='bold')
    
    if has_balanced_acc:
        axes = axes.flatten()
    
    # Accuracy por execução
    ax = axes[0] if has_balanced_acc else axes[0,0]
    ax.plot(df.index + 1, df['accuracy'], 'o-', color='blue', linewidth=2, markersize=6)
    ax.set_title('Accuracy por Execução', fontweight='bold')
    ax.set_xlabel('Execução')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['accuracy'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df["accuracy"].mean():.4f}')
    ax.legend()
    
    # Balanced Accuracy por execução (se existe)
    if has_balanced_acc:
        axes[1].plot(df.index + 1, df['balanced_accuracy'], 'o-', color='cyan', linewidth=2, markersize=6)
        axes[1].set_title('Balanced Accuracy por Execução', fontweight='bold')
        axes[1].set_xlabel('Execução')
        axes[1].set_ylabel('Balanced Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=df['balanced_accuracy'].mean(), color='red', linestyle='--', alpha=0.7, 
                        label=f'Média: {df["balanced_accuracy"].mean():.4f}')
        axes[1].legend()
        idx_f1, idx_prec, idx_recall = 2, 3, 4
    else:
        idx_f1, idx_prec, idx_recall = (0,1), (1,0), (1,1)
    
    # F1-Score por execução
    ax = axes[idx_f1] if has_balanced_acc else axes[idx_f1]
    ax.plot(df.index + 1, df['f1_score'], 'o-', color='green', linewidth=2, markersize=6)
    ax.set_title('F1-Score por Execução', fontweight='bold')
    ax.set_xlabel('Execução')
    ax.set_ylabel('F1-Score')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['f1_score'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df["f1_score"].mean():.4f}')
    ax.legend()
    
    # Precision por execução
    ax = axes[idx_prec] if has_balanced_acc else axes[idx_prec]
    ax.plot(df.index + 1, df['precision'], 'o-', color='orange', linewidth=2, markersize=6)
    ax.set_title('Precision por Execução', fontweight='bold')
    ax.set_xlabel('Execução')
    ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['precision'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df["precision"].mean():.4f}')
    ax.legend()
    
    # Recall por execução
    ax = axes[idx_recall] if has_balanced_acc else axes[idx_recall]
    ax.plot(df.index + 1, df['recall'], 'o-', color='purple', linewidth=2, markersize=6)
    ax.set_title('Recall por Execução', fontweight='bold')
    ax.set_xlabel('Execução')
    ax.set_ylabel('Recall')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['recall'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df["recall"].mean():.4f}')
    ax.legend()
    
    # Remover subplot extra se tiver balanced_accuracy (2x3 = 6 plots, mas usamos apenas 5)
    if has_balanced_acc:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_parameter_impact(df, plots_dir, algorithm_name):
    """Analisa o impacto dos parâmetros na performance"""
    if 'params' not in df.columns or len(df) < 2:
        return
    
    # Extrair parâmetros únicos
    param_keys = set()
    for params in df['params']:
        if isinstance(params, dict):
            param_keys.update(params.keys())
    
    if not param_keys:
        return
    
    # Criar subplots baseado no número de parâmetros
    n_params = len(param_keys)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{algorithm_name} - Impacto dos Parâmetros no F1-Score', fontsize=16, fontweight='bold')
    
    for i, param_key in enumerate(param_keys):
        if i >= len(axes):
            break
            
        # Extrair valores do parâmetro
        param_values = []
        f1_scores = []
        
        for idx, row in df.iterrows():
            if isinstance(row['params'], dict) and param_key in row['params']:
                param_values.append(row['params'][param_key])
                f1_scores.append(row['f1_score'])
        
        if len(set(param_values)) > 1:  # Só plotar se há variação
            # Agrupar por valor do parâmetro
            param_df = pd.DataFrame({'param': param_values, 'f1': f1_scores})
            param_grouped = param_df.groupby('param')['f1'].agg(['mean', 'std', 'count']).reset_index()
            
            axes[i].bar(range(len(param_grouped)), param_grouped['mean'], 
                       yerr=param_grouped['std'], capsize=5, alpha=0.7)
            axes[i].set_title(f'Impacto de {param_key}', fontweight='bold')
            axes[i].set_xlabel(param_key)
            axes[i].set_ylabel('F1-Score Médio')
            axes[i].set_xticks(range(len(param_grouped)))
            axes[i].set_xticklabels([str(x) for x in param_grouped['param']], rotation=45)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'{param_key}\n(valor fixo)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{param_key} (constante)', fontweight='bold')
    
    # Remover subplots vazios
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'parameter_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_confusion_matrix_analysis(df, plots_dir, algorithm_name):
    """Gera análise detalhada das matrizes de confusão"""
    if 'confusion_matrix' not in df.columns:
        return
    
    # Calcular matriz de confusão média
    cms = []
    for cm_data in df['confusion_matrix']:
        if isinstance(cm_data, dict):
            cm = np.array([[cm_data.get('tn', 0), cm_data.get('fp', 0)],
                          [cm_data.get('fn', 0), cm_data.get('tp', 0)]])
            cms.append(cm)
    
    if not cms:
        return
    
    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{algorithm_name} - Análise da Matriz de Confusão', fontsize=16, fontweight='bold')
    
    # Matriz de confusão média
    sns.heatmap(mean_cm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'],
                ax=axes[0])
    axes[0].set_title('Matriz de Confusão Média', fontweight='bold')
    axes[0].set_ylabel('Verdadeiro')
    axes[0].set_xlabel('Predito')
    
    # Desvio padrão
    sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Reds', 
                xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'],
                ax=axes[1])
    axes[1].set_title('Desvio Padrão', fontweight='bold')
    axes[1].set_ylabel('Verdadeiro')
    axes[1].set_xlabel('Predito')
    
    # Estabilidade (CV)
    cv_cm = np.divide(std_cm, mean_cm, out=np.zeros_like(std_cm), where=mean_cm!=0)
    sns.heatmap(cv_cm, annot=True, fmt='.3f', cmap='Oranges', 
                xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'],
                ax=axes[2])
    axes[2].set_title('Coeficiente de Variação', fontweight='bold')
    axes[2].set_ylabel('Verdadeiro')
    axes[2].set_xlabel('Predito')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_metrics_distribution(df, plots_dir, algorithm_name):
    """Gera análise da distribuição das métricas"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'balanced_accuracy' in df.columns:
        metrics.insert(1, 'balanced_accuracy')
    
    n_rows, n_cols = (2, 3) if len(metrics) == 5 else (2, 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12) if len(metrics) == 5 else (15, 10))
    axes = axes.flatten()
    fig.suptitle(f'{algorithm_name} - Distribuição das Métricas', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            # Histograma + densidade
            axes[i].hist(df[metric], bins=min(10, len(df)), alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Linha de densidade suavizada se houver dados suficientes
            if len(df) > 3:
                from scipy import stats
                density = stats.gaussian_kde(df[metric])
                xs = np.linspace(df[metric].min(), df[metric].max(), 200)
                axes[i].plot(xs, density(xs), color='red', linewidth=2)
            
            # Estatísticas
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.4f}')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1σ')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            
            metric_title = 'Balanced Accuracy' if metric == 'balanced_accuracy' else metric.replace("_", " ").title()
            axes[i].set_title(metric_title, fontweight='bold')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Densidade')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remover subplot extra se tiver 5 métricas em grid 2x3
    if len(metrics) == 5:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_execution_time_analysis(df, plots_dir, algorithm_name):
    """Analisa os tempos de execução"""
    if 'training_time' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{algorithm_name} - Análise de Tempo de Execução', fontsize=16, fontweight='bold')
    
    # Tempo por execução
    axes[0,0].plot(df.index + 1, df['training_time'], 'o-', color='red', linewidth=2, markersize=6)
    axes[0,0].set_title('Tempo de Treinamento por Execução', fontweight='bold')
    axes[0,0].set_xlabel('Execução')
    axes[0,0].set_ylabel('Tempo (s)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=df['training_time'].mean(), color='blue', linestyle='--', alpha=0.7, 
                     label=f'Média: {df["training_time"].mean():.2f}s')
    axes[0,0].legend()
    
    # Distribuição dos tempos
    axes[0,1].hist(df['training_time'], bins=min(10, len(df)), alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Distribuição dos Tempos', fontweight='bold')
    axes[0,1].set_xlabel('Tempo (s)')
    axes[0,1].set_ylabel('Frequência')
    axes[0,1].grid(True, alpha=0.3)
    
    # Eficiência (F1/tempo)
    efficiency = df['f1_score'] / df['training_time'].replace(0, 0.001)
    axes[1,0].plot(df.index + 1, efficiency, 'o-', color='green', linewidth=2, markersize=6)
    axes[1,0].set_title('Eficiência (F1-Score/Tempo)', fontweight='bold')
    axes[1,0].set_xlabel('Execução')
    axes[1,0].set_ylabel('F1/Segundo')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=efficiency.mean(), color='blue', linestyle='--', alpha=0.7, 
                     label=f'Média: {efficiency.mean():.4f}')
    axes[1,0].legend()
    
    # Correlação tempo vs performance
    axes[1,1].scatter(df['training_time'], df['f1_score'], alpha=0.7, s=60, color='purple')
    axes[1,1].set_title('Tempo vs F1-Score', fontweight='bold')
    axes[1,1].set_xlabel('Tempo de Treinamento (s)')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].grid(True, alpha=0.3)
    
    # Linha de tendência
    if len(df) > 2:
        z = np.polyfit(df['training_time'], df['f1_score'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(df['training_time'], p(df['training_time']), "r--", alpha=0.8)
        
        # Correlação
        corr = df['training_time'].corr(df['f1_score'])
        axes[1,1].text(0.05, 0.95, f'Correlação: {corr:.3f}', transform=axes[1,1].transAxes, 
                      bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'execution_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_tables(df, summary, tables_dir, algorithm_name):
    """Gera tabelas detalhadas dos resultados"""
    
    # Tabela de estatísticas descritivas
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time']
    if 'balanced_accuracy' in df.columns:
        metrics.insert(1, 'balanced_accuracy')
    stats_data = []
    
    for metric in metrics:
        if metric in df.columns:
            metric_title = 'Balanced Accuracy' if metric == 'balanced_accuracy' else metric.replace('_', ' ').title()
            stats_data.append({
                'Métrica': metric_title,
                'Média': df[metric].mean(),
                'Desvio Padrão': df[metric].std(),
                'Mínimo': df[metric].min(),
                'Máximo': df[metric].max(),
                'Mediana': df[metric].median(),
                'CV (%)': (df[metric].std() / max(df[metric].mean(), 0.0001)) * 100
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(tables_dir / 'descriptive_statistics.csv', index=False)
    
    # Tabela detalhada de todos os experimentos
    detailed_df = df.copy()
    if 'params' in detailed_df.columns:
        # Expandir parâmetros em colunas separadas
        params_df = pd.json_normalize(detailed_df['params'])
        detailed_df = pd.concat([detailed_df.drop('params', axis=1), params_df], axis=1)
    
    detailed_df.to_csv(tables_dir / 'detailed_results.csv', index=False)
    
    # Tabela de ranking por execução
    ranking_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'balanced_accuracy' in df.columns:
        ranking_cols.insert(1, 'balanced_accuracy')
    ranking_df = df[ranking_cols].copy()
    ranking_df['execution'] = range(1, len(df) + 1)
    ranking_df['rank_accuracy'] = ranking_df['accuracy'].rank(ascending=False)
    ranking_df['rank_f1'] = ranking_df['f1_score'].rank(ascending=False)
    ranking_df.to_csv(tables_dir / 'execution_ranking.csv', index=False)
    
    # Salvar markdown das estatísticas
    with open(tables_dir / 'statistics_summary.md', 'w') as f:
        f.write(f"# 📊 Estatísticas Detalhadas - {algorithm_name}\n\n")
        f.write("## Estatísticas Descritivas\n\n")
        f.write(stats_df.to_markdown(index=False, floatfmt='.4f'))
        f.write(f"\n\n## Resumo Executivo\n\n")
        f.write(f"- **Total de Execuções**: {len(df)}\n")
        f.write(f"- **Melhor F1-Score**: {df['f1_score'].max():.4f}\n")
        f.write(f"- **F1-Score Médio**: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}\n")
        f.write(f"- **Tempo Médio**: {df['training_time'].mean():.2f}s\n")
        f.write(f"- **Eficiência Média**: {(df['f1_score'] / df['training_time'].replace(0, 0.001)).mean():.4f} F1/s\n")

def generate_individual_report(df, summary, report_dir, algorithm_name, test_mode):
    """Gera relatório individual completo"""
    
    mode_str = "TESTE" if test_mode else "COMPLETO"
    
    # Calcular estatísticas principais
    best_f1_idx = df['f1_score'].idxmax()
    best_result = df.loc[best_f1_idx]
    
    stability = df['f1_score'].std()
    efficiency = (df['f1_score'] / df['training_time'].replace(0, 0.001)).mean()
    
    report_content = f"""# 📊 Relatório Individual - {algorithm_name}

**Modo de Execução**: {mode_str}  
**Data de Geração**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total de Execuções**: {len(df)}

## 🎯 Resumo Executivo

### Melhor Performance
- **F1-Score**: {best_result['f1_score']:.4f}
- **Accuracy**: {best_result['accuracy']:.4f}
- **Precision**: {best_result['precision']:.4f}
- **Recall**: {best_result['recall']:.4f}
- **Tempo de Treinamento**: {best_result['training_time']:.2f}s

### Performance Média
- **F1-Score Médio**: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}
- **Accuracy Média**: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}
- **Precision Média**: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}
- **Recall Médio**: {df['recall'].mean():.4f} ± {df['recall'].std():.4f}

### Métricas de Qualidade
- **Estabilidade (Desvio F1)**: {stability:.4f} {'🟢 Excelente' if stability < 0.01 else '🟡 Boa' if stability < 0.05 else '🔴 Instável'}
- **Eficiência Média**: {efficiency:.4f} F1/segundo
- **Tempo Médio**: {df['training_time'].mean():.2f}s ± {df['training_time'].std():.2f}s

## 📈 Análise Detalhada

### Distribuição das Métricas
"""

    # Adicionar estatísticas detalhadas
    report_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'balanced_accuracy' in df.columns:
        report_metrics.insert(1, 'balanced_accuracy')
    
    for metric in report_metrics:
        if metric in df.columns:
            q25 = df[metric].quantile(0.25)
            q75 = df[metric].quantile(0.75)
            metric_title = 'Balanced Accuracy' if metric == 'balanced_accuracy' else metric.replace('_', ' ').title()
            report_content += f"""
#### {metric_title}
- **Mínimo**: {df[metric].min():.4f}
- **Q1**: {q25:.4f}
- **Mediana**: {df[metric].median():.4f}
- **Q3**: {q75:.4f}
- **Máximo**: {df[metric].max():.4f}
- **IQR**: {q75 - q25:.4f}
"""

    # Análise de parâmetros se disponível
    if 'params' in df.columns and len(df) > 1:
        report_content += "\n### Análise de Parâmetros\n\n"
        
        # Extrair parâmetros únicos
        param_keys = set()
        for params in df['params']:
            if isinstance(params, dict):
                param_keys.update(params.keys())
        
        for param_key in param_keys:
            param_values = []
            f1_scores = []
            
            for idx, row in df.iterrows():
                if isinstance(row['params'], dict) and param_key in row['params']:
                    param_values.append(row['params'][param_key])
                    f1_scores.append(row['f1_score'])
            
            if len(set(param_values)) > 1:
                param_df = pd.DataFrame({'param': param_values, 'f1': f1_scores})
                param_grouped = param_df.groupby('param')['f1'].agg(['mean', 'std', 'count'])
                
                best_param = param_grouped['mean'].idxmax()
                report_content += f"""
#### {param_key}
- **Melhor valor**: {best_param} (F1: {param_grouped.loc[best_param, 'mean']:.4f})
- **Variação observada**: {len(set(param_values))} valores diferentes
- **Impacto no F1**: {param_grouped['mean'].max() - param_grouped['mean'].min():.4f}
"""

    # Recomendações
    report_content += f"""
## 🎯 Recomendações

### Pontos Fortes
"""
    
    if df['f1_score'].mean() > 0.8:
        report_content += "- ✅ **Excelente performance geral** (F1 > 0.8)\n"
    elif df['f1_score'].mean() > 0.6:
        report_content += "- ✅ **Boa performance geral** (F1 > 0.6)\n"
    
    if stability < 0.02:
        report_content += "- ✅ **Alta estabilidade** entre execuções\n"
    
    if efficiency > 0.01:
        report_content += "- ✅ **Boa eficiência computacional**\n"

    report_content += "\n### Áreas de Melhoria\n"
    
    if df['f1_score'].mean() < 0.6:
        report_content += "- 🔴 **Performance baixa** - considerar ajuste de hiperparâmetros\n"
    
    if stability > 0.05:
        report_content += "- 🔴 **Alta variabilidade** - resultados inconsistentes\n"
    
    if df['training_time'].mean() > 60:
        report_content += "- 🟡 **Tempo de treinamento elevado** - considerar otimizações\n"

    report_content += f"""
## 📊 Arquivos Gerados

### Gráficos
- `plots/performance_evolution.png` - Evolução das métricas
- `plots/parameter_impact.png` - Impacto dos parâmetros
- `plots/confusion_matrix_analysis.png` - Análise da matriz de confusão
- `plots/metrics_distribution.png` - Distribuição das métricas
- `plots/execution_time_analysis.png` - Análise de tempo

### Tabelas
- `tables/descriptive_statistics.csv` - Estatísticas descritivas
- `tables/detailed_results.csv` - Resultados detalhados
- `tables/execution_ranking.csv` - Ranking por execução

---
*Relatório gerado automaticamente pelo sistema de análise individual*
"""

    # Salvar relatório
    with open(report_dir / 'individual_report.md', 'w') as f:
        f.write(report_content)
    
    # Salvar resumo JSON
    summary_json = {
        'algorithm': algorithm_name,
        'mode': mode_str,
        'executions': len(df),
        'best_performance': {
            'f1_score': float(best_result['f1_score']),
            'accuracy': float(best_result['accuracy']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'training_time': float(best_result['training_time'])
        },
        'average_performance': {
            'f1_score': float(df['f1_score'].mean()),
            'accuracy': float(df['accuracy'].mean()),
            'precision': float(df['precision'].mean()),
            'recall': float(df['recall'].mean()),
            'training_time': float(df['training_time'].mean())
        },
        'stability': {
            'f1_std': float(df['f1_score'].std()),
            'accuracy_std': float(df['accuracy'].std()),
            'stability_score': float(stability)
        },
        'efficiency': float(efficiency),
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    with open(report_dir / 'summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("❌ Uso: python3 individual_analysis.py <results_directory>")
        sys.exit(1)
    
    results_directory = sys.argv[1]
    success = analyze_single_algorithm(results_directory)
    sys.exit(0 if success else 1)
