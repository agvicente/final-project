#!/usr/bin/env python3
"""
Plots Bayesianos conforme Brodersen et al. (2010)
Visualizações de distribuições posteriores e intervalos de credibilidade
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def plot_posterior_distributions(detailed_df, plots_dir):
    """
    Plota distribuições posteriores de Balanced Accuracy.
    
    Figura principal do artigo Brodersen et al. (2010) mostrando
    as densidades posteriores via convolução de distribuições Beta.
    
    Args:
        detailed_df: DataFrame com resultados detalhados (precisa ter tp, tn, fp, fn)
        plots_dir: Diretório para salvar plots
    """
    print("   📊 Gerando distribuições posteriores Bayesianas (Brodersen)...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("husl", len(detailed_df['algorithm'].unique()))
    
    for (algorithm, color) in zip(detailed_df['algorithm'].unique(), colors):
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        # Agregar confusion matrix de todas as execuções
        total_tp = algo_data['tp'].sum()
        total_tn = algo_data['tn'].sum()
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        
        # Calcular posterior (distribuição Beta para cada classe)
        sens_post = stats.beta(total_tp + 1, total_fn + 1)
        spec_post = stats.beta(total_tn + 1, total_fp + 1)
        
        # Amostrar BA posterior via convolução
        n_samples = 100000
        sens_samples = sens_post.rvs(n_samples)
        spec_samples = spec_post.rvs(n_samples)
        ba_samples = 0.5 * (sens_samples + spec_samples)
        
        # Densidade via KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ba_samples)
        x_range = np.linspace(max(0.5, ba_samples.min()-0.05), 
                              min(1.0, ba_samples.max()+0.05), 1000)
        density = kde(x_range)
        
        # Plot
        ax.plot(x_range, density, label=algorithm, linewidth=2.5, color=color)
        ax.fill_between(x_range, density, alpha=0.2, color=color)
        
        # Marcar média
        mean_ba = np.mean(ba_samples)
        ax.axvline(mean_ba, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Marcar IC 95%
        ci = np.percentile(ba_samples, [2.5, 97.5])
        ax.axvspan(ci[0], ci[1], alpha=0.1, color=color)
    
    ax.set_xlabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Posterior Density', fontsize=14, fontweight='bold')
    ax.set_title('Posterior Distributions of Balanced Accuracy\n' + 
                 '(Bayesian approach - Brodersen et al., 2010)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 1.0)
    
    # Adicionar linhas de referência
    ax.axvline(0.9, color='green', linestyle=':', alpha=0.5, linewidth=2, label='Target: 0.9')
    ax.axvline(0.8, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='Acceptable: 0.8')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'bayesian_posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Distribuições posteriores Bayesianas geradas")


def plot_credibility_intervals(detailed_df, plots_dir):
    """
    Plota intervalos de credibilidade 95% para Balanced Accuracy.
    
    Mostra os intervalos Bayesianos que respeitam os limites naturais [0,1]
    e fornecem interpretação probabilística direta.
    
    Args:
        detailed_df: DataFrame com resultados detalhados
        plots_dir: Diretório para salvar plots
    """
    print("   📊 Gerando intervalos de credibilidade Bayesianos...")
    
    algorithms = detailed_df['algorithm'].unique()
    results = []
    
    for algorithm in algorithms:
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        # Agregar
        total_tp = algo_data['tp'].sum()
        total_tn = algo_data['tn'].sum()
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        
        # Posterior
        sens_post = stats.beta(total_tp + 1, total_fn + 1)
        spec_post = stats.beta(total_tn + 1, total_fp + 1)
        
        n_samples = 100000
        sens_samples = sens_post.rvs(n_samples)
        spec_samples = spec_post.rvs(n_samples)
        ba_samples = 0.5 * (sens_samples + spec_samples)
        
        mean_ba = np.mean(ba_samples)
        median_ba = np.median(ba_samples)
        ci = np.percentile(ba_samples, [2.5, 97.5])
        
        # Calcular P(BA > 0.90)
        prob_above_90 = np.mean(ba_samples > 0.90)
        
        results.append({
            'algorithm': algorithm,
            'mean': mean_ba,
            'median': median_ba,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'prob_above_90': prob_above_90
        })
    
    # Ordenar por mean
    results_sorted = sorted(results, key=lambda x: x['mean'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(algorithms) * 0.6)))
    
    y_pos = np.arange(len(results_sorted))
    means = [r['mean'] for r in results_sorted]
    algos = [r['algorithm'] for r in results_sorted]
    errors_lower = [r['mean'] - r['ci_lower'] for r in results_sorted]
    errors_upper = [r['ci_upper'] - r['mean'] for r in results_sorted]
    
    # Barras horizontais com erro
    bars = ax.barh(y_pos, means, xerr=[errors_lower, errors_upper],
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Linhas de referência
    ax.axvline(0.9, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target: 0.9')
    ax.axvline(0.8, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Acceptable: 0.8')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algos, fontsize=11)
    ax.set_xlabel('Balanced Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Balanced Accuracy with 95% Bayesian Credibility Intervals\n' +
                 '(Brodersen et al., 2010 approach)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 1.0)
    
    # Adicionar valores e P(BA > 0.90)
    for i, r in enumerate(results_sorted):
        ax.text(r['mean'] + 0.02, i, 
                f"{r['mean']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]\nP(>0.9)={r['prob_above_90']:.2f}",
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'bayesian_credibility_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Intervalos de credibilidade gerados")


def plot_probabilistic_comparison_matrix(detailed_df, plots_dir):
    """
    Matriz de comparação probabilística: P(Algoritmo_i > Algoritmo_j)
    
    Permite comparações rigorosas entre algoritmos usando a abordagem Bayesiana.
    Valores > 0.95 indicam evidência forte de superioridade.
    
    Args:
        detailed_df: DataFrame com resultados detalhados
        plots_dir: Diretório para salvar plots
    """
    print("   📊 Gerando matriz de comparação probabilística...")
    
    algorithms = detailed_df['algorithm'].unique()
    n_algos = len(algorithms)
    
    # Calcular posteriors para cada algoritmo
    posteriors = {}
    for algorithm in algorithms:
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        total_tp = algo_data['tp'].sum()
        total_tn = algo_data['tn'].sum()
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        
        sens_post = stats.beta(total_tp + 1, total_fn + 1)
        spec_post = stats.beta(total_tn + 1, total_fp + 1)
        
        n_samples = 100000
        sens_samples = sens_post.rvs(n_samples)
        spec_samples = spec_post.rvs(n_samples)
        ba_samples = 0.5 * (sens_samples + spec_samples)
        
        posteriors[algorithm] = ba_samples
    
    # Matriz de comparações
    comparison_matrix = pd.DataFrame(
        index=algorithms,
        columns=algorithms,
        dtype=float
    )
    
    for alg1 in algorithms:
        for alg2 in algorithms:
            if alg1 == alg2:
                comparison_matrix.loc[alg1, alg2] = 0.5
            else:
                # P(alg1 > alg2)
                prob = np.mean(posteriors[alg1] > posteriors[alg2])
                comparison_matrix.loc[alg1, alg2] = prob
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(comparison_matrix.astype(float), annot=True, fmt='.3f', 
                cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
                square=True, linewidths=1,
                ax=ax, cbar_kws={"shrink": 0.8, 'label': 'P(Row > Column)'})
    
    ax.set_title('Probabilistic Comparison Matrix: P(Row Algorithm > Column Algorithm)\n' +
                 '(Bayesian approach - Brodersen et al., 2010)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Algorithm (Column)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Algorithm (Row)', fontsize=12, fontweight='bold')
    
    # Legenda explicativa
    textstr = 'Interpretation:\n' \
              'Green (>0.95): Strong evidence Row > Column\n' \
              'Red (<0.05): Strong evidence Column > Row\n' \
              'Yellow (~0.5): No clear difference'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'bayesian_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvar matriz como CSV
    tables_dir = plots_dir.parent / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_matrix.to_csv(tables_dir / 'bayesian_comparison_matrix.csv')
    
    # Gerar markdown com interpretação
    with open(tables_dir / 'bayesian_comparison_matrix.md', 'w') as f:
        f.write("# Probabilistic Comparison Matrix (Bayesian)\n\n")
        f.write("## P(Row Algorithm > Column Algorithm)\n\n")
        f.write(comparison_matrix.to_markdown(floatfmt='.3f'))
        f.write("\n\n## Interpretation\n\n")
        f.write("- **P > 0.95**: Strong evidence that Row algorithm is better than Column\n")
        f.write("- **P < 0.05**: Strong evidence that Column algorithm is better than Row\n")
        f.write("- **0.05 < P < 0.95**: Inconclusive or small difference\n")
        f.write("- **P ≈ 0.5**: No difference between algorithms\n\n")
        f.write("Reference: Brodersen et al. (2010) - Bayesian posterior distributions\n")
    
    print("   ✅ Matriz de comparação probabilística gerada")


def generate_bayesian_statistics_table(detailed_df, plots_dir):
    """
    Gera tabela com estatísticas Bayesianas completas.
    
    Args:
        detailed_df: DataFrame com resultados detalhados
        plots_dir: Diretório para salvar (vai salvar em tables/)
    """
    print("   📊 Gerando tabela de estatísticas Bayesianas...")
    
    algorithms = detailed_df['algorithm'].unique()
    bayesian_stats = []
    
    for algorithm in algorithms:
        algo_data = detailed_df[detailed_df['algorithm'] == algorithm]
        
        # Agregar confusion matrix
        total_tp = algo_data['tp'].sum()
        total_tn = algo_data['tn'].sum()
        total_fp = algo_data['fp'].sum()
        total_fn = algo_data['fn'].sum()
        
        # Posteriors
        sens_post = stats.beta(total_tp + 1, total_fn + 1)
        spec_post = stats.beta(total_tn + 1, total_fp + 1)
        
        # Amostrar BA
        n_samples = 100000
        sens_samples = sens_post.rvs(n_samples)
        spec_samples = spec_post.rvs(n_samples)
        ba_samples = 0.5 * (sens_samples + spec_samples)
        
        # Estatísticas
        bayesian_stats.append({
            'algorithm': algorithm,
            'ba_mean': np.mean(ba_samples),
            'ba_median': np.median(ba_samples),
            'ba_std': np.std(ba_samples),
            'ba_ci_lower': np.percentile(ba_samples, 2.5),
            'ba_ci_upper': np.percentile(ba_samples, 97.5),
            'sensitivity_mean': sens_post.mean(),
            'specificity_mean': spec_post.mean(),
            'prob_ba_above_80': np.mean(ba_samples > 0.80),
            'prob_ba_above_85': np.mean(ba_samples > 0.85),
            'prob_ba_above_90': np.mean(ba_samples > 0.90),
            'prob_ba_above_95': np.mean(ba_samples > 0.95)
        })
    
    # Criar DataFrame
    stats_df = pd.DataFrame(bayesian_stats).sort_values('ba_mean', ascending=False)
    
    # Salvar CSV
    tables_dir = plots_dir.parent / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(tables_dir / 'bayesian_statistics.csv', index=False, float_format='%.4f')
    
    # Salvar Markdown
    with open(tables_dir / 'bayesian_statistics.md', 'w') as f:
        f.write("# Bayesian Statistics (Brodersen et al., 2010)\n\n")
        f.write("## Balanced Accuracy - Posterior Distributions\n\n")
        f.write(stats_df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n\n## Columns Description\n\n")
        f.write("- **ba_mean**: Mean of BA posterior distribution\n")
        f.write("- **ba_median**: Median of BA posterior\n")
        f.write("- **ba_std**: Standard deviation of BA posterior\n")
        f.write("- **ba_ci_lower/upper**: 95% Bayesian credibility interval\n")
        f.write("- **sensitivity_mean**: Mean sensitivity (TP rate)\n")
        f.write("- **specificity_mean**: Mean specificity (TN rate)\n")
        f.write("- **prob_ba_above_X**: Probability that BA > X threshold\n\n")
        f.write("Reference: Brodersen, K.H., et al. (2010). 'The balanced accuracy and its posterior distribution'. ICPR.\n")
    
    print("   ✅ Tabela de estatísticas Bayesianas gerada")


def generate_all_bayesian_plots(detailed_df, plots_dir):
    """
    Gera todos os plots e tabelas Bayesianos.
    
    Args:
        detailed_df: DataFrame com resultados detalhados (precisa ter tp, tn, fp, fn)
        plots_dir: Diretório para salvar plots
    """
    print("\n🔬 ANÁLISES BAYESIANAS (Brodersen et al., 2010)")
    print("=" * 60)
    
    try:
        plot_posterior_distributions(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar distribuições posteriores: {e}")
    
    try:
        plot_credibility_intervals(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar intervalos de credibilidade: {e}")
    
    try:
        plot_probabilistic_comparison_matrix(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar matriz de comparação: {e}")
    
    try:
        generate_bayesian_statistics_table(detailed_df, plots_dir)
    except Exception as e:
        print(f"   ⚠️  Erro ao gerar tabela de estatísticas: {e}")
    
    print("=" * 60)
    print("✅ Análises Bayesianas completas!\n")

