import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import yaml
import json
import os
from scipy.stats import f_oneway

def load_config():
    """
    Carrega configuração para EDA seguindo padrão do pipeline DVC
    """
    config = {
        'input_file': '../data/processed/sampled.csv',
        'output_dir': 'src/eda/results',
        'plots_dir': 'src/eda/results/plots',
        'tables_dir': 'src/eda/results/tables',
        'label_col': 'Label',
        'random_state': 42
    }
    
    # Criar configuração se não existir
    os.makedirs('configs', exist_ok=True)
    config_path = 'configs/eda.yaml'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    return config

def setup_visualization():
    """
    Configuração inicial para visualizações
    """
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")
    warnings.filterwarnings('ignore')
    
    # Setup seed for reproducibility
    np.random.seed(42)
    
    # Setup pandas to display more columns
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)

def load_ciciot_dataset(data_path, sample_size=None, files_to_load=None):
    """
    Load and consolidate data from CICIOT dataset.
    Args:
        data_path (str): Path to the data file.
        sample_size (int): Number of samples to load.
        files_to_load (list): List of files to load.

    Returns:
        pd.DataFrame: Consolidated dataframe with all the data.
    """
    print(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        
        # Apply sampling if specified
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
        print(f"Dataset loaded: {df.shape}")
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def initial_data_inspection(df, output_dir):
    """Initial data inspection"""
    print("="*50)
    print("Initial Data Inspection")
    print("="*50)

    # Basic info
    dimensions = df.shape
    memory_usage = float(round(df.memory_usage().sum() / 1024**2, 2))
    
    print(f"Dimensions: {dimensions}")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Data types
    data_types = df.dtypes.value_counts()
    print("\nData types:")
    print(data_types)

    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values: {missing_values.sum()}")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found")

    # Infinite values
    infinite_values = df.isin([np.inf, -np.inf]).sum()
    if infinite_values.sum() > 0:
        print(f"\nInfinite values: {infinite_values.sum()}")
        print(infinite_values[infinite_values > 0])
    else:
        print("\nNo infinite values found")

    # Constant columns
    constant_columns = df.nunique() == 1
    if constant_columns.sum() > 0:
        print(f"\nConstant columns: {constant_columns.sum()}")
        print(constant_columns[constant_columns])
    else:
        print("\nNo constant columns found")

    # Duplicated rows
    duplicated_rows = df.duplicated().sum()
    print(f"\nDuplicated rows: {duplicated_rows} ({(duplicated_rows/df.shape[0])*100:.2f}%)")

    # Columns information
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f" - {col}")
    
    # Save inspection results
    inspection_results = {
        "dimensions": [dimensions[0], dimensions[1]],
        "memory_usage_mb": memory_usage,
        "data_types": {str(dtype): count for dtype, count in data_types.to_dict().items()},
        "missing_values": missing_values.to_dict(),
        "infinite_values": infinite_values.to_dict(),
        "constant_columns": constant_columns[constant_columns].index.tolist(),
        "duplicated_rows": int(duplicated_rows),
        "duplicated_percentage": float((duplicated_rows/df.shape[0])*100),
        "columns": df.columns.tolist()
    }

    print("="*50)
    print("Inspection results")
    print("="*50)
    print(inspection_results)
    
    with open(f"{output_dir}/tables/data_inspection.json", 'w') as f:
        json.dump(inspection_results, f, indent=2)
    
    return inspection_results

def analyze_class_distribution(df, label_col, output_dir):
    """Detailed analysis of class distribution"""
    print("="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)

    # Class count
    class_counts = df[label_col].value_counts()
    class_percentages = df[label_col].value_counts(normalize=True) * 100

    # Create dataframe for visualization
    class_summary = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    }).sort_values('Count', ascending=False)

    # Display summary
    print("\nClass Distribution Summary:")
    print(class_summary)

    # Save class summary
    class_summary.to_csv(f"{output_dir}/tables/class_distribution.csv")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Bar chart
    class_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Class Distribution')
    axes[0,0].set_xlabel('Class')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Pie chart - group non-representative categories as 'others'
    threshold = 0.01
    class_percentages_grouped = class_percentages.copy()
    
    # Identify categories below threshold
    small_categories = class_percentages_grouped[class_percentages_grouped < threshold * 100]
    
    if len(small_categories) > 1:
        # Sum small categories
        others_sum = small_categories.sum()
        # Remove small categories
        class_percentages_grouped = class_percentages_grouped[class_percentages_grouped >= threshold * 100]
        # Add 'Others' category
        class_percentages_grouped['Others'] = others_sum
    
    # Convert back to counts for pie chart
    total_samples = len(df)
    class_counts_grouped = (class_percentages_grouped / 100 * total_samples).round().astype(int)
        
    class_counts_grouped.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Class Distribution (%) - Grouped')
    axes[0,1].set_ylabel('')

    # Log Bar chart
    class_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral', logy=True)
    axes[1,0].set_title('Class Distribution - Logarithmic Scale')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Count (log scale)')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Horizontal bar chart
    class_counts.plot(kind='barh', ax=axes[1,1], color='lightgreen')
    axes[1,1].set_title('Class Distribution - Horizontal')
    axes[1,1].set_xlabel('Count')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    return class_summary

def balance_analysis(df, label_col, output_dir):
    """Detailed analysis of class balance"""
    print("="*50)
    print("BALANCE ANALYSIS")
    print("="*50)

    class_counts = df[label_col].value_counts()

    total_samples = len(df)
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[-1]

    inbalance_ratio = class_counts.max() / class_counts.min()

    print(f"Balance Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Majority class: {majority_class} ({class_counts.max()} samples)")
    print(f"Minority class: {minority_class} ({class_counts.min()} samples)")
    print(f"Inbalance ratio: {inbalance_ratio:.2f}")

    # Save balance analysis
    balance_results = {
        'total_samples': total_samples,
        'majority_class': majority_class,
        'majority_count': int(class_counts.max()),
        'minority_class': minority_class,
        'minority_count': int(class_counts.min()),
        'inbalance_ratio': float(inbalance_ratio)
    }
    
    with open(f"{output_dir}/tables/balance_analysis.json", 'w') as f:
        json.dump(balance_results, f, indent=2)

    return balance_results

def comprehensive_statistical_analysis(df, output_dir):
    """
    Descriptive statistical analysis of the numerical columns of the dataset.
    """
    print("="*50)
    print("Descriptive statistical analysis of the numerical columns of the dataset.")
    print("="*50)

    # Separate numerical and categorical columns
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    if 'ts' in numeric_features:
        numeric_features.remove('ts')

    print(f"Analyzing {len(numeric_features)} numerical columns...")

    # Basic statistics
    desc_stats = df[numeric_features].describe()
    print(desc_stats)

    # Additional statistics
    additional_stats = pd.DataFrame({
        'Skewness': df[numeric_features].skew(),
        'Kurtosis': df[numeric_features].kurtosis(),
        'CV': df[numeric_features].std() / df[numeric_features].mean(),
        'IQR': df[numeric_features].quantile(0.75) - df[numeric_features].quantile(0.25),
        'Range': df[numeric_features].max() - df[numeric_features].min(),
        'Min': df[numeric_features].min(),
        'Max': df[numeric_features].max(),
    })

    # Combine statistics
    full_stats = pd.concat([desc_stats.T, additional_stats], axis=1)

    # Save statistics to csv
    full_stats.to_csv(f'{output_dir}/tables/descriptive_analysis.csv')

    # Identify special behaviored features
    print("\nHigh variability features: (CV > 1)")
    high_cv = additional_stats[additional_stats['CV'] > 1].sort_values('CV', ascending=False)
    print(high_cv)

    print("\nHigh skewness features: (|Skewness|> 2)")
    high_skew = additional_stats[abs(additional_stats['Skewness']) > 2].sort_values('Skewness', ascending=False)
    print(high_skew)

    # Save specific analyses
    high_cv.to_csv(f'{output_dir}/tables/high_cv_features.csv')
    high_skew.to_csv(f'{output_dir}/tables/high_skew_features.csv')

    # Plot feature distributions
    plot_feature_distribution(df, numeric_features, output_dir)

    return full_stats

def plot_feature_distribution(df, features, output_dir):
    """
    Plot the distribution of features in the dataset.
    """
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

    for i, feature in enumerate(features):
        if i < len(axes):
            df[feature].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue')
            axes[i].set_title(f"{feature}")
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

            # Add mean and median lines
            mean_value = df[feature].mean()
            median_value = df[feature].median()
            axes[i].axvline(mean_value, color='red', linestyle='--', label='Mean')
            axes[i].axvline(median_value, color='green', linestyle='--', label='Median')
            axes[i].legend()

    # Remove extra subplots if any
    for i in range(n_features, len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def correlation_analysis(df, output_dir):
    """
    Correlation analysis of the dataset.
    """
    print("="*50)
    print("Correlation analysis of the dataset.")
    print("="*50)

    # Select numeric features
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    if 'ts' in numeric_features:
        numeric_features.remove('ts')

    # Compute correlation matrix
    correlation_matrix = df[numeric_features].corr()

    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix - Features CICIoT2023')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Identify highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'Feature1': correlation_matrix.columns[i], 
                    'Feature2': correlation_matrix.columns[j], 
                    'Correlation': corr_val
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(by='Correlation', key=abs, ascending=False)

    print(f"\nHigh correlation pairs (|corr| > 0.7):")
    print(high_corr_df)

    # Save correlation analysis
    correlation_matrix.to_csv(f'{output_dir}/tables/correlation_matrix.csv')
    high_corr_df.to_csv(f'{output_dir}/tables/high_corr_pairs.csv', index=False)

    return correlation_matrix, high_corr_df

def attack_class_analysis(df, label_col, output_dir):
    """Analyze features by attack class"""
    print("=" * 50)
    print("Attack Class Analysis")
    print("=" * 50)
    
    # Select important numerical features
    important_features = ['Rate', 'AVG', 'Min', 'Max', 'Std', 'Tot size']
    
    # Statistics by class
    class_stats = df.groupby(label_col)[important_features].agg(['mean', 'std', 'median']).round(4)
    
    # Comparative visualizations
    n_features = len(important_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
    
    for i, feature in enumerate(important_features):
        if i < len(axes):
            df.boxplot(column=feature, by=label_col, ax=axes[i])
            axes[i].set_title(f'{feature} by Attack Class')
            axes[i].set_xlabel('Attack Class')
            axes[i].set_ylabel(feature)
            axes[i].tick_params(axis='x', rotation=90)
    
    # Remove extra subplots
    for i in range(n_features, len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/attack_class_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ANOVA test for significant differences
    anova_results = []
    classes = df[label_col].unique()

    for feature in important_features:
        groups = [df[df[label_col] == cls][feature].dropna() for cls in classes]
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({
            'Feature': feature,
            'F-statistic': f_stat,
            'p-value': p_value,
            'Significant': p_value < 0.05
        })

    anova_df = pd.DataFrame(anova_results)

    print("\nANOVA Results:")
    print(anova_df)

    # Save results
    class_stats.to_csv(f'{output_dir}/tables/class_stats.csv')
    anova_df.to_csv(f'{output_dir}/tables/anova_results.csv', index=False)

    return class_stats, anova_results

def run_eda(config_file='configs/eda.yaml'):
    """
    Run complete EDA analysis following DVC pipeline pattern
    """
    # Load configuration
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config()
    
    # Setup directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    os.makedirs(config['tables_dir'], exist_ok=True)
    
    # Setup visualization
    setup_visualization()
    
    print("Starting EDA Analysis...")
    print(f"Input file: {config['input_file']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Load data
    df = load_ciciot_dataset(config['input_file'])

    df = handle_missing_values(df)
    
    # Run all analyses
    results = {}
    
    # 1. Initial data inspection
    results['data_inspection'] = initial_data_inspection(df, config['output_dir'])
    
    # 2. Class distribution analysis
    results['class_distribution'] = analyze_class_distribution(df, config['label_col'], config['output_dir'])
    
    # 3. Balance analysis
    results['balance_analysis'] = balance_analysis(df, config['label_col'], config['output_dir'])
    
    # 4. Statistical analysis
    results['statistical_analysis'] = comprehensive_statistical_analysis(df, config['output_dir'])
    
    # 5. Correlation analysis
    corr_matrix, high_corr_df = correlation_analysis(df, config['output_dir'])
    results['correlation_analysis'] = {
        'high_correlation_pairs': len(high_corr_df),
        'max_correlation': float(high_corr_df['Correlation'].abs().max()) if len(high_corr_df) > 0 else 0
    }
    
    # 6. Attack class analysis
    class_stats, anova_results = attack_class_analysis(df, config['label_col'], config['output_dir'])
    results['attack_class_analysis'] = {
        'features_analyzed': len(anova_results),
        'significant_features': sum([r['Significant'] for r in anova_results])
    }
    
    # Save summary results
    with open(f"{config['output_dir']}/eda_summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("EDA Analysis completed successfully!")
    print(f"Results saved in: {config['output_dir']}")
    
    return results

def handle_missing_values(df):
    """
    Substitui valores NaN e infinitos pela moda de cada coluna.
    Exclui a coluna 'label' do tratamento.

    Args:
        df (pd.DataFrame): DataFrame de entrada
        
    Returns:
        pd.DataFrame: Novo DataFrame com valores ausentes e infinitos substituídos
    """
    # Substituir valores infinitos por NaN primeiro (criar novo DataFrame)
    df_processed = df.replace([np.inf, -np.inf], np.nan)

    # Para cada coluna, calcular a moda e substituir valores ausentes
    # Excluir a coluna 'label' do tratamento
    columns_to_process = [col for col in df_processed.columns if col != 'Label']

    for column in columns_to_process:
        print(f"Handling missing values for column: {column}")
        if df_processed[column].isnull().any():
            # Calcular a moda da coluna (ignorando valores NaN)
            mode_value = df_processed[column].mode()
            print(f"Mode value: {mode_value}")
            
            # Se a coluna tem moda, usar o primeiro valor da moda
            if not mode_value.empty:
                fill_value = mode_value.iloc[0]
            else:
                # Se não há valores válidos na coluna, usar 0 como fallback
                fill_value = 0
            
            # Substituir valores NaN pela moda (criar nova série)
            df_processed[column] = df_processed[column].fillna(fill_value)

    print(f"Valores ausentes tratados. Shape final: {df_processed.shape}")
    print(f"Colunas processadas: {columns_to_process}")
    return df_processed

if __name__ == "__main__":
    config = load_config()
    results = run_eda()
    print("EDA pipeline completed!")
