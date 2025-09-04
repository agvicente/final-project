import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import json
import os

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

def load_config():
    config = {
        'input_file': 'data/processed/sampled.csv',
        'output_dir': 'data/processed',
        # 'test_size': 0.2,
        # 'random_state': 42,
        # 'features_to_encode': ['device_type', 'protocol'], #TODO: analisar graficos sem encode e scaling para ver a diferenca e necesidade
        # 'features_to_scale': ['packet_size', 'duration', 'packet_count', 'bytes_per_packet', 'packets_per_second']
    }
    
    os.makedirs('config', exist_ok=True)
    with open('configs/preprocessing.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return config

def preprocess_data(config_file='configs/preprocessing.yaml'):
    with open(config_file) as f:
        #NOTE: safeload impede execucao de codigo malicioso (impede a criação de objetos arbitrários e execução de código)
        config = yaml.safe_load(f)
        
    print('Loading data...')
    df = pd.read_csv(config['input_file'])

    df = handle_missing_values(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Salvar o dataset preprocessado
    os.makedirs(config['output_dir'], exist_ok=True)
    output_file = os.path.join(config['output_dir'], 'preprocessed.csv')
    df.to_csv(output_file, index=False)
    print(f"Dataset preprocessado salvo em: {output_file}")
    
    # Criar estatísticas do dataset
    stats = {
        'shape': df.shape,
        'label_distribution': df['Label'].value_counts().to_dict(),
        'missing_values_per_column': df.isnull().sum().to_dict(),
        'total_missing_values': df.isnull().sum().sum()
    }
    
    return stats

if __name__ == "__main__":
    config = load_config()
    stats = preprocess_data()
    print(stats)