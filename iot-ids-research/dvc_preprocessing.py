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

def normalize_data(df, output_dir):
    """
    Normaliza as features usando StandardScaler, excluindo a coluna 'Label'.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        output_dir (str): Diretório para salvar o scaler
        
    Returns:
        pd.DataFrame: DataFrame com features normalizadas
    """
    # Separar features da target variable
    feature_columns = [col for col in df.columns if col != 'Label']
    features = df[feature_columns]
    target = df['Label']
    
    print(f"Normalizando {len(feature_columns)} features...")
    print(f"Features: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
    
    # Aplicar StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Criar DataFrame com features normalizadas
    df_normalized = pd.DataFrame(features_scaled, columns=feature_columns, index=df.index)
    
    # Adicionar a coluna target de volta
    df_normalized['Label'] = target
    
    # Salvar o scaler para uso futuro
    os.makedirs(output_dir, exist_ok=True)
    scaler_file = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_file)
    print(f"Scaler salvo em: {scaler_file}")
    
    print(f"Normalização concluída. Shape final: {df_normalized.shape}")
    
    # Mostrar estatísticas das features normalizadas
    print("Estatísticas após normalização:")
    print(f"Média das features: {df_normalized[feature_columns].mean().mean():.6f}")
    print(f"Desvio padrão das features: {df_normalized[feature_columns].std().mean():.6f}")
    
    return df_normalized

def create_binary_labels(df):
    """
    Cria labels binárias para detecção de anomalias:
    - BENIGN → 0 (benigno)
    - Qualquer outra label → 1 (malicioso)
    
    Args:
        df (pd.DataFrame): DataFrame com coluna 'Label' original
        
    Returns:
        pd.DataFrame: DataFrame com coluna adicional 'Binary_Label'
    """
    df_binary = df.copy()
    
    # Criar labels binárias
    df_binary['Binary_Label'] = df['Label'].apply(
        lambda x: 0 if 'BENIGN' in str(x).upper() else 1
    )
    
    return df_binary

def split_and_normalize_data(df, config):
    """
    ✅ ABORDAGEM CORRETA: Separa treino/teste ANTES da normalização
    
    Evita data leakage fazendo com que o scaler aprenda parâmetros 
    APENAS dos dados de treino.
    
    Args:
        df (pd.DataFrame): DataFrame limpo (após handle_missing_values)
        config (dict): Configurações com test_size, random_state, etc.
        
    Returns:
        dict: Contém X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
    """
    print(f"\n{'='*60}")
    print("✅ SEPARAÇÃO TREINO/TESTE ANTES DA NORMALIZAÇÃO")
    print("   (Evitando Data Leakage)")
    print("="*60)
    
    # Separar features e targets
    feature_columns = [col for col in df.columns if col not in ['Label', 'Binary_Label']]
    X = df[feature_columns]
    y = df['Label']
    y_binary = df['Binary_Label']
    
    print(f"📊 Dataset completo:")
    print(f"   Total de amostras: {len(df):,}")
    print(f"   Número de features: {len(feature_columns)}")
    print(f"   Distribuição de labels: {dict(y.value_counts())}")
    print(f"   Distribuição binária: {dict(y_binary.value_counts())}")
    
    # 1️⃣ PRIMEIRO: Separar treino e teste
    stratify_param = y_binary if config.get('stratify', True) else None
    
    X_train, X_test, y_train, y_test, y_binary_train, y_binary_test = train_test_split(
        X, y, y_binary,
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=stratify_param
    )
    
    print(f"\n🔄 Separação realizada:")
    print(f"   Treino: {len(X_train):,} amostras ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Teste:  {len(X_test):,} amostras ({len(X_test)/len(df)*100:.1f}%)")
    print(f"   Labels treino: {dict(y_train.value_counts())}")
    print(f"   Labels teste:  {dict(y_test.value_counts())}")
    print(f"   Binary treino: {dict(y_binary_train.value_counts())}")
    print(f"   Binary teste:  {dict(y_binary_test.value_counts())}")
    
    # Estatísticas ANTES da normalização
    print(f"\n📈 Estatísticas ANTES da normalização:")
    print(f"   Treino - Média: {X_train.mean().mean():.4f}, Std: {X_train.std().mean():.4f}")
    print(f"   Teste  - Média: {X_test.mean().mean():.4f}, Std: {X_test.std().mean():.4f}")
    
    # 2️⃣ SEGUNDO: Normalizar (scaler aprende APENAS do treino)
    print(f"\n🔧 Normalizando dados...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit_transform no treino
    X_test_scaled = scaler.transform(X_test)        # apenas transform no teste
    
    # Estatísticas APÓS a normalização
    print(f"\n📈 Estatísticas APÓS a normalização:")
    print(f"   Treino - Média: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    print(f"   Teste  - Média: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")
    
    print(f"\n⚠️  IMPORTANTE:")
    print(f"   - Treino tem média ≈ 0 e std ≈ 1 (esperado)")
    print(f"   - Teste pode ter valores ligeiramente diferentes (NORMAL!)")
    print(f"   - Isso simula o cenário real de produção")
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_binary_train': y_binary_train,
        'y_binary_test': y_binary_test,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'train_indices': X_train.index,
        'test_indices': X_test.index
    }

def load_config():
    config = {
        'input_file': 'data/processed/sampled.csv',
        'output_dir': 'data/processed',
        'test_size': 0.2,  # 80/20 split (configurável)
        'random_state': 42,
        'stratify': True,  # Manter proporção de classes
        # 'features_to_encode': ['device_type', 'protocol'], #TODO: analisar graficos sem encode e scaling para ver a diferenca e necesidade
        # 'features_to_scale': ['packet_size', 'duration', 'packet_count', 'bytes_per_packet', 'packets_per_second']
    }
    
    os.makedirs('config', exist_ok=True)
    with open('configs/preprocessing.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return config

def save_split_data(split_results, config):
    """
    Salva os dados separados e normalizados em arquivos CSV e pickle
    
    Args:
        split_results (dict): Resultado da função split_and_normalize_data
        config (dict): Configurações do preprocessing
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair dados
    X_train_scaled = split_results['X_train_scaled']
    X_test_scaled = split_results['X_test_scaled']
    y_train = split_results['y_train']
    y_test = split_results['y_test']
    feature_columns = split_results['feature_columns']
    scaler = split_results['scaler']
    
    print(f"\n💾 Salvando arquivos em: {output_dir}")
    
    # 1. Salvar dados de treino como CSV
    train_df = pd.DataFrame(X_train_scaled, columns=feature_columns, index=y_train.index)
    train_df['Label'] = y_train
    train_file = os.path.join(output_dir, 'train_normalized.csv')
    train_df.to_csv(train_file, index=False)
    print(f"   ✅ Treino: {train_file}")
    
    # 2. Salvar dados de teste como CSV
    test_df = pd.DataFrame(X_test_scaled, columns=feature_columns, index=y_test.index)
    test_df['Label'] = y_test
    test_file = os.path.join(output_dir, 'test_normalized.csv')
    test_df.to_csv(test_file, index=False)
    print(f"   ✅ Teste:  {test_file}")
    
    # 3. Salvar scaler (CRUCIAL para novos dados)
    scaler_file = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_file)
    print(f"   ✅ Scaler: {scaler_file}")
    
    # 4. Salvar arrays numpy para ML (mais eficiente)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)
    print(f"   ✅ Arrays NumPy salvos para ML")
    
    # 5. Salvar metadados
    metadata = {
        'feature_columns': feature_columns,
        'train_indices': split_results['train_indices'].tolist(),
        'test_indices': split_results['test_indices'].tolist(),
        'scaler_params': {
            'mean': [float(x) for x in scaler.mean_],
            'scale': [float(x) for x in scaler.scale_]
        },
        'config': config
    }
    
    metadata_file = os.path.join(output_dir, 'preprocessing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadados: {metadata_file}")
    
    return {
        'train_file': train_file,
        'test_file': test_file,
        'scaler_file': scaler_file,
        'metadata_file': metadata_file
    }

def save_binary_data(split_results, config):
    """
    Salva os dados com labels binárias para detecção de anomalias
    
    Args:
        split_results (dict): Resultado da função split_and_normalize_data
        config (dict): Configurações do preprocessing
    """
    # Criar diretório para dados binários
    binary_dir = os.path.join(config['output_dir'], 'binary')
    os.makedirs(binary_dir, exist_ok=True)
    
    # Extrair dados
    X_train_scaled = split_results['X_train_scaled']
    X_test_scaled = split_results['X_test_scaled']
    y_binary_train = split_results['y_binary_train']
    y_binary_test = split_results['y_binary_test']
    feature_columns = split_results['feature_columns']
    scaler = split_results['scaler']
    
    # 1. Salvar dados de treino binários como CSV
    train_binary_df = pd.DataFrame(X_train_scaled, columns=feature_columns, index=y_binary_train.index)
    train_binary_df['Binary_Label'] = y_binary_train
    train_binary_file = os.path.join(binary_dir, 'train_binary.csv')
    train_binary_df.to_csv(train_binary_file, index=False)
    
    # 2. Salvar dados de teste binários como CSV
    test_binary_df = pd.DataFrame(X_test_scaled, columns=feature_columns, index=y_binary_test.index)
    test_binary_df['Binary_Label'] = y_binary_test
    test_binary_file = os.path.join(binary_dir, 'test_binary.csv')
    test_binary_df.to_csv(test_binary_file, index=False)
    
    # 3. Salvar arrays numpy para ML - versão binária
    np.save(os.path.join(binary_dir, 'X_train_binary.npy'), X_train_scaled)
    np.save(os.path.join(binary_dir, 'X_test_binary.npy'), X_test_scaled)
    np.save(os.path.join(binary_dir, 'y_train_binary.npy'), y_binary_train.values)
    np.save(os.path.join(binary_dir, 'y_test_binary.npy'), y_binary_test.values)
    
    # 4. Copiar scaler para o diretório binário
    scaler_binary_file = os.path.join(binary_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_binary_file)
    
    # 5. Salvar metadados específicos para versão binária
    binary_metadata = {
        'feature_columns': feature_columns,
        'binary_mapping': {'0': 'BENIGN', '1': 'MALICIOUS'},
        'binary_distribution_train': {str(k): int(v) for k, v in y_binary_train.value_counts().items()},
        'binary_distribution_test': {str(k): int(v) for k, v in y_binary_test.value_counts().items()},
        'config': config
    }
    
    binary_metadata_file = os.path.join(binary_dir, 'binary_metadata.json')
    with open(binary_metadata_file, 'w') as f:
        json.dump(binary_metadata, f, indent=2)
    
    return {
        'train_binary_file': train_binary_file,
        'test_binary_file': test_binary_file,
        'scaler_binary_file': scaler_binary_file,
        'binary_metadata_file': binary_metadata_file
    }

def preprocess_data(config_file='configs/preprocessing.yaml'):
    with open(config_file) as f:
        #NOTE: safeload impede execucao de codigo malicioso (impede a criação de objetos arbitrários e execução de código)
        config = yaml.safe_load(f)
        
    print('Loading data...')
    df = pd.read_csv(config['input_file'])

    print(f"Original shape: {df.shape}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Tratar valores ausentes
    df_clean = handle_missing_values(df)
    
    # Criar labels binárias para detecção de anomalias
    df_with_binary = create_binary_labels(df_clean)
    
    # ✅ Separar treino/teste ANTES da normalização
    split_results = split_and_normalize_data(df_with_binary, config)
    
    # Salvar arquivos com labels originais
    file_paths = save_split_data(split_results, config)
    
    # Salvar arquivos com labels binárias
    binary_paths = save_binary_data(split_results, config)
    
    # Criar estatísticas finais
    stats = {
        'original_shape': list(df.shape),
        'clean_shape': list(df_clean.shape),
        'train_shape': list(split_results['X_train_scaled'].shape),
        'test_shape': list(split_results['X_test_scaled'].shape),
        'label_distribution': {str(k): int(v) for k, v in df['Label'].value_counts().items()},
        'train_label_distribution': {str(k): int(v) for k, v in split_results['y_train'].value_counts().items()},
        'test_label_distribution': {str(k): int(v) for k, v in split_results['y_test'].value_counts().items()},
        'binary_train_distribution': {str(k): int(v) for k, v in split_results['y_binary_train'].value_counts().items()},
        'binary_test_distribution': {str(k): int(v) for k, v in split_results['y_binary_test'].value_counts().items()},
        'missing_values_treated': int(df.isnull().sum().sum()),
        'features_count': len(split_results['feature_columns']),
        'test_size_used': config['test_size'],
        'random_state': config['random_state'],
        'files_created': file_paths,
        'binary_files_created': binary_paths
    }
    
    print(f"\n📊 RESUMO DO PREPROCESSING:")
    print(f"   Amostras originais: {stats['original_shape'][0]:,}")
    print(f"   Features: {stats['features_count']}")
    print(f"   Treino: {stats['train_shape'][0]:,} amostras")
    print(f"   Teste: {stats['test_shape'][0]:,} amostras")
    print(f"   Split usado: {(1-config['test_size'])*100:.0f}/{config['test_size']*100:.0f}")
    
    return stats

if __name__ == "__main__":
    config = load_config()
    stats = preprocess_data()
    print(stats)