import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import json
import os

'''
Dúvidas:
    - É necessário amostrar os dados de forma que a proporção de dados que são a média na janela?
'''

'''
    - Criar arquivo metrics
    - Para cada arquivo:
        - Contar quantas amostras tem em cada arquivo ((filename)_samples) e salvar o arquivo metrics
        - Contar quantas amostras tem cada tipo de ataque em cada arquivo ((attack_name)_(filename)_attack_samples) e salvar o arquivo metrics
    - Com base nos valores por arquivo, calcular:
        - Quantas amostras tem contando todos os arquivos (total_samples) e salvar o arquivo metrics
        - Quantas amostras tem cada tipo de ataque juntando todos os arquivos ((attack_name)_attack_samples_total) e salvar o arquivo metrics
        - Porcentagem de cada tipo de ataque em relação ao total de dados ((attack_name)_attack_percentage_total) e salvar o arquivo metrics
    
Amostrar dados
    - Criar dataframe que será usado para juntar as amostras
    - Carregar o arquivo metrics
    - ler variável que define o tamanho da amostra (sampling_rate)
    - Para cada tipo de ataque:
        - Calcular quantas amostras devem ser tiradas para manter a proporção ((attack_name)_attack_percentage_total * total_samples * sampling_rate) ((attack_name)_samples_to_take)
    Enquanto o numero de amostras tiradas for menor que o numero de amostras que devem ser tiradas:
        - Escolher arquivo aleatoriamente
        - Para cada tipo de ataque:
            - Ler variável que define quantas amostras devem ser tiradas para aquele tipo de ataque ((attack_name)_samples_to_take)
            - Ler variável que define quantas amostras já foram tiradas para aquele tipo de ataque ((attack_name)_samples_taken)
            - se ((attack_name)_samples_taken) for menor que ((attack_name)_samples_to_take):
                - Retirar o numero de amostras para aquele tipo de ataque de forma que não ultrapasse o numero de amostras que devem ser tiradas para aquele tipo de ataque
                - Adicionar as amostras ao dataframe
                - Atualizar arquivo metrics com o numero de amostras tiradas total (em memoria)
                - Atualizar arquivo metrics com o numero de amostras tiradas para aquele tipo de ataque ((attack_name)_samples_taken) (em memoria)
            - se ((attack_name)_samples_taken) for maior que ((attack_name)_samples_to_take):
                - Remover amostras excedentes para aquele tipo de ataque
                - Atualizar arquivo metrics com o numero de amostras tiradas total (em memoria)
                - Atualizar arquivo metrics com o numero de amostras tiradas para aquele tipo de ataque ((attack_name)_samples_taken) (em memoria)
                - Sair do loop
        - Remover arquivo da memoria para carregar outro e reptir o processo ate sair do enquanto (while)
        - Consolidar dataframe
        - Consolidar arquivos metrics
    - Salvar dataframe com as amostras
    - Salvar arquivo sampled.csv
    - Salvar arquivo metrics
'''

def load_config():
    config = {
        'data_dir': 'data/raw/CSV/MERGED_CSV',
        'output_dir': 'data/processed',
        'sampling_rate': 0.1
    }
    
    os.makedirs('config', exist_ok=True)
    with open('configs/sampling.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return config

def preprocess_data(config_file='configs/sampling.yaml'):
    with open(config_file) as f:
        #NOTE: safeload impede execucao de codigo malicioso (impede a criação de objetos arbitrários e execução de código)
        config = yaml.safe_load(f)
        
    print('Loading data...')
    df = pd.read_csv(config['data_dir'])
    
    print(f"Original shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    feature_cols = [
        'device_type',
        'protocol', 
        'packet_size', 
        'duration', 
        'src_port', 
        'dst_port', 
        'packet_count',
        'hour',
        'day_of_week',
        'bytes_per_packet',
        'packets_per_second'
    ]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    print("Applying categoric encoding")
    encoders = {}
    for col in config['features_to_encode']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    print("Applying normalization...")
    scaler = StandardScaler()
    X[config['features_to_scale']] = scaler.fit_transform(X[config['features_to_scale']])
    
    y_binary = (y != 'normal').astype(int)
    y_multiclass = LabelEncoder().fit_transform(y)
    
    
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multiclass,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y_binary #TODO: ver problemas de balanceamento que essa linha pode causar (se o modelo multiclasse for treinado com esses dados eu posso ter problemas de balanceamento)
                            #TODO: ver StratifiedKFold do scikit-learn para validação cruzada estratificada multiclasse.
    )
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train_bin,
        'y_test_binary': y_test_bin,
        'y_train_multiclass': y_train_multi,
        'y_test_multiclass': y_test_multi
    }
    
    for name, data in datasets.items():
        filepath = f"{config['output_dir']}/encoders.pkl"
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.Series(data).to_csv(filepath, index=False, header=[name])
            
    joblib.dump(encoders, f"{config['output_dir']}/encoders.pkl")
    joblib.dump(scaler, f"{config['output_dir']}/scaler.pkl")
    
    stats = {
        'original_shape': df.shape,
        'processed_shape': X.shape,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'anomaly_rate': int(y_binary.mean()),
        'feature_names': list(X.columns),
        'target_distribution': {
            'normal': int((y_binary == 0).sum()),
            'anomaly': int((y_binary == 1).sum())
        }
    }
    
    with open(f"{config['output_dir']}/../metrics/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Preprocessing finished")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Anomaly rate: {stats['anomaly_rate']:.3f}")
    
    return stats

if __name__ == "__main__":
    config = load_config()
    stats = preprocess_data()
    
    