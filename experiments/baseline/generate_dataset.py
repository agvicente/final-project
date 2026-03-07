import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_iot_traffic_dataset(n_samples=100000):
    """Generate sintetic data"""

    np.random.seed(42)

    device_types = ['camera', 'senson', 'thermostat', 'smart_light', 'door_lock']
    protocols = ['TCP', 'UDP', 'HTTP', 'MQTT', 'CoAP']

    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(seconds=i*10) for i in range(n_samples)]

    normal_samples = int(n_samples * 0.9)

    normal_data = {
        'timestamp': timestamps[:normal_samples],
        'device_type': np.random.choice(device_types, normal_samples),
        'protocol': np.random.choice(protocols, normal_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'packet_size': np.random.normal(512, 128, normal_samples).astype(int),
        'duration': np.random.exponential(2.0, normal_samples),
        'src_port': np.random.randint(1024, 65535, normal_samples),
        'dst_port': np.random.choice([80, 443, 1883, 8080, 5683], normal_samples),
        'packet_count': np.random.poisson(10, normal_samples),
        'label': ['normal'] * normal_samples
    }

    anomaly_samples = n_samples - normal_samples
    attack_types = ['ddos', 'mirai', 'mitm', 'recon', 'spoofing']

    anomaly_data = {
        'timestamp': timestamps[normal_samples:],
        'device_type': np.random.choice(device_types, anomaly_samples),
        'protocol': np.random.choice(protocols, anomaly_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'packet_size': np.random.normal(1024, 512, anomaly_samples).astype(int),
        'duration': np.random.exponential(5.0, anomaly_samples),
        'src_port': np.random.randint(1, 1024, anomaly_samples),
        'dst_port': np.random.randint(1, 65535, anomaly_samples),
        'packet_count': np.random.poisson(50, anomaly_samples),
        'label': np.random.choice(attack_types, anomaly_samples)
    }

    all_data = {}

    for key in normal_data.keys():
        all_data[key] = list(normal_data[key]) + list(anomaly_data[key])
    
    df = pd.DataFrame(all_data)

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['bytes_per_packet'] = df['packet_size'] * df['packet_count']
    df['packets_per_second'] = df['packet_count'] / (df['duration'] + 0.001)

    df = df.sample(frac=1).reset_index(drop=True)

    return df

def create_multiple_datasets():

    datasets = {
        'v1_small': 10000,
        'v2_medium': 50000,
        'v3_large': 100000
    }
    
    os.makedirs('../data/raw', exist_ok=True)
    
    for version, size in datasets.items():
        print(f"Generating {version} with {size} samples...")
        df = generate_iot_traffic_dataset(size)
        
        filepath = f'../data/raw/iot_traffic_{version}.csv'
        df.to_csv(filepath, index=False)
        
        print(f"Saved: {filepath} ({df.shape[0]} rows, {df.shape[1]} columns)")
        print(f"Labels distribution: {df['label'].value_counts().to_dict()}")
        print()

if __name__ == "__main__":
    create_multiple_datasets()