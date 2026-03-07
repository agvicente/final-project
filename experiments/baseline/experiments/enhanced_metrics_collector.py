#!/usr/bin/env python3
"""
Sistema aprimorado de coleta de métricas para análise IoT-IDS
Coleta dados detalhados para análises avançadas de performance computacional
"""

import time
import psutil
import platform
import numpy as np
from datetime import datetime
import json

def get_system_info():
    """
    Coleta informações detalhadas do sistema para contexto das análises
    
    Returns:
        dict: Informações completas do sistema
    """
    try:
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        system_info = {
            # Sistema operacional
            'os': platform.system(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            
            # CPU
            'cpu_model': platform.processor(),
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'cpu_freq_current_mhz': cpu_freq.current if cpu_freq else None,
            'cpu_freq_max_mhz': cpu_freq.max if cpu_freq else None,
            
            # Memória
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent,
            
            # Python
            'python_version': platform.python_version(),
            
            # Timestamp
            'collection_timestamp': datetime.now().isoformat(),
        }
        
        return system_info
        
    except Exception as e:
        return {
            'error': str(e),
            'collection_timestamp': datetime.now().isoformat()
        }

def measure_prediction_latency_detailed(model, X_test, batch_sizes=[1, 10, 100, 1000]):
    """
    Mede latência de predição em diferentes cenários
    Importante para IoT: latência por amostra e throughput
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        batch_sizes: Tamanhos de batch para testar
        
    Returns:
        dict: Métricas detalhadas de latência
    """
    results = {
        'single_sample_latencies': [],
        'batch_latencies': {},
        'percentiles': {},
        'throughput': {}
    }
    
    # 1. Latência por amostra individual (primeira 100 amostras ou menos)
    n_samples = min(100, len(X_test))
    single_latencies = []
    
    for i in range(n_samples):
        start = time.perf_counter()
        _ = model.predict(X_test[i:i+1])
        latency = time.perf_counter() - start
        single_latencies.append(latency)
    
    results['single_sample_latencies'] = single_latencies
    
    # 2. Percentis de latência
    results['percentiles'] = {
        'p50': np.percentile(single_latencies, 50),
        'p90': np.percentile(single_latencies, 90),
        'p95': np.percentile(single_latencies, 95),
        'p99': np.percentile(single_latencies, 99),
        'min': np.min(single_latencies),
        'max': np.max(single_latencies),
        'mean': np.mean(single_latencies),
        'std': np.std(single_latencies)
    }
    
    # 3. Latência e throughput por tamanho de batch
    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue
            
        batch_times = []
        n_batches = min(10, len(X_test) // batch_size)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            start = time.perf_counter()
            _ = model.predict(X_test[start_idx:end_idx])
            batch_time = time.perf_counter() - start
            batch_times.append(batch_time)
        
        avg_batch_time = np.mean(batch_times)
        results['batch_latencies'][f'batch_{batch_size}'] = {
            'total_time': avg_batch_time,
            'time_per_sample': avg_batch_time / batch_size,
            'samples': batch_size
        }
        
        # Throughput (amostras por segundo)
        results['throughput'][f'batch_{batch_size}'] = batch_size / avg_batch_time
    
    return results

def monitor_resource_usage_detailed(process=None):
    """
    Monitora uso de recursos de forma detalhada durante execução
    
    Args:
        process: Processo a monitorar (None = processo atual)
        
    Returns:
        dict: Métricas detalhadas de recursos
    """
    if process is None:
        process = psutil.Process()
    
    try:
        cpu_times = process.cpu_times()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Informações de sistema
        system_memory = psutil.virtual_memory()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        return {
            # Memória do processo
            'process_memory_rss_mb': memory_info.rss / (1024**2),
            'process_memory_vms_mb': memory_info.vms / (1024**2),
            'process_memory_percent': memory_percent,
            
            # CPU do processo
            'process_cpu_percent': cpu_percent,
            'process_cpu_user_time': cpu_times.user,
            'process_cpu_system_time': cpu_times.system,
            'process_cpu_total_time': cpu_times.user + cpu_times.system,
            
            # Sistema geral
            'system_memory_available_mb': system_memory.available / (1024**2),
            'system_memory_percent': system_memory.percent,
            'system_cpu_percent': psutil.cpu_percent(interval=0.1),
            
            # Timestamp
            'timestamp': time.time()
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': time.time()
        }

def calculate_complexity_metrics(X_train, X_test, training_time, prediction_time):
    """
    Calcula métricas de complexidade computacional
    
    Args:
        X_train: Dados de treino
        X_test: Dados de teste
        training_time: Tempo de treinamento
        prediction_time: Tempo de predição
        
    Returns:
        dict: Métricas de complexidade
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    
    return {
        # Dataset
        'n_train_samples': n_train,
        'n_test_samples': n_test,
        'n_features': n_features,
        'dataset_size_mb': (X_train.nbytes + X_test.nbytes) / (1024**2),
        
        # Performance normalizada por tamanho
        'training_time_per_sample': training_time / max(n_train, 1),
        'training_time_per_sample_feature': training_time / max(n_train * n_features, 1),
        'prediction_time_per_sample': prediction_time / max(n_test, 1),
        
        # Throughput
        'training_throughput': n_train / max(training_time, 0.001),  # samples/sec
        'prediction_throughput': n_test / max(prediction_time, 0.001),  # samples/sec
        
        # Estimativas de escalabilidade
        'estimated_time_1M_samples': (training_time / n_train) * 1_000_000 if n_train > 0 else None,
        'estimated_memory_1M_samples_mb': (X_train.nbytes / n_train) * 1_000_000 / (1024**2) if n_train > 0 else None,
    }

def calculate_iot_specific_metrics(result_dict, system_info):
    """
    Calcula métricas específicas para ambientes IoT
    
    Args:
        result_dict: Dicionário com resultados básicos
        system_info: Informações do sistema
        
    Returns:
        dict: Métricas IoT específicas
    """
    # Extrair métricas básicas
    f1 = result_dict.get('f1_score', 0)
    training_time = result_dict.get('training_time', 0)
    prediction_time = result_dict.get('prediction_time', 0)
    memory_mb = result_dict.get('memory_usage_mb', 0)
    
    fp = result_dict.get('confusion_matrix', {}).get('fp', 0)
    fn = result_dict.get('confusion_matrix', {}).get('fn', 0)
    tn = result_dict.get('confusion_matrix', {}).get('tn', 0)
    tp = result_dict.get('confusion_matrix', {}).get('tp', 0)
    
    total_time = training_time + prediction_time
    
    return {
        # Eficiência de recursos
        'resource_efficiency_score': f1 / max(total_time * memory_mb, 0.001),
        'memory_efficiency': f1 / max(memory_mb, 0.001),
        'time_efficiency': f1 / max(total_time, 0.001),
        
        # Custo de erros
        'false_alarm_rate': fp / max(fp + tn, 1),
        'miss_rate': fn / max(fn + tp, 1),
        'false_alarm_cost': fp,  # Pode ser ponderado depois
        'miss_cost': fn,  # Pode ser ponderado depois
        
        # Adequação para tempo real
        'is_realtime_capable_strict': prediction_time < 0.01,  # <10ms
        'is_realtime_capable_moderate': prediction_time < 0.1,  # <100ms
        'is_realtime_capable_relaxed': prediction_time < 1.0,  # <1s
        
        # Score de adequação para IoT (0-1, maior é melhor)
        'iot_suitability_score': calculate_iot_score(
            f1, training_time, prediction_time, memory_mb
        ),
        
        # Características do ambiente
        'cpu_cores_available': system_info.get('cpu_cores_logical', 1),
        'memory_constraint_ratio': memory_mb / max(system_info.get('total_memory_gb', 1) * 1024, 1),
    }

def calculate_iot_score(f1, training_time, prediction_time, memory_mb):
    """
    Calcula score de adequação para IoT (0-1)
    Balanceia performance, velocidade e uso de recursos
    
    Fórmula: IoT_Score = (Performance * Speed * Memory_Efficiency)^(1/3)
    """
    # Normalizar performance (0-1)
    perf_score = min(f1, 1.0)
    
    # Normalizar velocidade (inversamente proporcional ao tempo)
    # Penalizar mais fortemente tempos altos
    total_time = training_time + prediction_time
    speed_score = 1.0 / (1.0 + np.log10(max(total_time, 0.1)))
    
    # Normalizar memória (inversamente proporcional)
    # Assumir 1GB como limite razoável para IoT
    memory_score = 1.0 / (1.0 + (memory_mb / 1024.0))
    
    # Score composto (média geométrica)
    iot_score = (perf_score * speed_score * memory_score) ** (1/3)
    
    return float(iot_score)

def collect_enhanced_metrics(model, X_train, X_test, y_train, y_test, y_pred,
                             training_time, prediction_time, memory_usage_mb,
                             basic_results, batch_sizes=[1, 10, 100]):
    """
    Função principal para coletar todas as métricas aprimoradas
    
    Args:
        model: Modelo treinado
        X_train, X_test, y_train, y_test: Dados
        y_pred: Predições
        training_time, prediction_time, memory_usage_mb: Métricas básicas
        basic_results: Dicionário com resultados básicos já calculados
        batch_sizes: Tamanhos de batch para teste de latência
        
    Returns:
        dict: Métricas completas aprimoradas
    """
    # Coletar informações do sistema
    system_info = get_system_info()
    
    # Métricas de complexidade
    complexity_metrics = calculate_complexity_metrics(
        X_train, X_test, training_time, prediction_time
    )
    
    # Métricas IoT específicas
    iot_metrics = calculate_iot_specific_metrics(basic_results, system_info)
    
    # Latência detalhada (apenas se não for muito custoso)
    latency_metrics = {}
    try:
        # Limitar tamanhos de batch no test mode
        if len(X_test) < 10000:  # Test mode
            batch_sizes = [1, 10, 100]
        else:  # Full mode
            batch_sizes = [1, 10, 100, 1000]
        
        latency_metrics = measure_prediction_latency_detailed(
            model, X_test, batch_sizes=batch_sizes
        )
    except Exception as e:
        latency_metrics = {'error': str(e)}
    
    # Monitoramento de recursos final
    resource_snapshot = monitor_resource_usage_detailed()
    
    # Compilar tudo
    enhanced_results = {
        **basic_results,
        'system_info': system_info,
        'complexity_metrics': complexity_metrics,
        'iot_metrics': iot_metrics,
        'latency_metrics': latency_metrics,
        'resource_snapshot': resource_snapshot,
    }
    
    return enhanced_results

# Para compatibilidade com código existente
__all__ = [
    'get_system_info',
    'measure_prediction_latency_detailed',
    'monitor_resource_usage_detailed',
    'calculate_complexity_metrics',
    'calculate_iot_specific_metrics',
    'calculate_iot_score',
    'collect_enhanced_metrics'
]

