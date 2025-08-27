import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import json
import os

#NOTE: Posso fazer diferentes tipos de amostragem, tentando simular diferentes cenários do munto real. por exemplo, no mundo real a maioria dos dados são benignos, e a maioria dos ataques são raros.
'''
Dúvidas:
    - É necessário amostrar os dados de forma que a proporção de dados que são a média na janela?
'''

'''
Pontos de melhoria:
    - Selecão aleatoria de arquivos é ineficiente, pois pode haver arquivos com muitas amostras e outros com poucas amostras
    - Multiplas cargas e descargas de arquivos
    - O Loop while pode executar muitas vezes sem progresso
    - Remover amostras excedentes pode quebrar a aleatoriedade
    - Sair do loop pode deixar outros tipos de ataques incompletos
    - Não está claro como as amostras excedentes são selecionadas para remoção
    - Se um arquivo não contem determinado tipo de ataque, o loop pode ficar em loop infinito
    - Não á verificação se é possível atingir as quotas desejadas
    - Alguns arquivos podem ser sobreamostrados enquanto outros podem ser ignorados
    - Não está sendo garantido a cobertura equilibrada dos arquivos
'''

'''
Sugestoes de melhoria:
    - Calcular exatamente quantas amostras tirar de cada arquivo para cada tipo de ataque
    - Fazer uma unica passada por cada arqivo
    - Usar pandas.sample com random_state para reprodutibilidade
    - verificar se as quootas são atingiveis antes de iniciar
    - Redistribuir quotas quando um arquivo não tem amostras suficientes
    - Implementar fallback para tipos de ataques raros
'''

'''
FASE 1: Análise e coleta de métricas
    - Carregar configurações do arquivo config/sampling.yaml (metodo load_config)
    - Criar arquivo metrics.json
    - Para cada arquivo:
        - Contar quantas amostras tem em cada arquivo ((filename)_samples) e salvar no metrics.json
        - Contar quantas amostras tem cada tipo de ataque em cada arquivo ((attack_name)_(filename)_attack_samples) e salvar no metrics.json
    - Com base nos valores por arquivo, calcular:
        - Quantas amostras tem contando todos os arquivos (total_samples) e salvar no metrics.json
        - Quantas amostras tem cada tipo de ataque juntando todos os arquivos ((attack_name)_attack_samples_total) e salvar no metrics.json
        - Porcentagem de cada tipo de ataque em relação ao total de dados ((attack_name)_attack_percentage_total) e salvar no metrics.json

FASE 2: Planejamento de amostragem estratificada
    - Carregar arquivo metrics.json
    - Ler variável que define o tamanho da amostra (sampling_rate)
    - Para cada tipo de ataque:
        - Calcular quantas amostras totais devem ser tiradas para manter a proporção ((attack_name)_attack_percentage_total * total_samples * sampling_rate) ((attack_name)_samples_target_total)
    
    - Para cada arquivo e cada tipo de ataque:
        - Calcular quota proporcional inicial: ((attack_name)_(filename)_attack_samples / (attack_name)_attack_samples_total) * (attack_name)_samples_target_total ((attack_name)_(filename)_quota_initial)
        - Ajustar quota para não exceder amostras disponíveis: min((attack_name)_(filename)_quota_initial, (attack_name)_(filename)_attack_samples) ((attack_name)_(filename)_quota_final)

FASE 3: Verificação de viabilidade e redistribuição
    - Para cada tipo de ataque:
        - Somar todas as quotas finais por arquivo ((attack_name)_total_achievable)
        - Se ((attack_name)_total_achievable) for menor que ((attack_name)_samples_target_total):
            - Identificar arquivos com quotas não saturadas (ainda têm amostras disponíveis)
            - Calcular déficit: ((attack_name)_samples_target_total) - ((attack_name)_total_achievable)
            - Redistribuir o déficit proporcionalmente entre arquivos não saturados, respeitando limites de amostras disponíveis
            - Atualizar ((attack_name)_(filename)_quota_final) para arquivos beneficiados
        - Se após redistribuição ainda houver déficit:
            - Marcar tipo de ataque como "raro" e aplicar fallback
            - Para tipos de ataque raros: usar todas as amostras disponíveis de todos os arquivos
            - Registrar no metrics.json o tipo de ataque como "fallback_applied"

FASE 4: Amostragem determinística
    - Criar dataframe vazio para consolidar amostras
    - Definir random_state fixo para reprodutibilidade
    - Para cada arquivo:
        - Carregar arquivo uma única vez
        - Para cada tipo de ataque presente no arquivo:
            - Obter ((attack_name)_(filename)_quota_final)
            - Se quota for maior que zero:
                - Filtrar dados do tipo de ataque específico
                - Usar pandas.sample(n=quota_final, random_state=random_state) para amostrar
                - Adicionar amostras ao dataframe consolidado
                - Atualizar métricas de amostras coletadas ((attack_name)_samples_collected_total)
        - Descarregar arquivo da memória

FASE 5: Consolidação e salvamento
    - Verificar se todas as quotas foram atingidas ou se houve aplicação de fallback
    - Consolidar métricas finais:
        - Total de amostras coletadas (total_samples_collected)
        - Amostras coletadas por tipo de ataque ((attack_name)_samples_collected_total)
        - Proporção final de cada tipo de ataque ((attack_name)_final_percentage)
        - Lista de tipos de ataque com fallback aplicado (fallback_attacks)
    - Salvar dataframe consolidado como sampled.csv
    - Salvar arquivo metrics.json atualizado com métricas de coleta
    - Registrar log de operação com detalhes da amostragem realizada
'''

def load_config():
    """
    Carrega ou cria configurações padrão para o processo de amostragem.
    
    Returns:
        dict: Dicionário com configurações padrão incluindo diretórios de dados,
              arquivo de métricas e taxa de amostragem.
    """
    config = {
        'data_dir': 'data/raw/CSV/MERGED_CSV',
        'data_output_dir': 'data/processed',
        'metrics_file': 'data/metrics/metrics.json',
        'sampling_rate': 0.1
    }
    
    os.makedirs('config', exist_ok=True)
    with open('configs/sampling.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return config


def analyze_and_collect_metrics(config=None, label_column='Label'):
    """
    FASE 1: Análise e coleta de métricas do dataset CICIoT.
    
    Esta função implementa a primeira fase do algoritmo de amostragem estratificada,
    realizando a análise completa dos arquivos CSV disponíveis e coletando métricas
    essenciais sobre distribuição de amostras e tipos de ataques.
    
    Args:
        config (dict, optional): Configurações do sistema. Se None, carrega configurações
                               padrão através de load_config().
        label_column (str): Nome da coluna que contém os rótulos dos tipos de ataque.
                           Default: 'Label'.
    
    Returns:
        dict: Dicionário contendo todas as métricas coletadas, incluindo:
            - Contagem de amostras por arquivo: {filename}_samples
            - Contagem de ataques por arquivo: {attack_name}_{filename}_attack_samples  
            - Total de amostras: total_samples
            - Total por tipo de ataque: {attack_name}_attack_samples_total
            - Porcentagens: {attack_name}_attack_percentage_total
    
    Raises:
        FileNotFoundError: Se o diretório de dados não for encontrado.
        ValueError: Se nenhum arquivo CSV for encontrado no diretório.
        KeyError: Se a coluna de rótulos não existir nos arquivos.
    
    Example:
        >>> config = load_config()
        >>> metrics = analyze_and_collect_metrics(config)
        >>> print(f"Total de amostras: {metrics['total_samples']}")
    """
    # Carrega configurações se não fornecidas
    if config is None:
        config = load_config()
    
    # Cria diretório para métricas se não existir
    metrics_dir = os.path.dirname(config['metrics_file'])
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Verifica se diretório de dados existe
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")
    
    # Lista arquivos CSV no diretório
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em: {data_dir}")
    
    print(f"Iniciando análise de {len(csv_files)} arquivos CSV...")
    
    # Dicionário para armazenar métricas
    metrics = {}
    metrics['perfile'] = {}
    metrics['total'] = {}
    metrics['total']['attack_samples'] = {}
    metrics['total']['attack_percentage'] = {}
    
    # Estruturas para acumular totais
    total_samples = 0
    attack_totals = {}
    
    # FASE 1.1: Análise por arquivo
    for i, filename in enumerate(csv_files, 1):
        print(f"Processando arquivo {i}/{len(csv_files)}: {filename}")
        
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Carrega apenas a coluna de rótulos para eficiência
            df = pd.read_csv(file_path, usecols=[label_column])
            
            # Remove extensão do filename para usar como identificador
            file_id = filename.replace('.csv', '')
            metrics['perfile'][f"{file_id}"] = {}
            
            # Conta total de amostras no arquivo
            file_samples = len(df)
            metrics['perfile'][f"{file_id}"]["samples"] = file_samples
            total_samples += file_samples
            
            # Conta amostras por tipo de ataque no arquivo
            attack_counts = df[label_column].value_counts()
            
            for attack_type, count in attack_counts.items():
                # Sanitiza nome do tipo de ataque (remove caracteres especiais)
                clean_attack_name = str(attack_type).replace(' ', '_').replace('-', '_').replace('.', '_')
                
                # Métrica por arquivo e tipo de ataque
                metric_key = f"{clean_attack_name}"
                metrics['perfile'][f"{file_id}"][metric_key] = int(count)
                
                # Acumula para total geral
                if clean_attack_name not in attack_totals:
                    attack_totals[clean_attack_name] = 0
                attack_totals[clean_attack_name] += int(count)
            
            print(f"  - {file_samples:,} amostras, {len(attack_counts)} tipos de ataque únicos")
            
        except KeyError:
            raise KeyError(f"Coluna '{label_column}' não encontrada no arquivo {filename}")
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
            continue
    
    # FASE 1.2: Cálculo de totais e porcentagens
    print("\nCalculando métricas totais...")
    
    # Total de amostras em todos os arquivos
    metrics['total']['samples'] = total_samples
    
    # Total de amostras por tipo de ataque
    for attack_type, total_count in attack_totals.items():
        metrics['total']['attack_samples'][f"{attack_type}"] = total_count
        
        # Calcula porcentagem do tipo de ataque em relação ao total
        percentage = (total_count / total_samples) * 100 if total_samples > 0 else 0
        metrics['total']['attack_percentage'][f"{attack_type}"] = round(percentage, 4)
    
    # FASE 1.3: Salvamento do arquivo metrics.json
    print(f"\nSalvando métricas em: {config['metrics_file']}")
    
    with open(config['metrics_file'], 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Resumo das métricas coletadas
    unique_attacks = len(attack_totals)
    print(f"\n--- RESUMO DA FASE 1 ---")
    print(f"Arquivos processados: {len(csv_files)}")
    print(f"Total de amostras: {total_samples:,}")
    print(f"Tipos de ataque únicos: {unique_attacks}")
    print(f"Arquivo de métricas salvo: {config['metrics_file']}")
    
    # Exibe distribuição dos tipos de ataque mais comuns
    print(f"\nTop 5 tipos de ataque mais frequentes:")
    sorted_attacks = sorted(attack_totals.items(), key=lambda x: x[1], reverse=True)[:5]
    for attack_type, count in sorted_attacks:
        percentage = (count / total_samples) * 100
        print(f"  {attack_type}: {count:,} ({percentage:.2f}%)")
    
    return metrics



if __name__ == "__main__":
    config = load_config()
    metrics = analyze_and_collect_metrics(config)
    print(metrics)