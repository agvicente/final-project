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
        'data_dir': 'data/raw/CSV/MERGED_CSV/test',
        'data_output_dir': 'data/processed/test',
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
    metrics['per_file'] = {}
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
            metrics['per_file'][f"{file_id}"] = {}
            
            # Conta total de amostras no arquivo
            file_samples = len(df)
            metrics['per_file'][f"{file_id}"]["samples"] = file_samples
            total_samples += file_samples
            
            # Conta amostras por tipo de ataque no arquivo
            attack_counts = df[label_column].value_counts()
            
            for attack_type, count in attack_counts.items():
                # Sanitiza nome do tipo de ataque (remove caracteres especiais)
                clean_attack_name = str(attack_type).replace(' ', '_').replace('-', '_').replace('.', '_')
                
                # Métrica por arquivo e tipo de ataque
                metric_key = f"{clean_attack_name}"
                metrics['per_file'][f"{file_id}"][metric_key] = int(count)
                
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


def stratified_sampling_planning(config=None, metrics_file_path=None):
    """
    FASE 2: Planejamento de amostragem estratificada.
    
    Esta função implementa a segunda fase do algoritmo de amostragem estratificada,
    calculando quotas proporcionais de amostras por arquivo e tipo de ataque para
    manter a distribuição original dos dados durante a amostragem.
    
    Args:
        config (dict, optional): Configurações do sistema. Se None, carrega configurações
                               padrão através de load_config().
        metrics_file_path (str, optional): Caminho para o arquivo metrics.json. Se None,
                                         usa o caminho definido no config.
    
    Returns:
        dict: Dicionário contendo o planejamento de amostragem com estrutura:
            - sampling_config: configurações usadas (sampling_rate, etc.)
            - target_samples: amostras alvo por tipo de ataque
            - quotas: estrutura aninhada com quotas por arquivo e tipo de ataque
                - per_file: {file_id: {attack_type: {initial: X, final: Y, available: Z}}}
                - summary: resumo das quotas por tipo de ataque
    
    Raises:
        FileNotFoundError: Se o arquivo metrics.json não for encontrado.
        ValueError: Se o sampling_rate for inválido (deve estar entre 0 e 1).
        KeyError: Se a estrutura do metrics.json estiver incorreta.
    
    Example:
        >>> config = load_config()
        >>> # Primeiro execute a FASE 1
        >>> metrics = analyze_and_collect_metrics(config)
        >>> # Depois execute a FASE 2
        >>> sampling_plan = stratified_sampling_planning(config)
        >>> print(f"Taxa de amostragem: {sampling_plan['sampling_config']['sampling_rate']}")
    """
    # Carrega configurações se não fornecidas
    if config is None:
        config = load_config()
    
    # Define caminho do arquivo de métricas
    if metrics_file_path is None:
        metrics_file_path = config['metrics_file']
    
    # Verifica se arquivo de métricas existe
    if not os.path.exists(metrics_file_path):
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {metrics_file_path}")
    
    # Valida sampling_rate
    sampling_rate = config.get('sampling_rate', 0.1)
    if not (0 < sampling_rate <= 1):
        raise ValueError(f"sampling_rate deve estar entre 0 e 1, recebido: {sampling_rate}")
    
    print(f"Iniciando FASE 2: Planejamento de amostragem estratificada")
    print(f"Taxa de amostragem: {sampling_rate:.1%}")
    
    # Carrega métricas da FASE 1
    with open(metrics_file_path, 'r') as f:
        metrics = json.load(f)
    
    # Valida estrutura do arquivo de métricas
    required_keys = ['per_file', 'total']
    for key in required_keys:
        if key not in metrics:
            raise KeyError(f"Chave '{key}' não encontrada no arquivo de métricas")
    
    if 'attack_samples' not in metrics['total'] or 'attack_percentage' not in metrics['total']:
        raise KeyError("Estrutura 'attack_samples' ou 'attack_percentage' não encontrada em metrics['total']")
    
    # Inicializa estrutura de planejamento
    sampling_plan = {
        'sampling_config': {
            'sampling_rate': sampling_rate,
            'total_samples_original': metrics['total']['samples'],
            'total_samples_target': int(metrics['total']['samples'] * sampling_rate)
        },
        'target_samples': {},
        'quotas': {
            'per_file': {},
            'summary': {}
        }
    }
    
    total_samples = metrics['total']['samples']
    attack_samples = metrics['total']['attack_samples']
    attack_percentages = metrics['total']['attack_percentage']
    
    print(f"\nTotal de amostras originais: {total_samples:,}")
    print(f"Total de amostras alvo: {sampling_plan['sampling_config']['total_samples_target']:,}")
    
    # FASE 2.1: Calcular amostras alvo por tipo de ataque
    print(f"\n--- CALCULANDO AMOSTRAS ALVO POR TIPO DE ATAQUE ---")
    
    for attack_type, percentage in attack_percentages.items():
        # Calcula quantas amostras totais devem ser tiradas para manter a proporção
        samples_target = int((percentage / 100) * total_samples * sampling_rate)
        sampling_plan['target_samples'][attack_type] = samples_target
        
        print(f"{attack_type}: {samples_target:,} amostras alvo ({percentage:.2f}%)")
    
    # FASE 2.2: Calcular quotas por arquivo e tipo de ataque
    print(f"\n--- CALCULANDO QUOTAS POR ARQUIVO ---")
    
    for file_id, file_metrics in metrics['per_file'].items():
        print(f"\nProcessando arquivo: {file_id}")
        sampling_plan['quotas']['per_file'][file_id] = {}
        
        file_samples = file_metrics['samples']
        print(f"  Amostras no arquivo: {file_samples:,}")
        
        for attack_type in attack_samples.keys():
            # Inicializa estrutura para este tipo de ataque
            sampling_plan['quotas']['per_file'][file_id][attack_type] = {
                'available': 0,
                'initial': 0,
                'final': 0
            }
            
            # Verifica se este arquivo tem amostras deste tipo de ataque
            available_samples = file_metrics.get(attack_type, 0)
            sampling_plan['quotas']['per_file'][file_id][attack_type]['available'] = available_samples
            
            if available_samples > 0 and attack_type in sampling_plan['target_samples']:
                total_attack_samples = attack_samples[attack_type]
                target_samples = sampling_plan['target_samples'][attack_type]
                
                # Calcula quota proporcional inicial
                if total_attack_samples > 0:
                    proportion = available_samples / total_attack_samples
                    initial_quota = int(proportion * target_samples)
                else:
                    initial_quota = 0
                
                # Ajusta quota para não exceder amostras disponíveis
                final_quota = min(initial_quota, available_samples)
                
                sampling_plan['quotas']['per_file'][file_id][attack_type]['initial'] = initial_quota
                sampling_plan['quotas']['per_file'][file_id][attack_type]['final'] = final_quota
                
                if final_quota > 0:
                    print(f"    {attack_type}: {final_quota:,}/{available_samples:,} (inicial: {initial_quota:,})")
    
    # FASE 2.3: Calcular resumo das quotas por tipo de ataque
    print(f"\n--- RESUMO DAS QUOTAS POR TIPO DE ATAQUE ---")
    
    for attack_type in attack_samples.keys():
        # Soma quotas finais de todos os arquivos para este tipo de ataque
        total_quota_final = 0
        total_quota_initial = 0
        total_available = 0
        files_with_samples = 0
        
        for file_id in sampling_plan['quotas']['per_file'].keys():
            file_data = sampling_plan['quotas']['per_file'][file_id][attack_type]
            total_quota_final += file_data['final']
            total_quota_initial += file_data['initial']
            total_available += file_data['available']
            
            if file_data['available'] > 0:
                files_with_samples += 1
        
        target_samples = sampling_plan['target_samples'].get(attack_type, 0)
        
        sampling_plan['quotas']['summary'][attack_type] = {
            'target': target_samples,
            'achievable': total_quota_final,
            'deficit': max(0, target_samples - total_quota_final),
            'available_total': total_available,
            'files_with_samples': files_with_samples,
            'initial_total': total_quota_initial
        }
        
        # Calcula taxa de atingimento
        achievement_rate = (total_quota_final / target_samples * 100) if target_samples > 0 else 0
        
        print(f"{attack_type}:")
        print(f"  Alvo: {target_samples:,} | Atingível: {total_quota_final:,} ({achievement_rate:.1f}%)")
        if total_quota_final < target_samples:
            print(f"  ⚠️  Déficit: {target_samples - total_quota_final:,} amostras")
    
    # FASE 2.4: Identificar tipos de ataque com problemas de amostragem
    problematic_attacks = []
    for attack_type, summary in sampling_plan['quotas']['summary'].items():
        if summary['deficit'] > 0:
            problematic_attacks.append(attack_type)
    
    if problematic_attacks:
        print(f"\n⚠️  ATENÇÃO: {len(problematic_attacks)} tipos de ataque têm déficit de amostras:")
        for attack_type in problematic_attacks[:5]:  # Mostra apenas os primeiros 5
            deficit = sampling_plan['quotas']['summary'][attack_type]['deficit']
            print(f"    {attack_type}: déficit de {deficit:,} amostras")
        
        if len(problematic_attacks) > 5:
            print(f"    ... e mais {len(problematic_attacks) - 5} tipos de ataque")
        
        print(f"\n📝 Estes casos serão tratados na FASE 3 (redistribuição e fallback)")
    
    # Salva o planejamento em arquivo para a próxima fase
    planning_file = config['metrics_file'].replace('metrics.json', 'sampling_plan.json')
    print(f"\nSalvando planejamento em: {planning_file}")
    
    with open(planning_file, 'w') as f:
        json.dump(sampling_plan, f, indent=2, ensure_ascii=False)
    
    print(f"\n--- RESUMO DA FASE 2 ---")
    print(f"Taxa de amostragem: {sampling_rate:.1%}")
    print(f"Tipos de ataque únicos: {len(attack_samples)}")
    print(f"Arquivos processados: {len(metrics['per_file'])}")
    print(f"Tipos com déficit: {len(problematic_attacks)}")
    print(f"Planejamento salvo: {planning_file}")
    
    return sampling_plan


if __name__ == "__main__":
    config = load_config()
    metrics = analyze_and_collect_metrics(config)
    sampling_plan = stratified_sampling_planning(config)
    print(metrics)