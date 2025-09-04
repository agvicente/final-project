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
                # clean_attack_name = str(attack_type).replace(' ', '_').replace('-', '_').replace('.', '_')
                clean_attack_name = attack_type
                
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


def feasibility_check_and_redistribution(config=None, sampling_plan_file_path=None):
    """
    FASE 3: Verificação de viabilidade e redistribuição.
    
    Esta função implementa a terceira fase do algoritmo de amostragem estratificada,
    realizando verificação de viabilidade das quotas e redistribuindo déficits entre
    arquivos disponíveis. Aplica fallback para tipos de ataques raros quando necessário.
    
    Args:
        config (dict, optional): Configurações do sistema. Se None, carrega configurações
                               padrão através de load_config().
        sampling_plan_file_path (str, optional): Caminho para o arquivo sampling_plan.json.
                                               Se None, deriva do config['metrics_file'].
    
    Returns:
        dict: Dicionário contendo o plano final de amostragem com estrutura:
            - sampling_config: configurações originais da FASE 2
            - target_samples: amostras alvo por tipo de ataque  
            - quotas: quotas finais após redistribuição e fallback
                - per_file: {file_id: {attack_type: {available, initial, final, redistributed}}}
                - summary: resumo final com status de redistribuição
            - redistribution_log: log detalhado das redistribuições realizadas
            - fallback_attacks: lista de tipos de ataque com fallback aplicado
    
    Raises:
        FileNotFoundError: Se o arquivo sampling_plan.json não for encontrado.
        KeyError: Se a estrutura do sampling_plan.json estiver incorreta.
        ValueError: Se quotas ou déficits forem inválidos.
    
    Example:
        >>> config = load_config()
        >>> # Execute FASE 1 e 2 primeiro
        >>> metrics = analyze_and_collect_metrics(config)
        >>> sampling_plan = stratified_sampling_planning(config)
        >>> # Depois execute FASE 3
        >>> final_plan = feasibility_check_and_redistribution(config)
        >>> print(f"Ataques com fallback: {len(final_plan['fallback_attacks'])}")
    """
    # Carrega configurações se não fornecidas
    if config is None:
        config = load_config()
    
    # Define caminho do arquivo de planejamento
    if sampling_plan_file_path is None:
        sampling_plan_file_path = config['metrics_file'].replace('metrics.json', 'sampling_plan.json')
    
    # Verifica se arquivo de planejamento existe
    if not os.path.exists(sampling_plan_file_path):
        raise FileNotFoundError(f"Arquivo de planejamento não encontrado: {sampling_plan_file_path}")
    
    print(f"Iniciando FASE 3: Verificação de viabilidade e redistribuição")
    
    # Carrega planejamento da FASE 2
    with open(sampling_plan_file_path, 'r') as f:
        sampling_plan = json.load(f)
    
    # Valida estrutura do arquivo de planejamento
    required_keys = ['sampling_config', 'target_samples', 'quotas']
    for key in required_keys:
        if key not in sampling_plan:
            raise KeyError(f"Chave '{key}' não encontrada no arquivo de planejamento")
    
    # Inicializa estruturas para FASE 3
    final_plan = sampling_plan.copy()
    final_plan['redistribution_log'] = {}
    final_plan['fallback_attacks'] = []
    
    # Adiciona campo 'redistributed' para rastrear redistribuições
    for file_id in final_plan['quotas']['per_file']:
        for attack_type in final_plan['quotas']['per_file'][file_id]:
            final_plan['quotas']['per_file'][file_id][attack_type]['redistributed'] = 0
    
    redistribution_log = {}
    fallback_attacks = []
    
    print(f"Analisando {len(final_plan['quotas']['summary'])} tipos de ataque...")
    
    # FASE 3.1: Identificar tipos de ataque com déficit
    attacks_with_deficit = []
    for attack_type, summary in final_plan['quotas']['summary'].items():
        if summary['deficit'] > 0:
            attacks_with_deficit.append((attack_type, summary['deficit']))
    
    print(f"\n--- TIPOS DE ATAQUE COM DÉFICIT: {len(attacks_with_deficit)} ---")
    for attack_type, deficit in attacks_with_deficit:
        target = final_plan['quotas']['summary'][attack_type]['target']
        achievable = final_plan['quotas']['summary'][attack_type]['achievable']
        print(f"{attack_type}: déficit de {deficit:,} amostras (alvo: {target:,}, atingível: {achievable:,})")
    
    # FASE 3.2: Redistribuição para cada tipo de ataque com déficit
    for attack_type, original_deficit in attacks_with_deficit:
        print(f"\n=== REDISTRIBUINDO: {attack_type} ===")
        redistribution_log[attack_type] = {
            'original_deficit': original_deficit,
            'redistributed': 0,
            'final_deficit': original_deficit,
            'files_benefited': [],
            'fallback_applied': False
        }
        
        current_deficit = original_deficit
        
        # Identifica arquivos não saturados (com amostras disponíveis além da quota atual)
        unsaturated_files = []
        for file_id in final_plan['quotas']['per_file']:
            file_data = final_plan['quotas']['per_file'][file_id][attack_type]
            available = file_data['available']
            current_quota = file_data['final']
            
            if available > current_quota:  # Arquivo tem mais amostras disponíveis
                remaining_capacity = available - current_quota
                unsaturated_files.append({
                    'file_id': file_id,
                    'current_quota': current_quota,
                    'available': available,
                    'remaining_capacity': remaining_capacity
                })
        
        if not unsaturated_files:
            print(f"  ⚠️  Nenhum arquivo não saturado encontrado para {attack_type}")
            redistribution_log[attack_type]['fallback_applied'] = True
            fallback_attacks.append(attack_type)
            continue
        
        # Calcula capacidade total restante
        total_remaining_capacity = sum(f['remaining_capacity'] for f in unsaturated_files)
        
        print(f"  Déficit original: {original_deficit:,}")
        print(f"  Arquivos não saturados: {len(unsaturated_files)}")
        print(f"  Capacidade restante total: {total_remaining_capacity:,}")
        
        if total_remaining_capacity >= current_deficit:
            # CASO 1: Capacidade suficiente para cobrir todo o déficit
            print(f"  ✅ Capacidade suficiente para redistribuição completa")
            
            redistributed_total = 0
            for file_info in unsaturated_files:
                file_id = file_info['file_id']
                remaining_capacity = file_info['remaining_capacity']
                
                # Calcula proporção baseada na capacidade restante
                proportion = remaining_capacity / total_remaining_capacity
                redistribution_amount = min(
                    int(current_deficit * proportion),
                    remaining_capacity
                )
                
                if redistribution_amount > 0:
                    # Atualiza quota final
                    old_quota = final_plan['quotas']['per_file'][file_id][attack_type]['final']
                    new_quota = old_quota + redistribution_amount
                    final_plan['quotas']['per_file'][file_id][attack_type]['final'] = new_quota
                    final_plan['quotas']['per_file'][file_id][attack_type]['redistributed'] = redistribution_amount
                    
                    redistributed_total += redistribution_amount
                    redistribution_log[attack_type]['files_benefited'].append({
                        'file_id': file_id,
                        'old_quota': old_quota,
                        'new_quota': new_quota,
                        'redistribution_amount': redistribution_amount
                    })
                    
                    print(f"    {file_id}: {old_quota:,} → {new_quota:,} (+{redistribution_amount:,})")
            
            # Ajuste fino: distribui amostras restantes
            remaining_deficit = current_deficit - redistributed_total
            if remaining_deficit > 0:
                for file_info in unsaturated_files:
                    if remaining_deficit <= 0:
                        break
                    
                    file_id = file_info['file_id']
                    file_data = final_plan['quotas']['per_file'][file_id][attack_type]
                    
                    if file_data['final'] < file_data['available']:
                        additional = min(remaining_deficit, file_data['available'] - file_data['final'])
                        if additional > 0:
                            file_data['final'] += additional
                            file_data['redistributed'] += additional
                            redistributed_total += additional
                            remaining_deficit -= additional
                            print(f"    {file_id}: ajuste fino +{additional:,}")
            
            redistribution_log[attack_type]['redistributed'] = redistributed_total
            redistribution_log[attack_type]['final_deficit'] = max(0, current_deficit - redistributed_total)
            
        else:
            # CASO 2: Capacidade insuficiente - usa toda capacidade disponível e aplica fallback
            print(f"  ⚠️  Capacidade insuficiente. Usando toda capacidade disponível e aplicando fallback")
            
            redistributed_total = 0
            # Primeira usa toda a capacidade restante
            for file_info in unsaturated_files:
                file_id = file_info['file_id']
                remaining_capacity = file_info['remaining_capacity']
                
                if remaining_capacity > 0:
                    old_quota = final_plan['quotas']['per_file'][file_id][attack_type]['final']
                    new_quota = file_info['available']  # Usa todas as amostras disponíveis
                    final_plan['quotas']['per_file'][file_id][attack_type]['final'] = new_quota
                    final_plan['quotas']['per_file'][file_id][attack_type]['redistributed'] = remaining_capacity
                    
                    redistributed_total += remaining_capacity
                    redistribution_log[attack_type]['files_benefited'].append({
                        'file_id': file_id,
                        'old_quota': old_quota,
                        'new_quota': new_quota,
                        'redistribution_amount': remaining_capacity
                    })
                    
                    print(f"    {file_id}: {old_quota:,} → {new_quota:,} (+{remaining_capacity:,}) [SATURADO]")
            
            redistribution_log[attack_type]['redistributed'] = redistributed_total
            redistribution_log[attack_type]['final_deficit'] = current_deficit - redistributed_total
            
            # Aplica fallback - usar todas as amostras disponíveis de todos os arquivos
            if redistribution_log[attack_type]['final_deficit'] > 0:
                print(f"    🔄 Aplicando FALLBACK para {attack_type}")
                redistribution_log[attack_type]['fallback_applied'] = True
                fallback_attacks.append(attack_type)
                
                # Força usar todas as amostras disponíveis
                total_fallback = 0
                for file_id in final_plan['quotas']['per_file']:
                    file_data = final_plan['quotas']['per_file'][file_id][attack_type]
                    if file_data['available'] > 0:
                        old_quota = file_data['final']
                        file_data['final'] = file_data['available']
                        total_fallback += file_data['available']
                        if file_data['available'] > old_quota:
                            additional = file_data['available'] - old_quota
                            file_data['redistributed'] += additional
                            print(f"    {file_id}: FALLBACK {old_quota:,} → {file_data['available']:,}")
                
                redistribution_log[attack_type]['final_deficit'] = 0  # Fallback elimina déficit
    
    # FASE 3.3: Atualiza resumo final
    print(f"\n--- ATUALIZANDO RESUMO FINAL ---")
    
    for attack_type in final_plan['quotas']['summary']:
        # Recalcula totais após redistribuição
        total_final = 0
        for file_id in final_plan['quotas']['per_file']:
            total_final += final_plan['quotas']['per_file'][file_id][attack_type]['final']
        
        # Atualiza summary
        old_achievable = final_plan['quotas']['summary'][attack_type]['achievable']
        final_plan['quotas']['summary'][attack_type]['achievable'] = total_final
        final_plan['quotas']['summary'][attack_type]['deficit'] = max(
            0, final_plan['quotas']['summary'][attack_type]['target'] - total_final
        )
        
        # Adiciona informações de redistribuição
        if attack_type in redistribution_log:
            final_plan['quotas']['summary'][attack_type]['redistribution'] = redistribution_log[attack_type]
            final_plan['quotas']['summary'][attack_type]['fallback_applied'] = redistribution_log[attack_type]['fallback_applied']
        else:
            final_plan['quotas']['summary'][attack_type]['redistribution'] = None
            final_plan['quotas']['summary'][attack_type]['fallback_applied'] = False
        
        improvement = total_final - old_achievable
        if improvement > 0:
            target = final_plan['quotas']['summary'][attack_type]['target']
            achievement_rate = (total_final / target * 100) if target > 0 else 0
            print(f"{attack_type}: {old_achievable:,} → {total_final:,} (+{improvement:,}) [{achievement_rate:.1f}%]")
    
    # Salva logs de redistribuição
    final_plan['redistribution_log'] = redistribution_log
    final_plan['fallback_attacks'] = fallback_attacks
    
    # FASE 3.4: Salvamento do plano final
    final_plan_file = config['metrics_file'].replace('metrics.json', 'final_sampling_plan.json')
    print(f"\nSalvando plano final em: {final_plan_file}")
    
    with open(final_plan_file, 'w') as f:
        json.dump(final_plan, f, indent=2, ensure_ascii=False)
    
    # Estatísticas finais
    total_redistributions = sum(1 for log in redistribution_log.values() if log['redistributed'] > 0)
    total_redistributed_samples = sum(log['redistributed'] for log in redistribution_log.values())
    successful_redistributions = sum(1 for log in redistribution_log.values() 
                                   if log['redistributed'] > 0 and not log['fallback_applied'])
    
    print(f"\n--- RESUMO DA FASE 3 ---")
    print(f"Tipos de ataque processados: {len(attacks_with_deficit)}")
    print(f"Redistribuições bem-sucedidas: {successful_redistributions}")
    print(f"Total de amostras redistribuídas: {total_redistributed_samples:,}")
    print(f"Tipos com fallback aplicado: {len(fallback_attacks)}")
    if fallback_attacks:
        print(f"Ataques com fallback: {', '.join(fallback_attacks[:5])}")
        if len(fallback_attacks) > 5:
            print(f"  ... e mais {len(fallback_attacks) - 5}")
    print(f"Plano final salvo: {final_plan_file}")
    
    return final_plan


def deterministic_sampling(config=None, final_sampling_plan_file_path=None, label_column='Label'):
    """
    FASE 4: Amostragem determinística.
    
    Esta função implementa a quarta fase do algoritmo de amostragem estratificada,
    realizando a amostragem determinística dos dados baseada nas quotas finais
    calculadas na FASE 3. Utiliza pandas.sample com random_state fixo para
    garantir reprodutibilidade dos resultados.
    
    Args:
        config (dict, optional): Configurações do sistema. Se None, carrega configurações
                               padrão através de load_config().
        final_sampling_plan_file_path (str, optional): Caminho para o arquivo final_sampling_plan.json.
                                                     Se None, deriva do config['metrics_file'].
        label_column (str): Nome da coluna que contém os rótulos dos tipos de ataque.
                           Default: 'Label'.
    
    Returns:
        dict: Dicionário contendo os resultados da amostragem com estrutura:
            - sampling_config: configurações originais das fases anteriores
            - target_samples: amostras alvo por tipo de ataque
            - quotas: quotas finais utilizadas na amostragem
            - collection_results: métricas de amostras efetivamente coletadas
                - samples_collected_total: total de amostras coletadas
                - samples_collected_by_attack: amostras coletadas por tipo de ataque
                - collection_log: log detalhado da coleta por arquivo e tipo de ataque
            - sampled_data: DataFrame consolidado com todas as amostras coletadas
            - execution_metadata: informações sobre a execução (random_state, timestamp, etc.)
    
    Raises:
        FileNotFoundError: Se o arquivo final_sampling_plan.json não for encontrado.
        FileNotFoundError: Se algum arquivo CSV especificado não for encontrado.
        KeyError: Se a estrutura do final_sampling_plan.json estiver incorreta.
        KeyError: Se a coluna de rótulos não existir nos arquivos CSV.
        ValueError: Se quotas ou configurações forem inválidas.
    
    Example:
        >>> config = load_config()
        >>> # Execute FASES 1, 2 e 3 primeiro
        >>> metrics = analyze_and_collect_metrics(config)
        >>> sampling_plan = stratified_sampling_planning(config)
        >>> final_plan = feasibility_check_and_redistribution(config)
        >>> # Depois execute FASE 4
        >>> sampling_results = deterministic_sampling(config)
        >>> print(f"Amostras coletadas: {sampling_results['collection_results']['samples_collected_total']:,}")
    """
    # Carrega configurações se não fornecidas
    if config is None:
        config = load_config()
    
    # Define caminho do arquivo de planejamento final
    if final_sampling_plan_file_path is None:
        final_sampling_plan_file_path = config['metrics_file'].replace('metrics.json', 'final_sampling_plan.json')
    
    # Verifica se arquivo de planejamento final existe
    if not os.path.exists(final_sampling_plan_file_path):
        raise FileNotFoundError(f"Arquivo de planejamento final não encontrado: {final_sampling_plan_file_path}")
    
    print(f"Iniciando FASE 4: Amostragem determinística")
    
    # Carrega planejamento final da FASE 3
    with open(final_sampling_plan_file_path, 'r') as f:
        final_plan = json.load(f)
    
    # Valida estrutura do arquivo de planejamento final
    required_keys = ['sampling_config', 'target_samples', 'quotas']
    for key in required_keys:
        if key not in final_plan:
            raise KeyError(f"Chave '{key}' não encontrada no arquivo de planejamento final")
    
    # Inicializa estruturas para FASE 4
    sampling_results = {
        'sampling_config': final_plan['sampling_config'],
        'target_samples': final_plan['target_samples'],
        'quotas': final_plan['quotas'],
        'collection_results': {
            'samples_collected_total': 0,
            'samples_collected_by_attack': {},
            'collection_log': {}
        },
        'sampled_data': None,
        'execution_metadata': {
            'random_state': 42,  # Random state fixo para reprodutibilidade
            'label_column': label_column,
            'data_directory': config['data_dir']
        }
    }
    
    # Define random_state fixo para reprodutibilidade
    random_state = sampling_results['execution_metadata']['random_state']
    
    # Cria DataFrame vazio para consolidar amostras
    consolidated_df = pd.DataFrame()
    
    # Inicializa contadores
    samples_collected_by_attack = {}
    for attack_type in final_plan['target_samples'].keys():
        samples_collected_by_attack[attack_type] = 0
    
    # Lista arquivos CSV no diretório
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em: {data_dir}")
    
    print(f"Processando {len(csv_files)} arquivos CSV...")
    
    # FASE 4.1: Amostragem por arquivo
    for i, filename in enumerate(csv_files, 1):
        print(f"\nProcessando arquivo {i}/{len(csv_files)}: {filename}")
        
        # Remove extensão para obter file_id
        file_id = filename.replace('.csv', '')
        
        # Verifica se arquivo está no planejamento
        if file_id not in final_plan['quotas']['per_file']:
            print(f"  ⚠️  Arquivo {file_id} não encontrado no planejamento. Pulando...")
            continue
        
        file_path = os.path.join(data_dir, filename)
        sampling_results['collection_results']['collection_log'][file_id] = {}
        
        try:
            # Carrega arquivo uma única vez
            print(f"  📂 Carregando arquivo completo...")
            df = pd.read_csv(file_path)
            
            # Verifica se coluna de rótulos existe
            if label_column not in df.columns:
                raise KeyError(f"Coluna '{label_column}' não encontrada no arquivo {filename}")
            
            file_total_samples = len(df)
            file_collected_samples = 0
            
            print(f"  📊 Arquivo carregado: {file_total_samples:,} amostras totais")
            
            # FASE 4.2: Amostragem por tipo de ataque no arquivo atual
            file_quotas = final_plan['quotas']['per_file'][file_id]
            
            for attack_type, quota_info in file_quotas.items():
                quota_final = quota_info['final']
                available_samples = quota_info['available']
                
                # Inicializa log para este tipo de ataque
                sampling_results['collection_results']['collection_log'][file_id][attack_type] = {
                    'quota_final': quota_final,
                    'available_samples': available_samples,
                    'samples_collected': 0,
                    'success': False
                }
                
                # Se quota for maior que zero, realiza amostragem
                if quota_final > 0:
                    print(f"    🎯 {attack_type}: amostrando {quota_final:,}/{available_samples:,} amostras")
                    
                    try:
                        # Filtra dados do tipo de ataque específico
                        attack_data = df[df[label_column] == attack_type]
                        actual_available = len(attack_data)
                        
                        if actual_available != available_samples:
                            print(f"      ⚠️  Divergência: esperado {available_samples:,}, encontrado {actual_available:,}")
                        
                        if actual_available == 0:
                            print(f"      ❌ Nenhuma amostra encontrada para {attack_type}")
                            continue
                        
                        # Ajusta quota se necessário
                        effective_quota = min(quota_final, actual_available)
                        
                        if effective_quota < quota_final:
                            print(f"      ⚠️  Quota ajustada: {quota_final:,} → {effective_quota:,}")
                        
                        # Amostragem determinística usando pandas.sample
                        if effective_quota == actual_available:
                            # Se quota igual ao disponível, usa todas as amostras
                            sampled_data = attack_data.copy()
                        else:
                            # Amostragem aleatória com random_state fixo
                            sampled_data = attack_data.sample(
                                n=effective_quota,
                                random_state=random_state,
                                replace=False  # Sem reposição
                            )
                        
                        actual_collected = len(sampled_data)
                        
                        # Adiciona amostras ao DataFrame consolidado
                        consolidated_df = pd.concat([consolidated_df, sampled_data], ignore_index=True)
                        
                        # Atualiza contadores
                        samples_collected_by_attack[attack_type] += actual_collected
                        file_collected_samples += actual_collected
                        
                        # Atualiza log
                        sampling_results['collection_results']['collection_log'][file_id][attack_type].update({
                            'samples_collected': actual_collected,
                            'success': True,
                            'effective_quota': effective_quota,
                            'actual_available': actual_available
                        })
                        
                        print(f"      ✅ Coletadas {actual_collected:,} amostras")
                        
                    except Exception as e:
                        print(f"      ❌ Erro ao amostrar {attack_type}: {str(e)}")
                        sampling_results['collection_results']['collection_log'][file_id][attack_type]['error'] = str(e)
                        continue
                
                elif available_samples > 0:
                    print(f"    ⏭️  {attack_type}: quota zero, pulando {available_samples:,} amostras disponíveis")
            
            print(f"  📋 Resumo do arquivo: {file_collected_samples:,} amostras coletadas")
            
            # Descarrega arquivo da memória (Python garbage collector cuidará disso)
            del df
            
        except Exception as e:
            print(f"  ❌ Erro ao processar {filename}: {str(e)}")
            sampling_results['collection_results']['collection_log'][file_id]['error'] = str(e)
            continue
    
    # FASE 4.3: Consolidação final dos resultados
    print(f"\n--- CONSOLIDANDO RESULTADOS ---")
    
    total_collected = len(consolidated_df)
    sampling_results['collection_results']['samples_collected_total'] = total_collected
    sampling_results['collection_results']['samples_collected_by_attack'] = samples_collected_by_attack
    sampling_results['sampled_data'] = consolidated_df
    
    print(f"Total de amostras consolidadas: {total_collected:,}")
    
    # Exibe estatísticas por tipo de ataque
    print(f"\nAmostras coletadas por tipo de ataque:")
    for attack_type, collected in samples_collected_by_attack.items():
        target = final_plan['target_samples'].get(attack_type, 0)
        achievement_rate = (collected / target * 100) if target > 0 else 0
        fallback_status = " [FALLBACK]" if attack_type in final_plan.get('fallback_attacks', []) else ""
        print(f"  {attack_type}: {collected:,}/{target:,} ({achievement_rate:.1f}%){fallback_status}")
    
    # FASE 4.4: Salvamento dos resultados
    # Salva DataFrame consolidado
    output_dir = config.get('data_output_dir', 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    
    sampled_csv_path = os.path.join(output_dir, 'sampled.csv')
    print(f"\nSalvando dados amostrados em: {sampled_csv_path}")
    consolidated_df.to_csv(sampled_csv_path, index=False)
    
    # Salva resultados da coleta (sem o DataFrame para evitar redundância)
    results_to_save = sampling_results.copy()
    results_to_save['sampled_data'] = f"Dados salvos em: {sampled_csv_path}"
    
    sampling_results_file = config['metrics_file'].replace('metrics.json', 'sampling_results.json')
    print(f"Salvando resultados da amostragem em: {sampling_results_file}")
    
    with open(sampling_results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # Estatísticas finais
    target_total = sum(final_plan['target_samples'].values())
    overall_achievement = (total_collected / target_total * 100) if target_total > 0 else 0
    files_processed = len([f for f in sampling_results['collection_results']['collection_log'].keys() 
                          if 'error' not in sampling_results['collection_results']['collection_log'][f]])
    
    print(f"\n--- RESUMO DA FASE 4 ---")
    print(f"Arquivos processados com sucesso: {files_processed}/{len(csv_files)}")
    print(f"Total de amostras coletadas: {total_collected:,}")
    print(f"Total de amostras alvo: {target_total:,}")
    print(f"Taxa de atingimento geral: {overall_achievement:.1f}%")
    print(f"Tipos de ataque únicos coletados: {len([k for k, v in samples_collected_by_attack.items() if v > 0])}")
    print(f"Dados consolidados salvos: {sampled_csv_path}")
    print(f"Resultados salvos: {sampling_results_file}")
    print(f"Random state utilizado: {random_state}")
    
    return sampling_results


def consolidation_and_saving(config=None, sampling_results=None, sampling_results_file_path=None):
    """
    FASE 5: Consolidação e salvamento.
    
    Esta função implementa a quinta e última fase do algoritmo de amostragem estratificada,
    realizando a consolidação final de todas as métricas, verificação de atingimento de quotas,
    salvamento dos dados amostrados e geração de log detalhado de toda a operação.
    
    Args:
        config (dict, optional): Configurações do sistema. Se None, carrega configurações
                               padrão através de load_config().
        sampling_results (dict, optional): Resultados da FASE 4. Se None, carrega do arquivo
                                         sampling_results.json.
        sampling_results_file_path (str, optional): Caminho para o arquivo sampling_results.json.
                                                   Se None, deriva do config['metrics_file'].
    
    Returns:
        dict: Dicionário contendo o resultado final consolidado com estrutura:
            - sampling_config: configurações finais utilizadas
            - original_metrics: métricas originais do dataset
            - target_samples: amostras alvo por tipo de ataque
            - collection_results: resultados efetivos da coleta
            - final_metrics: métricas consolidadas finais
                - total_samples_collected: total de amostras coletadas
                - samples_collected_by_attack: amostras coletadas por tipo de ataque
                - final_percentage_by_attack: proporção final de cada tipo de ataque
                - fallback_attacks: lista de tipos de ataque com fallback aplicado
                - quota_achievement: análise de atingimento das quotas
            - operation_log: log detalhado de toda a operação de amostragem
            - file_paths: caminhos dos arquivos gerados
    
    Raises:
        FileNotFoundError: Se arquivos necessários não forem encontrados.
        ValueError: Se dados de entrada forem inválidos.
        KeyError: Se estrutura dos dados estiver incorreta.
    
    Example:
        >>> config = load_config()
        >>> # Execute todas as fases anteriores
        >>> metrics = analyze_and_collect_metrics(config)
        >>> sampling_plan = stratified_sampling_planning(config)
        >>> final_plan = feasibility_check_and_redistribution(config)
        >>> sampling_results = deterministic_sampling(config)
        >>> # Finalize com FASE 5
        >>> final_results = consolidation_and_saving(config, sampling_results)
        >>> print(f"Operação concluída: {final_results['final_metrics']['total_samples_collected']:,} amostras")
    """
    import datetime
    
    # Carrega configurações se não fornecidas
    if config is None:
        config = load_config()
    
    print(f"Iniciando FASE 5: Consolidação e salvamento")
    
    # Carrega resultados da FASE 4 se não fornecidos
    if sampling_results is None:
        if sampling_results_file_path is None:
            sampling_results_file_path = config['metrics_file'].replace('metrics.json', 'sampling_results.json')
        
        if not os.path.exists(sampling_results_file_path):
            raise FileNotFoundError(f"Arquivo de resultados da amostragem não encontrado: {sampling_results_file_path}")
        
        print(f"Carregando resultados da FASE 4: {sampling_results_file_path}")
        with open(sampling_results_file_path, 'r') as f:
            sampling_results = json.load(f)
    
    # Valida estrutura dos resultados da FASE 4
    required_keys = ['sampling_config', 'target_samples', 'collection_results']
    for key in required_keys:
        if key not in sampling_results:
            raise KeyError(f"Chave '{key}' não encontrada nos resultados da amostragem")
    
    # Carrega métricas originais para comparação
    original_metrics_file = config['metrics_file']
    if os.path.exists(original_metrics_file):
        with open(original_metrics_file, 'r') as f:
            original_metrics = json.load(f)
    else:
        original_metrics = {}
    
    # Inicializa estrutura de resultados finais
    final_results = {
        'sampling_config': sampling_results['sampling_config'],
        'original_metrics': original_metrics,
        'target_samples': sampling_results['target_samples'],
        'collection_results': sampling_results['collection_results'],
        'final_metrics': {},
        'operation_log': {
            'timestamp': datetime.datetime.now().isoformat(),
            'phases_executed': ['FASE 1', 'FASE 2', 'FASE 3', 'FASE 4', 'FASE 5'],
            'execution_summary': {},
            'quota_analysis': {},
            'fallback_analysis': {},
            'file_generation': {}
        },
        'file_paths': {}
    }
    
    # FASE 5.1: Verificação de quotas e fallbacks
    print(f"\n--- VERIFICANDO ATINGIMENTO DE QUOTAS ---")
    
    target_samples = sampling_results['target_samples']
    collected_samples = sampling_results['collection_results']['samples_collected_by_attack']
    total_collected = sampling_results['collection_results']['samples_collected_total']
    
    quota_achievement = {}
    fallback_attacks = []
    perfectly_achieved = 0
    partially_achieved = 0
    failed_attacks = 0
    
    # Carrega informações de fallback do planejamento final se disponível
    final_plan_file = config['metrics_file'].replace('metrics.json', 'final_sampling_plan.json')
    if os.path.exists(final_plan_file):
        with open(final_plan_file, 'r') as f:
            final_plan = json.load(f)
            fallback_attacks = final_plan.get('fallback_attacks', [])
    
    for attack_type, target in target_samples.items():
        collected = collected_samples.get(attack_type, 0)
        achievement_rate = (collected / target * 100) if target > 0 else 0
        
        quota_achievement[attack_type] = {
            'target': target,
            'collected': collected,
            'achievement_rate': round(achievement_rate, 2),
            'deficit': max(0, target - collected),
            'surplus': max(0, collected - target),
            'fallback_applied': attack_type in fallback_attacks
        }
        
        # Categoriza o atingimento
        if achievement_rate >= 100:
            perfectly_achieved += 1
        elif achievement_rate >= 50:
            partially_achieved += 1
        else:
            failed_attacks += 1
        
        status = "✅ PERFEITO" if achievement_rate >= 100 else \
                "🟡 PARCIAL" if achievement_rate >= 50 else "❌ FALHOU"
        fallback_info = " [FALLBACK]" if attack_type in fallback_attacks else ""
        
        print(f"  {attack_type}: {collected:,}/{target:,} ({achievement_rate:.1f}%) {status}{fallback_info}")
    
    # FASE 5.2: Consolidação de métricas finais
    print(f"\n--- CONSOLIDANDO MÉTRICAS FINAIS ---")
    
    # Calcula proporções finais
    final_percentages = {}
    for attack_type, collected in collected_samples.items():
        percentage = (collected / total_collected * 100) if total_collected > 0 else 0
        final_percentages[attack_type] = round(percentage, 4)
    
    # Monta métricas finais
    final_results['final_metrics'] = {
        'total_samples_collected': total_collected,
        'samples_collected_by_attack': collected_samples,
        'final_percentage_by_attack': final_percentages,
        'fallback_attacks': fallback_attacks,
        'quota_achievement': quota_achievement,
        'achievement_summary': {
            'perfectly_achieved': perfectly_achieved,
            'partially_achieved': partially_achieved,
            'failed_attacks': failed_attacks,
            'total_attack_types': len(target_samples),
            'fallback_count': len(fallback_attacks)
        }
    }
    
    # Estatísticas gerais
    original_total = original_metrics.get('total', {}).get('samples', 0)
    reduction_rate = ((original_total - total_collected) / original_total * 100) if original_total > 0 else 0
    target_total = sum(target_samples.values())
    overall_achievement = (total_collected / target_total * 100) if target_total > 0 else 0
    
    print(f"Total de amostras coletadas: {total_collected:,}")
    print(f"Total de amostras originais: {original_total:,}")
    print(f"Taxa de redução: {reduction_rate:.1f}%")
    print(f"Taxa de atingimento geral: {overall_achievement:.1f}%")
    print(f"Tipos de ataque com fallback: {len(fallback_attacks)}")
    
    # FASE 5.3: Salvamento do DataFrame consolidado
    print(f"\n--- SALVANDO DADOS CONSOLIDADOS ---")
    
    output_dir = config.get('data_output_dir', 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva DataFrame consolidado como sampled.csv
    sampled_csv_path = os.path.join(output_dir, 'sampled.csv')
    
    # Verifica se temos dados para salvar
    if 'sampled_data' in sampling_results and sampling_results['sampled_data'] is not None:
        if isinstance(sampling_results['sampled_data'], pd.DataFrame):
            sampling_results['sampled_data'].to_csv(sampled_csv_path, index=False)
            print(f"✅ DataFrame consolidado salvo: {sampled_csv_path}")
        else:
            print(f"⚠️  Dados já salvos anteriormente: {sampling_results['sampled_data']}")
    else:
        print(f"⚠️  Nenhum DataFrame encontrado para salvar")
    
    final_results['file_paths']['sampled_data'] = sampled_csv_path
    
    # FASE 5.4: Atualização e salvamento do metrics.json
    print(f"\n--- ATUALIZANDO ARQUIVO DE MÉTRICAS ---")
    
    # Atualiza métricas originais com resultados da amostragem
    updated_metrics = original_metrics.copy()
    updated_metrics['sampling'] = {
        'executed': True,
        'timestamp': final_results['operation_log']['timestamp'],
        'sampling_rate': sampling_results['sampling_config']['sampling_rate'],
        'total_samples_collected': total_collected,
        'samples_collected_by_attack': collected_samples,
        'final_percentage_by_attack': final_percentages,
        'quota_achievement_summary': final_results['final_metrics']['achievement_summary'],
        'fallback_attacks': fallback_attacks
    }
    
    # Salva metrics.json atualizado
    updated_metrics_file = config['metrics_file']
    with open(updated_metrics_file, 'w') as f:
        json.dump(updated_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Arquivo de métricas atualizado: {updated_metrics_file}")
    final_results['file_paths']['updated_metrics'] = updated_metrics_file
    
    # FASE 5.5: Geração de log detalhado da operação
    print(f"\n--- GERANDO LOG DETALHADO DA OPERAÇÃO ---")
    
    operation_log = final_results['operation_log']
    
    # Resumo da execução
    operation_log['execution_summary'] = {
        'original_samples': original_total,
        'target_samples': target_total,
        'collected_samples': total_collected,
        'reduction_rate_percent': round(reduction_rate, 2),
        'overall_achievement_percent': round(overall_achievement, 2),
        'sampling_rate': sampling_results['sampling_config']['sampling_rate']
    }
    
    # Análise detalhada de quotas
    operation_log['quota_analysis'] = quota_achievement
    
    # Análise de fallbacks
    operation_log['fallback_analysis'] = {
        'fallback_attacks': fallback_attacks,
        'fallback_count': len(fallback_attacks),
        'fallback_percentage': round(len(fallback_attacks) / len(target_samples) * 100, 2) if target_samples else 0,
        'fallback_reasons': "Tipos de ataque raros com amostras insuficientes para atingir quotas proporcionais"
    }
    
    # Informações sobre arquivos gerados
    operation_log['file_generation'] = {
        'sampled_data_csv': sampled_csv_path,
        'updated_metrics_json': updated_metrics_file,
        'sampling_results_json': config['metrics_file'].replace('metrics.json', 'sampling_results.json'),
        'final_sampling_plan_json': final_plan_file
    }
    
    # Salva log completo da operação
    operation_log_file = config['metrics_file'].replace('metrics.json', 'operation_log.json')
    with open(operation_log_file, 'w') as f:
        json.dump(operation_log, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Log da operação salvo: {operation_log_file}")
    final_results['file_paths']['operation_log'] = operation_log_file
    
    # Salva resultado final consolidado
    final_results_file = config['metrics_file'].replace('metrics.json', 'final_results.json')
    final_results_to_save = final_results.copy()
    
    with open(final_results_file, 'w') as f:
        json.dump(final_results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados finais salvos: {final_results_file}")
    final_results['file_paths']['final_results'] = final_results_file
    
    # FASE 5.6: Relatório final da operação
    print(f"\n{'='*60}")
    print(f"🎉 AMOSTRAGEM ESTRATIFICADA CONCLUÍDA COM SUCESSO!")
    print(f"{'='*60}")
    
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"   • Dataset original: {original_total:,} amostras")
    print(f"   • Taxa de amostragem: {sampling_results['sampling_config']['sampling_rate']:.1%}")
    print(f"   • Amostras alvo: {target_total:,}")
    print(f"   • Amostras coletadas: {total_collected:,}")
    print(f"   • Taxa de atingimento: {overall_achievement:.1f}%")
    print(f"   • Redução de dados: {reduction_rate:.1f}%")
    
    print(f"\n🎯 ANÁLISE DE QUOTAS:")
    print(f"   • Perfeitamente atingidas: {perfectly_achieved}/{len(target_samples)} ({perfectly_achieved/len(target_samples)*100:.1f}%)")
    print(f"   • Parcialmente atingidas: {partially_achieved}/{len(target_samples)} ({partially_achieved/len(target_samples)*100:.1f}%)")
    print(f"   • Falhas: {failed_attacks}/{len(target_samples)} ({failed_attacks/len(target_samples)*100:.1f}%)")
    print(f"   • Fallbacks aplicados: {len(fallback_attacks)}/{len(target_samples)} ({len(fallback_attacks)/len(target_samples)*100:.1f}%)")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    for file_type, file_path in final_results['file_paths'].items():
        print(f"   • {file_type}: {file_path}")
    
    print(f"\n✨ A amostragem estratificada foi concluída com sucesso!")
    print(f"   Os dados estão prontos para as próximas etapas do pipeline.")
    
    return final_results


if __name__ == "__main__":
    config = load_config()
    metrics = analyze_and_collect_metrics(config)
    sampling_plan = stratified_sampling_planning(config)
    final_plan = feasibility_check_and_redistribution(config)
    sampling_results = deterministic_sampling(config)
    final_results = consolidation_and_saving(config, sampling_results)
    print(metrics)