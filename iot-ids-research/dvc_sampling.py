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
    - Criar arquivo metrics
    - Para cada arquivo:
        - Contar quantas amostras tem em cada arquivo ((filename)_samples) e salvar no metrics
        - Contar quantas amostras tem cada tipo de ataque em cada arquivo ((attack_name)_(filename)_attack_samples) e salvar no metrics
    - Com base nos valores por arquivo, calcular:
        - Quantas amostras tem contando todos os arquivos (total_samples) e salvar no metrics
        - Quantas amostras tem cada tipo de ataque juntando todos os arquivos ((attack_name)_attack_samples_total) e salvar no metrics
        - Porcentagem de cada tipo de ataque em relação ao total de dados ((attack_name)_attack_percentage_total) e salvar no metrics

FASE 2: Planejamento de amostragem estratificada
    - Carregar arquivo metrics
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
            - Registrar no metrics o tipo de ataque como "fallback_applied"

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
    - Salvar arquivo metrics atualizado com métricas de coleta
    - Registrar log de operação com detalhes da amostragem realizada
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