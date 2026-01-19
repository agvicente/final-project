"""
Configuracoes do Consumer para IoT IDS Streaming

Este modulo define as configuracoes do Kafka Consumer e do
processador de flows usando dataclasses.

Arquitetura:
    packets (topic) --> Consumer --> Flow Aggregator --> flows (topic)
"""

# ============================================================
# IMPORTS
# ============================================================

from dataclasses import dataclass, field
from typing import Optional, List
import os


# ============================================================
# KAFKA CONSUMER CONFIG
# ============================================================
# Configuracoes especificas para consumidores Kafka
#
# DIFERENCA PRODUCER vs CONSUMER:
#   Producer: envia mensagens, controla batching e compressao
#   Consumer: le mensagens, controla offset e grupo
#
# CONCEITO: Consumer Group
#   Kafka distribui particoes entre consumers do mesmo grupo.
#   Se topico tem 3 particoes e grupo tem 3 consumers,
#   cada consumer le 1 particao (paralelismo).
#
#   Grupo "flow-processor":
#   ┌─────────────────────────────────────────┐
#   │  Topic: packets (3 particoes)           │
#   │  ├── Particao 0 → Consumer 1            │
#   │  ├── Particao 1 → Consumer 2            │
#   │  └── Particao 2 → Consumer 3            │
#   └─────────────────────────────────────────┘
# ============================================================

@dataclass
class ConsumerKafkaConfig:
    """
    Configuracoes de conexao Kafka para o Consumer.

    Uso:
        config = ConsumerKafkaConfig()
        config = ConsumerKafkaConfig.from_env()
    """

    # Endereco do Kafka (mesmo do producer)
    bootstrap_servers: str = "localhost:9092"

    # Topico de onde ler pacotes
    topic_packets: str = "packets"

    # Topico para onde enviar flows processados
    topic_flows: str = "flows"

    # --------------------------------------------------------
    # GROUP_ID - Identificador do Consumer Group
    # --------------------------------------------------------
    # Consumers com mesmo group_id compartilham a leitura.
    # Cada particao e lida por apenas 1 consumer do grupo.
    #
    # Se group_id diferente: cada consumer le TODAS as mensagens do topico
    # Se group_id igual: mensagens sao distribuidas entre consumers via particao
    #
    # COMPARACAO COM FILAS TRADICIONAIS:
    #   - RabbitMQ: cada fila tem 1 consumer (ou compete)
    #   - Kafka: topico pode ter N grupos, cada um le tudo
    # --------------------------------------------------------
    group_id: str = "flow-processor"

    # --------------------------------------------------------
    # AUTO_OFFSET_RESET - O que fazer quando nao ha offset salvo?
    # --------------------------------------------------------
    # Opcoes:
    #   'earliest': comeca do inicio do topico (le tudo)
    #   'latest': comeca do fim (so mensagens novas)
    #   'none': erro se nao houver offset
    #
    # Para desenvolvimento: 'earliest' permite reprocessar
    # Para producao: depende do caso de uso
    # --------------------------------------------------------
    auto_offset_reset: str = "earliest"

    # --------------------------------------------------------
    # ENABLE_AUTO_COMMIT - Commit automatico de offsets?
    # --------------------------------------------------------
    # True: Kafka salva automaticamente o progresso
    # False: Voce controla quando marcar mensagens como lidas
    #
    # TRADE-OFF:
    #   True = mais simples, risco de perder mensagens se crash
    #   False = mais controle, garante processamento
    #
    # Para v0.1: True (simplicidade)
    # Para producao: False (garantias)
    # --------------------------------------------------------
    enable_auto_commit: bool = True

    # Intervalo de auto-commit em ms (se enable_auto_commit=True)
    auto_commit_interval_ms: int = 5000

    # --------------------------------------------------------
    # MAX_POLL_RECORDS - Quantas mensagens buscar por vez
    # --------------------------------------------------------
    # Controla o tamanho do "lote" que o consumer processa.
    # Maior = mais throughput, mais memoria
    # Menor = menor latencia, menos memoria
    # --------------------------------------------------------
    max_poll_records: int = 500

    # Timeout para poll() em ms
    # Se nenhuma mensagem em 1s, poll() retorna vazio
    poll_timeout_ms: int = 1000

    @classmethod
    def from_env(cls) -> "ConsumerKafkaConfig":
        """Cria config a partir de variaveis de ambiente."""
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_packets=os.getenv("KAFKA_TOPIC_PACKETS", "packets"),
            topic_flows=os.getenv("KAFKA_TOPIC_FLOWS", "flows"),
            group_id=os.getenv("KAFKA_GROUP_ID", "flow-processor"),
        )


# ============================================================
# FLOW AGGREGATION CONFIG
# ============================================================
# Configuracoes para agregacao de pacotes em flows.
#
# O QUE E UM FLOW?
#   Um flow e um conjunto de pacotes que compartilham:
#   - IP origem
#   - IP destino
#   - Porta origem
#   - Porta destino
#   - Protocolo (TCP/UDP/ICMP)
#
#   Essa tupla de 5 elementos e chamada "5-tuple" ou "flow key".
#
# POR QUE AGREGAR EM FLOWS?
#   - Pacotes individuais tem pouca informacao
#   - Flows capturam comportamento temporal
#   - Features de flow sao mais discriminativas para ML
#   - CICIoT2023 usa 47 features extraidas de flows
# ============================================================

@dataclass
class FlowConfig:
    """
    Configuracoes para agregacao de pacotes em flows.

    Um flow e definido pela 5-tuple:
        (src_ip, dst_ip, src_port, dst_port, protocol)
    """

    # --------------------------------------------------------
    # FLOW_TIMEOUT - Quanto tempo sem pacotes = flow terminou
    # --------------------------------------------------------
    # Se um flow nao recebe pacotes por X segundos,
    # consideramos que ele terminou e podemos extrair features.
    #
    # Valores tipicos:
    #   - 60s para TCP (conexoes longas)
    #   - 30s para UDP (sem estado)
    #   - 120s para analise mais conservadora
    # --------------------------------------------------------
    flow_timeout_seconds: float = 60.0

    # --------------------------------------------------------
    # ACTIVITY_TIMEOUT - Timeout de inatividade
    # --------------------------------------------------------
    # Diferente do flow_timeout, este considera gaps de atividade.
    # Se houver um gap > activity_timeout dentro do flow,
    # pode indicar comportamento anomalo.
    # --------------------------------------------------------
    activity_timeout_seconds: float = 5.0

    # --------------------------------------------------------
    # MAX_PACKETS_PER_FLOW - Limite de pacotes por flow
    # --------------------------------------------------------
    # Evita que um unico flow consuma toda a memoria.
    # Ataques DDoS podem gerar milhoes de pacotes no mesmo flow.
    #
    # Quando atinge o limite:
    #   1. Extrai features do flow atual
    #   2. Inicia novo flow com mesma 5-tuple
    # --------------------------------------------------------
    max_packets_per_flow: int = 10000

    # --------------------------------------------------------
    # MIN_PACKETS_PER_FLOW - Minimo de pacotes para ser flow
    # --------------------------------------------------------
    # Flows com muito poucos pacotes podem ser ruido.
    # Ignoramos flows com menos de N pacotes.
    # --------------------------------------------------------
    min_packets_per_flow: int = 2

    # --------------------------------------------------------
    # WINDOW_SIZE - Janela para calculo de features
    # --------------------------------------------------------
    # CICIoT2023 usa janelas de 10 ou 100 pacotes para
    # calcular medias e agregacoes.
    #
    # 10 pacotes: ataques com poucos pacotes (web, brute force)
    # 100 pacotes: ataques volumetricos (DDoS, DoS)
    # --------------------------------------------------------
    window_size: int = 100

    # Se True, envia flow para topico 'flows' apos processar
    publish_flows: bool = True

    # Se True, imprime flows no console (debug)
    verbose: bool = False


@dataclass
class ConsumerConfig:
    """
    Configuracao completa do Consumer.

    Agrupa todas as configs relacionadas em um unico objeto.

    COMPARACAO COM JAVA:
        Similar a um objeto de configuracao com @ConfigurationProperties
        do Spring Boot, onde voce agrupa configs relacionadas.
    """

    kafka: ConsumerKafkaConfig = field(default_factory=ConsumerKafkaConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    # --------------------------------------------------------
    # field(default_factory=...) - O QUE E?
    # --------------------------------------------------------
    # Em dataclasses, nao podemos usar objetos mutaveis como default:
    #   kafka: KafkaConfig = KafkaConfig()  # ERRO!
    #
    # Isso porque o mesmo objeto seria compartilhado entre instancias.
    #
    # default_factory resolve isso criando um NOVO objeto para cada
    # instancia da dataclass.
    #
    # COMPARACAO COM JAVA:
    #   Python: field(default_factory=KafkaConfig)
    #   Java:   private KafkaConfig kafka = new KafkaConfig();
    #           (Java cria novo objeto por instancia naturalmente)
    # --------------------------------------------------------

    @classmethod
    def from_env(cls) -> "ConsumerConfig":
        """Cria config completa a partir de variaveis de ambiente."""
        return cls(
            kafka=ConsumerKafkaConfig.from_env(),
            flow=FlowConfig(),
        )


# ============================================================
# CONFIGURACOES PADRAO PARA DESENVOLVIMENTO
# ============================================================

DEV_CONFIG = ConsumerConfig()
