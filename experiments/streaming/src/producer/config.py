"""
Configuracoes do Producer para IoT IDS Streaming

Este arquivo define as configuracoes usando dataclasses do Python.
"""

# ============================================================
# IMPORTS
# ============================================================

# dataclass: Decorador que gera automaticamente __init__, __repr__, __eq__
# Similar ao 'record' do Java 14+ ou @Data do Lombok
# Sem dataclass, teriamos que escrever ~15 linhas de codigo repetitivo
from dataclasses import dataclass

# Optional: Indica que um valor pode ser None
# Optional[str] = str | None (string ou nulo)
# Similar ao @Nullable do Java
from typing import Optional

# os: Acesso a variaveis de ambiente do sistema operacional
# os.getenv("VAR") = System.getenv("VAR") em Java
import os


# ============================================================
# @dataclass - O QUE E?
# ============================================================
# Decorador que transforma uma classe em um "container de dados"
#
# O que o @dataclass gera automaticamente:
#   - __init__(): Construtor com todos os campos
#   - __repr__(): Representacao em string para debug
#   - __eq__(): Comparacao entre instancias
#
# COMPARACAO COM JAVA:
#   Python @dataclass  ≈  Java record (14+) ou POJO com Lombok @Data
#
# Exemplo equivalente em Java:
#   public record KafkaConfig(
#       String bootstrapServers,
#       String topicPackets
#   ) {}
# ============================================================

@dataclass
class KafkaConfig:
    """
    Configuracoes de conexao com Kafka.

    Uso:
        # Forma 1: Construtor padrao (usa valores default)
        config = KafkaConfig()

        # Forma 2: Especificando valores
        config = KafkaConfig(bootstrap_servers="kafka:9092")

        # Forma 3: A partir de variaveis de ambiente
        config = KafkaConfig.from_env()
    """

    # --------------------------------------------------------
    # CAMPOS DA DATACLASS
    # --------------------------------------------------------
    # Sintaxe: nome: tipo = valor_padrao
    # O @dataclass gera __init__ automaticamente com esses campos
    # --------------------------------------------------------

    # Endereco do Kafka (host:porta)
    # Em producao, pode ser lista: "kafka1:9092,kafka2:9092"
    bootstrap_servers: str = "localhost:9092"

    # Topico onde o producer envia pacotes raw do PCAP
    topic_packets: str = "packets"

    # Topico onde o consumer envia flows processados (47 features)
    topic_flows: str = "flows"

    # --------------------------------------------------------
    # CONFIGURACOES DO PRODUCER KAFKA
    # --------------------------------------------------------

    # batch_size: Tamanho maximo do lote em bytes antes de enviar
    # 16384 = 16KB. Kafka acumula mensagens ate esse tamanho
    batch_size: int = 16384

    # linger_ms: Tempo maximo de espera para acumular mensagens
    # Mesmo que o batch nao encha, envia apos 10ms
    # Trade-off: maior = mais eficiente, menor = menor latencia
    linger_ms: int = 10

    # compression_type: Algoritmo de compressao
    # Opcoes: 'none', 'gzip', 'snappy', 'lz4', 'zstd'
    # gzip: boa compressao, mais CPU. Ideal para alto volume.
    # TODO: Estudar os outros algoritmos de compressao e escolher o melhor para o nosso caso.
    compression_type: str = "gzip"

    # acks: Quantas confirmacoes o producer espera
    #   0 = nenhuma (fire and forget, pode perder dados)
    #   1 = apenas do leader (balanco entre seguranca e velocidade)
    #   'all' = de todas as replicas (mais seguro, mais lento)
    # TODO: Estudar diferencase trade offs entre tipos de acks.
    acks: int = 1

    # --------------------------------------------------------
    # @classmethod - O QUE E?
    # --------------------------------------------------------
    # Metodo que pertence a CLASSE, nao a uma instancia.
    #
    # DIFERENCA PARA @staticmethod:
    #   @classmethod  - recebe 'cls' (a classe) como 1o parametro
    #   @staticmethod - nao recebe nada (como 'static' do Java)
    #
    # COMPARACAO COM JAVA:
    #   Python @staticmethod ≈ Java static
    #   Python @classmethod  ≈ Nao existe direto em Java
    #
    # POR QUE USAR @classmethod?
    #   - Factory methods (formas alternativas de criar a classe)
    #   - Funciona corretamente com heranca (cls = subclasse)
    #
    # O QUE E 'cls'?
    #   - Primeiro parametro implicito (como 'self' para instancias)
    #   - self = a instancia (this em Java)
    #   - cls  = a classe em si (NomeClasse.class em Java)
    # --------------------------------------------------------

    @classmethod
    def from_env(cls) -> "KafkaConfig":
        # --------------------------------------------------------
        # FORWARD REFERENCE: -> "KafkaConfig" (com aspas)
        # --------------------------------------------------------
        # Por que aspas? Python le o arquivo linha por linha.
        # Quando chega aqui, ainda esta definindo KafkaConfig,
        # entao o nome 'KafkaConfig' ainda nao existe!
        #
        # Com aspas, Python entende: "vou resolver depois"
        #
        # JAVA NAO PRECISA DISSO porque o compilador faz 2 passagens:
        #   1a passagem: coleta todos os nomes de classes
        #   2a passagem: resolve as referencias
        # --------------------------------------------------------
        """
        Cria config a partir de variaveis de ambiente.

        Factory method - forma alternativa de construir a classe.
        Util para ambientes diferentes (dev, staging, prod).

        Variaveis de ambiente:
            KAFKA_BOOTSTRAP_SERVERS: Endereco do Kafka
            KAFKA_TOPIC_PACKETS: Nome do topico de pacotes
            KAFKA_TOPIC_FLOWS: Nome do topico de flows

        Exemplo:
            export KAFKA_BOOTSTRAP_SERVERS=kafka-prod:9092
            config = KafkaConfig.from_env()
        """
        # cls(...) = KafkaConfig(...) - cria nova instancia
        # Usar 'cls' ao inves de 'KafkaConfig' permite que subclasses
        # funcionem corretamente (cls sera a subclasse)
        return cls(
            # os.getenv(nome, valor_padrao)
            # Se a variavel nao existir, usa o valor padrao
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_packets=os.getenv("KAFKA_TOPIC_PACKETS", "packets"),
            topic_flows=os.getenv("KAFKA_TOPIC_FLOWS", "flows"),
        )


@dataclass
class ProducerConfig:
    """
    Configuracoes do PCAP Producer.

    Controla como o producer le e processa arquivos PCAP.
    """

    # --------------------------------------------------------
    # Optional[tipo] - O QUE E?
    # --------------------------------------------------------
    # Indica que o valor pode ser None (nulo)
    # Optional[str] equivale a: str | None
    #
    # Similar ao @Nullable do Java ou String? do Kotlin
    # --------------------------------------------------------

    # Caminho para o arquivo PCAP (None = nao definido ainda)
    pcap_path: Optional[str] = None

    # Quantos pacotes processar antes de dar flush no Kafka
    # Maior = mais eficiente (menos I/O), Menor = menos risco de perda
    # flush() forca o envio de todas as mensagens no buffer e aguarda confirmacao
    flush_interval: int = 1000

    # Limite de pacotes a processar (None = todos)
    # Util para testes: max_packets=100 processa so 100
    max_packets: Optional[int] = None

    # A cada quantos pacotes mostrar log de progresso
    log_interval: int = 10000

    # Se True, mostra logs detalhados (debug)
    verbose: bool = False


# ============================================================
# CONFIGURACOES PADRAO PARA DESENVOLVIMENTO
# ============================================================

# Instancia padrao para uso rapido em dev
# Como @dataclass gera __init__ com defaults, KafkaConfig() funciona
DEV_CONFIG = KafkaConfig()

# Caminho relativo ao diretorio streaming/
# Usado para testes locais com PCAP pequeno (9MB)
DEV_PCAP = "../data/raw/PCAP/SqlInjection/SqlInjection.pcap"
