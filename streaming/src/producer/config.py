"""
Configuracoes do Producer para IoT IDS Streaming
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class KafkaConfig:
    """Configuracoes de conexao com Kafka"""
    bootstrap_servers: str = "localhost:9092"
    topic_packets: str = "packets"
    topic_flows: str = "flows"

    # Producer configs
    batch_size: int = 16384  # 16KB
    linger_ms: int = 10  # Aguarda ate 10ms para batch
    compression_type: str = "gzip"  # Compressao para alto volume
    acks: int = 1  # Aguarda confirmacao do leader (0, 1, ou 'all')

    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """Cria config a partir de variaveis de ambiente"""
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_packets=os.getenv("KAFKA_TOPIC_PACKETS", "packets"),
            topic_flows=os.getenv("KAFKA_TOPIC_FLOWS", "flows"),
        )


@dataclass
class ProducerConfig:
    """Configuracoes do PCAP Producer"""
    # Paths
    pcap_path: Optional[str] = None

    # Processing
    batch_size: int = 1000  # Pacotes por batch
    max_packets: Optional[int] = None  # Limite de pacotes (None = todos)

    # Logging
    log_interval: int = 10000  # Log a cada N pacotes
    verbose: bool = False


# Configs padrao para desenvolvimento local
DEV_CONFIG = KafkaConfig()
DEV_PCAP = "../data/raw/PCAP/SqlInjection/SqlInjection.pcap"  # Relativo a streaming/
