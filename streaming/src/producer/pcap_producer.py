"""
PCAP Producer v0.1 - Le arquivos PCAP e publica pacotes no Kafka

Funcionalidades:
- Le PCAPs usando dpkt (leve e rapido)
- Extrai metadados basicos de cada pacote
- Publica no topico 'packets' em formato JSON
- Suporta rate limiting e batching
"""
import json
import time
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from dataclasses import asdict

import dpkt
from kafka import KafkaProducer
from kafka.errors import KafkaError

from .config import KafkaConfig, ProducerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PCAPProducer:
    """
    Producer que le PCAPs e publica pacotes no Kafka.

    Uso:
        producer = PCAPProducer()
        producer.process_pcap("path/to/file.pcap")
    """

    def __init__(
        self,
        kafka_config: Optional[KafkaConfig] = None,
        producer_config: Optional[ProducerConfig] = None
    ):
        self.kafka_config = kafka_config or KafkaConfig()
        self.producer_config = producer_config or ProducerConfig()
        self._producer: Optional[KafkaProducer] = None

        # Stats
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors = 0
        self.start_time: Optional[float] = None

    def connect(self) -> None:
        """Conecta ao Kafka"""
        logger.info(f"Conectando ao Kafka em {self.kafka_config.bootstrap_servers}...")

        self._producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            batch_size=self.kafka_config.batch_size,
            linger_ms=self.kafka_config.linger_ms,
            compression_type=self.kafka_config.compression_type,
            acks=self.kafka_config.acks,
        )

        logger.info("Conectado ao Kafka!")

    def close(self) -> None:
        """Fecha conexao com Kafka"""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Conexao com Kafka fechada")

    def _extract_packet_info(
        self,
        timestamp: float,
        buf: bytes,
        packet_num: int
    ) -> Dict[str, Any]:
        """
        Extrai informacoes basicas de um pacote.

        Args:
            timestamp: Timestamp do pacote (epoch)
            buf: Bytes raw do pacote
            packet_num: Numero sequencial do pacote

        Returns:
            Dict com metadados do pacote
        """
        packet_info = {
            "packet_num": packet_num,
            "timestamp": timestamp,
            "length": len(buf),
            "protocol": "unknown",
            "src_ip": None,
            "dst_ip": None,
            "src_port": None,
            "dst_port": None,
        }

        try:
            # Parse Ethernet frame
            eth = dpkt.ethernet.Ethernet(buf)

            # Check if IP packet
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                packet_info["src_ip"] = self._ip_to_str(ip.src)
                packet_info["dst_ip"] = self._ip_to_str(ip.dst)
                packet_info["ip_len"] = ip.len
                packet_info["ttl"] = ip.ttl

                # TCP
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    packet_info["protocol"] = "TCP"
                    packet_info["src_port"] = tcp.sport
                    packet_info["dst_port"] = tcp.dport
                    packet_info["tcp_flags"] = tcp.flags
                    packet_info["tcp_seq"] = tcp.seq
                    packet_info["tcp_ack"] = tcp.ack
                    packet_info["payload_len"] = len(tcp.data)

                # UDP
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    packet_info["protocol"] = "UDP"
                    packet_info["src_port"] = udp.sport
                    packet_info["dst_port"] = udp.dport
                    packet_info["payload_len"] = len(udp.data)

                # ICMP
                elif isinstance(ip.data, dpkt.icmp.ICMP):
                    packet_info["protocol"] = "ICMP"
                    packet_info["icmp_type"] = ip.data.type
                    packet_info["icmp_code"] = ip.data.code

                else:
                    packet_info["protocol"] = f"IP-{ip.p}"

            # IPv6
            elif isinstance(eth.data, dpkt.ip6.IP6):
                packet_info["protocol"] = "IPv6"
                # Simplificado por ora

        except Exception as e:
            packet_info["parse_error"] = str(e)

        return packet_info

    @staticmethod
    def _ip_to_str(ip_bytes: bytes) -> str:
        """Converte bytes de IP para string"""
        try:
            return '.'.join(str(b) for b in ip_bytes)
        except Exception:
            return "unknown"

    def _read_pcap(self, pcap_path: str) -> Iterator[tuple]:
        """
        Generator que le pacotes de um PCAP.

        Yields:
            (timestamp, packet_bytes)
        """
        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP nao encontrado: {pcap_path}")

        logger.info(f"Lendo PCAP: {pcap_path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                yield ts, buf

    def process_pcap(
        self,
        pcap_path: str,
        max_packets: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processa um arquivo PCAP e publica pacotes no Kafka.

        Args:
            pcap_path: Caminho para o arquivo PCAP
            max_packets: Limite de pacotes (None = todos)

        Returns:
            Dict com estatisticas de processamento
        """
        if not self._producer:
            self.connect()

        self.start_time = time.time()
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors = 0

        max_packets = max_packets or self.producer_config.max_packets
        topic = self.kafka_config.topic_packets

        logger.info(f"Iniciando processamento do PCAP...")
        if max_packets:
            logger.info(f"Limite de pacotes: {max_packets}")

        try:
            for packet_num, (ts, buf) in enumerate(self._read_pcap(pcap_path), 1):
                # Limite de pacotes
                if max_packets and packet_num > max_packets:
                    break

                # Extrai info do pacote
                packet_info = self._extract_packet_info(ts, buf, packet_num)

                # Cria chave baseada no flow (src_ip:src_port -> dst_ip:dst_port)
                key = None
                if packet_info["src_ip"] and packet_info["dst_ip"]:
                    key = f"{packet_info['src_ip']}:{packet_info.get('src_port', 0)}-{packet_info['dst_ip']}:{packet_info.get('dst_port', 0)}"

                # Publica no Kafka
                try:
                    self._producer.send(topic, key=key, value=packet_info)
                    self.packets_sent += 1
                    self.bytes_sent += len(buf)
                except KafkaError as e:
                    self.errors += 1
                    if self.producer_config.verbose:
                        logger.warning(f"Erro ao enviar pacote {packet_num}: {e}")

                # Log de progresso
                if packet_num % self.producer_config.log_interval == 0:
                    elapsed = time.time() - self.start_time
                    rate = packet_num / elapsed
                    logger.info(
                        f"Progresso: {packet_num:,} pacotes | "
                        f"{rate:.0f} pkt/s | "
                        f"{self.bytes_sent / 1024 / 1024:.2f} MB"
                    )

            # Flush final
            self._producer.flush()

        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            raise

        # Estatisticas finais
        elapsed = time.time() - self.start_time
        stats = {
            "pcap_path": pcap_path,
            "packets_sent": self.packets_sent,
            "bytes_sent": self.bytes_sent,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
            "packets_per_second": self.packets_sent / elapsed if elapsed > 0 else 0,
            "mb_per_second": (self.bytes_sent / 1024 / 1024) / elapsed if elapsed > 0 else 0,
        }

        logger.info(f"Processamento concluido!")
        logger.info(f"  Pacotes enviados: {stats['packets_sent']:,}")
        logger.info(f"  Dados enviados: {stats['bytes_sent'] / 1024 / 1024:.2f} MB")
        logger.info(f"  Tempo total: {stats['elapsed_seconds']:.2f}s")
        logger.info(f"  Taxa: {stats['packets_per_second']:.0f} pkt/s")

        return stats


def main():
    """CLI para testar o producer"""
    import argparse

    parser = argparse.ArgumentParser(description="PCAP Producer para Kafka")
    parser.add_argument("pcap", help="Caminho para o arquivo PCAP")
    parser.add_argument("-n", "--max-packets", type=int, help="Limite de pacotes")
    parser.add_argument("-b", "--bootstrap-servers", default="localhost:9092")
    parser.add_argument("-t", "--topic", default="packets")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    kafka_config = KafkaConfig(
        bootstrap_servers=args.bootstrap_servers,
        topic_packets=args.topic,
    )
    producer_config = ProducerConfig(
        max_packets=args.max_packets,
        verbose=args.verbose,
    )

    producer = PCAPProducer(kafka_config, producer_config)

    try:
        stats = producer.process_pcap(args.pcap, args.max_packets)
        print(json.dumps(stats, indent=2))
    finally:
        producer.close()


if __name__ == "__main__":
    main()
