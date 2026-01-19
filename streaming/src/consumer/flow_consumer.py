"""
Flow Consumer v0.1 - Consome pacotes do Kafka e agrega em flows

Funcionalidades:
- Consome do topico 'packets'
- Agrupa pacotes por 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)
- Extrai features basicas dos flows
- Publica flows processados no topico 'flows'

Arquitetura:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Kafka     │────►│  Consumer   │────►│   Kafka     │
    │  (packets)  │     │  (este)     │     │  (flows)    │
    └─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │    Flow     │
                        │ Aggregator  │
                        └─────────────┘
"""

# ============================================================
# IMPORTS - BIBLIOTECAS PADRAO
# ============================================================

import json
import time
import logging
import signal
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

# ============================================================
# IMPORTS - KAFKA
# ============================================================
# KafkaConsumer: Cliente para ler mensagens
# KafkaProducer: Para publicar flows processados
# ============================================================

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# ============================================================
# IMPORTS LOCAIS
# ============================================================

from .config import ConsumerConfig, ConsumerKafkaConfig, FlowConfig


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# FLOW KEY - Identificador unico de um flow
# ============================================================
# Um flow e identificado pela 5-tuple:
#   (src_ip, dst_ip, src_port, dst_port, protocol)
#
# COMPARACAO COM JAVA:
#   Python usa tuplas como chaves de dicionario (hashable)
#   Java usaria uma classe FlowKey com equals() e hashCode()
#
# Tipo: Tuple[str, str, int, int, str]
# Exemplo: ("192.168.1.1", "10.0.0.1", 54321, 80, "TCP")
# ============================================================

FlowKey = Tuple[str, str, int, int, str]


# ============================================================
# FLOW DATA - Dados agregados de um flow
# ============================================================

@dataclass
class FlowData:
    """
    Armazena dados agregados de um flow.

    Esta classe acumula informacoes de todos os pacotes
    que pertencem ao mesmo flow (mesma 5-tuple).

    Campos calculados em tempo real conforme pacotes chegam.
    """

    # --------------------------------------------------------
    # IDENTIFICACAO DO FLOW
    # --------------------------------------------------------
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------
    # Primeiro e ultimo pacote do flow
    first_packet_time: float = 0.0
    last_packet_time: float = 0.0

    # --------------------------------------------------------
    # CONTADORES
    # --------------------------------------------------------
    packet_count: int = 0
    total_bytes: int = 0

    # Forward = src -> dst, Backward = dst -> src
    fwd_packet_count: int = 0
    bwd_packet_count: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    # --------------------------------------------------------
    # TAMANHOS DE PACOTES
    # --------------------------------------------------------
    # Listas para calcular estatisticas (min, max, mean, std)
    # NOTA: Em producao, usariamos algoritmos online para
    # calcular estatisticas sem armazenar todos os valores
    # --------------------------------------------------------
    packet_sizes: List[int] = field(default_factory=list)
    fwd_packet_sizes: List[int] = field(default_factory=list)
    bwd_packet_sizes: List[int] = field(default_factory=list)

    # --------------------------------------------------------
    # INTER-ARRIVAL TIMES (IAT)
    # --------------------------------------------------------
    # Tempo entre pacotes consecutivos
    # Importante para detectar ataques (DDoS tem IAT muito baixo)
    # --------------------------------------------------------
    inter_arrival_times: List[float] = field(default_factory=list)
    fwd_inter_arrival_times: List[float] = field(default_factory=list)
    bwd_inter_arrival_times: List[float] = field(default_factory=list)

    # --------------------------------------------------------
    # TCP FLAGS
    # --------------------------------------------------------
    # Contagem de flags TCP (importante para detectar scans)
    # SYN sem ACK = possivel SYN scan
    # Muitos RST = possivel port scan
    # --------------------------------------------------------
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0

    # Ultimo timestamp de pacote forward/backward (para IAT)
    _last_fwd_time: float = field(default=0.0, repr=False)
    _last_bwd_time: float = field(default=0.0, repr=False)

    def add_packet(self, packet: Dict[str, Any], is_forward: bool = True) -> None:
        """
        Adiciona um pacote ao flow e atualiza estatisticas.

        Args:
            packet: Dicionario com dados do pacote (do producer)
            is_forward: True se pacote vai de src->dst, False se dst->src
        """
        timestamp = packet.get("timestamp", time.time())
        size = packet.get("length", 0)

        # Atualiza timestamps
        if self.packet_count == 0:
            self.first_packet_time = timestamp
        self.last_packet_time = timestamp

        # Atualiza contadores gerais
        self.packet_count += 1
        self.total_bytes += size
        self.packet_sizes.append(size)

        # Atualiza contadores direcionais
        if is_forward:
            self.fwd_packet_count += 1
            self.fwd_bytes += size
            self.fwd_packet_sizes.append(size)

            # Calcula IAT forward
            if self._last_fwd_time > 0:
                iat = timestamp - self._last_fwd_time
                self.fwd_inter_arrival_times.append(iat)
                self.inter_arrival_times.append(iat)
            self._last_fwd_time = timestamp
        else:
            self.bwd_packet_count += 1
            self.bwd_bytes += size
            self.bwd_packet_sizes.append(size)

            # Calcula IAT backward
            if self._last_bwd_time > 0:
                iat = timestamp - self._last_bwd_time
                self.bwd_inter_arrival_times.append(iat)
                self.inter_arrival_times.append(iat)
            self._last_bwd_time = timestamp

        # --------------------------------------------------------
        # ATUALIZA FLAGS TCP (se disponivel)
        # --------------------------------------------------------
        # tcp_flags e um inteiro (bitmask), nao um dicionario!
        # Cada bit representa uma flag:
        #   FIN = 0x01 (bit 0)
        #   SYN = 0x02 (bit 1)
        #   RST = 0x04 (bit 2)
        #   PSH = 0x08 (bit 3)
        #   ACK = 0x10 (bit 4)
        #   URG = 0x20 (bit 5)
        #
        # OPERACAO BITWISE: flags & 0x02 retorna 0x02 se SYN esta setado, 0 se nao
        # bool(flags & 0x02) converte para True/False
        # int(True) = 1, int(False) = 0
        # --------------------------------------------------------
        tcp_flags = packet.get("tcp_flags", 0)
        if tcp_flags and isinstance(tcp_flags, int):
            self.fin_count += 1 if (tcp_flags & 0x01) else 0
            self.syn_count += 1 if (tcp_flags & 0x02) else 0
            self.rst_count += 1 if (tcp_flags & 0x04) else 0
            self.psh_count += 1 if (tcp_flags & 0x08) else 0
            self.ack_count += 1 if (tcp_flags & 0x10) else 0
            self.urg_count += 1 if (tcp_flags & 0x20) else 0

    @property
    def duration(self) -> float:
        """Duracao do flow em segundos."""
        return self.last_packet_time - self.first_packet_time

    @property
    def flow_key(self) -> FlowKey:
        """Retorna a 5-tuple que identifica este flow."""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

    # TODO: Implementar a extracao de features do flow
    def to_features(self) -> Dict[str, Any]:
        """
        Extrai features do flow para ML.

        Retorna um dicionario com features similares ao CICIoT2023.
        Estas features serao usadas pelo modelo de deteccao.

        NOTA: Esta e uma versao simplificada. O CICIoT2023 usa
        47 features. Implementaremos mais features no futuro.
        """
        # --------------------------------------------------------
        # FUNCOES AUXILIARES PARA ESTATISTICAS
        # --------------------------------------------------------
        def safe_mean(lst: List[float]) -> float:
            """Media segura (retorna 0 se lista vazia)."""
            return sum(lst) / len(lst) if lst else 0.0

        def safe_std(lst: List[float]) -> float:
            """Desvio padrao seguro."""
            if len(lst) < 2:
                return 0.0
            mean = safe_mean(lst)
            variance = sum((x - mean) ** 2 for x in lst) / len(lst)
            return variance ** 0.5

        def safe_min(lst: List[float]) -> float:
            return min(lst) if lst else 0.0

        def safe_max(lst: List[float]) -> float:
            return max(lst) if lst else 0.0

        # --------------------------------------------------------
        # EXTRACAO DE FEATURES
        # --------------------------------------------------------
        duration = self.duration

        return {
            # Identificacao
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,

            # Temporais
            "flow_duration": duration,
            "first_packet_time": self.first_packet_time,
            "last_packet_time": self.last_packet_time,

            # Contadores
            "packet_count": self.packet_count,
            "total_bytes": self.total_bytes,
            "fwd_packet_count": self.fwd_packet_count,
            "bwd_packet_count": self.bwd_packet_count,
            "fwd_bytes": self.fwd_bytes,
            "bwd_bytes": self.bwd_bytes,

            # Taxas (packets/s, bytes/s)
            "packets_per_second": self.packet_count / duration if duration > 0 else 0,
            "bytes_per_second": self.total_bytes / duration if duration > 0 else 0,

            # Estatisticas de tamanho de pacote
            "packet_size_mean": safe_mean(self.packet_sizes),
            "packet_size_std": safe_std(self.packet_sizes),
            "packet_size_min": safe_min(self.packet_sizes),
            "packet_size_max": safe_max(self.packet_sizes),

            # Estatisticas de tamanho forward
            "fwd_packet_size_mean": safe_mean(self.fwd_packet_sizes),
            "fwd_packet_size_std": safe_std(self.fwd_packet_sizes),

            # Estatisticas de tamanho backward
            "bwd_packet_size_mean": safe_mean(self.bwd_packet_sizes),
            "bwd_packet_size_std": safe_std(self.bwd_packet_sizes),

            # Inter-arrival time
            "iat_mean": safe_mean(self.inter_arrival_times),
            "iat_std": safe_std(self.inter_arrival_times),
            "iat_min": safe_min(self.inter_arrival_times),
            "iat_max": safe_max(self.inter_arrival_times),

            # TCP Flags
            "syn_count": self.syn_count,
            "ack_count": self.ack_count,
            "fin_count": self.fin_count,
            "rst_count": self.rst_count,
            "psh_count": self.psh_count,
            "urg_count": self.urg_count,

            # Ratios
            "fwd_bwd_ratio": self.fwd_packet_count / self.bwd_packet_count if self.bwd_packet_count > 0 else 0,

            # Timestamp de extracao
            "extracted_at": datetime.now().isoformat(),
        }


# ============================================================
# FLOW CONSUMER - Classe principal
# ============================================================

class FlowConsumer:
    """
    Consumer que le pacotes do Kafka e agrega em flows.

    Fluxo:
        1. Conecta ao Kafka como consumer
        2. Le pacotes do topico 'packets'
        3. Agrupa pacotes por 5-tuple em FlowData
        4. Quando flow "fecha", extrai features
        5. Publica features no topico 'flows'

    Um flow "fecha" quando:
        - Timeout sem novos pacotes (flow_timeout)
        - Atinge max_packets_per_flow
        - Recebe FIN/RST (TCP)
    """

    def __init__(self, config: Optional[ConsumerConfig] = None):
        """
        Inicializa o consumer.

        Args:
            config: Configuracoes do consumer (None = defaults)
        """
        self.config = config or ConsumerConfig()
        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaProducer] = None

        # --------------------------------------------------------
        # DICIONARIO DE FLOWS ATIVOS
        # --------------------------------------------------------
        # Chave: FlowKey (5-tuple)
        # Valor: FlowData
        #
        # defaultdict: Cria FlowData automaticamente se chave nao existe
        # Similar a computeIfAbsent do Java Map
        # --------------------------------------------------------
        self._active_flows: Dict[FlowKey, FlowData] = {}

        # Estatisticas
        self.packets_processed = 0
        self.flows_completed = 0
        self.start_time: Optional[float] = None

        # Flag para shutdown graceful
        self._running = False

    def connect(self) -> None:
        """Conecta ao Kafka (consumer e producer)."""
        kafka_cfg = self.config.kafka

        logger.info(f"Conectando consumer ao Kafka em {kafka_cfg.bootstrap_servers}...")

        # --------------------------------------------------------
        # KAFKA CONSUMER
        # --------------------------------------------------------
        # Parametros principais:
        #   bootstrap_servers: endereco do Kafka
        #   group_id: nome do consumer group
        #   auto_offset_reset: onde comecar se nao houver offset
        #   value_deserializer: como converter bytes -> objeto
        # --------------------------------------------------------
        self._consumer = KafkaConsumer(
            kafka_cfg.topic_packets,  # Topico(s) para assinar
            bootstrap_servers=kafka_cfg.bootstrap_servers,
            group_id=kafka_cfg.group_id,
            auto_offset_reset=kafka_cfg.auto_offset_reset,
            enable_auto_commit=kafka_cfg.enable_auto_commit,
            auto_commit_interval_ms=kafka_cfg.auto_commit_interval_ms,
            max_poll_records=kafka_cfg.max_poll_records,

            # Deserializa JSON -> Dict
            # Inverso do producer: bytes -> string -> dict
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),

            # Key tambem como string
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
        )

        logger.info(f"Consumer conectado! Grupo: {kafka_cfg.group_id}")

        # --------------------------------------------------------
        # KAFKA PRODUCER (para publicar flows)
        # --------------------------------------------------------
        if self.config.flow.publish_flows:
            self._producer = KafkaProducer(
                bootstrap_servers=kafka_cfg.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            logger.info(f"Producer conectado para topico '{kafka_cfg.topic_flows}'")

    def close(self) -> None:
        """Fecha conexoes de forma segura."""
        self._running = False

        # Processa flows restantes antes de fechar
        self._flush_all_flows()

        if self._consumer:
            self._consumer.close()
            logger.info("Consumer fechado")

        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Producer fechado")

    def _get_flow_key(self, packet: Dict[str, Any]) -> Optional[FlowKey]:
        """
        Extrai a 5-tuple de um pacote.

        Args:
            packet: Dicionario com dados do pacote

        Returns:
            FlowKey ou None se pacote nao tem info suficiente
        """
        src_ip = packet.get("src_ip")
        dst_ip = packet.get("dst_ip")
        src_port = packet.get("src_port", 0)
        dst_port = packet.get("dst_port", 0)
        protocol = packet.get("protocol", "UNKNOWN")

        # Valida campos obrigatorios
        if not src_ip or not dst_ip:
            return None

        return (src_ip, dst_ip, src_port, dst_port, protocol)

    def _get_or_create_flow(self, key: FlowKey) -> FlowData:
        """
        Obtem flow existente ou cria um novo.

        Similar ao computeIfAbsent do Java Map.
        """
        if key not in self._active_flows:
            self._active_flows[key] = FlowData(
                src_ip=key[0],
                dst_ip=key[1],
                src_port=key[2],
                dst_port=key[3],
                protocol=key[4],
            )
        return self._active_flows[key]

    def _is_forward_packet(self, packet: Dict[str, Any], flow: FlowData) -> bool:
        """
        Determina se pacote e forward (src->dst) ou backward (dst->src).

        Compara o IP origem do pacote com o IP origem do flow.
        """
        return packet.get("src_ip") == flow.src_ip

    def _process_packet(self, packet: Dict[str, Any]) -> None:
        """
        Processa um pacote: adiciona ao flow correspondente.

        Args:
            packet: Dados do pacote do Kafka
        """
        # Extrai flow key
        key = self._get_flow_key(packet)
        if key is None:
            return  # Pacote sem info suficiente

        # Verifica se e o flow normal ou reverso
        # Flow (A->B) e flow (B->A) sao o MESMO flow bidirecional
        reverse_key = (key[1], key[0], key[3], key[2], key[4])

        # Usa a chave que ja existe, ou a normal se nenhuma existe
        if reverse_key in self._active_flows:
            actual_key = reverse_key
            is_forward = False
        else:
            actual_key = key
            is_forward = True

        # Obtem ou cria o flow
        flow = self._get_or_create_flow(actual_key)

        # Adiciona pacote ao flow
        flow.add_packet(packet, is_forward)
        self.packets_processed += 1

        # Verifica se flow deve ser fechado
        if flow.packet_count >= self.config.flow.max_packets_per_flow:
            self._complete_flow(actual_key, reason="max_packets")

    def _complete_flow(self, key: FlowKey, reason: str = "timeout") -> None:
        """
        Completa um flow: extrai features e publica.

        Args:
            key: FlowKey do flow a completar
            reason: Motivo do fechamento (timeout, max_packets, fin, rst)
        """
        if key not in self._active_flows:
            return

        flow = self._active_flows.pop(key)

        # Ignora flows muito pequenos
        if flow.packet_count < self.config.flow.min_packets_per_flow:
            return

        # Extrai features
        features = flow.to_features()
        features["close_reason"] = reason

        self.flows_completed += 1

        # Log se verbose
        if self.config.flow.verbose:
            logger.info(
                f"Flow completado: {flow.src_ip}:{flow.src_port} -> "
                f"{flow.dst_ip}:{flow.dst_port} ({flow.protocol}) - "
                f"{flow.packet_count} pacotes, {flow.duration:.2f}s"
            )

        # Publica no Kafka
        if self._producer and self.config.flow.publish_flows:
            flow_key_str = f"{flow.src_ip}-{flow.dst_ip}-{flow.protocol}"
            self._producer.send(
                self.config.kafka.topic_flows,
                key=flow_key_str,
                value=features,
            )

    def _check_flow_timeouts(self) -> None:
        """
        Verifica e fecha flows que atingiram timeout.

        Chamado periodicamente durante o consumo.
        """
        current_time = time.time()
        timeout = self.config.flow.flow_timeout_seconds

        # Lista de flows a fechar (nao podemos modificar dict durante iteracao)
        flows_to_close = []

        for key, flow in self._active_flows.items():
            if current_time - flow.last_packet_time > timeout:
                flows_to_close.append(key)

        # Fecha flows
        for key in flows_to_close:
            self._complete_flow(key, reason="timeout")

    def _flush_all_flows(self) -> None:
        """Fecha todos os flows ativos (usado no shutdown)."""
        keys = list(self._active_flows.keys())
        for key in keys:
            self._complete_flow(key, reason="shutdown")

    def run(self, max_messages: Optional[int] = None) -> Dict[str, Any]:
        """
        Executa o consumer em loop.

        Args:
            max_messages: Limite de mensagens (None = infinito)

        Returns:
            Estatisticas de execucao
        """
        if not self._consumer:
            self.connect()

        self._running = True
        self.start_time = time.time()
        last_timeout_check = time.time()

        logger.info("Iniciando consumo de pacotes...")
        logger.info(f"Topico: {self.config.kafka.topic_packets}")
        logger.info(f"Grupo: {self.config.kafka.group_id}")

        try:
            # --------------------------------------------------------
            # LOOP PRINCIPAL DE CONSUMO
            # --------------------------------------------------------
            # poll() retorna um dict de {TopicPartition: [mensagens]}
            # Processamos cada mensagem individualmente
            # --------------------------------------------------------
            while self._running:
                # Poll com timeout
                records = self._consumer.poll(
                    timeout_ms=self.config.kafka.poll_timeout_ms
                )

                # Processa cada mensagem
                for topic_partition, messages in records.items():
                    for message in messages:
                        self._process_packet(message.value)

                        # Verifica limite de mensagens
                        if max_messages and self.packets_processed >= max_messages:
                            logger.info(f"Limite de {max_messages} mensagens atingido")
                            self._running = False
                            break

                    if not self._running:
                        break

                # Verifica timeouts periodicamente (a cada segundo)
                if time.time() - last_timeout_check > 1.0:
                    self._check_flow_timeouts()
                    last_timeout_check = time.time()

                # Log de progresso
                if self.packets_processed > 0 and self.packets_processed % 10000 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.packets_processed / elapsed
                    logger.info(
                        f"Progresso: {self.packets_processed} pacotes, "
                        f"{self.flows_completed} flows, "
                        f"{rate:.0f} pkt/s"
                    )

        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuario (Ctrl+C)")

        finally:
            self.close()

        # Retorna estatisticas
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "packets_processed": self.packets_processed,
            "flows_completed": self.flows_completed,
            "active_flows_at_end": len(self._active_flows),
            "elapsed_seconds": elapsed,
            "packets_per_second": self.packets_processed / elapsed if elapsed > 0 else 0,
        }


# ============================================================
# SIGNAL HANDLERS - Shutdown graceful
# ============================================================

_consumer_instance: Optional[FlowConsumer] = None


def _signal_handler(signum, frame):
    """Handler para SIGINT e SIGTERM."""
    logger.info(f"Recebido sinal {signum}, iniciando shutdown...")
    if _consumer_instance:
        _consumer_instance._running = False


# ============================================================
# MAIN - Ponto de entrada
# ============================================================

if __name__ == "__main__":
    """
    Execucao direta do consumer.

    Uso:
        cd streaming
        python -m src.consumer.flow_consumer

    Ou com argumentos:
        python -m src.consumer.flow_consumer --max-messages 1000 --verbose
    """
    import argparse

    # --------------------------------------------------------
    # PARSE DE ARGUMENTOS
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Flow Consumer - Consome pacotes e agrega em flows"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Numero maximo de mensagens a processar (default: infinito)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra cada flow)"
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Nao publica flows no Kafka (apenas processa)"
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="flow-processor",
        help="ID do consumer group (default: flow-processor)"
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Comeca do inicio do topico (auto_offset_reset=earliest)"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CONFIGURACAO
    # --------------------------------------------------------
    config = ConsumerConfig()
    config.flow.verbose = args.verbose
    config.flow.publish_flows = not args.no_publish
    config.kafka.group_id = args.group_id

    if args.from_beginning:
        config.kafka.auto_offset_reset = "earliest"

    # --------------------------------------------------------
    # SETUP SIGNAL HANDLERS
    # --------------------------------------------------------
    # Registra handlers para shutdown graceful
    # SIGINT = Ctrl+C, SIGTERM = kill
    # --------------------------------------------------------
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --------------------------------------------------------
    # EXECUTA CONSUMER
    # --------------------------------------------------------
    consumer = FlowConsumer(config)
    _consumer_instance = consumer  # Para signal handler

    logger.info("=" * 60)
    logger.info("Flow Consumer v0.1")
    logger.info("=" * 60)

    try:
        stats = consumer.run(max_messages=args.max_messages)

        logger.info("=" * 60)
        logger.info("Estatisticas finais:")
        logger.info(f"  Pacotes processados: {stats['packets_processed']}")
        logger.info(f"  Flows completados: {stats['flows_completed']}")
        logger.info(f"  Tempo total: {stats['elapsed_seconds']:.2f}s")
        logger.info(f"  Taxa: {stats['packets_per_second']:.0f} pkt/s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise
