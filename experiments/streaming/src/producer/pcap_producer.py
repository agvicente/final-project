"""
PCAP Producer v0.1 - Le arquivos PCAP e publica pacotes no Kafka

Funcionalidades:
- Le PCAPs usando dpkt (leve e rapido)
- Extrai metadados basicos de cada pacote
- Publica no topico 'packets' em formato JSON
- Suporta rate limiting e batching

Fluxo de dados:
    PCAP file -> dpkt (parse) -> Dict (metadados) -> JSON -> Kafka

Arquitetura:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Arquivo    │────►│  Producer   │────►│    Kafka    │
    │   PCAP      │     │  (este)     │     │  (packets)  │
    └─────────────┘     └─────────────┘     └─────────────┘
"""

# ============================================================
# IMPORTS - BIBLIOTECAS PADRAO DO PYTHON
# ============================================================

# json: Serializa objetos Python para JSON e vice-versa
# Similar ao Jackson ou Gson do Java
# json.dumps(obj) = objectMapper.writeValueAsString(obj)
import json

# time: Funcoes relacionadas a tempo
# time.time() = System.currentTimeMillis() / 1000 (em segundos)
import time

# logging: Sistema de logs do Python
# Similar ao SLF4J/Logback do Java
# Niveis: DEBUG < INFO < WARNING < ERROR < CRITICAL
import logging

# Path: Manipulacao de caminhos de arquivo orientada a objetos
# Similar ao java.nio.file.Path
# Mais moderno que os.path (strings)
from pathlib import Path

# ============================================================
# TYPE HINTS - ANOTACOES DE TIPO
# ============================================================
# Python e dinamicamente tipado, mas permite anotacoes de tipo
# para documentacao e checagem estatica (mypy, IDEs)
#
# COMPARACAO COM JAVA:
#   Python                    Java
#   ───────                   ────
#   str                       String
#   int                       int/Integer
#   float                     double/Double
#   bool                      boolean/Boolean
#   list[str]                 List<String>
#   dict[str, int]            Map<String, Integer>
#   Optional[str]             @Nullable String
#   Any                       Object
# ============================================================
from typing import (
    Iterator,   # Similar a Iterator<T> do Java - para generators
    Dict,       # Similar a Map<K,V> do Java
    Any,        # Similar a Object do Java - qualquer tipo
    Optional,   # Indica que pode ser None (null)
)

# asdict: Converte dataclass para dicionario
# Util para serializar dataclasses como JSON
from dataclasses import asdict

# ============================================================
# IMPORTS - BIBLIOTECAS EXTERNAS (pip install)
# ============================================================

# dpkt: Biblioteca leve para parsing de pacotes de rede
# Alternativas: scapy (mais pesado), pyshark (wrapper do tshark)
# Escolhemos dpkt por ser rapido e ter baixo consumo de memoria
# TODO: Estudar alternativas e escolher a melhor para o nosso caso.
import dpkt

# kafka-python: Cliente Kafka oficial para Python
# Similar ao kafka-clients do Java
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ============================================================
# IMPORT RELATIVO - MODULOS DO NOSSO PROJETO
# ============================================================
# O ponto (.) significa "deste mesmo pacote"
#   from .config    = from src.producer.config
#   from ..utils    = from src.utils (um nivel acima)
#
# COMPARACAO COM JAVA:
#   Python: from .config import KafkaConfig
#   Java:   import com.projeto.producer.config.KafkaConfig;
# ============================================================
from .config import KafkaConfig, ProducerConfig


# ============================================================
# CONFIGURACAO DE LOGGING
# ============================================================
# logging.basicConfig: Configura o sistema de logs global
#
# Parametros:
#   level: Nivel minimo de log a exibir (INFO = ignora DEBUG)
#   format: Template da mensagem de log
#       %(asctime)s    = timestamp (2024-01-15 10:30:00,123)
#       %(name)s       = nome do logger (src.producer.pcap_producer)
#       %(levelname)s  = nivel (INFO, ERROR, etc)
#       %(message)s    = a mensagem em si
#
# COMPARACAO COM JAVA (logback.xml):
#   <pattern>%d{yyyy-MM-dd HH:mm:ss} - %logger - %level - %msg%n</pattern>
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================
# __name__ - VARIAVEL ESPECIAL DO PYTHON
# ============================================================
# __name__ contem o nome do modulo atual
# - Se executar direto: __name__ = "__main__"
# - Se importar:        __name__ = "src.producer.pcap_producer"
#
# getLogger(__name__) cria um logger com o nome do modulo
# Isso permite filtrar logs por modulo no futuro
# ============================================================
logger = logging.getLogger(__name__)


class PCAPProducer:
    """
    Producer que le PCAPs e publica pacotes no Kafka.

    Esta classe encapsula toda a logica de:
    1. Conexao com Kafka
    2. Leitura de arquivos PCAP
    3. Extracao de metadados de pacotes
    4. Publicacao no topico Kafka

    Uso basico:
        producer = PCAPProducer()
        producer.process_pcap("path/to/file.pcap")
        producer.close()

    Uso com configuracao customizada:
        kafka_cfg = KafkaConfig(bootstrap_servers="kafka:9092")
        producer_cfg = ProducerConfig(max_packets=1000)
        producer = PCAPProducer(kafka_cfg, producer_cfg)

    COMPARACAO COM JAVA:
        Esta classe seria similar a uma classe com injecao de dependencias
        onde KafkaConfig e ProducerConfig seriam injetados via construtor.
    """

    # ============================================================
    # __init__ - CONSTRUTOR
    # ============================================================
    # Metodo especial chamado quando cria uma nova instancia
    # Similar ao construtor em Java
    #
    # COMPARACAO:
    #   Python: def __init__(self, param):
    #   Java:   public ClassName(Type param) { }
    #
    # self = referencia a instancia atual (como 'this' em Java)
    # Diferente de Java, em Python o 'self' e EXPLICITO
    # ============================================================
    def __init__(
        self,
        kafka_config: Optional[KafkaConfig] = None,
        producer_config: Optional[ProducerConfig] = None
    ):
        """
        Inicializa o producer com configuracoes opcionais.

        Args:
            kafka_config: Configuracoes do Kafka (None = usa defaults)
            producer_config: Configuracoes do producer (None = usa defaults)
        """
        # --------------------------------------------------------
        # OPERADOR 'or' PARA VALORES DEFAULT
        # --------------------------------------------------------
        # kafka_config or KafkaConfig() significa:
        #   - Se kafka_config for "truthy" (nao None), usa ele
        #   - Se for None/False/""/0, usa KafkaConfig()
        #
        # COMPARACAO COM JAVA:
        #   this.kafkaConfig = kafkaConfig != null ? kafkaConfig : new KafkaConfig();
        #
        # Alternativa mais explicita em Python:
        #   self.kafka_config = kafka_config if kafka_config is not None else KafkaConfig()
        # --------------------------------------------------------
        self.kafka_config = kafka_config or KafkaConfig()
        self.producer_config = producer_config or ProducerConfig()

        # --------------------------------------------------------
        # ATRIBUTO PRIVADO (CONVENCAO)
        # --------------------------------------------------------
        # O underscore (_) no inicio indica "privado por convencao"
        # Python NAO tem private/protected como Java
        # E apenas uma convencao: "nao mexa neste atributo diretamente"
        #
        # COMPARACAO COM JAVA:
        #   Python: self._producer = None     # convencao
        #   Java:   private KafkaProducer producer = null;  # enforced
        # --------------------------------------------------------
        self._producer: Optional[KafkaProducer] = None

        # Estatisticas de processamento (publicas para leitura)
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors = 0
        self.start_time: Optional[float] = None

    # ============================================================
    # METODO DE INSTANCIA
    # ============================================================
    # Metodos que operam em uma instancia especifica
    # Sempre recebem 'self' como primeiro parametro
    #
    # -> None: Indica que o metodo nao retorna nada (void em Java)
    # ============================================================
    def connect(self) -> None:
        """Conecta ao Kafka."""

        # --------------------------------------------------------
        # F-STRING (FORMATTED STRING LITERAL)
        # --------------------------------------------------------
        # f"texto {variavel}" = interpolacao de string
        # Introduzido no Python 3.6
        #
        # COMPARACAO COM JAVA:
        #   Python: f"Conectando ao Kafka em {self.kafka_config.bootstrap_servers}"
        #   Java:   "Conectando ao Kafka em " + this.kafkaConfig.getBootstrapServers()
        #   Java:   String.format("Conectando ao Kafka em %s", ...)
        #   Java 15+: "Conectando ao Kafka em %s".formatted(...)
        # --------------------------------------------------------
        logger.info(f"Conectando ao Kafka em {self.kafka_config.bootstrap_servers}...")

        # --------------------------------------------------------
        # CRIANDO O KAFKA PRODUCER
        # --------------------------------------------------------
        self._producer = KafkaProducer(
            # Endereco do broker Kafka
            bootstrap_servers=self.kafka_config.bootstrap_servers,

            # --------------------------------------------------------
            # LAMBDA - FUNCAO ANONIMA
            # --------------------------------------------------------
            # lambda args: expressao
            # E uma funcao inline de uma linha
            #
            # value_serializer: Funcao que transforma o valor antes de enviar
            # Aqui: converte Dict -> JSON string -> bytes UTF-8
            #
            # COMPARACAO COM JAVA:
            #   Python: lambda v: json.dumps(v).encode('utf-8')
            #   Java:   (v) -> objectMapper.writeValueAsBytes(v)
            #
            # Equivalente a:
            #   def serialize_value(v):
            #       json_string = json.dumps(v)  # Dict -> "{"key": "value"}"
            #       return json_string.encode('utf-8')  # String -> bytes
            # --------------------------------------------------------
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),

            # key_serializer: Serializa a chave (pode ser None)
            # 'if k else None' = operador ternario
            # Se k existir, codifica; senao, retorna None
            key_serializer=lambda k: k.encode('utf-8') if k else None,

            # Configuracoes de performance (vem do KafkaConfig)
            batch_size=self.kafka_config.batch_size,
            linger_ms=self.kafka_config.linger_ms,
            compression_type=self.kafka_config.compression_type,
            acks=self.kafka_config.acks,
        )

        logger.info("Conectado ao Kafka!")

    def close(self) -> None:
        """Fecha conexao com Kafka de forma segura."""

        # --------------------------------------------------------
        # VERIFICACAO DE NONE
        # --------------------------------------------------------
        # if self._producer:  = if self._producer is not None:
        # Em Python, None, 0, "", [], {} sao "falsy"
        # Qualquer outro valor e "truthy"
        # --------------------------------------------------------
        if self._producer:
            # flush(): Aguarda todas as mensagens pendentes serem enviadas
            # Importante chamar antes de fechar para nao perder dados
            self._producer.flush()

            # close(): Libera recursos (conexoes, threads)
            self._producer.close()

            logger.info("Conexao com Kafka fechada")

    def _extract_packet_info(
        self,
        timestamp: float,
        buf: bytes,
        packet_num: int
    ) -> Dict[str, Any]:
        """
        Extrai informacoes basicas de um pacote de rede.

        Este metodo faz o "parsing" do pacote, extraindo campos como:
        - IPs de origem e destino
        - Portas (para TCP/UDP)
        - Protocolo (TCP, UDP, ICMP, etc)
        - Flags TCP, numero de sequencia, etc

        Args:
            timestamp: Timestamp do pacote em segundos (epoch Unix)
            buf: Bytes brutos do pacote (como capturado da rede)
            packet_num: Numero sequencial do pacote no arquivo

        Returns:
            Dict[str, Any]: Dicionario com metadados extraidos
                Exemplo: {"src_ip": "192.168.1.1", "protocol": "TCP", ...}

        ESTRUTURA DE UM PACOTE DE REDE:
            ┌──────────────┬────────────┬─────────────┬──────────┐
            │   Ethernet   │     IP     │   TCP/UDP   │ Payload  │
            │   (14 bytes) │ (20 bytes) │ (20+ bytes) │ (dados)  │
            └──────────────┴────────────┴─────────────┴──────────┘
        """

        # --------------------------------------------------------
        # DICIONARIO (Dict) - ESTRUTURA CHAVE-VALOR
        # --------------------------------------------------------
        # Similar ao HashMap/Map do Java
        #
        # COMPARACAO:
        #   Python: {"key": "value", "num": 123}
        #   Java:   Map.of("key", "value", "num", 123)
        #
        # Tipos misturados sao permitidos (Dict[str, Any])
        # --------------------------------------------------------
        packet_info = {
            "packet_num": packet_num,
            "timestamp": timestamp,
            "length": len(buf),  # len() = tamanho/length
            "protocol": "unknown",
            "src_ip": None,
            "dst_ip": None,
            "src_port": None,
            "dst_port": None,
        }

        # --------------------------------------------------------
        # TRY/EXCEPT - TRATAMENTO DE EXCECOES
        # --------------------------------------------------------
        # Similar ao try/catch do Java
        #
        # COMPARACAO:
        #   Python:                     Java:
        #   try:                        try {
        #       ...                         ...
        #   except Exception as e:      } catch (Exception e) {
        #       ...                         ...
        #   finally:                    } finally {
        #       ...                         ...
        # --------------------------------------------------------
        try:
            # --------------------------------------------------------
            # PARSING DO PACOTE COM DPKT
            # --------------------------------------------------------
            # dpkt.ethernet.Ethernet(buf) interpreta os bytes como
            # um frame Ethernet e extrai os campos
            #
            # Estrutura do objeto eth:
            #   eth.src  = MAC de origem (6 bytes)
            #   eth.dst  = MAC de destino (6 bytes)
            #   eth.type = Tipo do protocolo interno (0x0800 = IP)
            #   eth.data = Conteudo (pacote IP, ARP, etc)
            # --------------------------------------------------------
            eth = dpkt.ethernet.Ethernet(buf)

            # --------------------------------------------------------
            # isinstance() - VERIFICA TIPO EM RUNTIME
            # --------------------------------------------------------
            # Similar ao 'instanceof' do Java
            #
            # COMPARACAO:
            #   Python: isinstance(obj, Classe)
            #   Java:   obj instanceof Classe
            #
            # Aqui verificamos se o conteudo do Ethernet e um pacote IP
            # --------------------------------------------------------
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data

                # Extrai IPs (bytes -> string "192.168.1.1")
                packet_info["src_ip"] = self._ip_to_str(ip.src)
                packet_info["dst_ip"] = self._ip_to_str(ip.dst)
                packet_info["ip_len"] = ip.len  # Tamanho total do pacote IP
                packet_info["ttl"] = ip.ttl     # Time To Live (saltos restantes)

                # --------------------------------------------------------
                # VERIFICACAO DO PROTOCOLO DE TRANSPORTE
                # --------------------------------------------------------
                # ip.data contem o pacote TCP, UDP ou ICMP
                # ip.p contem o numero do protocolo (6=TCP, 17=UDP, 1=ICMP)
                # --------------------------------------------------------

                # TCP - Transmission Control Protocol
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    packet_info["protocol"] = "TCP"
                    packet_info["src_port"] = tcp.sport   # Porta origem
                    packet_info["dst_port"] = tcp.dport   # Porta destino
                    packet_info["tcp_flags"] = tcp.flags  # SYN, ACK, FIN, etc
                    packet_info["tcp_seq"] = tcp.seq      # Numero de sequencia
                    packet_info["tcp_ack"] = tcp.ack      # Numero de ACK
                    packet_info["payload_len"] = len(tcp.data)  # Dados da aplicacao

                # UDP - User Datagram Protocol
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    packet_info["protocol"] = "UDP"
                    packet_info["src_port"] = udp.sport
                    packet_info["dst_port"] = udp.dport
                    packet_info["payload_len"] = len(udp.data)

                # ICMP - Internet Control Message Protocol (ping, etc)
                elif isinstance(ip.data, dpkt.icmp.ICMP):
                    packet_info["protocol"] = "ICMP"
                    packet_info["icmp_type"] = ip.data.type  # Tipo (8=echo request)
                    packet_info["icmp_code"] = ip.data.code  # Subtipo

                # Outro protocolo IP (GRE, ESP, etc)
                else:
                    # ip.p = numero do protocolo IP
                    packet_info["protocol"] = f"IP-{ip.p}"

            # IPv6 (simplificado por ora)
            elif isinstance(eth.data, dpkt.ip6.IP6):
                packet_info["protocol"] = "IPv6"

        except Exception as e:
            # Se der erro no parsing, guarda o erro mas nao interrompe
            # str(e) converte a excecao para string legivel
            packet_info["parse_error"] = str(e)

        return packet_info

    # ============================================================
    # @staticmethod - METODO ESTATICO
    # ============================================================
    # Metodo que NAO recebe 'self' nem 'cls'
    # Pertence a classe mas nao opera em instancias
    # Igual ao 'static' do Java
    #
    # QUANDO USAR:
    #   - Funcao utilitaria relacionada a classe
    #   - Nao precisa de dados da instancia
    #   - Poderia ser uma funcao solta, mas faz sentido no contexto
    # ============================================================
    @staticmethod
    def _ip_to_str(ip_bytes: bytes) -> str:
        """
        Converte bytes de IP para string legivel.

        Args:
            ip_bytes: 4 bytes representando o IP (ex: b'\\xc0\\xa8\\x01\\x01')

        Returns:
            String no formato "192.168.1.1"

        Exemplo:
            bytes: b'\\xc0\\xa8\\x01\\x01'  (192=0xc0, 168=0xa8, 1=0x01, 1=0x01)
            string: "192.168.1.1"
        """
        try:
            # --------------------------------------------------------
            # GENERATOR EXPRESSION + join()
            # --------------------------------------------------------
            # str(b) for b in ip_bytes = gera string de cada byte
            # '.'.join(...) = junta com pontos
            #
            # Exemplo passo a passo:
            #   ip_bytes = b'\xc0\xa8\x01\x01' (192, 168, 1, 1)
            #   str(b) for b in ip_bytes -> "192", "168", "1", "1"
            #   '.'.join(...) -> "192.168.1.1"
            #
            # COMPARACAO COM JAVA:
            #   String.join(".", Arrays.stream(bytes).mapToObj(String::valueOf).toArray(String[]::new))
            # --------------------------------------------------------
            return '.'.join(str(b) for b in ip_bytes)
        except Exception:
            return "unknown"

    # ============================================================
    # GENERATOR - FUNCAO QUE USA 'yield'
    # ============================================================
    # Generator e uma funcao que "produz" valores sob demanda
    # ao inves de retornar tudo de uma vez
    #
    # VANTAGENS:
    #   - Memoria: Nao carrega todo o arquivo na RAM
    #   - Lazy: So processa quando o valor e consumido
    #
    # COMPARACAO COM JAVA:
    #   Similar a Stream<T> ou Iterator<T>
    #   Python: yield valor
    #   Java:   Stream.of(...) ou implementar Iterator
    #
    # COMO FUNCIONA:
    #   1. Chama _read_pcap() -> retorna um objeto generator
    #   2. for ts, buf in generator -> a cada iteracao:
    #      - Executa ate o proximo 'yield'
    #      - Retorna o valor do yield
    #      - Pausa a funcao (guarda estado)
    #   3. Proxima iteracao -> continua de onde parou
    # ============================================================
    def _read_pcap(self, pcap_path: str) -> Iterator[tuple]:
        """
        Generator que le pacotes de um arquivo PCAP.

        Este metodo NAO carrega todo o arquivo na memoria!
        Ele le e retorna um pacote por vez (streaming).

        Args:
            pcap_path: Caminho para o arquivo PCAP

        Yields:
            tuple: (timestamp, packet_bytes) para cada pacote

        Raises:
            FileNotFoundError: Se o arquivo nao existir

        Exemplo de uso:
            for timestamp, packet_bytes in self._read_pcap("file.pcap"):
                print(f"Pacote em {timestamp}: {len(packet_bytes)} bytes")
        """

        # Path() cria um objeto Path (mais poderoso que string)
        path = Path(pcap_path)

        # path.exists() verifica se o arquivo existe
        if not path.exists():
            # raise = throw em Java
            raise FileNotFoundError(f"PCAP nao encontrado: {pcap_path}")

        # path.stat().st_size = tamanho do arquivo em bytes
        logger.info(f"Lendo PCAP: {pcap_path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

        # --------------------------------------------------------
        # CONTEXT MANAGER (with statement)
        # --------------------------------------------------------
        # 'with' garante que o arquivo sera fechado mesmo com erro
        # Similar ao try-with-resources do Java
        #
        # COMPARACAO:
        #   Python:
        #       with open(path, 'rb') as f:
        #           # usa f
        #       # f.close() automatico
        #
        #   Java:
        #       try (var f = new FileInputStream(path)) {
        #           // usa f
        #       } // f.close() automatico
        #
        # 'rb' = read binary (le bytes, nao texto)
        # --------------------------------------------------------
        with open(pcap_path, 'rb') as f:
            # dpkt.pcap.Reader le o formato PCAP
            pcap = dpkt.pcap.Reader(f)

            # --------------------------------------------------------
            # YIELD - RETORNA E PAUSA
            # --------------------------------------------------------
            # yield (ts, buf) retorna a tupla e PAUSA aqui
            # Na proxima iteracao do for, continua deste ponto
            #
            # Se fosse 'return', retornaria TUDO de uma vez
            # e carregaria todos os pacotes na memoria!
            # --------------------------------------------------------
            for ts, buf in pcap:
                yield ts, buf

    def process_pcap(
        self,
        pcap_path: str,
        max_packets: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processa um arquivo PCAP e publica pacotes no Kafka.

        Este e o metodo principal da classe. Ele:
        1. Conecta ao Kafka (se necessario)
        2. Le o PCAP pacote por pacote (streaming)
        3. Extrai metadados de cada pacote
        4. Publica no topico Kafka
        5. Retorna estatisticas

        Args:
            pcap_path: Caminho para o arquivo PCAP
            max_packets: Limite de pacotes a processar (None = todos)

        Returns:
            Dict com estatisticas: packets_sent, bytes_sent, errors, etc.
        """

        # Lazy connection: so conecta se ainda nao conectou
        if not self._producer:
            self.connect()

        # Reseta estatisticas para este processamento
        self.start_time = time.time()  # Timestamp atual em segundos
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors = 0

        # Prioridade: parametro > config > None (todos)
        max_packets = max_packets or self.producer_config.max_packets
        topic = self.kafka_config.topic_packets

        logger.info(f"Iniciando processamento do PCAP...")
        if max_packets:
            logger.info(f"Limite de pacotes: {max_packets}")

        try:
            # --------------------------------------------------------
            # enumerate() - ITERACAO COM INDICE
            # --------------------------------------------------------
            # enumerate(iterable, start) retorna (indice, valor)
            #
            # COMPARACAO COM JAVA:
            #   Python:
            #       for i, item in enumerate(lista, 1):
            #
            #   Java:
            #       int i = 1;
            #       for (var item : lista) { i++; ... }
            #
            # O segundo parametro (1) e o valor inicial do indice
            # --------------------------------------------------------
            for packet_num, (ts, buf) in enumerate(self._read_pcap(pcap_path), 1):

                # Verifica limite de pacotes
                if max_packets and packet_num > max_packets:
                    break  # Sai do loop

                # Extrai metadados do pacote
                packet_info = self._extract_packet_info(ts, buf, packet_num)

                # --------------------------------------------------------
                # CHAVE DO KAFKA (KEY)
                # --------------------------------------------------------
                # A chave determina em qual particao a mensagem vai
                # Mensagens com mesma chave vao para mesma particao
                # Isso garante ordem para mensagens do mesmo "flow"
                #
                # Flow = conexao identificada por (src_ip, src_port, dst_ip, dst_port)
                # --------------------------------------------------------
                key = None
                if packet_info["src_ip"] and packet_info["dst_ip"]:
                    # .get(key, default) retorna o valor ou default se nao existir
                    # Similar ao Map.getOrDefault() do Java
                    key = f"{packet_info['src_ip']}:{packet_info.get('src_port', 0)}-{packet_info['dst_ip']}:{packet_info.get('dst_port', 0)}"

                # Publica no Kafka
                try:
                    # send() e assincrono - retorna Future
                    # A mensagem e adicionada a um buffer interno
                    # Kafka envia em batches para performance
                    self._producer.send(topic, key=key, value=packet_info)
                    self.packets_sent += 1
                    self.bytes_sent += len(buf)

                except KafkaError as e:
                    # Conta o erro mas continua processando
                    self.errors += 1
                    if self.producer_config.verbose:
                        logger.warning(f"Erro ao enviar pacote {packet_num}: {e}")

                # --------------------------------------------------------
                # LOG DE PROGRESSO
                # --------------------------------------------------------
                # % e o operador modulo (resto da divisao)
                # packet_num % 10000 == 0 -> True a cada 10000 pacotes
                # --------------------------------------------------------
                if packet_num % self.producer_config.log_interval == 0:
                    elapsed = time.time() - self.start_time
                    rate = packet_num / elapsed

                    # --------------------------------------------------------
                    # FORMATACAO DE STRINGS
                    # --------------------------------------------------------
                    # {:,}  = separador de milhares (10,000)
                    # {:.0f} = float com 0 casas decimais
                    # {:.2f} = float com 2 casas decimais
                    # --------------------------------------------------------
                    logger.info(
                        f"Progresso: {packet_num:,} pacotes | "
                        f"{rate:.0f} pkt/s | "
                        f"{self.bytes_sent / 1024 / 1024:.2f} MB"
                    )

            # Flush: aguarda todas as mensagens pendentes serem enviadas
            # Importante antes de encerrar para nao perder dados
            self._producer.flush()

        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            # Re-raise: propaga a excecao para quem chamou
            raise

        # Calcula estatisticas finais
        elapsed = time.time() - self.start_time
        stats = {
            "pcap_path": pcap_path,
            "packets_sent": self.packets_sent,
            "bytes_sent": self.bytes_sent,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
            # Operador ternario: valor_se_true if condicao else valor_se_false
            "packets_per_second": self.packets_sent / elapsed if elapsed > 0 else 0,
            "mb_per_second": (self.bytes_sent / 1024 / 1024) / elapsed if elapsed > 0 else 0,
        }

        logger.info(f"Processamento concluido!")
        logger.info(f"  Pacotes enviados: {stats['packets_sent']:,}")
        logger.info(f"  Dados enviados: {stats['bytes_sent'] / 1024 / 1024:.2f} MB")
        logger.info(f"  Tempo total: {stats['elapsed_seconds']:.2f}s")
        logger.info(f"  Taxa: {stats['packets_per_second']:.0f} pkt/s")

        return stats


# ============================================================
# FUNCAO MAIN - PONTO DE ENTRADA CLI
# ============================================================
def main():
    """
    CLI (Command Line Interface) para testar o producer.

    Uso:
        python -m src.producer.pcap_producer arquivo.pcap
        python -m src.producer.pcap_producer arquivo.pcap -n 1000
        python -m src.producer.pcap_producer arquivo.pcap --verbose
    """

    # --------------------------------------------------------
    # argparse - PARSING DE ARGUMENTOS DE LINHA DE COMANDO
    # --------------------------------------------------------
    # Similar ao Apache Commons CLI ou picocli do Java
    #
    # Define os argumentos aceitos pelo programa
    # argparse cuida de:
    #   - Parsing dos argumentos
    #   - Validacao de tipos
    #   - Geracao de --help automatico
    #   - Mensagens de erro
    # --------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(description="PCAP Producer para Kafka")

    # Argumento posicional (obrigatorio)
    parser.add_argument("pcap", help="Caminho para o arquivo PCAP")

    # Argumentos opcionais (flags)
    # -n ou --max-packets, tipo int
    parser.add_argument("-n", "--max-packets", type=int, help="Limite de pacotes")

    # -b ou --bootstrap-servers, default se nao especificado
    parser.add_argument("-b", "--bootstrap-servers", default="localhost:9092")

    # -t ou --topic
    parser.add_argument("-t", "--topic", default="packets")

    # -v ou --verbose: flag booleana (action="store_true")
    # Se presente, args.verbose = True; senao, False
    parser.add_argument("-v", "--verbose", action="store_true")

    # parse_args() processa sys.argv e retorna objeto com os valores
    args = parser.parse_args()

    # Cria configuracoes a partir dos argumentos
    kafka_config = KafkaConfig(
        bootstrap_servers=args.bootstrap_servers,
        topic_packets=args.topic,
    )
    producer_config = ProducerConfig(
        max_packets=args.max_packets,
        verbose=args.verbose,
    )

    # Cria e executa o producer
    producer = PCAPProducer(kafka_config, producer_config)

    # --------------------------------------------------------
    # TRY/FINALLY - GARANTIR LIMPEZA
    # --------------------------------------------------------
    # finally SEMPRE executa, mesmo com erro ou return
    # Usado para liberar recursos (fechar conexoes, arquivos)
    #
    # COMPARACAO COM JAVA: identico
    # --------------------------------------------------------
    try:
        stats = producer.process_pcap(args.pcap, args.max_packets)
        # json.dumps com indent=2 formata bonito (pretty print)
        print(json.dumps(stats, indent=2))
    finally:
        producer.close()


# ============================================================
# if __name__ == "__main__" - PADRAO DE EXECUCAO
# ============================================================
# Este bloco SO executa se o arquivo for rodado diretamente
# NAO executa se for importado como modulo
#
# COMO FUNCIONA:
#   - Quando Python executa um arquivo diretamente:
#     __name__ = "__main__"
#
#   - Quando Python importa um arquivo:
#     __name__ = "nome.do.modulo"
#
# COMPARACAO COM JAVA:
#   Similar ao public static void main(String[] args)
#   Mas em Java nao ha equivalente a "importar" uma classe
#   e rodar seu main automaticamente
#
# EXEMPLO:
#   python pcap_producer.py arquivo.pcap    -> executa main()
#   from producer import PCAPProducer       -> NAO executa main()
# ============================================================
if __name__ == "__main__":
    main()
