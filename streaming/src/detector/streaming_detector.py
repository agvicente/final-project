"""
Streaming Detector - Integra Consumer Kafka com TEDA para deteccao em tempo real.

Este modulo conecta o pipeline de streaming com o detector TEDA:
    1. Consome flows do topico 'flows'
    2. Extrai features numericas de cada flow
    3. Passa para o TEDADetector
    4. Publica alertas no topico 'alerts'

Arquitetura:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Kafka     │────►│  Streaming  │────►│   Kafka     │
    │  (flows)    │     │  Detector   │     │  (alerts)   │
    └─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │    TEDA     │
                        │  Detector   │
                        └─────────────┘

Uso:
    python -m src.detector.streaming_detector --verbose
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import time
import logging
import signal
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from .teda import TEDADetector, TEDAResult

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACAO
# ============================================================

@dataclass
class StreamingDetectorConfig:
    """
    Configuracoes do StreamingDetector.

    Agrupa configuracoes de Kafka e TEDA em um unico objeto.
    """
    # Kafka
    bootstrap_servers: str = "localhost:9092"
    topic_flows: str = "flows"
    topic_alerts: str = "alerts"
    group_id: str = "teda-detector"
    auto_offset_reset: str = "earliest"

    # TEDA
    teda_m: float = 3.0  # Desvios padrao para threshold
    teda_min_samples: int = 10  # Minimo de amostras antes de detectar

    # Features a usar (None = todas numericas)
    feature_names: Optional[List[str]] = None

    # Comportamento
    publish_alerts: bool = True
    publish_all_results: bool = False  # Se True, publica normal tambem
    verbose: bool = False
    log_interval: int = 100  # Log a cada N flows


# ============================================================
# FEATURES PARA TEDA
# ============================================================

# Features numericas extraidas do flow que serao usadas pelo TEDA
# Estas features foram selecionadas por serem:
# 1. Numericas (TEDA trabalha com vetores)
# 2. Relevantes para deteccao de anomalias
# 3. Disponiveis no Consumer v0.1

DEFAULT_FEATURES = [
    # Contadores basicos
    "packet_count",
    "total_bytes",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_bytes",
    "bwd_bytes",

    # Taxas
    "packets_per_second",
    "bytes_per_second",

    # Estatisticas de tamanho
    "packet_size_mean",
    "packet_size_std",

    # Inter-arrival time
    "iat_mean",
    "iat_std",

    # TCP Flags (importantes para detectar scans)
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",

    # Ratio
    "fwd_bwd_ratio",
]


# ============================================================
# STREAMING DETECTOR
# ============================================================

class StreamingDetector:
    """
    Detector de anomalias em streaming usando TEDA.

    Consome flows do Kafka, aplica TEDA para detectar anomalias,
    e publica alertas.

    Fluxo:
        1. Conecta ao Kafka (consumer para 'flows', producer para 'alerts')
        2. Para cada flow recebido:
           a. Extrai features numericas
           b. Normaliza features (opcional)
           c. Passa para TEDADetector.update()
           d. Se anomalia, publica alerta
        3. Mantem estatisticas de execucao

    Uso:
        detector = StreamingDetector()
        detector.run()  # Loop infinito
        # ou
        detector.run(max_flows=1000)  # Processa 1000 flows
    """

    def __init__(self, config: Optional[StreamingDetectorConfig] = None):
        """
        Inicializa o StreamingDetector.

        Args:
            config: Configuracoes (None = defaults)
        """
        self.config = config or StreamingDetectorConfig()

        # Kafka clients
        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaProducer] = None

        # TEDA detector
        self._teda = TEDADetector(
            m=self.config.teda_m,
            min_samples=self.config.teda_min_samples,
        )

        # Features a usar
        self._feature_names = self.config.feature_names or DEFAULT_FEATURES

        # Estatisticas
        self.flows_processed = 0
        self.anomalies_detected = 0
        self.start_time: Optional[float] = None

        # Controle
        self._running = False

    def connect(self) -> None:
        """Conecta ao Kafka."""
        logger.info(f"Conectando ao Kafka em {self.config.bootstrap_servers}...")

        # Consumer para flows
        self._consumer = KafkaConsumer(
            self.config.topic_flows,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.group_id,
            auto_offset_reset=self.config.auto_offset_reset,
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
        )
        logger.info(f"Consumer conectado ao topico '{self.config.topic_flows}'")

        # Producer para alerts
        if self.config.publish_alerts:
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            logger.info(f"Producer conectado ao topico '{self.config.topic_alerts}'")

    def close(self) -> None:
        """Fecha conexoes."""
        self._running = False

        if self._consumer:
            self._consumer.close()
            logger.info("Consumer fechado")

        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Producer fechado")

    def _extract_features(self, flow: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extrai features numericas de um flow.

        Args:
            flow: Dicionario com dados do flow (do Consumer)

        Returns:
            Array numpy com features, ou None se erro
        """
        try:
            features = []
            for name in self._feature_names:
                value = flow.get(name, 0)
                # Converte para float, tratando None
                if value is None:
                    value = 0.0
                features.append(float(value))

            return np.array(features, dtype=np.float64)

        except Exception as e:
            logger.warning(f"Erro extraindo features: {e}")
            return None

    def _create_alert(
        self,
        flow: Dict[str, Any],
        result: TEDAResult
    ) -> Dict[str, Any]:
        """
        Cria mensagem de alerta para publicar no Kafka.

        Args:
            flow: Dados originais do flow
            result: Resultado do TEDA

        Returns:
            Dicionario com alerta formatado
        """
        return {
            # Identificacao do flow
            "flow_id": f"{flow.get('src_ip', '?')}:{flow.get('src_port', '?')}->"
                      f"{flow.get('dst_ip', '?')}:{flow.get('dst_port', '?')}",
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "src_port": flow.get("src_port"),
            "dst_port": flow.get("dst_port"),
            "protocol": flow.get("protocol"),

            # Metricas do flow
            "packet_count": flow.get("packet_count"),
            "total_bytes": flow.get("total_bytes"),
            "flow_duration": flow.get("flow_duration"),

            # Resultado TEDA
            # Converte para tipos Python nativos (numpy types nao sao JSON serializable)
            "eccentricity": float(result.eccentricity),
            "typicality": float(result.typicality),
            "threshold": float(result.threshold),
            "is_anomaly": bool(result.is_anomaly),

            # Contexto
            "sample_number": result.sample_count,
            "detected_at": datetime.now().isoformat(),

            # Severidade (baseada em quao acima do threshold)
            "severity": self._calculate_severity(result),
        }

    def _calculate_severity(self, result: TEDAResult) -> str:
        """
        Calcula severidade da anomalia baseada na eccentricity.

        Quanto maior a eccentricity acima do threshold, mais severa.

        Returns:
            "low", "medium", "high", ou "critical"
        """
        if not result.is_anomaly:
            return "normal"

        # Ratio de eccentricity sobre threshold
        ratio = result.normalized_eccentricity / result.threshold

        if ratio < 1.5:
            return "low"
        elif ratio < 2.0:
            return "medium"
        elif ratio < 3.0:
            return "high"
        else:
            return "critical"

    def _process_flow(self, flow: Dict[str, Any]) -> Optional[TEDAResult]:
        """
        Processa um flow: extrai features, aplica TEDA, publica alerta.

        Args:
            flow: Dados do flow do Kafka

        Returns:
            TEDAResult ou None se erro
        """
        # Extrai features
        features = self._extract_features(flow)
        if features is None:
            return None

        # Aplica TEDA
        result = self._teda.update(features)

        self.flows_processed += 1

        # Log verbose
        if self.config.verbose:
            status = "ANOMALIA!" if result.is_anomaly else "normal"
            logger.info(
                f"Flow {self.flows_processed}: "
                f"ξ={result.eccentricity:.4f}, "
                f"τ={result.typicality:.4f}, "
                f"threshold={result.threshold:.4f} → {status}"
            )

        # Se anomalia, publica alerta
        if result.is_anomaly:
            self.anomalies_detected += 1
            alert = self._create_alert(flow, result)

            if self._producer and self.config.publish_alerts:
                self._producer.send(
                    self.config.topic_alerts,
                    key=alert["flow_id"],
                    value=alert,
                )

            if self.config.verbose:
                logger.warning(
                    f"🚨 ANOMALIA DETECTADA: {alert['flow_id']} "
                    f"(ξ={result.eccentricity:.4f}, severity={alert['severity']})"
                )

        # Publica todos os resultados (se configurado)
        elif self._producer and self.config.publish_all_results:
            alert = self._create_alert(flow, result)
            self._producer.send(
                self.config.topic_alerts,
                key=alert["flow_id"],
                value=alert,
            )

        return result

    def run(self, max_flows: Optional[int] = None) -> Dict[str, Any]:
        """
        Executa o detector em loop.

        Args:
            max_flows: Limite de flows (None = infinito)

        Returns:
            Estatisticas de execucao
        """
        if not self._consumer:
            self.connect()

        self._running = True
        self.start_time = time.time()

        logger.info("=" * 60)
        logger.info("Streaming Detector iniciado")
        logger.info(f"  Topico entrada: {self.config.topic_flows}")
        logger.info(f"  Topico alertas: {self.config.topic_alerts}")
        logger.info(f"  TEDA m={self.config.teda_m}, min_samples={self.config.teda_min_samples}")
        logger.info(f"  Features: {len(self._feature_names)}")
        logger.info("=" * 60)

        try:
            while self._running:
                # Poll com timeout de 1s
                records = self._consumer.poll(timeout_ms=1000)

                for topic_partition, messages in records.items():
                    for message in messages:
                        self._process_flow(message.value)

                        # Verifica limite
                        if max_flows and self.flows_processed >= max_flows:
                            logger.info(f"Limite de {max_flows} flows atingido")
                            self._running = False
                            break

                    if not self._running:
                        break

                # Log de progresso periodico
                if (self.flows_processed > 0 and
                    self.flows_processed % self.config.log_interval == 0):
                    elapsed = time.time() - self.start_time
                    rate = self.flows_processed / elapsed if elapsed > 0 else 0
                    anomaly_rate = (self.anomalies_detected / self.flows_processed * 100
                                   if self.flows_processed > 0 else 0)
                    logger.info(
                        f"Progresso: {self.flows_processed} flows, "
                        f"{self.anomalies_detected} anomalias ({anomaly_rate:.1f}%), "
                        f"{rate:.1f} flows/s"
                    )

        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuario (Ctrl+C)")

        finally:
            self.close()

        # Estatisticas finais
        elapsed = time.time() - self.start_time if self.start_time else 0
        stats = {
            "flows_processed": self.flows_processed,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": (self.anomalies_detected / self.flows_processed * 100
                           if self.flows_processed > 0 else 0),
            "elapsed_seconds": elapsed,
            "flows_per_second": self.flows_processed / elapsed if elapsed > 0 else 0,
            "teda_stats": self._teda.get_statistics(),
        }

        return stats


# ============================================================
# SIGNAL HANDLERS
# ============================================================

_detector_instance: Optional[StreamingDetector] = None


def _signal_handler(signum, frame):
    """Handler para SIGINT e SIGTERM."""
    logger.info(f"Recebido sinal {signum}, iniciando shutdown...")
    if _detector_instance:
        _detector_instance._running = False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    """
    Execucao direta do StreamingDetector.

    Uso:
        cd streaming
        python -m src.detector.streaming_detector

    Ou com argumentos:
        python -m src.detector.streaming_detector --max-flows 100 --verbose
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming Detector - TEDA para deteccao de anomalias em tempo real"
    )
    parser.add_argument(
        "--max-flows",
        type=int,
        default=None,
        help="Numero maximo de flows a processar (default: infinito)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra cada flow)"
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="teda-detector",
        help="ID do consumer group (default: teda-detector)"
    )
    parser.add_argument(
        "--m",
        type=float,
        default=3.0,
        help="Parametro m do TEDA (desvios padrao, default: 3.0)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimo de amostras antes de detectar (default: 10)"
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Nao publica alertas no Kafka"
    )

    args = parser.parse_args()

    # Configuracao
    config = StreamingDetectorConfig(
        group_id=args.group_id,
        teda_m=args.m,
        teda_min_samples=args.min_samples,
        publish_alerts=not args.no_publish,
        verbose=args.verbose,
    )

    # Signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Executa
    detector = StreamingDetector(config)
    _detector_instance = detector

    logger.info("=" * 60)
    logger.info("Streaming Detector v0.1 - TEDA")
    logger.info("=" * 60)

    try:
        stats = detector.run(max_flows=args.max_flows)

        logger.info("=" * 60)
        logger.info("Estatisticas finais:")
        logger.info(f"  Flows processados: {stats['flows_processed']}")
        logger.info(f"  Anomalias detectadas: {stats['anomalies_detected']}")
        logger.info(f"  Taxa de anomalias: {stats['anomaly_rate']:.2f}%")
        logger.info(f"  Tempo total: {stats['elapsed_seconds']:.2f}s")
        logger.info(f"  Taxa: {stats['flows_per_second']:.1f} flows/s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise
