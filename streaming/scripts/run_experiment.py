"""
Orquestrador de Experimentos - Semana 5 (Fase A)

Script leve que coordena componentes existentes para executar experimentos:
    1. Injeta PCAP no Kafka (PCAPProducer)
    2. Inicia FlowConsumer (background)
    3. Executa StreamingDetector com ground_truth + métricas
    4. Salva resultados estruturados

Arquitetura:
    ┌──────────┐     ┌─────────┐     ┌──────────┐     ┌──────────────┐
    │   PCAP   │────►│  Kafka  │────►│  Kafka   │────►│  Streaming   │
    │   File   │     │(packets)│     │ (flows)  │     │   Detector   │
    └──────────┘     └─────────┘     └──────────┘     └──────────────┘
                           │                                  │
                           ▼                                  │
                    ┌─────────────┐                          │
                    │    Flow     │                          │
                    │  Consumer   │                          │
                    │ (background)│                          │
                    └─────────────┘                          │
                                                              ▼
                                                       ┌─────────────┐
                                                       │ GroundTruth │
                                                       │  + Metrics  │
                                                       └─────────────┘
                                                              │
                                                              ▼
                                                       results.json

Uso:
    python run_experiment.py \\
        --pcap data/pcaps/benign/Benign_Final.pcap \\
        --output results/exp1.json

    # Validação rápida (poucos pacotes)
    python run_experiment.py \\
        --pcap data/pcaps/ddos/DDoS-ICMP.pcap \\
        --max-packets 1000 \\
        --verbose

Referência:
    docs/methodology/experiment-methodology.md - Seção 8.0.5 (Semana 5)
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
import json
import time
import logging
import argparse
import subprocess
import csv
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Adiciona src/ ao path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from producer.pcap_producer import PCAPProducer
from producer.config import KafkaConfig as ProducerKafkaConfig, ProducerConfig
from detector.streaming_detector import StreamingDetector, StreamingDetectorConfig, DetectorAlgorithm
from metrics.ground_truth import GroundTruthProvider
from metrics.prequential_metrics import PrequentialMetrics

# Kafka imports para pre-flight check
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from kafka_utils import purge_kafka_topics


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def check_kafka_connection(bootstrap_servers: str = "localhost:9092", timeout_ms: int = 5000) -> bool:
    """
    Verifica se o Kafka está acessível antes de iniciar o experimento.

    Args:
        bootstrap_servers: Endereço do Kafka
        timeout_ms: Timeout de conexão em milissegundos

    Returns:
        True se conectou com sucesso

    Raises:
        SystemExit: Se Kafka não está acessível
    """
    logger.info("Verificando conexão com Kafka...")

    try:
        # Tenta criar um consumidor temporário apenas para verificar conexão
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=timeout_ms,
            api_version_auto_timeout_ms=timeout_ms
        )
        consumer.close()
        logger.info("✅ Kafka acessível")
        return True

    except NoBrokersAvailable:
        logger.error("=" * 70)
        logger.error("❌ ERRO: Kafka não está rodando ou não está acessível")
        logger.error("=" * 70)
        logger.error("")
        logger.error("Para iniciar o Kafka, execute:")
        logger.error("  cd /Users/augusto/mestrado/final-project")
        logger.error("  docker-compose up -d kafka zookeeper")
        logger.error("")
        logger.error("Aguarde ~30 segundos para inicialização completa.")
        logger.error("")
        logger.error("Para verificar se está rodando:")
        logger.error("  docker ps | grep kafka")
        logger.error("")
        logger.error("=" * 70)
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Erro ao verificar Kafka: {e}")
        logger.error("Verifique se o endereço está correto: {bootstrap_servers}")
        sys.exit(1)


def inject_pcap(
    pcap_path: str,
    bootstrap_servers: str = "localhost:9092",
    max_packets: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Injeta PCAP no Kafka usando PCAPProducer.

    Args:
        pcap_path: Caminho do arquivo PCAP
        bootstrap_servers: Endereço do Kafka
        max_packets: Limite de pacotes (None = todos)
        verbose: Modo verboso

    Returns:
        Estatísticas da injeção
    """
    logger.info("=" * 60)
    logger.info("ETAPA 1: Injetando PCAP no Kafka")
    logger.info("=" * 60)

    kafka_config = ProducerKafkaConfig(
        bootstrap_servers=bootstrap_servers,
        topic_packets="packets"
    )

    producer_config = ProducerConfig(
        max_packets=max_packets,
        verbose=verbose
    )

    producer = PCAPProducer(kafka_config, producer_config)

    try:
        stats = producer.process_pcap(pcap_path, max_packets=max_packets)
        logger.info(f"✅ PCAP injetado: {stats['packets_sent']} pacotes em {stats['elapsed_seconds']:.2f}s")
        return stats

    finally:
        producer.close()


def start_flow_consumer(experiment_id: str, verbose: bool = False) -> subprocess.Popen:
    """
    Inicia FlowConsumer em processo separado.

    Args:
        experiment_id: ID único do experimento
        verbose: Modo verboso

    Returns:
        Processo do FlowConsumer
    """
    logger.info("=" * 60)
    logger.info("ETAPA 2: Iniciando FlowConsumer (background)")
    logger.info("=" * 60)

    cmd = [
        "python", "-m", "src.consumer.flow_consumer",
        "--group-id", f"flow-consumer-{experiment_id}",
        "--from-beginning",  # Consome desde o início
    ]

    if verbose:
        cmd.append("--verbose")

    process = subprocess.Popen(
        cmd,
        cwd=Path(__file__).parent.parent,  # streaming/
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    logger.info(f"✅ FlowConsumer iniciado (PID: {process.pid})")
    time.sleep(3)  # Aguarda inicialização

    return process


def run_detector(
    experiment_id: str,
    pcap_path: str,
    bootstrap_servers: str = "localhost:9092",
    algorithm: str = "micro_teda",
    r0: float = 0.1,
    min_samples: int = 10,
    window_size: int = 1000,
    alpha: float = 0.01,
    max_flows: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Executa StreamingDetector com ground truth e métricas.

    Args:
        experiment_id: ID único do experimento
        pcap_path: Caminho do PCAP (para ground truth)
        bootstrap_servers: Endereço do Kafka
        algorithm: "teda" ou "micro_teda"
        r0: Variância mínima (MicroTEDAclus)
        min_samples: Amostras mínimas antes de detectar
        window_size: Janela deslizante (prequential)
        alpha: Fading factor (prequential)
        max_flows: Limite de flows (None = todos disponíveis)
        verbose: Modo verboso

    Returns:
        Estatísticas completas (detector + métricas)
    """
    logger.info("=" * 60)
    logger.info("ETAPA 3: Executando detecção com validação")
    logger.info("=" * 60)

    # Ground truth provider
    ground_truth = GroundTruthProvider(pcap_path)
    logger.info(f"Ground truth: {ground_truth.get_attack_type().value}")

    # Métricas prequential
    metrics = PrequentialMetrics(window_size=window_size, alpha=alpha)

    # Configuração do detector
    algo_enum = (DetectorAlgorithm.TEDA if algorithm == "teda"
                 else DetectorAlgorithm.MICRO_TEDA)

    config = StreamingDetectorConfig(
        bootstrap_servers=bootstrap_servers,
        topic_flows="flows",
        topic_alerts="alerts",
        group_id=f"detector-{experiment_id}",
        auto_offset_reset="earliest",  # Começa do início
        algorithm=algo_enum,
        micro_teda_r0=r0,
        micro_teda_min_samples=min_samples,
        publish_alerts=False,  # Não publica alerts em experimentos
        verbose=verbose,
        log_interval=100,
    )

    # Cria detector COM ground_truth e metrics
    detector = StreamingDetector(config, ground_truth=ground_truth, metrics=metrics)

    try:
        # Executa detecção
        stats = detector.run(max_flows=max_flows)

        # Log de métricas finais
        if metrics:
            global_metrics = metrics.get_global_metrics()
            logger.info("=" * 60)
            logger.info("MÉTRICAS PREQUENTIAL:")
            logger.info(f"  Precision: {global_metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {global_metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score: {global_metrics.get('f1', 0):.4f}")

            mttd = metrics.get_mttd()
            if mttd is not None:
                logger.info(f"  MTTD: {mttd:.2f}s")
            else:
                logger.info(f"  MTTD: N/A (sem ataques ou não detectados)")

            logger.info("=" * 60)

        return stats

    finally:
        detector.close()


def save_structured_results(
    output_dir: Path,
    results: Dict[str, Any],
    pcap_paths: Dict[str, str],
    pcap_stats: List[Dict[str, Any]],
    config_used: Dict[str, Any],
    system_usage: List[Dict[str, Any]],
    cluster_snapshots: List[Dict[str, Any]]
) -> None:
    """
    Salva os 5 artefatos estruturados (Semana 5 Fase B).

    Artefatos:
    1. run_meta.json - Metadata do experimento
    2. detection_results.json - Resultados de detecção
    3. metrics_windowed.csv - Métricas por janela
    4. clusters_state.jsonl - Snapshots de clusters
    5. system_usage.csv - CPU/memória ao longo do tempo

    Args:
        output_dir: Diretório de saída
        results: Resultados do detector
        pcap_paths: Caminhos dos PCAPs usados
        pcap_stats: Estatísticas de injeção dos PCAPs
        config_used: Configuração do experimento
        system_usage: Lista de medições de sistema
        cluster_snapshots: Lista de snapshots de clusters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. run_meta.json
    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=output_dir.parent.parent
    ).stdout.strip() or "unknown"

    run_meta = {
        "git_commit": git_commit,
        "algorithm": config_used.get("algorithm", "micro_teda"),
        "params": {
            "r0": config_used.get("r0", 0.1),
            "min_samples": config_used.get("min_samples", 10),
            "window_size": config_used.get("window_size", 1000),
            "alpha": config_used.get("alpha", 0.01),
        },
        "pcaps": pcap_paths,
        "execution": {
            "start_time": datetime.now().isoformat(),
            "duration_seconds": results.get("elapsed_seconds", 0),
        },
        "volumes": {
            "total_flows": results.get("flows_processed", 0),
            "anomalies_detected": results.get("anomalies_detected", 0),
        }
    }

    with open(output_dir / "run_meta.json", 'w') as f:
        json.dump(run_meta, f, indent=2)

    # 2. detection_results.json
    with open(output_dir / "detection_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 3. metrics_windowed.csv (se houver métricas)
    if "prequential_metrics" in results:
        metrics = results["prequential_metrics"]
        with open(output_dir / "metrics_windowed.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'window', 'precision', 'recall', 'f1', 'fpr',
                'cumulative_error', 'window_error', 'fading_error'
            ])
            writer.writeheader()
            writer.writerow({
                'window': 'final',
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'fpr': metrics.get('fpr', 0),
                'cumulative_error': metrics.get('cumulative_error', 0),
                'window_error': metrics.get('window_error', 0),
                'fading_error': metrics.get('fading_error', 0),
            })

    # 4. clusters_state.jsonl
    with open(output_dir / "clusters_state.jsonl", 'w') as f:
        for snapshot in cluster_snapshots:
            f.write(json.dumps(snapshot) + '\n')

    # 5. system_usage.csv
    if system_usage:
        with open(output_dir / "system_usage.csv", 'w', newline='') as f:
            fieldnames = ['timestamp', 'elapsed_sec', 'flows_processed', 'rss_mb', 'cpu_percent']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(system_usage)

    logger.info(f"✅ 5 artefatos salvos em: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Interface CLI para executar experimentos."""
    parser = argparse.ArgumentParser(
        description="Orquestrador de Experimentos - Semana 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # PCAP
    parser.add_argument(
        "--pcap",
        required=True,
        help="Caminho para PCAP benign (warm-up)"
    )
    parser.add_argument(
        "--attack-pcap",
        default=None,
        help="Caminho para PCAP de ataque (opcional, processado após benign)"
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Limite de pacotes a injetar do PCAP benign (None = todos)"
    )
    parser.add_argument(
        "--max-packets-attack",
        type=int,
        default=None,
        help="Limite de pacotes do PCAP de ataque (None = todos)"
    )

    # Kafka
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Endereço do Kafka"
    )

    # Detector
    parser.add_argument(
        "--algorithm",
        choices=["teda", "micro_teda"],
        default="micro_teda",
        help="Algoritmo de detecção"
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=0.1,
        help="Variância mínima (MicroTEDAclus)"
    )
    # TODO: Para que serve este min_samples e onde ele é usado?
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Amostras mínimas antes de detectar"
    )

    # Métricas Prequential
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Tamanho da janela deslizante"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Fading factor (0 < α ≤ 1)"
    )

    # Execução
    parser.add_argument(
        "--max-flows",
        type=int,
        default=None,
        help="Limite de flows a processar (None = todos disponíveis)"
    )

    # Output
    parser.add_argument(
        "--output",
        default="results/experiment.json",
        help="Caminho para salvar resultados"
    )

    # Flags
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra cada flow)"
    )
    parser.add_argument(
        "--skip-purge",
        action="store_true",
        help="Pular purga de tópicos (DEBUG: permite reusar dados)"
    )

    args = parser.parse_args()

    # Sanitize attack-pcap: tratar "none" como None
    if args.attack_pcap and args.attack_pcap.lower() == "none":
        args.attack_pcap = None

    # Gera ID único para este experimento
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Salva configuração usada
    config_used = vars(args)

    # PRÉ-REQUISITO: Verificar se Kafka está rodando
    check_kafka_connection(bootstrap_servers=args.bootstrap_servers)

    # ISOLAMENTO: Limpar tópicos para garantir experimento independente
    if not args.skip_purge:
        logger.info("=" * 60)
        logger.info("ISOLAMENTO DE EXPERIMENTO")
        logger.info("=" * 60)
        purge_kafka_topics(bootstrap_servers=args.bootstrap_servers)
        logger.info("=" * 60)
    else:
        logger.warning("⚠️ Purga de tópicos DESABILITADA (--skip-purge)")
        logger.warning("⚠️ Experimento pode ter interferência de dados antigos")

    flow_consumer_process = None
    start_time = time.time()
    system_usage = []
    cluster_snapshots = []
    pcap_stats_list = []

    try:
        logger.info("=" * 60)
        logger.info("INICIANDO EXPERIMENTO")
        logger.info("=" * 60)
        logger.info(f"PCAP Benign: {args.pcap}")
        if args.attack_pcap:
            logger.info(f"PCAP Attack: {args.attack_pcap}")
        logger.info(f"Algoritmo: {args.algorithm}")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 60)

        # ETAPA 1: Injetar PCAP benign
        pcap_stats_benign = inject_pcap(
            pcap_path=args.pcap,
            bootstrap_servers=args.bootstrap_servers,
            max_packets=args.max_packets,
            verbose=args.verbose
        )
        pcap_stats_list.append(pcap_stats_benign)

        # ETAPA 1b: Injetar PCAP de ataque (se fornecido)
        if args.attack_pcap:
            time.sleep(2)  # Pausa entre PCAPs
            pcap_stats_attack = inject_pcap(
                pcap_path=args.attack_pcap,
                bootstrap_servers=args.bootstrap_servers,
                max_packets=args.max_packets_attack,
                verbose=args.verbose
            )
            pcap_stats_list.append(pcap_stats_attack)

        # ETAPA 2: Iniciar FlowConsumer
        flow_consumer_process = start_flow_consumer(experiment_id=experiment_id, verbose=args.verbose)

        # ETAPA 3: Executar detector com validação
        # Usar primeiro PCAP para ground truth (ou attack se disponível)
        gt_pcap = args.attack_pcap if args.attack_pcap else args.pcap

        results = run_detector(
            experiment_id=experiment_id,
            pcap_path=gt_pcap,
            bootstrap_servers=args.bootstrap_servers,
            algorithm=args.algorithm,
            r0=args.r0,
            min_samples=args.min_samples,
            window_size=args.window_size,
            alpha=args.alpha,
            max_flows=args.max_flows,
            verbose=args.verbose
        )

        # ETAPA 4: Coletar snapshots de clusters (simplificado)
        if "detector_stats" in results:
            cluster_snapshots.append({
                "timestamp": time.time(),
                "flows_processed": results.get("flows_processed", 0),
                "num_clusters": results["detector_stats"].get("num_clusters", 0),
                "clusters": results["detector_stats"].get("clusters", [])
            })

        # ETAPA 5: System usage (simplificado - apenas final)
        process = psutil.Process()
        system_usage.append({
            "timestamp": time.time(),
            "elapsed_sec": time.time() - start_time,
            "flows_processed": results.get("flows_processed", 0),
            "rss_mb": process.memory_info().rss / 1e6,
            "cpu_percent": process.cpu_percent()
        })

        # Salvar 5 artefatos estruturados
        output_dir = Path(args.output)
        pcap_paths = {
            "benign": args.pcap,
            "attack": args.attack_pcap if args.attack_pcap else None
        }

        save_structured_results(
            output_dir=output_dir,
            results=results,
            pcap_paths=pcap_paths,
            pcap_stats=pcap_stats_list,
            config_used=config_used,
            system_usage=system_usage,
            cluster_snapshots=cluster_snapshots
        )

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("✅ EXPERIMENTO CONCLUÍDO")
        logger.info(f"Tempo total: {elapsed:.2f}s")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Erro fatal no experimento: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Encerra FlowConsumer
        if flow_consumer_process:
            logger.info("Encerrando FlowConsumer...")
            flow_consumer_process.terminate()
            try:
                flow_consumer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                flow_consumer_process.kill()


if __name__ == "__main__":
    main()
