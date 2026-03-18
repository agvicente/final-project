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
from detector.streaming_detector import (
    StreamingDetector, StreamingDetectorConfig, DetectorAlgorithm,
    DetectionGranularity, FEATURE_SETS,
)
from metrics.prequential_metrics import PrequentialMetrics

# Kafka imports para pre-flight check
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from kafka_utils import purge_kafka_topics, wait_for_flow_consumer


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
        logger.error("  cd experiments/streaming/docker")
        logger.error("  docker compose up -d")
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


def load_attack_ips(attack_ips_path: Optional[str] = None) -> Optional[set]:
    """
    Carrega conjunto de IPs de atacantes do arquivo attack_ips.json.

    Busca em ordem:
      1. Caminho explícito (--attack-ips-file)
      2. data/attack_ips.json (relativo ao repo root)

    Args:
        attack_ips_path: Caminho explícito (opcional)

    Returns:
        Set de IPs ou None se arquivo não encontrado
    """
    search_paths = []

    if attack_ips_path:
        search_paths.append(Path(attack_ips_path))

    # Padrão: data/attack_ips_campaign02.json, depois attack_ips.json
    repo_root = Path(__file__).parent.parent.parent.parent
    search_paths.append(repo_root / "data" / "attack_ips_campaign02.json")
    search_paths.append(repo_root / "data" / "attack_ips.json")

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            attack_ips = set(data.get("attack_ips", []))
            logger.info(f"Carregados {len(attack_ips)} IPs de atacantes de: {path}")
            return attack_ips

    return None


def run_detector(
    experiment_id: str,
    bootstrap_servers: str = "localhost:9092",
    algorithm: str = "micro_teda",
    r0: float = 0.1,
    min_samples: int = 10,
    window_size: int = 1000,
    alpha: float = 0.01,
    max_flows: Optional[int] = None,
    verbose: bool = False,
    benign_packets_sent: int = 0,
    attack_packets_sent: int = 0,
    ground_truth_mode: str = "phase",
    attack_ips: Optional[set] = None,
    feature_names: Optional[List[str]] = None,
    granularity: str = "flow",
    window_seconds: float = 10.0,
    min_flows_per_window: int = 5,
    window_feature_version: str = "v1",
) -> Dict[str, Any]:
    """
    Executa StreamingDetector e avalia resultados.

    O detector é cego a labels (puramente não-supervisionado).
    A avaliação é feita externamente aqui, usando um de dois modos:

    Modo "phase" (legado):
        Divisão por posição ordinal baseada na proporção de pacotes injetados.
        Limitações documentadas em methodology.md seção 4.4 (L1-L4).

    Modo "ip" (recomendado):
        Rotulação por IP de origem — usa mapeamento de IPs de atacantes
        extraído dos PCAPs (data/attack_ips.json).
        Elimina limitações L1-L4 do modo phase.

    Args:
        experiment_id: ID único do experimento
        bootstrap_servers: Endereço do Kafka
        algorithm: "teda" ou "micro_teda"
        r0: Variância mínima (MicroTEDAclus)
        min_samples: Amostras mínimas antes de detectar
        window_size: Janela deslizante (prequential)
        alpha: Fading factor (prequential)
        max_flows: Limite de flows (None = todos disponíveis)
        verbose: Modo verboso
        benign_packets_sent: Pacotes injetados da fase benign
        attack_packets_sent: Pacotes injetados da fase ataque (0 = benign-only)
        ground_truth_mode: "phase" (legado) ou "ip" (por IP de atacante)
        attack_ips: Set de IPs de atacantes (obrigatório se ground_truth_mode="ip")

    Returns:
        Estatísticas completas (detector + métricas prequential)
    """
    logger.info("=" * 60)
    logger.info("ETAPA 3: Executando detecção")
    logger.info("=" * 60)

    # Configuração do detector
    algo_enum = (DetectorAlgorithm.TEDA if algorithm == "teda"
                 else DetectorAlgorithm.MICRO_TEDA)

    gran_enum = (DetectionGranularity.WINDOW if granularity == "window"
                 else DetectionGranularity.FLOW)

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
        feature_names=feature_names,
        granularity=gran_enum,
        window_size_seconds=window_seconds,
        min_flows_per_window=min_flows_per_window,
        window_feature_version=window_feature_version,
    )

    # Detector puramente não-supervisionado — sem ground truth
    detector = StreamingDetector(config)

    try:
        stats = detector.run(max_flows=max_flows)

    finally:
        detector.close()

    # -----------------------------------------------------------------------
    # Avaliação — fora do detector (ground truth aplicado externamente)
    # -----------------------------------------------------------------------
    detection_results = stats.pop("detection_results", [])
    metrics = PrequentialMetrics(window_size=window_size, alpha=alpha)

    if ground_truth_mode == "ip" and attack_ips:
        # ── Modo IP: rotulação por IP de origem/destino ──
        # Cada flow é rotulado individualmente com base nos IPs de atacantes
        # conhecidos do CICIoT2023 (extraídos por extract_attack_ips.py).
        logger.info(f"Ground truth: modo IP ({len(attack_ips)} IPs de atacantes)")

        attack_flow_count = 0
        for result in detection_results:
            is_anomaly = result["is_anomaly"]
            timestamp = result["first_packet_time"]
            src_ip = result.get("src_ip")
            dst_ip = result.get("dst_ip")
            y_true = (src_ip in attack_ips or dst_ip in attack_ips)
            if y_true:
                attack_flow_count += 1
            metrics.update(is_anomaly, y_true, timestamp)

        logger.info(
            f"Divisão por IP: {len(detection_results) - attack_flow_count} flows benign / "
            f"{attack_flow_count} flows ataque"
        )
        stats["ground_truth_mode"] = "ip"
        stats["attack_flows_by_ip"] = attack_flow_count
        stats["benign_flows_by_ip"] = len(detection_results) - attack_flow_count

    else:
        # ── Modo Phase (legado): rotulação por posição ordinal ──
        # Limitações L1-L4 documentadas em methodology.md seção 4.4.
        if ground_truth_mode == "ip":
            logger.warning("⚠️ Modo IP solicitado mas attack_ips não disponível — "
                          "usando fallback para modo phase")

        total_packets = benign_packets_sent + attack_packets_sent
        if total_packets > 0 and attack_packets_sent > 0:
            benign_ratio = benign_packets_sent / total_packets
            benign_flow_count = int(len(detection_results) * benign_ratio)
            logger.info(
                f"Ground truth: modo phase (legado)"
            )
            logger.info(
                f"Divisão de fases: {benign_flow_count} flows benign / "
                f"{len(detection_results) - benign_flow_count} flows ataque"
                f" (proporção: {benign_ratio:.2%} benign)"
            )
        else:
            # Experimento benign-only: todos os flows têm y_true=False
            benign_flow_count = len(detection_results)

        for i, result in enumerate(detection_results):
            is_anomaly = result["is_anomaly"]
            timestamp = result["first_packet_time"]
            y_true = i >= benign_flow_count  # False=benign, True=ataque
            metrics.update(is_anomaly, y_true, timestamp)

        stats["ground_truth_mode"] = "phase"

    # Adiciona métricas ao stats
    global_metrics = metrics.get_global_metrics()
    stats["prequential_metrics"] = global_metrics

    # Log de métricas finais
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
            "features": config_used.get("features", "v1"),
            "granularity": config_used.get("granularity", "flow"),
            "window_seconds": config_used.get("window_seconds", None),
            "min_flows_per_window": config_used.get("min_flows_per_window", None),
            "window_features": config_used.get("window_features", "v1"),
            "ground_truth": config_used.get("ground_truth", "ip"),
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
        "--features",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Conjunto de features: v1 (17), v2 (25), v3 (32)"
    )
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

    # Granularidade de detecção
    parser.add_argument(
        "--granularity",
        choices=["flow", "window"],
        default="flow",
        help="Granularidade: 'flow' (per-flow) ou 'window' (agregado por IP/janela temporal)"
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Tamanho da janela temporal em segundos (modo window)"
    )
    parser.add_argument(
        "--min-flows-per-window",
        type=int,
        default=5,
        help="Mínimo de flows por janela para emitir vetor (modo window)"
    )

    # Window features version
    parser.add_argument(
        "--window-features",
        choices=["v1", "v2"],
        default="v1",
        help="Window feature set: 'v1' (12 basic) or 'v2' (19 = basic + behavioral)"
    )

    # Ground truth
    parser.add_argument(
        "--ground-truth",
        choices=["phase", "ip"],
        default="ip",
        help="Modo de ground truth: 'phase' (legado, por posição ordinal) "
             "ou 'ip' (por IP de atacante, recomendado). Default: ip. "
             "Se 'ip' mas attack_ips.json não existir, faz fallback para 'phase'."
    )
    parser.add_argument(
        "--attack-ips-file",
        default=None,
        help="Caminho para attack_ips.json (default: data/attack_ips.json)"
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

    # Carregar IPs de atacantes se modo IP
    attack_ips = None
    if args.ground_truth == "ip":
        attack_ips = load_attack_ips(args.attack_ips_file)
        if attack_ips is None:
            logger.warning("⚠️ attack_ips.json não encontrado — fallback para modo 'phase'")
            logger.warning("   Execute extract_attack_ips.py primeiro para habilitar modo IP")

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
        logger.info(f"Ground truth: {args.ground_truth}"
                    f"{' (com ' + str(len(attack_ips)) + ' IPs)' if attack_ips else ''}")
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

        # ETAPA 2.5: Sincronização — aguardar FlowConsumer processar todos os pacotes
        benign_packets = pcap_stats_benign.get("packets_sent", 0)
        attack_packets = pcap_stats_list[1].get("packets_sent", 0) if len(pcap_stats_list) > 1 else 0

        logger.info("=" * 60)
        logger.info("SINCRONIZAÇÃO: Aguardando FlowConsumer")
        logger.info("=" * 60)
        logger.info(f"Pacotes injetados: {benign_packets + attack_packets}")

        try:
            # Fase 1: Esperar FlowConsumer consumir todos os pacotes
            # (flows topic para de crescer quando não há mais timeouts de evento)
            wait_for_flow_consumer(
                bootstrap_servers=args.bootstrap_servers,
                stable_seconds=5.0,
                timeout_seconds=600,
            )

            # Fase 2: Terminar FlowConsumer → flush dos flows ativos restantes
            logger.info("Encerrando FlowConsumer (flush de flows ativos)...")
            flow_consumer_process.terminate()
            try:
                flow_consumer_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                flow_consumer_process.kill()
                flow_consumer_process.wait(timeout=5)
            flow_consumer_process = None  # Evita duplo terminate no finally

            # Fase 3: Esperar flows do flush aparecerem no Kafka
            time.sleep(2)
            total_flows = wait_for_flow_consumer(
                bootstrap_servers=args.bootstrap_servers,
                stable_seconds=3.0,
                timeout_seconds=30,
            )
            logger.info(f"✅ FlowConsumer finalizado: {total_flows} flows produzidos")

        except TimeoutError as e:
            logger.warning(f"⚠️ {e} — prosseguindo mesmo assim")

        # ETAPA 3: Executar detector + avaliação por ground truth
        selected_features = FEATURE_SETS.get(args.features)
        results = run_detector(
            experiment_id=experiment_id,
            bootstrap_servers=args.bootstrap_servers,
            algorithm=args.algorithm,
            r0=args.r0,
            min_samples=args.min_samples,
            window_size=args.window_size,
            alpha=args.alpha,
            max_flows=args.max_flows,
            verbose=args.verbose,
            benign_packets_sent=benign_packets,
            attack_packets_sent=attack_packets,
            ground_truth_mode=args.ground_truth,
            attack_ips=attack_ips,
            feature_names=selected_features,
            granularity=args.granularity,
            window_seconds=args.window_seconds,
            min_flows_per_window=args.min_flows_per_window,
            window_feature_version=args.window_features,
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
