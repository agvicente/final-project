#!/usr/bin/env python3
"""
Extrai IPs de atacantes dos PCAPs do CICIoT2023.

Estratégia:
  1. Lê IPs únicos do PCAP benigno (baseline de IPs legítimos)
  2. Lê IPs únicos de cada PCAP de ataque
  3. IPs que aparecem APENAS em PCAPs de ataque = atacantes
  4. Salva mapeamento em data/attack_ips.json

O arquivo gerado é carregado automaticamente por run_experiment.py
para rotulação precisa por IP (substituindo rotulação por fase).

Uso (executar na máquina Linux onde os PCAPs existem):
    cd experiments/streaming
    source venv/bin/activate
    python3 scripts/extract_attack_ips.py --pcap-dir ../../data/pcaps/

Saída:
    data/attack_ips.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, Any

# dpkt para parsing de PCAPs (mesmo usado pelo PCAPProducer)
import dpkt
import socket

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diretório padrão para salvar attack_ips.json (relativo ao repo root)
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent.parent / "data" / "attack_ips.json"


def ip_to_str(ip_bytes: bytes) -> str:
    """Converte bytes de IP para string (ex: b'\\xc0\\xa8\\x01\\x01' -> '192.168.1.1')."""
    try:
        return socket.inet_ntoa(ip_bytes)
    except (ValueError, OSError):
        return None


def extract_ips_from_pcap(pcap_path: str, max_packets: int = None) -> Set[str]:
    """
    Extrai todos os IPs únicos (src + dst) de um PCAP.

    Args:
        pcap_path: Caminho do arquivo PCAP
        max_packets: Limite de pacotes a ler (None = todos)

    Returns:
        Set de IPs encontrados
    """
    ips = set()
    count = 0

    with open(pcap_path, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            # Tenta formato pcapng
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for timestamp, buf in pcap:
            if max_packets and count >= max_packets:
                break
            count += 1

            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    src = ip_to_str(ip.src)
                    dst = ip_to_str(ip.dst)
                    if src:
                        ips.add(src)
                    if dst:
                        ips.add(dst)
            except Exception:
                continue

    logger.info(f"  {pcap_path}: {count} packets → {len(ips)} IPs únicos")
    return ips


def classify_pcap(name: str) -> str:
    """
    Classifica um PCAP pelo nome do arquivo ou diretório-pai.

    Usa heurísticas baseadas nos nomes reais do CICIoT2023.
    Aceita tanto nome de arquivo (BenignTraffic.pcap) quanto
    nome de diretório (Benign_Final, DDoS-ICMP_Flood).

    Args:
        name: Nome do arquivo ou diretório (case-insensitive)

    Returns:
        Categoria: "benign", "ddos", "dos", "mirai", "recon", "spoofing", "other"
    """
    name_lower = name.lower()

    if "benign" in name_lower:
        return "benign"
    elif "ddos" in name_lower:
        return "ddos"
    elif "dos" in name_lower and "ddos" not in name_lower:
        return "dos"
    elif "mirai" in name_lower:
        return "mirai"
    elif "recon" in name_lower or "scan" in name_lower or "portscan" in name_lower:
        return "recon"
    elif "spoof" in name_lower or "mitm" in name_lower:
        return "spoofing"
    else:
        return "other"


def find_pcaps(pcap_dir: Path) -> Dict[str, list]:
    """
    Encontra PCAPs e classifica por categoria.

    Aceita qualquer estrutura de diretórios:
        data/pcaps/Benign_Final/BenignTraffic.pcap   → benign
        data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap → ddos
        data/pcaps/benign/file.pcap                   → benign
        data/pcaps/flat_file_ddos.pcap                → ddos

    Classifica por: nome do diretório-pai primeiro, nome do arquivo como fallback.
    """
    categories = defaultdict(list)

    for pcap in sorted(pcap_dir.rglob("*.pcap")):
        # Tenta classificar pelo diretório-pai primeiro
        parent_name = pcap.parent.name
        category = classify_pcap(parent_name)

        # Se o diretório-pai não deu match útil, tenta pelo nome do arquivo
        if category == "other":
            category = classify_pcap(pcap.stem)

        categories[category].append(str(pcap))

    return dict(categories)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai IPs de atacantes dos PCAPs do CICIoT2023"
    )
    parser.add_argument(
        "--pcap-dir",
        required=True,
        help="Diretório raiz dos PCAPs (ex: ../../data/pcaps/)"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Arquivo de saída (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Limite de pacotes por PCAP (None = todos, use para teste rápido)"
    )
    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    if not pcap_dir.exists():
        logger.error(f"Diretório não encontrado: {pcap_dir}")
        sys.exit(1)

    # 1. Encontrar PCAPs por categoria
    logger.info("=" * 60)
    logger.info("PASSO 1: Descobrindo PCAPs")
    logger.info("=" * 60)

    categories = find_pcaps(pcap_dir)
    if "benign" not in categories:
        logger.error("Nenhum PCAP benigno encontrado! Necessário para baseline de IPs.")
        sys.exit(1)

    for cat, pcaps in categories.items():
        logger.info(f"  {cat}: {len(pcaps)} PCAPs")

    # 2. Extrair IPs do tráfego benigno (baseline)
    logger.info("=" * 60)
    logger.info("PASSO 2: Extraindo IPs do tráfego benigno")
    logger.info("=" * 60)

    benign_ips = set()
    for pcap in categories["benign"]:
        ips = extract_ips_from_pcap(pcap, max_packets=args.max_packets)
        benign_ips.update(ips)

    logger.info(f"Total IPs benignos: {len(benign_ips)}")

    # 3. Extrair IPs de cada categoria de ataque
    logger.info("=" * 60)
    logger.info("PASSO 3: Extraindo IPs dos PCAPs de ataque")
    logger.info("=" * 60)

    attack_categories = {k: v for k, v in categories.items() if k != "benign"}
    attack_ips_by_category = {}
    all_attack_ips = set()

    for category, pcaps in sorted(attack_categories.items()):
        logger.info(f"\nCategoria: {category}")
        category_ips = set()
        for pcap in pcaps:
            ips = extract_ips_from_pcap(pcap, max_packets=args.max_packets)
            category_ips.update(ips)

        # IPs exclusivos desta categoria (não presentes no benigno)
        exclusive = category_ips - benign_ips
        attack_ips_by_category[category] = sorted(exclusive)
        all_attack_ips.update(exclusive)

        logger.info(f"  IPs totais: {len(category_ips)}, "
                    f"exclusivos (não-benignos): {len(exclusive)}")

    # 4. IPs compartilhados entre benign e ataque (tráfego de fundo IoT)
    shared_ips = set()
    for category, pcaps in attack_categories.items():
        for pcap in pcaps:
            ips = extract_ips_from_pcap(pcap, max_packets=args.max_packets)
            shared_ips.update(ips & benign_ips)

    # 5. Montar resultado
    logger.info("=" * 60)
    logger.info("PASSO 4: Gerando mapeamento")
    logger.info("=" * 60)

    result = {
        "_metadata": {
            "description": "IPs de atacantes extraídos dos PCAPs do CICIoT2023. "
                           "Gerado por extract_attack_ips.py.",
            "method": "IPs presentes em PCAPs de ataque mas ausentes no PCAP benigno",
            "pcap_dir": str(pcap_dir.resolve()),
            "max_packets_per_pcap": args.max_packets,
            "benign_pcaps": categories["benign"],
            "total_benign_ips": len(benign_ips),
            "total_attack_only_ips": len(all_attack_ips),
            "total_shared_ips": len(shared_ips),
        },
        "attack_ips": sorted(all_attack_ips),
        "attack_ips_by_category": attack_ips_by_category,
        "benign_ips": sorted(benign_ips),
    }

    # 6. Salvar
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n✅ Salvo em: {output_path}")
    logger.info(f"   IPs de ataque (exclusivos): {len(all_attack_ips)}")
    logger.info(f"   IPs benignos: {len(benign_ips)}")
    logger.info(f"   IPs compartilhados (vítimas): {len(shared_ips)}")
    logger.info(f"\nPróximo passo: re-rodar experimentos com --ground-truth ip")


if __name__ == "__main__":
    main()
