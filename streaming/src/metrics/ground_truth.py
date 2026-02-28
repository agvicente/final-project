"""
Ground truth heurístico para validação de experimentos.

Semana 5: Inferir label do nome do arquivo PCAP.
Semana 6+: Integrar com CSVs do CICIoT2023 para labels exatos.

Baseado em: docs/methodology/experiment-methodology.md seção 8.0.5.4
"""

from pathlib import Path
from typing import Dict
from enum import Enum


class AttackType(Enum):
    """Tipos de ataque do CICIoT2023."""
    BENIGN = "benign"
    DDOS = "ddos"
    DOS = "dos"
    MIRAI = "mirai"
    RECON = "reconnaissance"
    SPOOFING = "spoofing"
    WEB = "web_based"
    BRUTE_FORCE = "brute_force"


class GroundTruthProvider:
    """
    Fornece ground truth para flows baseado em heurísticas.

    Semana 5: Inferência por nome de arquivo PCAP (simplificado).
    Limitação: Todos os flows do mesmo PCAP recebem o mesmo label.

    Exemplo:
        >>> gt = GroundTruthProvider("data/pcaps/ddos/DDoS-ICMP_Flood.pcap")
        >>> gt.get_attack_type()
        <AttackType.DDOS: 'ddos'>
        >>> gt.get_flow_label({})
        True
    """

    def __init__(self, pcap_path: str):
        """
        Inicializa provider com path do PCAP.

        Args:
            pcap_path: Path do arquivo PCAP sendo processado
        """
        self.pcap_path = Path(pcap_path)
        self.attack_type = self._infer_attack_type()
        self.is_attack = self.attack_type != AttackType.BENIGN

    def _infer_attack_type(self) -> AttackType:
        """
        Infere tipo de ataque do nome do arquivo.

        Heurística: Procura keywords no filename (case-insensitive).
        Baseado nos nomes reais do CICIoT2023.

        Returns:
            AttackType correspondente
        """
        filename_lower = self.pcap_path.name.lower()

        # Benign (verificar primeiro)
        if 'benign' in filename_lower or 'normal' in filename_lower:
            return AttackType.BENIGN

        # DDoS (16 tipos no CICIoT2023)
        elif 'ddos' in filename_lower:
            return AttackType.DDOS

        # DoS (4 tipos no CICIoT2023)
        elif 'dos-' in filename_lower:  # Hífen para não pegar DDoS
            return AttackType.DOS

        # Mirai (3 tipos no CICIoT2023)
        elif 'mirai' in filename_lower:
            return AttackType.MIRAI

        # Reconnaissance (5 tipos: Recon-* + VulnerabilityScan)
        elif 'recon' in filename_lower or 'scan' in filename_lower or 'discovery' in filename_lower:
            return AttackType.RECON

        # Spoofing (2 tipos: MITM-ArpSpoofing, DNS_Spoofing)
        elif 'spoof' in filename_lower or 'mitm' in filename_lower:
            return AttackType.SPOOFING

        # Web-based (6 tipos)
        elif any(keyword in filename_lower for keyword in [
            'sql', 'xss', 'command', 'upload', 'backdoor', 'browser', 'injection'
        ]):
            return AttackType.WEB

        # Brute Force (1 tipo)
        elif 'brute' in filename_lower or 'dictionary' in filename_lower:
            return AttackType.BRUTE_FORCE

        else:
            # Default: se não reconhecido, assume ataque genérico (DDOS)
            # Conservador para IDS: preferir falso positivo a falso negativo
            return AttackType.DDOS

    def get_flow_label(self, flow: Dict) -> bool:
        """
        Retorna label para um flow específico.

        Semana 5: Retorna mesmo label para todos os flows do PCAP.
        Limitação: Não distingue flows individuais.

        Args:
            flow: Dicionário com dados do flow (não usado na v5)

        Returns:
            True se ataque, False se benign
        """
        return self.is_attack

    def get_attack_type(self) -> AttackType:
        """Retorna tipo de ataque do PCAP."""
        return self.attack_type

    def get_metadata(self) -> Dict:
        """
        Retorna metadata do ground truth.

        Útil para logging e debugging.

        Returns:
            Dict com informações sobre o ground truth
        """
        return {
            'pcap_path': str(self.pcap_path),
            'pcap_filename': self.pcap_path.name,
            'attack_type': self.attack_type.value,
            'is_attack': self.is_attack,
            'method': 'filename_heuristic',
            'version': '5.0'
        }


# ============================================================
# TESTES (executar com: python -m streaming.src.metrics.ground_truth)
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TESTES - Ground Truth Provider (CICIoT2023)")
    print("=" * 60)

    # Casos de teste baseados nos nomes reais do CICIoT2023
    test_cases = [
        # Benign
        ('Benign_Final.pcap', AttackType.BENIGN, False),
        ('data/pcaps/benign/BenignTraffic.pcap', AttackType.BENIGN, False),

        # DDoS (16 tipos)
        ('DDoS-ICMP_Flood.pcap', AttackType.DDOS, True),
        ('DDoS-HTTP_Flood.pcap', AttackType.DDOS, True),
        ('DDoS-SYN_Flood.pcap', AttackType.DDOS, True),
        ('DDoS-SlowLoris.pcap', AttackType.DDOS, True),

        # DoS (4 tipos)
        ('DoS-HTTP_Flood.pcap', AttackType.DOS, True),
        ('DoS-TCP_Flood.pcap', AttackType.DOS, True),
        ('DoS-UDP_Flood.pcap', AttackType.DOS, True),

        # Mirai (3 tipos)
        ('Mirai-greeth_flood.pcap', AttackType.MIRAI, True),
        ('Mirai-greip_flood.pcap', AttackType.MIRAI, True),
        ('Mirai-udpplain.pcap', AttackType.MIRAI, True),

        # Recon (5 tipos)
        ('Recon-HostDiscovery.pcap', AttackType.RECON, True),
        ('Recon-PortScan.pcap', AttackType.RECON, True),
        ('VulnerabilityScan.pcap', AttackType.RECON, True),

        # Spoofing (2 tipos)
        ('MITM-ArpSpoofing.pcap', AttackType.SPOOFING, True),
        ('DNS_Spoofing.pcap', AttackType.SPOOFING, True),

        # Web-based (6 tipos)
        ('SqlInjection.pcap', AttackType.WEB, True),
        ('XSS.pcap', AttackType.WEB, True),
        ('CommandInjection.pcap', AttackType.WEB, True),
        ('Uploading_Attack.pcap', AttackType.WEB, True),
        ('Backdoor_Malware.pcap', AttackType.WEB, True),
        ('BrowserHijacking.pcap', AttackType.WEB, True),

        # Brute Force (1 tipo)
        ('DictionaryBruteForce.pcap', AttackType.BRUTE_FORCE, True),
    ]

    passed = 0
    failed = 0

    for pcap_path, expected_type, expected_is_attack in test_cases:
        gt = GroundTruthProvider(pcap_path)

        # Verificar tipo de ataque
        if gt.get_attack_type() == expected_type:
            status_type = "✅"
            passed += 1
        else:
            status_type = "❌"
            failed += 1
            print(f"\n❌ FALHOU: {pcap_path}")
            print(f"   Esperado: {expected_type.value}, Obtido: {gt.get_attack_type().value}")

        # Verificar se é ataque
        if gt.get_flow_label({}) == expected_is_attack:
            status_label = "✅"
            passed += 1
        else:
            status_label = "❌"
            failed += 1

        if status_type == "✅" and status_label == "✅":
            print(f"✅ {pcap_path:40s} → {gt.get_attack_type().value}")

    print("\n" + "=" * 60)
    print(f"RESULTADOS: {passed}/{len(test_cases)*2} testes passaram")
    print("=" * 60)

    if failed == 0:
        print("✅ Todos os testes passaram!")
        exit(0)
    else:
        print(f"❌ {failed} testes falharam!")
        exit(1)
