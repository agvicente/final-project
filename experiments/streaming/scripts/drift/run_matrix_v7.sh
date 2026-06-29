#!/bin/bash
# Re-run da matriz do Experimento A com a LINHAGEM CORRETA do detector (corrected.py),
# substituindo o micro_teda.py com bug (variância atual vs hipotética) usado no run original.
#
# Derivado de /tmp/run_matrix.sh (orquestrador ad-hoc original, agora versionado).
# gen_comp() é BYTE-IDÊNTICO ao original → composições determinísticas idênticas →
# pareamento por `comp` legítimo na comparação Wilcoxon vs os 240 runs antigos.
#
# Matriz: 4 ataques × 30 composições × 2 regimes (r0) = 240 runs.
# Cenário realista benign->ataque, normalizado. Checkpoint por-run (pula feitos) → resiliente.
#
# Uso:
#   ./run_matrix_v7.sh <variant_name> [out_dir]
# Exemplos:
#   ./run_matrix_v7.sh V7_full_corrected /tmp/matrixA-V7corr
#   ./run_matrix_v7.sh V7_forgetting     /tmp/matrixA-V7forget
set -u

VARIANT="${1:?uso: run_matrix_v7.sh <variant_name> [out_dir]}"
OUT="${2:-/tmp/matrixA-${VARIANT}}"

cd ~/final-project/experiments/streaming

PCAP_BENIGN="../../data/pcaps/Benign_Final/BenignTraffic.pcap"
declare -A ATTACKS=(
  ["ddos_syn"]="../../data/pcaps/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap"
  ["mirai"]="../../data/pcaps/Mirai-greeth_flood/Mirai-greeth_flood.pcap"
  ["recon"]="../../data/pcaps/Recon-PortScan/Recon-PortScan.pcap"
  ["dos_syn"]="../../data/pcaps/DoS-SYN_Flood/DoS-SYN_Flood.pcap"
)
R0S=("0.10" "0.001")
mkdir -p "$OUT"
LOG="$OUT/matrix.log"
: > "$LOG"
echo "VARIANT=$VARIANT  OUT=$OUT  $(date)" >> "$LOG"

# 30 composicoes: grade deterministica de fatias (benign:attack em milhares de pacotes).
# IDENTICO ao /tmp/run_matrix.sh original — NÃO ALTERAR (garante pareamento por comp).
gen_comp() {  # arg: indice 0..29 -> ecoa "BENIGN_PKTS:ATTACK_PKTS"
  local i=$1
  local b=$(( 25000 + (i % 6) * 3000 ))   # 25k..40k
  local a=$(( 15000 + (i % 5) * 2500 ))   # 15k..25k
  echo "${b}:${a}"
}

total=0; done=0
for atk in "${!ATTACKS[@]}"; do
  for r0 in "${R0S[@]}"; do
    for i in $(seq 0 29); do
      total=$((total+1))
      comp=$(gen_comp $i); B=${comp%%:*}; A=${comp##*:}
      d="$OUT/${atk}_r${r0}_c${i}"
      # checkpoint: pula se ja tem serie por-fluxo
      if [ -f "$d/detection_results.json" ]; then
        echo "SKIP $d (ja existe)" >> "$LOG"; done=$((done+1)); continue
      fi
      echo "### RUN $atk r0=$r0 comp$i (b=$B a=$A) -> $d [$VARIANT]" >> "$LOG"
      venv/bin/python scripts/run_experiment.py \
        --phases "${PCAP_BENIGN}:benign:${B},${ATTACKS[$atk]}:${atk}:${A}" \
        --algorithm variant_micro_teda --variant-name "$VARIANT" \
        --normalize-features --r0 "$r0" \
        --output "$d/" >> "$LOG" 2>&1 \
        && { echo "OK $d" >> "$LOG"; done=$((done+1)); } \
        || echo "FAIL $d" >> "$LOG"
    done
  done
done
echo "MATRIX_DONE variant=$VARIANT total=$total done=$done $(date)" >> "$LOG"
