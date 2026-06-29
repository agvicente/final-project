#!/bin/bash
set -euo pipefail

# ============================================================
# Campaign-06: IoT Validation of Regime Transition Hypothesis
#
# Tests whether feature normalization (z-score) collapses V0 and V7
# into the same operational regime in real CICIoT2023 data, as
# predicted by the synthetic Exp 3 (regime-transition.md).
#
# Design (16 runs):
#   - Algorithms: V0 (variant_V0_original), V7 (micro_teda)
#   - Scenarios:  benign, ddos, mirai, recon
#   - Conditions: raw vs normalized
#   - Seed: 42 (single)
#   Total: 2 algos x 4 scenarios x 2 norm-modes = 16 runs
#
# Hypothesis predictions:
#   raw:        V0 fragments (~1000 clusters, FPR>40%) ; V7 collapses (low FPR)
#   normalized: V0 and V7 converge to similar regime, comparable FPR
#
# Run from:
#   cd ~/mestrado/final-project/experiments/streaming
#   source venv/bin/activate
#   bash scripts/run_campaign06_normalize.sh
#
# Pre-requisites:
#   - Kafka running: cd docker && docker-compose up -d
#   - PCAPs at ../../data/pcaps/  (Linux machine)
#   - attack_ips.json extracted (scripts/extract_attack_ips.py)
#
# Estimated time: 3-4 hours (V0 high-lambda saturation in raw mode is slow)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../venv/bin/python"
RUN_SCRIPT="${SCRIPT_DIR}/run_experiment.py"
RESULTS_BASE="../../experiments/results/campaign-06"

mkdir -p "${RESULTS_BASE}"

# Configuration
SEED=42
R0=0.10
MAX_PACKETS=50000
MAX_FLOWS=10000
WARMUP_SIZE=200  # ~2% of 10k flows; enough to estimate stats robustly

BENIGN_PCAP="../../data/pcaps/Benign_Final/BenignTraffic.pcap"

declare -A ATTACK_PCAPS
ATTACK_PCAPS[ddos]="../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap"
ATTACK_PCAPS[mirai]="../../data/pcaps/Mirai-greeth_flood/Mirai-greeth_flood.pcap"
ATTACK_PCAPS[recon]="../../data/pcaps/Recon-PortScan/Recon-PortScan.pcap"

ALGOS=(
    "variant_V0_original|--variant-name V0_original --algorithm variant_micro_teda"
    "micro_teda|--algorithm micro_teda"
)

NORM_MODES=("raw" "normalized")

run_count=0
total=$(( ${#ALGOS[@]} * (1 + ${#ATTACK_PCAPS[@]}) * ${#NORM_MODES[@]} ))

start_ts=$(date +%s)

for algo_entry in "${ALGOS[@]}"; do
    algo_name="${algo_entry%%|*}"
    algo_args="${algo_entry##*|}"

    for norm_mode in "${NORM_MODES[@]}"; do
        if [ "$norm_mode" = "normalized" ]; then
            norm_flag="--normalize-features --normalize-mode zscore --normalize-warmup-size ${WARMUP_SIZE}"
        else
            norm_flag=""
        fi

        # ── Benign baseline ────────────────────────────────────
        run_count=$((run_count + 1)) || true
        out_dir="${RESULTS_BASE}/${algo_name}-benign-${norm_mode}-seed${SEED}"
        echo ""
        echo "[${run_count}/${total}] ${algo_name} | benign | ${norm_mode} | seed=${SEED}"
        echo "  -> ${out_dir}"

        ${VENV_PYTHON} "${RUN_SCRIPT}" \
            --pcap "${BENIGN_PCAP}" \
            --attack-pcap none \
            ${algo_args} \
            --r0 "${R0}" \
            --max-packets "${MAX_PACKETS}" \
            --max-flows "${MAX_FLOWS}" \
            ${norm_flag} \
            --output "${out_dir}" \
            2>&1 | tail -20 || { echo "FAILED: ${out_dir}"; continue; }

        # ── Each attack scenario ──────────────────────────────
        for scenario in "${!ATTACK_PCAPS[@]}"; do
            attack_pcap="${ATTACK_PCAPS[$scenario]}"
            run_count=$((run_count + 1)) || true
            out_dir="${RESULTS_BASE}/${algo_name}-${scenario}-${norm_mode}-seed${SEED}"
            echo ""
            echo "[${run_count}/${total}] ${algo_name} | ${scenario} | ${norm_mode} | seed=${SEED}"
            echo "  -> ${out_dir}"

            ${VENV_PYTHON} "${RUN_SCRIPT}" \
                --pcap "${BENIGN_PCAP}" \
                --attack-pcap "${attack_pcap}" \
                ${algo_args} \
                --r0 "${R0}" \
                --max-packets "${MAX_PACKETS}" \
                --max-flows "${MAX_FLOWS}" \
                ${norm_flag} \
                --output "${out_dir}" \
                2>&1 | tail -20 || { echo "FAILED: ${out_dir}"; continue; }
        done
    done
done

end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

echo ""
echo "============================================================"
echo "Campaign-06 complete: ${run_count} runs in $((elapsed / 60))m $((elapsed % 60))s"
echo "Results: ${RESULTS_BASE}"
echo "Next: aggregate metrics, generate ANALYSIS.md, integrate into paper §V."
echo "============================================================"
