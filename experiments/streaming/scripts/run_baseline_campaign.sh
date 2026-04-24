#!/bin/bash
set -euo pipefail

# Note: arithmetic ((var++)) returns 1 when var=0 in some shells,
# which triggers set -e. Use || true to prevent premature exit.

# ============================================================
# Experiment 3: Baseline IF/OC-SVM + MicroTEDAclus Variants
# Campaign-05: Comparison with baselines for SoftCom 2026
#
# Run from:
#   cd ~/mestrado/final-project/experiments/streaming
#   source venv/bin/activate
#   bash scripts/run_baseline_campaign.sh
#
# Pre-requisites:
#   - Kafka running: cd docker && docker-compose up -d
#   - river installed: pip install river
#   - teda_hd installed: cd experiments/teda-high-dim && pip install -e .
#   - PCAPs at ../../data/pcaps/ (Linux machine)
#   - attack_ips.json extracted (scripts/extract_attack_ips.py)
#
# Total runs: 5 algorithms x 6 configs (5 attacks + benign) x 1 seed = 30
# Estimated time: 1-2 hours
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../venv/bin/python"
RUN_SCRIPT="${SCRIPT_DIR}/run_experiment.py"
RESULTS_BASE="../../experiments/results/campaign-05"

# Single seed for initial run (fast). Re-run with more seeds later for CI.
SEEDS=(42)

# PCAP paths (verified from campaign-04 run_meta.json)
BENIGN_PCAP="../../data/pcaps/Benign_Final/BenignTraffic.pcap"

declare -A ATTACK_PCAPS
ATTACK_PCAPS[ddos]="../../data/pcaps/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap"
ATTACK_PCAPS[syn]="../../data/pcaps/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap"
ATTACK_PCAPS[tcp]="../../data/pcaps/DDoS-TCP_Flood/DDoS-TCP_Flood.pcap"
ATTACK_PCAPS[mirai]="../../data/pcaps/Mirai-greeth_flood/Mirai-greeth_flood.pcap"
ATTACK_PCAPS[recon]="../../data/pcaps/Recon-PortScan/Recon-PortScan.pcap"

# Algorithms to test:
#   - halfspace_trees: river HST (genuinamente incremental) — Tan et al. 2011
#   - lof: river LOF (genuinamente incremental) — Breunig et al. 2000
#   - micro_teda: MicroTEDAclus proprio (referencia)
#   - variant_V0_original: Implementacao original (todas flags OFF)
#   - variant_V4_selective_update: Apenas update seletivo
ALGOS=("halfspace_trees" "lof" "micro_teda" "variant_V0_original" "variant_V4_selective_update")

# Attack configs (5 attacks + benign-only)
ATTACKS=("benign" "ddos" "syn" "tcp" "mirai" "recon")

echo "============================================================"
echo "Experiment 3: Baseline Campaign (Campaign-05)"
echo "============================================================"
echo "Algorithms: ${ALGOS[*]}"
echo "Attacks:    ${ATTACKS[*]}"
echo "Seeds:      ${SEEDS[*]}"
echo ""

# Pre-flight checks
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Python venv not found at $VENV_PYTHON"
    echo "       Run: cd experiments/streaming && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "ERROR: run_experiment.py not found at $RUN_SCRIPT"
    exit 1
fi

if [ ! -f "$BENIGN_PCAP" ]; then
    echo "ERROR: Benign PCAP not found at $BENIGN_PCAP"
    echo "       This script must run on the Linux machine with PCAPs"
    exit 1
fi

# Count total runs
total=0
for seed in "${SEEDS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            ((total++)) || true
        done
    done
done

echo "Total runs: $total"
echo "============================================================"
echo ""

done_count=0
skipped=0
failed=0
start_time=$(date +%s)

for seed in "${SEEDS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            ((done_count++)) || true
            outdir="${RESULTS_BASE}/${algo}-${attack}-seed${seed}"

            # Skip if already completed
            if [ -f "${outdir}/detection_results.json" ]; then
                echo "  [SKIP] ${algo}/${attack}/seed${seed} (already exists)"
                ((skipped++)) || true
                continue
            fi

            echo "  [${done_count}/${total}] Running ${algo} on ${attack} (seed=${seed})..."

            # Build command
            cmd="${VENV_PYTHON} ${RUN_SCRIPT}"
            cmd+=" --pcap ${BENIGN_PCAP}"

            if [ "$attack" != "benign" ]; then
                if [ -z "${ATTACK_PCAPS[$attack]}" ]; then
                    echo "  [WARN] Unknown attack type: ${attack}"
                    ((failed++)) || true
                    continue
                fi
                cmd+=" --attack-pcap ${ATTACK_PCAPS[$attack]}"
            fi

            # Handle algorithm name
            # variant_* algorithms use --algorithm variant_micro_teda + --variant-name
            if [[ "$algo" == variant_* ]]; then
                variant_name="${algo#variant_}"
                cmd+=" --algorithm variant_micro_teda --variant-name ${variant_name}"
            else
                cmd+=" --algorithm ${algo}"
            fi

            cmd+=" --r0 0.10"
            cmd+=" --max-packets 50000 --max-flows 10000"
            cmd+=" --output ${outdir}"

            # Run experiment
            if eval "$cmd"; then
                echo "  [OK]   ${algo}/${attack}/seed${seed}"
            else
                echo "  [FAIL] ${algo}/${attack}/seed${seed}"
                ((failed++))
            fi
        done
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "============================================================"
echo "Campaign-05 Complete"
echo "============================================================"
echo "Total runs:   $total"
echo "Completed:    $((done_count - skipped - failed))"
echo "Skipped:      $skipped"
echo "Failed:       $failed"
echo "Elapsed time: ${hours}h ${minutes}m"
echo "Results in:   ${RESULTS_BASE}/"
echo "============================================================"
