#!/usr/bin/env bash
# ============================================================
# Campaign-04: Original EvolvingClustering (Maia 2020)
#
# Compara a implementacao original do autor com a implementacao
# propria (micro_teda) usada em C01-C03.
#
# Frozen: GT=ip, features_per_flow=v1, min_samples=10
# Variable: algorithm=original_micro_teda
#
# Runs: 30 total
#   Block 1: Flow-level baseline (6 runs)
#     - A1-benign + 5 ataques, r0=0.10, flow-level
#   Block 2: Window v1, w={10,30}s, r0=0.10 (12 runs)
#     - A1-benign + 5 ataques × 2 windows
#   Block 3: Window v2, w={10,30}s, r0=0.10 (12 runs)
#     - A1-benign + 5 ataques × 2 windows
#
# Comparacao direta com:
#   - C02-S1 (flow-level, micro_teda)
#   - C03-S4 (window, micro_teda, wf v1/v2)
# ============================================================
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$STREAMING_DIR/../.." && pwd)"
PCAP_DIR="$REPO_ROOT/data/pcaps"
OUTPUT_BASE="$REPO_ROOT/experiments/results/campaign-04"
BENIGN_PCAP="$PCAP_DIR/Benign_Final/BenignTraffic.pcap"
RUN_SCRIPT="$SCRIPT_DIR/run_experiment.py"

# ── Counters ───────────────────────────────────────────────
TOTAL=0
SUCCESS=0
FAIL=0
SKIPPED=0
FAILED_LIST=()

# ── Common params (frozen) ─────────────────────────────────
ALGO="original_micro_teda"
COMMON="--algorithm $ALGO --ground-truth ip --min-samples 10 --window-size 1000 --alpha 0.01 --max-packets 50000 --max-packets-attack 50000 --max-flows 10000"

# ── Helper ─────────────────────────────────────────────────
run_exp() {
    local name="$1"
    shift
    local output_dir="$OUTPUT_BASE/$name"
    TOTAL=$((TOTAL + 1))

    # Skip if already completed
    if [[ -f "$output_dir/detection_results.json" && -f "$output_dir/run_meta.json" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP  #$TOTAL $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        SUCCESS=$((SUCCESS + 1))
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] START #$TOTAL  $name"
    echo "================================================================"

    local start_ts
    start_ts=$(date +%s)

    if python "$RUN_SCRIPT" $COMMON --output "$output_dir" "$@" 2>&1; then
        local end_ts
        end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[$(date '+%H:%M:%S')] DONE  #$TOTAL  $name  (${elapsed}s)"
        SUCCESS=$((SUCCESS + 1))
    else
        local end_ts
        end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[$(date '+%H:%M:%S')] FAIL  #$TOTAL  $name  (${elapsed}s)"
        FAIL=$((FAIL + 1))
        FAILED_LIST+=("$name")
    fi
}

# ── Pre-flight checks ─────────────────────────────────────
echo "Campaign-04 — Original EvolvingClustering"
echo "==========================================="

if [[ ! -f "$BENIGN_PCAP" ]]; then
    echo "ERROR: Benign PCAP not found: $BENIGN_PCAP"
    exit 1
fi

if [[ ! -f "$REPO_ROOT/data/attack_ips.json" && ! -f "$REPO_ROOT/data/attack_ips_campaign02.json" ]]; then
    echo "ERROR: attack_ips.json not found. Run extract_attack_ips.py first."
    exit 1
fi

# Verify original package is installed
python -c "from evolving.EvolvingClustering import EvolvingClustering" 2>/dev/null || {
    echo "ERROR: evolclustering package not installed. Run: pip install -e evolving_clustering/"
    exit 1
}

mkdir -p "$OUTPUT_BASE"

echo "PCAP_DIR:    $PCAP_DIR"
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "ALGORITHM:   $ALGO"
echo ""

# ── Attack PCAP map ───────────────────────────────────────
PCAP_DDOS="$PCAP_DIR/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap"
PCAP_SYN="$PCAP_DIR/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap"
PCAP_TCP="$PCAP_DIR/DDoS-TCP_Flood/DDoS-TCP_Flood.pcap"
PCAP_MIRAI="$PCAP_DIR/Mirai-greeth_flood/Mirai-greeth_flood.pcap"
PCAP_RECON="$PCAP_DIR/Recon-PortScan/Recon-PortScan.pcap"

for pcap_var in PCAP_DDOS PCAP_SYN PCAP_TCP PCAP_MIRAI PCAP_RECON; do
    if [[ ! -f "${!pcap_var}" ]]; then
        echo "WARNING: Missing PCAP: ${!pcap_var}"
    fi
done

declare -A ATK_PCAPS
ATK_PCAPS[ddos]="$PCAP_DDOS"
ATK_PCAPS[syn]="$PCAP_SYN"
ATK_PCAPS[tcp]="$PCAP_TCP"
ATK_PCAPS[mirai]="$PCAP_MIRAI"
ATK_PCAPS[recon]="$PCAP_RECON"

ATK_ORDER="ddos syn tcp mirai recon"

CAMPAIGN_START=$(date +%s)
echo ""
echo "============================================================"
echo "Campaign-04 starting at $(date)"
echo "============================================================"

# ============================================================
# BLOCK 1: Flow-level baseline — 6 runs
# granularity=flow, r0=0.10
# Direct comparison with C02-S1 (micro_teda flow-level)
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 1: Flow-level baseline — 6 runs                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

# A1-benign
run_exp "B1-A1-benign-flow-r0_0.10" \
    --pcap "$BENIGN_PCAP" \
    --granularity flow --r0 0.10

# A2 attacks
for atk in $ATK_ORDER; do
    run_exp "B1-A2-${atk}-flow-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
        --granularity flow --r0 0.10
done

# ============================================================
# BLOCK 2: Window features v1 — 12 runs
# granularity=window, window_features=v1, w={10,30}s, r0=0.10
# Direct comparison with C03-S4 Block 1 (micro_teda wf v1)
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 2: Window features v1 — 12 runs                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 10 30; do
    # A1-benign
    run_exp "B2-A1-benign-wfv1-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" \
        --granularity window --window-features v1 --window-seconds "$wsec" --r0 0.10

    # A2 attacks
    for atk in $ATK_ORDER; do
        run_exp "B2-A2-${atk}-wfv1-w${wsec}s-r0_0.10" \
            --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
            --granularity window --window-features v1 --window-seconds "$wsec" --r0 0.10
    done
done

# ============================================================
# BLOCK 3: Window features v2 — 12 runs
# granularity=window, window_features=v2, w={10,30}s, r0=0.10
# Direct comparison with C03-S4 Block 2 (micro_teda wf v2)
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 3: Window features v2 — 12 runs                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 10 30; do
    # A1-benign
    run_exp "B3-A1-benign-wfv2-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" \
        --granularity window --window-features v2 --window-seconds "$wsec" --r0 0.10

    # A2 attacks
    for atk in $ATK_ORDER; do
        run_exp "B3-A2-${atk}-wfv2-w${wsec}s-r0_0.10" \
            --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
            --granularity window --window-features v2 --window-seconds "$wsec" --r0 0.10
    done
done

# ============================================================
# SUMMARY
# ============================================================
CAMPAIGN_END=$(date +%s)
CAMPAIGN_ELAPSED=$((CAMPAIGN_END - CAMPAIGN_START))
CAMPAIGN_MIN=$((CAMPAIGN_ELAPSED / 60))

echo ""
echo "============================================================"
echo "  Campaign-04 COMPLETE"
echo "============================================================"
echo "  Algorithm:    $ALGO (Maia 2020 original)"
echo "  Total runs:   $TOTAL"
echo "  Success:      $SUCCESS (skipped: $SKIPPED)"
echo "  Failed:       $FAIL"
echo "  Elapsed:      ${CAMPAIGN_MIN}m $((CAMPAIGN_ELAPSED % 60))s"
echo "  Output:       $OUTPUT_BASE"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "Failed experiments:"
    for name in "${FAILED_LIST[@]}"; do
        echo "  - $name"
    done
    echo ""
fi

echo ""
echo "Next steps:"
echo "  1. Analyze results: compare with C02-S1 (flow) and C03-S4 (window)"
echo "  2. Key question: does original implementation produce different detection results?"
