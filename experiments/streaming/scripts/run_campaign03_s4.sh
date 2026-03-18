#!/usr/bin/env bash
# ============================================================
# Campaign-03 Step S4: Behavioral Window Features (v2)
#
# Tests 7 new behavioral features (entropy, ratios, rates)
# on top of the 12 base window features.
#
# Frozen from C02: algorithm=micro_teda, GT=ip, features=v1 (per-flow)
# Variable: window_features={v1,v2}, window_seconds={10,30}, r0={0.05,0.10,0.15}
#
# Runs: 48 total
#   - Control v1:  6 scenarios × 2 windows × r0=0.10        = 12
#   - Features v2: 6 scenarios × 2 windows × r0=0.10        = 12
#   - r0 sweep v2: 6 scenarios × 2 windows × r0={0.05,0.15} = 24
# ============================================================
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$STREAMING_DIR/../.." && pwd)"
PCAP_DIR="$REPO_ROOT/data/pcaps"
OUTPUT_BASE="$REPO_ROOT/experiments/results/campaign-03"
BENIGN_PCAP="$PCAP_DIR/Benign_Final/BenignTraffic.pcap"
RUN_SCRIPT="$SCRIPT_DIR/run_experiment.py"

# ── Counters ───────────────────────────────────────────────
TOTAL=0
SUCCESS=0
FAIL=0
SKIPPED=0
FAILED_LIST=()

# ── Common params (frozen from C02) ───────────────────────
COMMON_ARGS="--algorithm micro_teda --ground-truth ip --min-samples 10 --window-size 1000 --alpha 0.01 --max-packets 50000 --max-packets-attack 50000 --max-flows 10000 --granularity window"

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

    if python "$RUN_SCRIPT" $COMMON_ARGS --output "$output_dir" "$@" 2>&1; then
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
echo "Campaign-03 S4 — Pre-flight checks"
echo "===================================="

if [[ ! -f "$BENIGN_PCAP" ]]; then
    echo "ERROR: Benign PCAP not found: $BENIGN_PCAP"
    exit 1
fi

if [[ ! -f "$REPO_ROOT/data/attack_ips.json" && ! -f "$REPO_ROOT/data/attack_ips_campaign02.json" ]]; then
    echo "ERROR: attack_ips.json not found. Run extract_attack_ips.py first."
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "PCAP_DIR:    $PCAP_DIR"
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "BENIGN_PCAP: $BENIGN_PCAP"
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

CAMPAIGN_START=$(date +%s)
echo ""
echo "============================================================"
echo "Campaign-03 S4 starting at $(date)"
echo "============================================================"

# ── Attack scenarios ───────────────────────────────────────
# Declare associative arrays for attack names and PCAPs
declare -A ATK_PCAPS
ATK_PCAPS[ddos]="$PCAP_DDOS"
ATK_PCAPS[syn]="$PCAP_SYN"
ATK_PCAPS[tcp]="$PCAP_TCP"
ATK_PCAPS[mirai]="$PCAP_MIRAI"
ATK_PCAPS[recon]="$PCAP_RECON"

ATK_ORDER="ddos syn tcp mirai recon"

# ============================================================
# BLOCK 1: Control v1 — 12 runs
# window_features=v1, window_seconds={10,30}, r0=0.10
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 1: Control (window features v1) — 12 runs       ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 10 30; do
    # A1-benign
    run_exp "S4-A1-benign-wfv1-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" \
        --window-features v1 --window-seconds "$wsec" --r0 0.10

    # A2 attacks
    for atk in $ATK_ORDER; do
        run_exp "S4-A2-${atk}-wfv1-w${wsec}s-r0_0.10" \
            --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
            --window-features v1 --window-seconds "$wsec" --r0 0.10
    done
done

# ============================================================
# BLOCK 2: Features v2 — 12 runs
# window_features=v2, window_seconds={10,30}, r0=0.10
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 2: Behavioral features v2 — 12 runs             ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 10 30; do
    # A1-benign
    run_exp "S4-A1-benign-wfv2-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" \
        --window-features v2 --window-seconds "$wsec" --r0 0.10

    # A2 attacks
    for atk in $ATK_ORDER; do
        run_exp "S4-A2-${atk}-wfv2-w${wsec}s-r0_0.10" \
            --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
            --window-features v2 --window-seconds "$wsec" --r0 0.10
    done
done

# ============================================================
# BLOCK 3: r0 sweep with v2 — 24 runs
# window_features=v2, window_seconds={10,30}, r0={0.05,0.15}
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Block 3: r0 sweep with v2 — 24 runs                   ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 10 30; do
    for r0 in 0.05 0.15; do
        # A1-benign
        run_exp "S4-A1-benign-wfv2-w${wsec}s-r0_${r0}" \
            --pcap "$BENIGN_PCAP" \
            --window-features v2 --window-seconds "$wsec" --r0 "$r0"

        # A2 attacks
        for atk in $ATK_ORDER; do
            run_exp "S4-A2-${atk}-wfv2-w${wsec}s-r0_${r0}" \
                --pcap "$BENIGN_PCAP" --attack-pcap "${ATK_PCAPS[$atk]}" \
                --window-features v2 --window-seconds "$wsec" --r0 "$r0"
        done
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
echo "  Campaign-03 S4 COMPLETE"
echo "============================================================"
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

echo "Next: python $REPO_ROOT/experiments/results/campaign-03/generate_plots_s4.py"
