#!/usr/bin/env bash
# ============================================================
# Campaign-02: Incremental improvements over Campaign-01
#
# S1: Ground Truth por IP (16 runs)
# S2: Feature Expansion v2/v3 (32 runs)
# S3: Window Aggregation (24 runs)
# Total: 72 runs
# ============================================================
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$STREAMING_DIR/../.." && pwd)"
PCAP_DIR="$REPO_ROOT/data/pcaps"
OUTPUT_BASE="$REPO_ROOT/experiments/results/campaign-02"
BENIGN_PCAP="$PCAP_DIR/Benign_Final/BenignTraffic.pcap"
RUN_SCRIPT="$SCRIPT_DIR/run_experiment.py"

# ── Counters ───────────────────────────────────────────────
TOTAL=0
SUCCESS=0
FAIL=0
SKIPPED=0
FAILED_LIST=()

# ── Common params (same as campaign-01) ────────────────────
# Same params as campaign-01 (max-packets for Kafka stability, max-flows for consistency)
COMMON_ARGS="--algorithm micro_teda --min-samples 10 --window-size 1000 --alpha 0.01 --max-packets 50000 --max-packets-attack 50000 --max-flows 10000"

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
echo "Campaign-02 — Pre-flight checks"
echo "================================"

if [[ ! -f "$BENIGN_PCAP" ]]; then
    echo "ERROR: Benign PCAP not found: $BENIGN_PCAP"
    exit 1
fi

# Check attack_ips.json exists
if [[ ! -f "$REPO_ROOT/data/attack_ips.json" && ! -f "$REPO_ROOT/data/attack_ips_campaign02.json" ]]; then
    echo "ERROR: attack_ips.json not found. Run extract_attack_ips.py first."
    echo "  python3 scripts/extract_attack_ips.py --pcap-dir $PCAP_DIR"
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

# Verify attack PCAPs exist
for pcap_var in PCAP_DDOS PCAP_SYN PCAP_TCP PCAP_MIRAI PCAP_RECON; do
    if [[ ! -f "${!pcap_var}" ]]; then
        echo "WARNING: Missing PCAP: ${!pcap_var}"
    fi
done

CAMPAIGN_START=$(date +%s)
echo ""
echo "============================================================"
echo "Campaign-02 starting at $(date)"
echo "============================================================"

# ============================================================
# STEP 1 (S1): Ground Truth por IP — 16 runs
# Same experiments as campaign-01, now with IP-based ground truth
# (--ground-truth ip is the default)
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 1 (S1): Ground Truth por IP — 16 runs            ║"
echo "╚══════════════════════════════════════════════════════════╝"

# A1-benign: 4 r0 values
for r0 in 0.05 0.10 0.15 0.20; do
    run_exp "S1-A1-benign-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --r0 "$r0"
done

# A2-ddos: 4 r0 values
for r0 in 0.05 0.10 0.15 0.20; do
    run_exp "S1-A2-ddos-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_DDOS" --r0 "$r0"
done

# A2-syn: 2 r0 values
for r0 in 0.10 0.15; do
    run_exp "S1-A2-syn-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_SYN" --r0 "$r0"
done

# A2-tcp: 2 r0 values
for r0 in 0.10 0.15; do
    run_exp "S1-A2-tcp-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_TCP" --r0 "$r0"
done

# A2-mirai: 2 r0 values
for r0 in 0.10 0.15; do
    run_exp "S1-A2-mirai-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_MIRAI" --r0 "$r0"
done

# A2-recon: 2 r0 values
for r0 in 0.10 0.15; do
    run_exp "S1-A2-recon-r0_${r0}" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_RECON" --r0 "$r0"
done

# ============================================================
# STEP 2 (S2): Feature Expansion — 32 runs (v2 + v3)
# Same attacks/r0 as S1, with --features v2 and --features v3
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 2 (S2): Feature Expansion — 32 runs              ║"
echo "╚══════════════════════════════════════════════════════════╝"

for feat_ver in v2 v3; do
    # A1-benign: 4 r0 values
    for r0 in 0.05 0.10 0.15 0.20; do
        run_exp "S2-A1-benign-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --features "$feat_ver" --r0 "$r0"
    done

    # A2-ddos: 4 r0 values
    for r0 in 0.05 0.10 0.15 0.20; do
        run_exp "S2-A2-ddos-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_DDOS" \
            --features "$feat_ver" --r0 "$r0"
    done

    # A2-syn: 2 r0 values
    for r0 in 0.10 0.15; do
        run_exp "S2-A2-syn-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_SYN" \
            --features "$feat_ver" --r0 "$r0"
    done

    # A2-tcp: 2 r0 values
    for r0 in 0.10 0.15; do
        run_exp "S2-A2-tcp-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_TCP" \
            --features "$feat_ver" --r0 "$r0"
    done

    # A2-mirai: 2 r0 values
    for r0 in 0.10 0.15; do
        run_exp "S2-A2-mirai-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_MIRAI" \
            --features "$feat_ver" --r0 "$r0"
    done

    # A2-recon: 2 r0 values
    for r0 in 0.10 0.15; do
        run_exp "S2-A2-recon-${feat_ver}-r0_${r0}" \
            --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_RECON" \
            --features "$feat_ver" --r0 "$r0"
    done
done

# ============================================================
# STEP 3 (S3): Window Aggregation — 24 runs
# --granularity window --features v2, window sizes: 5, 10, 30, 60s
# All with r0=0.10
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 3 (S3): Window Aggregation — 24 runs             ║"
echo "╚══════════════════════════════════════════════════════════╝"

for wsec in 5 10 30 60; do
    # A1-benign
    run_exp "S3-A1-benign-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10

    # A2-ddos
    run_exp "S3-A2-ddos-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_DDOS" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10

    # A2-syn
    run_exp "S3-A2-syn-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_SYN" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10

    # A2-tcp
    run_exp "S3-A2-tcp-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_TCP" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10

    # A2-mirai
    run_exp "S3-A2-mirai-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_MIRAI" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10

    # A2-recon
    run_exp "S3-A2-recon-w${wsec}s-r0_0.10" \
        --pcap "$BENIGN_PCAP" --attack-pcap "$PCAP_RECON" \
        --features v2 --granularity window --window-seconds "$wsec" --r0 0.10
done

# ============================================================
# SUMMARY
# ============================================================
CAMPAIGN_END=$(date +%s)
CAMPAIGN_ELAPSED=$((CAMPAIGN_END - CAMPAIGN_START))
CAMPAIGN_MIN=$((CAMPAIGN_ELAPSED / 60))

echo ""
echo "============================================================"
echo "  Campaign-02 COMPLETE"
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

echo "Next: python $REPO_ROOT/experiments/results/campaign-02/generate_plots.py"
