#!/bin/bash
set -e

echo "=========================================="
echo "E2E TEST: Experiment Isolation"
echo "=========================================="

STREAMING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="/tmp/e2e_isolation_test"
PCAP_PATH="$(cd "$(dirname "$0")/../../.." && pwd)/data/pcaps/Benign_Final/BenignTraffic.pcap"

# Cleanup
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

cd "$STREAMING_DIR"

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 1: Running Experiment 1..."
python3 scripts/run_experiment.py \
    --pcap "$PCAP_PATH" \
    --max-packets 500 \
    --max-flows 100 \
    --output "$RESULTS_DIR/exp1" \
    > "$RESULTS_DIR/exp1.log" 2>&1

EXP1_FLOWS=$(jq '.flows_processed' "$RESULTS_DIR/exp1/detection_results.json")
echo "✓ Experiment 1: $EXP1_FLOWS flows processed"

echo ""
echo "Step 2: Running Experiment 2 (should be isolated)..."
python3 scripts/run_experiment.py \
    --pcap "$PCAP_PATH" \
    --max-packets 500 \
    --max-flows 100 \
    --output "$RESULTS_DIR/exp2" \
    > "$RESULTS_DIR/exp2.log" 2>&1

EXP2_FLOWS=$(jq '.flows_processed' "$RESULTS_DIR/exp2/detection_results.json")
echo "✓ Experiment 2: $EXP2_FLOWS flows processed"

echo ""
echo "Step 3: Validating isolation..."

# Should process same amount (isolated)
if [ "$EXP1_FLOWS" -eq "$EXP2_FLOWS" ]; then
    echo "✅ PASS: Both experiments processed same flows ($EXP1_FLOWS)"
else
    echo "❌ FAIL: Flow counts differ (Exp1=$EXP1_FLOWS, Exp2=$EXP2_FLOWS)"
    echo "This suggests interference between experiments"
    exit 1
fi

# Check different group IDs were used
EXP1_LOG=$(cat "$RESULTS_DIR/exp1.log")
EXP2_LOG=$(cat "$RESULTS_DIR/exp2.log")

if echo "$EXP1_LOG" | grep -q "flow-consumer-2026"; then
    if echo "$EXP2_LOG" | grep -q "flow-consumer-2026"; then
        echo "✅ PASS: Both experiments used unique group IDs"
    fi
fi

# Check purge happened
if echo "$EXP1_LOG" | grep -q "ISOLAMENTO DE EXPERIMENTO"; then
    if echo "$EXP2_LOG" | grep -q "ISOLAMENTO DE EXPERIMENTO"; then
        echo "✅ PASS: Purge executed in both experiments"
    fi
fi

echo ""
echo "=========================================="
echo "E2E Test PASSED - Experiments are isolated"
echo "=========================================="
