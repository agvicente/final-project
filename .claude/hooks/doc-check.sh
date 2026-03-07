#!/usr/bin/env bash
# Stop Hook - Checks if documentation needs updating and reminds Claude

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
MARKER_FILE="$PROJECT_ROOT/.claude/.session-start-commit"

cd "$PROJECT_ROOT" || exit 0

# Need git and a marker to compare against
[ -d ".git" ] || exit 0
[ -f "$MARKER_FILE" ] || exit 0

START_COMMIT=$(cat "$MARKER_FILE")

# Count files changed since session start (tracked only)
CHANGED_FILES=$(git diff --name-only "$START_COMMIT" HEAD 2>/dev/null | wc -l | tr -d ' ')
UNCOMMITTED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((CHANGED_FILES + UNCOMMITTED))

# No changes = nothing to report
[ "$TOTAL" -eq 0 ] && exit 0

# Check if STATUS.md was updated (committed or uncommitted changes)
STATUS_UPDATED="false"
if git diff --name-only "$START_COMMIT" HEAD 2>/dev/null | grep -q "^STATUS.md$"; then
    STATUS_UPDATED="true"
fi
if git diff --name-only 2>/dev/null | grep -q "^STATUS.md$"; then
    STATUS_UPDATED="true"
fi

if [ "$STATUS_UPDATED" = "false" ]; then
    echo "LEMBRETE: $TOTAL arquivo(s) modificado(s) nesta sessao e STATUS.md ainda nao foi atualizado. Atualize antes de encerrar."
fi
