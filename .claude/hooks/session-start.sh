#!/usr/bin/env bash
# Session Start Hook - Records baseline commit for progress tracking

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
MARKER_FILE="$PROJECT_ROOT/.claude/.session-start-commit"

cd "$PROJECT_ROOT" || exit 0

# Save current HEAD as session baseline
if [ -d ".git" ]; then
    git rev-parse HEAD 2>/dev/null > "$MARKER_FILE"
fi

echo "Success"
