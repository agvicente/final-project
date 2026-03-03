#!/usr/bin/env bash
# Auto-Save Hook - Triggered by Stop hook after each Claude response

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
cd "$PROJECT_ROOT" || exit 0

# Skip if not a git repo
[ -d ".git" ] || exit 0

# Skip if no unstaged or staged changes
if git diff-index --quiet HEAD -- 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    echo '{"status":"skipped","message":"No changes to save"}'
    exit 0
fi

TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# Add all changes and commit with wip: prefix on current branch
git add -A 2>/dev/null
git commit -m "wip: auto-save $TIMESTAMP" --no-verify 2>/dev/null || true

echo "{\"status\":\"success\",\"message\":\"Auto-saved on branch $BRANCH at $TIMESTAMP\"}"
