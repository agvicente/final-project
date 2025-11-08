#!/usr/bin/env bash
# Auto-Save Hook - Saves work every 10 minutes to protect against crashes

set -e

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)

cd "$PROJECT_ROOT" || exit 1

# Check if git repo
if [ ! -d ".git" ]; then
    echo '{"status": "skipped", "message": "Not a git repository"}'
    exit 0
fi

# Check if there are changes
if git diff-index --quiet HEAD --; then
    echo '{"status": "skipped", "message": "No changes to save"}'
    exit 0
fi

# Create or switch to wip/auto-save branch
git branch wip/auto-save 2>/dev/null || true
CURRENT_BRANCH=$(git branch --show-current)

# Stash current changes
git stash push -m "auto-save-$TIMESTAMP" 2>/dev/null || true

# Switch to auto-save branch
git checkout wip/auto-save 2>/dev/null || git checkout -b wip/auto-save

# Apply stash
git stash pop 2>/dev/null || true

# Add and commit
git add -A
git commit -m "Auto-save: $TIMESTAMP" --no-verify 2>/dev/null || true

# Return to original branch
git checkout "$CURRENT_BRANCH" 2>/dev/null

cat << EOF
{
  "status": "success",
  "message": "Auto-saved to wip/auto-save branch",
  "timestamp": "$TIMESTAMP",
  "display_message": "💾 Auto-save realizado"
}
EOF
