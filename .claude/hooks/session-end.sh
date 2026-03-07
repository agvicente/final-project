#!/usr/bin/env bash
# Session End Hook - Archives STATUS.md + git activity into progress log

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
MARKER_FILE="$PROJECT_ROOT/.claude/.session-start-commit"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M)
SESSION_LOG="$PROJECT_ROOT/docs/progress/${DATE}-session-${TIME}.md"
STATUS_FILE="$PROJECT_ROOT/STATUS.md"

cd "$PROJECT_ROOT" || exit 0
[ -d ".git" ] || exit 0

mkdir -p "$PROJECT_ROOT/docs/progress"

# Get start commit (fallback if no marker)
if [ -f "$MARKER_FILE" ]; then
    START_COMMIT=$(cat "$MARKER_FILE")
else
    START_COMMIT="HEAD~10"
fi

# Collect git data
COMMITS=$(git log --oneline "$START_COMMIT"..HEAD 2>/dev/null)
FILES_CHANGED=$(git diff --name-only "$START_COMMIT" HEAD 2>/dev/null)

# Include uncommitted changes
UNCOMMITTED=$(git diff --name-only 2>/dev/null)
if [ -n "$UNCOMMITTED" ]; then
    FILES_CHANGED=$(printf '%s\n%s' "$FILES_CHANGED" "$UNCOMMITTED" | sort -u)
fi

# Skip if no activity at all
if [ -z "$COMMITS" ] && [ -z "$FILES_CHANGED" ]; then
    rm -f "$MARKER_FILE"
    echo '{"status":"skipped","message":"No activity this session"}'
    exit 0
fi

# Archive: STATUS.md snapshot + git activity
cat > "$SESSION_LOG" << LOGEOF
# Session — $DATE $TIME

## STATUS.md (snapshot at session end)

$(cat "$STATUS_FILE" 2>/dev/null | sed '1,/^---$/d' || echo "_STATUS.md not found_")

## Git Activity

### Commits
$( [ -n "$COMMITS" ] && echo "$COMMITS" | sed 's/^/- /' || echo "_Nenhum commit_" )

### Files Modified
$( [ -n "$FILES_CHANGED" ] && echo "$FILES_CHANGED" | sed 's/^/- /' || echo "_Nenhum arquivo_" )
LOGEOF

# Clean up marker
rm -f "$MARKER_FILE"

echo "{\"status\":\"success\",\"message\":\"Progress archived: $SESSION_LOG\"}"
