#!/usr/bin/env bash
# Session End Hook - Saves session progress

set -e

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
SESSION_LOG="$PROJECT_ROOT/docs/progress/$(date +%Y-%m-%d)-session-$(date +%H%M).md"

# Create session log directory if not exists
mkdir -p "$PROJECT_ROOT/docs/progress"

# Create session log with placeholder
cat > "$SESSION_LOG" << 'EOF'
# Session Log
**Date:** $(date +%Y-%m-%d)
**Time:** $(date +%H:%M:%S)

## What Was Done
[Auto-generated - to be filled by Claude during session]

## Files Modified
[List of files changed]

## Next Steps
[What to do in next session]
EOF

# Output success
cat << EOF
{
  "status": "success",
  "message": "Session ended, log created at $SESSION_LOG",
  "display_message": "✅ Sessão encerrada. Progresso salvo."
}
EOF
