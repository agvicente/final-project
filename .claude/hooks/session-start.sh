#!/usr/bin/env bash
# Session Start Hook - Loads context and syncs Zotero

set -e

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
CONTEXT_FILE="$PROJECT_ROOT/docs/SESSION_CONTEXT.md"
ZOTERO_BIB="/Users/augusto/mestrado/references.bib"

# Check if running in Claude Code context
if [ -z "$CLAUDE_SESSION_ID" ]; then
    echo "Not in Claude Code session"
    exit 0
fi

# Create message for Claude
cat << EOF
{
  "status": "success",
  "message": "Session started successfully",
  "context": {
    "session_context_available": $([ -f "$CONTEXT_FILE" ] && echo "true" || echo "false"),
    "zotero_bib_available": $([ -f "$ZOTERO_BIB" ] && echo "true" || echo "false"),
    "timestamp": "$(date +%Y-%m-%d\ %H:%M:%S)"
  },
  "display_message": "🚀 Session iniciada! Use /resume para ver o contexto atual."
}
EOF
