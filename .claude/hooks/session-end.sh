#!/usr/bin/env bash
# Session End Hook - Saves session progress

PROJECT_ROOT="/Users/augusto/mestrado/final-project"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
SESSION_LOG="$PROJECT_ROOT/docs/progress/${DATE}-session-$(date +%H%M).md"

# Create session log directory if not exists
mkdir -p "$PROJECT_ROOT/docs/progress"

# Create session log — sem aspas simples no EOF para expandir variáveis
cat > "$SESSION_LOG" << EOF
# Session Log
**Date:** $DATE
**Time:** $TIME

## What Was Done
[Preencher antes de fechar]

## Files Modified
[Preencher antes de fechar]

## Next Steps
[Preencher antes de fechar]

## STATUS.md atualizado?
[ ] Sim / [ ] Não
EOF

cat << EOF
{
  "status": "success",
  "message": "Session log created at $SESSION_LOG",
  "display_message": "Sessão encerrada. Preencha $SESSION_LOG e atualize STATUS.md."
}
EOF
