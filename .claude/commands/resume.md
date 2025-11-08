---
description: Resume current research context from SESSION_CONTEXT.md. Shows phase, current task, last progress, and next steps.
---

# Command: /resume

You are resuming a research session. Read and present the current context clearly.

## Steps

1. **Read SESSION_CONTEXT.md:**
   ```
   /Users/augusto/mestrado/final-project/docs/SESSION_CONTEXT.md
   ```

2. **Extract key information:**
   - Current phase and week
   - What was done in last session
   - Current task/goal
   - Next immediate steps

3. **Check for interrupted session:**
   - Look for `wip/auto-save` branch: `git branch --list wip/auto-save`
   - If exists: "⚠️ Sessão anterior foi interrompida. Posso recuperar o trabalho salvo automaticamente. Deseja continuar de onde parou?"

4. **Present summary (in Portuguese):**
   ```markdown
   ## 📍 Contexto Atual

   **Fase:** [Phase X - Name]
   **Semana:** [Week N of M]
   **Tarefa Atual:** [What you're working on]

   **Última Sessão:**
   - [What was done]
   - [Files modified]

   **Próximos Passos:**
   1. [Next immediate task]
   2. [Following task]
   3. [Then...]

   **Esta Semana (até agora):**
   - [Progress summary]

   Pronto para continuar? Digite o que quer fazer ou peça sugestões.
   ```

5. **Always end with:** "O que você gostaria de fazer agora?"

## Notes

- Be concise but informative
- Use Portuguese for better communication with Augusto
- If SESSION_CONTEXT.md doesn't exist or is empty, say so and offer to create it
- Don't start working automatically, wait for Augusto's input
