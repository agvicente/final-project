---
description: Resume current research context from SESSION_CONTEXT.md. Shows phase, current task, last progress, and next steps.
---

# Command: /resume

You are resuming a research session. Read and present the current context clearly.

## Steps

1. **Read STATUS.md:**
   ```
   /Users/augusto/mestrado/final-project/STATUS.md
   ```

2. **Extract key information:**
   - Current phase and deadline
   - What was done in last session
   - Current task/goal
   - Next immediate steps (seção "Próxima sessão")

3. **Check recent progress logs:**
   - Read latest file in `docs/progress/` for detailed history

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
- If STATUS.md doesn't exist or is empty, say so and offer to create it
- Don't start working automatically, wait for Augusto's input
