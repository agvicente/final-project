---
description: Start new weekly research sprint. Set goals, update SESSION_CONTEXT, create new weekly report.
---

# Command: /start-sprint

Start a new weekly sprint for research.

## Steps

1. **Check previous week:**
   - Read `docs/weekly-reports/current-week.md`
   - Ask: "Deseja finalizar o relatório da semana passada primeiro? (/finalize-week)"

2. **Set week goals:**
   Ask Augusto:
   ```
   ## 🎯 Nova Sprint Semanal

   Qual é o objetivo principal desta semana?

   **Opções comuns:**
   A. Estudar conceitos/papers (30% teoria)
   B. Implementar código/experimentos (60% prática)
   C. Analisar resultados/escrever (10% revisão)
   D. Outro (descreva)
   ```

3. **Based on answer, suggest sprint plan:**
   ```markdown
   ## Sprint Plan - Semana X

   **Objetivo:** [Main goal]

   **Segunda (2-3h):**
   - [ ] [Theory/planning task]

   **Terça-Quinta (8-12h):**
   - [ ] [Implementation tasks]
   - [ ] [Experiments]

   **Sexta (2-3h):**
   - [ ] [Analysis/reporting]
   - [ ] Preparar relatório semanal

   **Entregável:** [What will be done by end of week]

   Aprova este plano? Posso ajustar.
   ```

4. **After approval:**
   - Update `docs/SESSION_CONTEXT.md` with new week info
   - Create fresh `docs/weekly-reports/current-week.md`
   - Commit: `git add docs/ && git commit -m "Start sprint: Week X - [goal]"`

5. **Start first task:**
   "Pronto! Vamos começar com: [first task from Monday]. O que você precisa para começar?"

## Notes

- Be flexible with time allocations (Augusto has 10-20h/week)
- Align with 30/60/10 distribution
- Always have clear deliverable for week
- Update SESSION_CONTEXT immediately
