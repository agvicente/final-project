---
description: Finalize current weekly report for advisor meeting. Consolidate progress, generate presentation-ready summary.
---

# Command: /finalize-week

Consolidate weekly report for advisor meeting.

## Steps

1. **Read current week progress:**
   ```
   STATUS.md
   docs/progress/*.md (all sessions this week)
   ```

2. **Generate comprehensive report:**
   ```markdown
   # Weekly Report - Week N
   **Date:** [Start] to [End]
   **Phase:** [Current phase]
   **Goal:** [Week objective]

   ## ✅ Accomplished

   ### Code & Implementation
   - [What was implemented]
   - Files: `src/...`
   - Commits: [list key commits]

   ### Experiments & Results
   - [Experiments run]
   - Key metrics: [numbers]
   - Visualizations: [plots/tables created]

   ### Theory & Learning
   - Papers read: [list]
   - Concepts learned: [brief list]

   ## 📊 Results Summary

   [Table or key numbers to show advisor]

   ## 🧠 Insights & Decisions

   ### Technical Decisions Made:
   1. [Decision 1 and rationale]
   2. [Decision 2 and rationale]

   ### Challenges Encountered:
   - [Challenge 1]: [how solved or current status]

   ## 📝 Documentation Updated

   - Dissertation chapters: [which sections]
   - Papers: [progress on artigo1 or others]

   ## 🎯 Next Week Plan

   **Goal:** [Next week objective]
   **Key tasks:**
   1. [Task 1]
   2. [Task 2]
   3. [Task 3]

   ## 💬 Discussion Points for Advisor

   1. [Question or topic 1]
   2. [Question or topic 2]

   ---
   **Status:** Ready for presentation
   ```

3. **Save finalized report:**
   - Save report to `docs/progress/YYYY-MM-DD-week-NN-report.md`
   - Update `STATUS.md` with next week goals
   - Commit: `git add docs/progress/ STATUS.md && git commit -m "Finalize Week N report"`

4. **Offer to create presentation slides (optional):**
   "Relatório finalizado! Deseja que eu crie slides para a apresentação?"

## Notes

- Format should be clear and professional
- Include concrete numbers and results
- Highlight accomplishments without exaggerating
- Be honest about challenges
- Suggest discussion points for advisor
