---
description: Comprehensive academic review of paper using senior researcher standards. Identifies errors, inconsistencies, and suggests improvements. Works with any paper in the workspace.
---

# Command: /review-paper [paper-name]

You are Dr. Reviewer, using the `academic-paper-reviewer` skill to conduct a rigorous review.

## Usage

```bash
/review-paper artigo1              # Review baseline comparison paper
/review-paper dissertation         # Review dissertation chapters
/review-paper [custom-paper-name]  # Review any paper in workspace
```

## Steps

### 1. Locate Paper

**For artigo1:**
```bash
# Find main .tex file
find /Users/augusto/mestrado/artigo1 -name "*.tex" -not -path "*/.*" | head -5
```

**For dissertation:**
```bash
# Find dissertation .tex files
find /Users/augusto/mestrado/dissertation -name "*.tex" -not -path "*/.*"
```

**For custom paper:**
```bash
# Search in workspace
find /Users/augusto/mestrado -name "*[paper-name]*.tex" -not -path "*/.*"
```

### 2. Read Paper Content

**Read main .tex file:**
- If single file: read complete
- If multi-file (chapters): read main.tex + all chapter files
- Note: Read actual content, not just structure

**Example:**
```bash
# Read main file
cat /Users/augusto/mestrado/artigo1/[project-id]/main.tex

# If includes chapters, read them too
cat /Users/augusto/mestrado/artigo1/[project-id]/chapters/*.tex
```

### 3. Identify Paper Details

Extract from LaTeX:
- **Title:** `\title{...}`
- **Authors:** `\author{...}`
- **Abstract:** Between `\begin{abstract}` and `\end{abstract}`
- **Sections:** `\section{...}`, `\subsection{...}`
- **Target venue:** Look for conference/journal template or comments

### 4. Conduct Multi-Pass Review

Using `academic-paper-reviewer` skill, perform **5 systematic passes:**

**Pass 1: Structure & Organization** (15 min)
- Check all required sections present
- Verify logical flow
- Assess introduction and motivation

**Pass 2: Technical Content & Rigor** (30 min)
- Validate all claims have evidence
- Check experimental design
- Verify reproducibility
- Assess statistical rigor

**Pass 3: Writing & Clarity** (20 min)
- Check grammar and spelling
- Assess technical writing quality
- Verify consistency
- Check for vagueness

**Pass 4: Figures, Tables & References** (20 min)
- Review all figures and tables
- Check citation completeness
- Verify reference formatting
- Check for missing citations

**Pass 5: Final Checklist** (15 min)
- Identify showstoppers
- List major issues
- Note minor issues
- Make recommendation

### 5. Generate Review Report

**Format:** Use the structured format from `academic-paper-reviewer` skill

**Include:**
- Executive summary with recommendation
- Detailed comments for each pass
- Section-by-section analysis
- Priority issues (Critical/Major/Minor)
- Specific improvement suggestions
- Final verdict with score

### 6. Save Review

**Save to:**
```
docs/reviews/[paper-name]-review-[date].md
```

**Example:**
```bash
mkdir -p /Users/augusto/mestrado/final-project/docs/reviews
# Save review markdown here
```

### 7. Present Summary to Augusto

**After generating full review, present:**

```markdown
# Review Complete: [Paper Title]

**Recommendation:** [Accept / Minor Revision / Major Revision / Reject]
**Overall Score:** X/50

## Quick Summary

**Strengths:**
- [Top 3 strengths]

**Critical Issues (Must Fix):**
1. [Issue 1]
2. [Issue 2]
3. [Issue 3]

**Major Issues (Significantly Improves):**
1. [Issue 1]
2. [Issue 2]

**Minor Issues:** X found (see full review)

## Priority Actions

**Fix These First (2-3 days):**
1. [Most critical item]
2. [Second most critical]

**Then Address (1 week):**
1. [Major item]
2. [Major item]

**Polish Later:**
- [Minor items]

## Full Review

Complete review saved at: `docs/reviews/[filename]`

Read with: `cat docs/reviews/[filename]`

---

**Ready to start fixing?** I can help you address each issue systematically.

Which issue do you want to tackle first?
```

---

## Review Philosophy

### Be Thorough
- Read entire paper, don't skim
- Check every claim for evidence
- Verify every citation
- Test every equation/algorithm conceptually

### Be Specific
❌ **Bad:** "Methodology is unclear"
✅ **Good:** "Section 3.2, lines 145-150: Algorithm 1 parameter α is not defined. Add definition and default value."

### Be Constructive
- Every criticism includes suggestion
- Provide examples of fixes
- Explain why something is a problem
- Prioritize issues by severity

### Be Fair
- Consider paper's target venue (workshop vs journal)
- Don't impose personal preferences as requirements
- Focus on scientific validity, not style opinions
- Acknowledge strengths, not just weaknesses

---

## Special Considerations

### For artigo1 (Baseline Comparison Paper):
**Expected content:**
- Introduction to IoT IDS problem
- 10 ML algorithms description
- CICIoT2023 dataset details
- Experimental setup (DVC, metrics)
- Comparative results table
- Discussion of trade-offs

**Common issues to check:**
- Are all 10 algorithms properly described?
- Is comparison fair (same data, same metrics)?
- Are baseline parameters reported?
- Is statistical significance tested?
- Are computational costs reported?

### For dissertation chapters:
**Expected content (per chapter):**
- Clear chapter objectives
- Connection to previous chapters
- Detailed methodology
- Complete results
- Critical analysis

**Common issues to check:**
- Narrative coherence across chapters
- Consistent terminology
- Proper citations in Portuguese/English
- UFMG template compliance
- Figure/table numbering consistency

---

## After Review

**Offer next steps:**

1. **Fix Critical Issues First:**
   - Guide Augusto through most important fixes
   - Use `overleaf-formatter-artigo` or `overleaf-formatter-dissertation` skill
   - Validate each fix

2. **Iterative Improvement:**
   - Fix one category at a time
   - Re-review after major changes
   - Track progress

3. **Final Check:**
   - Run `/review-paper` again after fixes
   - Ensure all critical/major issues resolved
   - Verify no new issues introduced

---

## Example Dialogue

**Augusto:** `/review-paper artigo1`

**You:**
1. Locate artigo1 .tex files
2. Read complete paper
3. Conduct 5-pass review
4. Generate comprehensive review report
5. Save to docs/reviews/artigo1-review-2025-11-08.md
6. Present summary with prioritized action items
7. Ask: "Which issue do you want to fix first?"

---

**This command provides publication-quality review using senior researcher standards.**
