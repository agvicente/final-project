---
name: scientific-paper-writer
description: Expert scientific writer for ML/IoT/Security papers. Writes incrementally during research, follows IMRAD structure, adapts tone for workshops/conferences/journals.
version: 1.0.0
activate_when:
  - "write paper"
  - "paper section"
  - "introduction"
  - "methodology"
  - "results"
  - "artigo1"
---

# Scientific Paper Writer

## Purpose
Write high-quality academic papers incrementally as research progresses. Focus on clarity, conciseness, and proper scientific communication.

## IMRAD Structure

### Introduction
- Problem statement (IoT security challenges)
- Research gap (what's missing in literature)
- Contribution (your solution)
- Paper organization

### Related Work / Background
- Group papers by theme
- Critical analysis (not just listing)
- Position your work vs others

### Methodology
- Dataset description (CICIoT2023)
- Preprocessing pipeline
- Algorithms implemented
- Experimental setup
- Evaluation metrics

### Results
- Present numbers with context
- Tables and figures with clear captions
- Statistical significance
- Comparison with baselines

### Discussion
- Interpret results
- Limitations
- Future work

### Conclusion
- Summarize contributions
- Key findings
- Impact

## Writing Style

**Good Scientific Writing:**
- Active voice when possible: "We implement X" not "X is implemented"
- Precise: "F1-score improved by 3.2%" not "significant improvement"
- Concise: Remove unnecessary words
- Logical flow: Each paragraph connects to next

**Bad patterns to avoid:**
- Vague claims: "very good results", "performs well"
- Overclaiming: "best ever", "revolutionary"
- Missing context: Numbers without comparison
- Passive voice overuse

## Incremental Writing

**Week by week during research:**
- Week 1-3: Draft introduction + related work
- Week 4-6: Write methodology as you implement
- Week 7-9: Results section with each experiment
- Week 10-12: Discussion + conclusion

**Don't wait until end to write everything!**

## Paper Types

### Workshop Paper (4-6 pages)
- Focus: Single clear contribution
- Style: Concise, preliminary results OK
- Example: Baseline comparison (artigo1)

### Conference Paper (8-10 pages)
- Focus: Complete solution with validation
- Style: Comprehensive, strong results required
- Example: Evolutionary clustering + streaming

### Journal Paper (12-20 pages)
- Focus: Deep analysis, multiple contributions
- Style: Thorough, extensive experiments
- Example: Full system with multiple datasets

## Templates

**Abstract template:**
```
[Context: 1-2 sentences on problem domain]
[Gap: 1 sentence on what's missing]
[Solution: 1-2 sentences on your approach]
[Results: 2-3 sentences with key numbers]
[Impact: 1 sentence on significance]
```

**Methodology section outline:**
```
3. Methodology
  3.1 Dataset
    - Source and characteristics
    - Preprocessing steps
    - Train/test split
  3.2 Proposed Approach
    - Algorithm description
    - Key innovations
    - Pseudocode or architecture diagram
  3.3 Baseline Methods
    - What you compare against
    - Why these baselines
  3.4 Evaluation Metrics
    - Metrics used
    - Why these metrics
  3.5 Experimental Setup
    - Hardware/software
    - Hyperparameters
    - Number of runs
```

**Results table template:**
```
| Algorithm | Accuracy | F1-Score | Time (s) |
|-----------|----------|----------|----------|
| K-means | 0.95±0.02 | 0.94±0.03 | 45.2±3.1 |
| Evolutionary | **0.97±0.01** | **0.96±0.02** | **38.5±2.8** |
| DBSCAN | 0.93±0.03 | 0.92±0.04 | 52.1±4.2 |

Bold indicates best performance. Mean±std over 5 runs.
```

## Integration with Overleaf

**artigo1 location:** `/Users/augusto/mestrado/artigo1/`

**Before editing LaTeX:**
1. Use `overleaf-formatter-artigo` skill to check format
2. Edit section by section
3. Validate compilation
4. Commit with descriptive message

## Citation Management

Papers are in `/Users/augusto/mestrado/references.bib`

```latex
\cite{Maia2020}  % Single citation
\cite{Maia2020,Lu2019,Wahab2022}  % Multiple
```

Always verify BibTeX key matches references.bib

## Common Sections for Your Research

**For baseline paper (artigo1):**
- Introduction: IoT IDS challenges
- Related Work: ML for IoT security
- Methodology: CICIoT2023 + 10 algorithms + DVC pipeline
- Results: Comparative analysis table
- Discussion: GradientBoosting best accuracy, SGDOneClassSVM fastest
- Conclusion: Baseline established for future work

**For evolutionary clustering paper:**
- Introduction: Concept drift problem in IoT
- Related Work: Evolutionary clustering + streaming IDS
- Methodology: Mixture of Typicalities implementation
- Results: Adaptation speed + detection accuracy
- Discussion: Advantages over static approaches
- Conclusion: Ready for streaming integration

---
**Use this to write papers incrementally as research progresses.**
