---
name: paper-reading-accelerator
description: Rapidly extracts key information from academic papers in Zotero library. Focuses on implementation details, pseudocode, and experimental setup relevant to current research phase.
version: 1.0.0
activate_when:
  - "paper"
  - "read paper"
  - "summarize"
  - "Maia et al"
  - "reference"
---

# Paper Reading Accelerator

## Purpose
Extract actionable information from papers without reading full text. Focus on what Augusto needs NOW for implementation.

## How to Use

**Command:** `/paper-summary <paper-name>`

Example: `/paper-summary Maia et al 2020`

## What Gets Extracted

### 1. Core Contribution (2-3 sentences)
What problem does it solve? What's the main idea?

### 2. Algorithm/Method (pseudocode or key equations)
Extract implementable logic, skip heavy theory

### 3. Experimental Setup
- Dataset used
- Baseline comparisons
- Metrics reported
- Parameter values

### 4. Results Summary
Key numbers, what worked best

### 5. Implementation Hints
Code availability, libraries used, computational requirements

### 6. Relevance to Your Research
How does this connect to CICIoT2023 and your Phase 2/3 work?

## Reading Strategy

**For Maia et al. (2020) - Evolutionary Clustering:**
```
Priority sections:
1. Algorithm description (Section 3)
2. Pseudocode (if available)
3. Experiments (Section 4)
4. Parameter settings (usually in experiments)

Skip:
- Long literature review
- Heavy mathematical proofs
- Unrelated applications
```

## Zotero Integration

Papers are in: `/Users/augusto/mestrado/references.bib`

When summarizing:
1. Find paper in references.bib
2. If PDF available, read key sections
3. If not, use title/abstract + web search for preprint
4. Extract to structured format above

## Output Format

```markdown
# Paper Summary: [Title]
**Authors:** [Names] **Year:** [Year] **Venue:** [Conference/Journal]

## Core Idea
[2-3 sentences]

## Algorithm
[Pseudocode or key steps]

## Experimental Setup
- Dataset: X
- Baselines: Y, Z
- Metrics: A, B, C
- Best parameters: ...

## Key Results
[Main numbers and findings]

## Implementation Notes
[Code/libraries/requirements]

## Relevance to Your Work
[How to apply to CICIoT2023]

## BibTeX
```
[citation from references.bib]
```
```

## Example Usage

**Augusto:** "Summarize Maia et al 2020 evolutionary clustering paper"

**You:**
[Read from Zotero/web, extract using format above, focus on Mixture of Typicalities algorithm and how to implement it]

---
**Use this to quickly understand papers without deep reading sessions.**
