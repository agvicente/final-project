---
description: Summarize academic paper from Zotero library. Extracts implementation details, pseudocode, experimental setup relevant to research.
---

# Command: /paper-summary [paper-name]

Quickly summarize a paper from Zotero for implementation purposes.

## Usage

```
/paper-summary Maia et al 2020
/paper-summary CICIoT2023
/paper-summary evolutionary clustering
```

## Steps

1. **Find paper in references.bib:**
   ```bash
   grep -i "[search term]" /Users/augusto/mestrado/references.bib
   ```

2. **If found, extract BibTeX entry:**
   Get title, authors, year, venue

3. **Search for paper content:**
   - Check if PDF available locally
   - If not, search web for preprint (arXiv, ResearchGate, author website)
   - Use WebSearch if needed

4. **Use `paper-reading-accelerator` skill to extract:**
   - Core contribution (2-3 sentences)
   - Algorithm/method (pseudocode)
   - Experimental setup (dataset, metrics, parameters)
   - Results summary
   - Implementation hints
   - Relevance to CICIoT2023 research

5. **Present structured summary:**
   ```markdown
   # 📄 Paper Summary: [Title]

   **Authors:** [Names]
   **Year:** [Year]
   **Venue:** [Conference/Journal]

   ## 🎯 Core Idea
   [2-3 sentences explaining the main contribution]

   ## 🔧 Algorithm/Method
   ```
   [Pseudocode or key equations]
   ```

   ## 🧪 Experimental Setup
   - **Dataset:** [What they used]
   - **Baselines:** [What they compared against]
   - **Metrics:** [How they evaluated]
   - **Parameters:** [Key parameter values]

   ## 📊 Key Results
   [Main numbers and findings]

   ## 💡 Implementation Notes
   - Code available: [Yes/No, where]
   - Libraries used: [List]
   - Computational requirements: [Hardware/time]

   ## 🔗 Relevance to Your Research
   [How this applies to CICIoT2023 and your Phase 2/3 work]

   [Specific suggestions: "You could adapt X by...", "This algorithm fits your streaming architecture because..."]

   ## 📚 Citation
   ```
   [BibTeX from references.bib]
   ```

   **Ready to implement? I can help you code this.**
   ```

6. **Offer next steps:**
   "Quer que eu ajude a implementar este algoritmo? Ou prefere outro paper?"

## Notes

- Focus on actionable information for implementation
- Skip heavy theory unless Augusto asks
- Connect directly to CICIoT2023 context
- Suggest how to integrate with existing code
- Be ready to dive deeper into specific sections if asked
