---
name: overleaf-formatter-dissertation
description: Maintains LaTeX formatting for dissertation (PT and EN versions). Manages multi-chapter structure, cross-references, bibliography. Ensures UFMG template compliance.
version: 1.0.0
activate_when:
  - "dissertation"
  - "dissertação"
  - "chapter"
  - "capítulo"
---

# Overleaf Formatter - Dissertation

## Purpose
Maintain formatting consistency across dissertation chapters (Portuguese and English versions).

## Structure

**Typical UFMG dissertation:**
```
dissertation/
├── main.tex  (master file)
├── chapters/
│   ├── 01-introducao.tex
│   ├── 02-fundamentacao.tex
│   ├── 03-metodologia.tex
│   ├── 04-resultados.tex
│   └── 05-conclusao.tex
├── figures/
├── tables/
└── references.bib
```

## Editing Protocol

**1. Always edit chapter files, not main.tex**

**2. Chapter template:**
```latex
\chapter{Introdução}
\label{chap:introducao}

\section{Contexto e Motivação}
\label{sec:contexto}

[Content here]

\section{Objetivos}
\label{sec:objetivos}

[Content here]
```

**3. Cross-references between chapters:**
```latex
% In Chapter 1:
\label{sec:objetivos}

% In Chapter 3:
Como descrito na Seção~\ref{sec:objetivos}, nosso objetivo é...
```

**4. Figures/Tables in chapters:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/fase1-results.png}
  \caption{Resultados dos experimentos baseline (Fase 1).}
  \label{fig:fase1-results}
\end{figure}

No Capítulo~\ref{chap:metodologia}, a Figura~\ref{fig:fase1-results} apresenta...
```

## PT ↔ EN Consistency

**When translating:**
- Keep ALL \label{} identical
- Keep ALL file/folder structure identical
- Only translate text content and captions

**Example:**
```latex
% PT version (chapters/01-introducao.tex):
\chapter{Introdução}
\label{chap:introducao}
O crescimento da Internet das Coisas...

% EN version (chapters/01-introduction.tex):
\chapter{Introduction}
\label{chap:introducao}  % SAME LABEL
The growth of the Internet of Things...
```

## Validation Checklist

**Before committing any chapter:**
- [ ] Syntax valid (no missing braces)
- [ ] All \ref{} have matching \label{}
- [ ] All \cite{} exist in references.bib
- [ ] Figures/tables exist in folders
- [ ] Captions are clear and complete
- [ ] Cross-references between chapters work
- [ ] Compilation succeeds (pdflatex + bibtex)

## Compilation Test

```bash
cd /Users/augusto/mestrado/dissertation/pt
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Check output
ls main.pdf
```

## Common Multi-Chapter Issues

❌ **Forward reference before label defined**
```latex
% Chapter 3 references Chapter 4 (not written yet)
Ver Capítulo~\ref{chap:conclusao}  % Will show "??" in PDF
```
Solution: Add placeholder or remove reference until chapter exists

❌ **Inconsistent numbering**
Dissertation template handles this, don't manually number

❌ **Duplicate labels**
```latex
% Chapter 1
\section{Metodologia}
\label{sec:metodologia}

% Chapter 3
\section{Metodologia}
\label{sec:metodologia}  % DUPLICATE!
```
Solution: Use unique labels: `\label{sec:intro-metodologia}` and `\label{sec:method-metodologia}`

## Template Compliance (UFMG)

Most UFMG templates have specific requirements:
- Cover page format
- Abstract (Resumo) in Portuguese
- Abstract in English
- List of figures/tables auto-generated
- Bibliography style (abntex2 usually)

**Don't modify main.tex structure unless necessary**

## Incremental Writing Strategy

1. Create chapter file with skeleton structure
2. Write section by section
3. Commit after each complete section
4. Cross-reference later when all chapters exist

**Commit messages:**
```
"Cap. 3: Add Section 3.1 (Dataset description)"
"Cap. 4: Complete Phase 1 results"
"Cap. 2: Fix citations in Related Work"
```

---
**Use this skill for all dissertation LaTeX edits. Consistency is key!**
