---
name: overleaf-formatter-artigo
description: Maintains LaTeX formatting consistency for artigo1 paper. Prevents compilation errors, validates references, checks structure before commits.
version: 1.0.0
activate_when:
  - "artigo1"
  - "paper"
  - "latex"
  - "overleaf"
---

# Overleaf Formatter - Artigo1

## Purpose
Prevent LaTeX errors and maintain formatting consistency in baseline comparison paper.

## Pre-Commit Checklist

Before ANY edit to artigo1:
1. ✅ Read current file to understand format
2. ✅ Make edit preserving indentation/structure
3. ✅ Check references match references.bib
4. ✅ Validate LaTeX syntax
5. ✅ Test compilation (if possible)
6. ✅ Commit with clear message

## Common LaTeX Patterns (Learn These)

**Sections:**
```latex
\section{Introduction}
\subsection{Background}
\subsubsection{IoT Security}
```

**Citations:**
```latex
\cite{Maia2020}  % Single
\cite{Maia2020,Lu2019}  % Multiple
```

**Figures:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\columnwidth]{figures/results.png}
  \caption{Performance comparison of ML algorithms}
  \label{fig:results}
\end{figure}

Reference in text: Figure~\ref{fig:results} shows...
```

**Tables:**
```latex
\begin{table}[htbp]
  \caption{Algorithm performance on CICIoT2023}
  \label{tab:results}
  \centering
  \begin{tabular}{lcccc}
    \toprule
    Algorithm & Accuracy & F1 & Time (s) \\
    \midrule
    K-means & 0.95 & 0.94 & 45.2 \\
    Evolutionary & \textbf{0.97} & \textbf{0.96} & \textbf{38.5} \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Math:**
```latex
Inline: $F_1 = 2 \times \frac{precision \times recall}{precision + recall}$

Display:
\begin{equation}
  \label{eq:f1score}
  F_1 = 2 \times \frac{precision \times recall}{precision + recall}
\end{equation}
```

## Validation Steps

**1. Check References:**
```bash
# Find all \cite{} in .tex
grep -o '\\cite{[^}]*}' *.tex

# Verify keys exist in references.bib
# Look for @article{KEY, or @inproceedings{KEY,
```

**2. Check Labels/References:**
```bash
# All \label{} should have matching \ref{}
grep -o '\\label{[^}]*}' *.tex
grep -o '\\ref{[^}]*}' *.tex
```

**3. Test Compilation:**
```bash
cd /Users/augusto/mestrado/artigo1/[project-dir]
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Common Errors to Prevent

❌ **Missing closing brace**
```latex
\section{Introduction  % Missing }
```

❌ **Undefined reference**
```latex
\cite{NonexistentPaper2024}  % Not in .bib
```

❌ **Special characters not escaped**
```latex
50% improvement  % Should be: 50\% improvement
```

❌ **Figure file not found**
```latex
\includegraphics{missing_file.png}
```

## Before Committing

1. Read modified section aloud (check flow)
2. Verify all references/labels
3. Check no TODOs left: `grep -i todo *.tex`
4. Git diff to review changes
5. Commit message: "Update [section]: [what changed]"

Example: `git commit -m "Update Results: Add Phase 1 baseline table"`

---
**Use this skill whenever editing artigo1 LaTeX files.**
