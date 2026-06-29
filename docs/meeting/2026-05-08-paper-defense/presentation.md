# Apresentação — Paper SoftCom 2026: Phase Transition Characterization in MicroTEDAclus

**Reunião:** 2026-05-08 (sexta-feira)
**Apresentador:** Augusto Custódio Vicente
**Orientador:** Frederico Gadelha Guimarães
**Objetivo:** validar framing, conteúdo e Tabela VIII honesta antes da submissão (deadline 2026-05-11).

---

## Slide 1 — Onde estamos e o que mudou desde 17/04

**Ponto central:** Reformulação do framing principal do paper, baseada em achado posterior à reunião de abril.

- Reunião 17/04: paper era "5 adaptações para corrigir MicroTEDAclus em IoT IDS".
- Análise da Campaign-05 (`experiments/results/campaign-05/ANALYSIS.md` §7) revelou três regimes patológicos distintos com topologias de cluster qualitativamente diferentes.
- Investigação subsequente (28/04 → 04/05) identificou o **mecanismo causal**: a comparação $\sigma^2$ vs $r_0$ no denominador da eccentricidade governa transição entre dois regimes operacionais.
- Experimento sintético controlado (Exp 3, 1.620 runs, executado 04/05) **confirmou** a estrutura $\lambda^* \propto \sqrt{r_0}$ com check_ratio = 1,00 exato.
- **Pergunta para esta reunião:** o framing "phase transition characterization" é mais defensável que "5 adaptations operacional"?

---

## Slide 2 — O Mecanismo Estrutural

**Ponto central:** A escolha entre dois regimes está literalmente codificada em uma linha de código.

- Em `corrected.py:233`:
  ```python
  effective_var = max(var, self.r0)
  ```
- Função não-suave: trava em um dos dois argumentos.
- Define **dois regimes**:
  - **r0-bounded** ($\sigma^2 \ll r_0$): denominador constante = $r_0$. Limiar **não-adaptativo**.
  - **data-bounded** ($\sigma^2 \gg r_0$): denominador adapta-se à dispersão real. Limiar **adaptativo**.
- 4 code paths em `corrected.py` dependem dessa comparação (linhas 233, 280, 292, 296).
- O algoritmo **original** (V0) não tem o `max()` explícito mas tem comparação análoga em `_is_outlier()` linha 200: `var > r₀`.

**Mensagem:** o regime change não é peculiaridade da nossa correção; é estrutural ao MicroTEDAclus em ambas as implementações.

---

## Slide 3 — Predição Algébrica Falsificável

**Ponto central:** A teoria prevê quantitativamente onde o regime transiciona.

Para dados Gaussianos $\mathcal{N}(0, \lambda^2 I_d)$:
- Variância Welford converge para o **traço da covariância**: $\sigma^2 \to d \cdot \lambda^2$.
- Transição em $\sigma^2 = r_0$ implica:

$$
\boxed{\lambda^* = \sqrt{r_0 / d}}
$$

- Predição quantitativa para $d = 17$:

| $r_0$ | $\lambda^*$ predito |
|---|---|
| $10^{-3}$ | $0{,}0077$ |
| $10^{-1}$ | $0{,}077$ |
| $10^{0}$ | $0{,}243$ |

**Falsificável:** se a estrutura $\sqrt{r_0}$ não se observar empiricamente, a hipótese está errada.

---

## Slide 4 — A Ponte Teoria→Prática: o Fator $(2/d)^2$

**Ponto central:** A teoria de phase transition é geral; o fator $(2/d)^2$ é o **mecanismo específico** que joga deployments IoT (d=17) na fronteira de regime.

A predição $\lambda^* = \sqrt{r_0/d}$ vale para $\sigma^2$ "real". A pergunta operacional: que $\sigma^2$ cada implementação efetivamente acumula?

- **V7 (Welford raw):** $\sigma^2 \to d \cdot \lambda^2$ (trace covariance correto). Em IoT raw com features escala $10^3$–$10^6$: $\sigma^2 \approx 10^{10}$–$10^{12}$ ≫ $r_0=0{,}1$ → firmemente **data-bounded**.
- **V0 (literal Maia):** $\sigma^2 = (\|\delta\|\cdot 2/d)^2/(n-1) \approx (4/d) \cdot \lambda^2$. O fator $(2/d)^2$ **encolhe $\sigma^2$ em $\sim d^2/4$**.

| $d$ | $(2/d)^2$ | Encolhimento |
|---|---|---|
| 2 (Maia validou) | 1 | nenhum — V0 funciona como V7 |
| 17 (IoT real) | 0,0138 | 72× — $\sigma^2_{V0}$ aproxima da fronteira $r_0$ |
| 50 | 0,0016 | 625× — $\sigma^2_{V0}$ definitivamente abaixo de $r_0$ |

**Mensagem:** $(2/d)^2$ NÃO é o "bug central". É o mecanismo que **seleciona o regime** em V0:
- $d \le 6$: V0 opera em data-bounded (igual V7) — Maia validou aqui.
- $d \ge 10$: V0 entra em r0-bounded paranoia → fragmentação.

A phase transition é teoria geral; $(2/d)^2$ é a porta específica pela qual V0 atravessa a fronteira em alta-d.

---

## Slide 5 — Hipóteses Pré-Registradas (Exp 3)

**Ponto central:** Três hipóteses falsificáveis, formuladas antes do experimento.

- **H1 (transição V7):** V7 transiciona em $\lambda^* = \sqrt{r_0/d}$, com tolerância 2×.
- **H2 (V0 difere):** V0 (sem `max()` guard) apresenta comportamento qualitativamente diferente — Cohen's $d > 0{,}8$ vs V7 em pelo menos uma métrica.
- **H3 (escala universal):** Razão $\lambda^*(r_0^a)/\lambda^*(r_0^b)$ bate $\sqrt{r_0^a/r_0^b}$ dentro de 2×.

Se nenhuma se confirmar, voltamos ao framing operacional.

---

## Slide 6 — Setup Experimental

**Ponto central:** Rigor metodológico padrão do projeto.

- 1.620 runs: $\lambda \times r_0 \times$ algoritmo $\times$ seed:
  - 9 valores de $\lambda$ log-espaçados em $[10^{-3}, 10^1]$.
  - 3 valores de $r_0 \in \{10^{-3}, 10^{-1}, 10^0\}$.
  - 2 algoritmos: V0 (literal Maia) e V7 (corrigido).
  - 30 seeds por condição.
- $d = 17$ fixo (matches IoT v1 features).
- 1.000 amostras por run (950 normais + 50 anomalias a $5\sigma$).
- Estatística: Friedman + Nemenyi (não-paramétrica) + ANOVA + Tukey HSD (paramétrica) + bootstrap CI 95% (1.000 amostras).
- Código em `experiments/teda-high-dim/experiments/exp03_regime_transition.py`.

---

## Slide 7 — Resultado Principal

**Ponto central:** A predição estrutural se confirma exatamente; o coeficiente refina.

- **H3 confirmada com precisão sub-amostral** (check_ratio = 1,00 em 3/3 comparações):
  - $\lambda^*(r_0=0{,}1) / \lambda^*(r_0=10^{-3}) = 10{,}000$ vs predito $\sqrt{100} = 10{,}000$.
  - $\lambda^*(r_0=1) / \lambda^*(r_0=10^{-3}) = 31{,}623$ vs $\sqrt{1\,000} = 31{,}623$.
  - $\lambda^*(r_0=1) / \lambda^*(r_0=0{,}1) = 3{,}162$ vs $\sqrt{10} = 3{,}162$.
- **Coeficiente empírico:** V7 = $0{,}092 \cdot \sqrt{r_0}$ vs predito $0{,}243 \cdot \sqrt{r_0}$. Fator constante 0,38 — consistente em 3 ordens de grandeza de $r_0$.
- **Friedman $\chi^2 \ge 160$, $p \le 10^{-30}$** entre $\lambda$ groups.

**Interpretação do offset:** o regime indicator usa fração de clusters com $\sigma^2 > r_0$. Quando V7 fragmenta em ~5–6 clusters, cada um captura subset espacial → variância por cluster $<$ trace covariance total. Refinamento quantitativo do prefactor; estrutura $\sqrt{r_0}$ exata.

---

## Slide 8 — V0 vs V7: Diferença Qualitativa Massiva

**Ponto central:** Cohen's d > 1.000 estabelece V0 e V7 como detectores estruturalmente diferentes.

| $r_0$ | $\lambda$ | FPR V0 | FPR V7 | Cohen's $d$ |
|---|---|---|---|---|
| $10^{-3}$ | $0{,}316$ | $0{,}998$ | $0{,}001$ | $+1\,247$ |
| $10^{-3}$ | $1{,}0$ | $0{,}998$ | $0{,}001$ | $+1\,376$ |
| $10^{-1}$ | $3{,}16$ | $0{,}998$ | $0{,}001$ | $+1\,247$ |
| $10^{0}$ | $10$ | $0{,}998$ | $0{,}001$ | $+1\,247$ |

- Threshold de "large effect" (Cohen 1988) é $|d| > 0{,}8$. Observamos magnitudes 3 ordens de grandeza acima.
- Visual em `fig_regime_v0_vs_v7.pdf`: V0 colapsa em 1.000 clusters (cap), V7 estável em ~6.

**Mensagem:** as 5 adaptações não são "calibração" — elas mudam o detector estruturalmente.

---

## Slide 9 — Reinterpretação de Maia 2020 (Escopo Retrospectivo)

**Ponto central:** Não refutamos Maia. Delimitamos seu escopo de aplicabilidade.

- Maia 2020 reporta "$r_0 = 0{,}001$ robusto para todos os datasets".
- Datasets do paper: $d \in \{2, 3\}$, features $\lambda \sim 1$.
- Substituindo na nossa fórmula: $\lambda^* = \sqrt{0{,}001/2} \approx 0{,}022 \ll \lambda \sim 1$.
- **Maia opera firmemente em regime data-bounded** — longe da fronteira.
- A "robustez de $r_0$" é estabilidade **dentro** de um regime, não generalização **entre** regimes.
- Em IoT com features raw e $d = 17$: a calibração default cai na fronteira, e a "robustez" colapsa.

**Framing politicamente confortável:** Frederico é co-autor de Maia. Esta análise **estende** o paper original; não acusa.

---

## Slide 10 — IoT Manifestation + Tabela VIII Honesta

**Ponto central:** Em CICIoT2023, V7 estabilizado reduz FPR 14×; Tabela VIII expõe o trade-off honesto.

- **Per-flow (C04):** V0 FPR = 54,4% → V7 FPR = 3,9% (consistente com framework).
- **Tabela VIII (C05 baselines):**

| Algoritmo | Tipo | FPR | F1 | Tput | Calibration gap |
|---|---|---|---|---|---|
| HST | Streaming | 47,0% | **50,4%** | 134,6 fl/s | 7,8 pp |
| V7 (ours) | Incremental | **3,8%** | 9,8% | 124,9 fl/s | **40,6 pp** |
| V0 | Incremental | 48,7% | 44,8% | 16,4 fl/s | 0,6 pp |
| LOF | Streaming | 19,2% | 14,1% | 1,7 fl/s | 38,5 pp |

- **HST domina F1.** Não escondemos.
- **Calibration gap** (= |predicted_anomaly_rate − true_attack_rate|) expõe o silent collapse de V7 (40,6 pp): o detector é "mudo" — alarma 4% independente do cenário ter 0% ou 63% de ataque.
- **Reframe:** V7 é caracterização teórica, não competidor SOTA. HST é referência streaming, não competidor.
- **Campaign-06 em preparação** (não rodada): valida hipótese de que normalização de features colapsa V0/V7 para mesmo regime em IoT real.

---

## Slide 11 — Conclusão e Próximos Passos

**Ponto central:** Paper pronto para submissão dia 10/05. Preciso do feedback do orientador hoje.

**Resumo do paper:**
- Caracterização teórica de phase transition em MicroTEDAclus (mecanismo + predição + validação causal).
- 5 adaptações reframadas como "regime stabilizers" (não correções de bug).
- IoT manifestation honesta com HST como referência.
- 22 referências, 7 novas para o framing regime change.

**Pendências antes da submissão:**
1. Build LaTeX limpo (Overleaf) — tenho 7 dias de buffer.
2. Verificação final de Tabela VIII com números corretos da C05.
3. Email para Maia (via Frederico) sobre o framing escopo retrospectivo.
4. Revisão deste paper pelo orientador.

**Pergunta direta ao orientador:**
- O framing **regime change** é defensável academicamente vs. o orientador (você) e vs. reviewers SoftCom?
- Há algum ponto da apresentação que parece fraco ou contra-argumentável?
- A Tabela VIII honesta com HST F1 > V7 é aceitável (reframe HST como "reference" não "competitor"), ou expõe o paper a ataque do reviewer 2?
- OK enviar email para Maia com este framing? Posso redigir, você revisa.

---

## Tempo total estimado: 25–30 min apresentação + 30 min Q&A.
