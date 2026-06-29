# Speaker Notes — Defesa Paper SoftCom (2026-05-08)

> **Como usar:** Leia `presentation.md` para o roteiro narrativo. Este arquivo expande cada slide com (1) **fala** sugerida, (2) **âncoras** numéricas/citações para não esquecer, (3) **se perguntarem** com respostas prontas.
>
> **Tempo:** ~25–30 min apresentação + 30 min Q&A. Slides 6 e 7 são o coração.

---

## Slide 1 — Onde estamos e o que mudou

**Fala:**
- "Bom dia. Continuação da Fase 2 do mestrado, com pivô importante desde nossa última reunião."
- "Em 17/04 apresentei o paper como '5 adaptações para corrigir MicroTEDAclus em alta dimensão'. Era um framing operacional, defensável mas modesto."
- "Análise da Campaign-05, executada entre 22 e 28 de abril, revelou três regimes patológicos com topologias de cluster qualitativamente diferentes — V0 com 1.580 singletons, micro_teda com 1 cluster gigante absorvendo 92% dos flows, V1 colapsando em 43 clusters."
- "Investigando essa diferença descobrimos que o algoritmo opera em dois regimes estruturais governados pela comparação $\sigma^2$ vs $r_0$. Daí vem o novo framing: phase transition characterization."
- "Rodei o experimento controlado — 1.620 runs sintéticos — em 04/05. A predição estrutural se confirmou exatamente. O paper está reescrito."

**Âncoras:** Reunião anterior 2026-04-17 | Pivô em 28/04 | Exp 3 executado 04/05 | 1.620 runs, 30 seeds, $d = 17$.

**Se perguntarem "por que mudou o framing tão tarde?":** Porque a evidência apareceu tarde. O framing antigo era correto mas operacional; o novo é teoricamente mais forte. Tive 7 dias para decidir e validar. O experimento foi rodado em 04/05.

---

## Slide 2 — O Mecanismo Estrutural

**Fala:**
- "A linha-chave está no `corrected.py:233`: `effective_var = max(var, self.r0)`."
- "A função `max` é não-suave. Para qualquer ponto $x$ entrando no detector, dois caminhos qualitativamente diferentes são possíveis."
- "Caso 1, $\sigma^2 \ll r_0$: o `max` retorna $r_0$. O denominador da eccentricidade é constante, idêntico para todo cluster. O detector usa um **limiar fixo de aceitação**, calibrado pelo parâmetro, não pelos dados. Chamamos isso de regime **r0-bounded**."
- "Caso 2, $\sigma^2 \gg r_0$: o `max` retorna $\sigma^2$. Cada cluster usa sua própria variância como denominador. O limiar de aceitação **adapta-se à dispersão local**. Regime **data-bounded**."
- "Não é gradiente; é phase transition no sentido operacional. A fronteira é nítida em $\sigma^2 = r_0$."
- "Em corrected.py há 4 code paths que executam essa comparação: linhas 233, 280 (eccentricidade hipotética em chebyshev_accepts), 292 (guarda n=2), 296 (guarda n<3 original). Todos os 4 são pontos onde o regime é selecionado."
- "No original.py (V0 literal), o `max(var, r0)` não existe explicitamente, mas a guarda `n<3` em `_is_outlier` linha 200 faz comparação análoga: `var > r0`. Mesmo regime structure, ativado de forma diferente."

**Âncoras:** corrected.py:233,280,292,296 | original.py:200 | função max não-suave | 2 regimes operacionais.

**Se perguntarem "essa estrutura é peculiar a MicroTEDAclus ou é geral em clustering streaming?":** É um tipo de **floor numérico** comum (StandardScaler, batch normalization também usam `max(eps, x)` para estabilidade). A novidade é mostrar que aqui o floor define dois **regimes operacionais distintos**, não apenas estabilidade numérica.

---

## Slide 3 — Predição Algébrica Falsificável

**Fala:**
- "Para fazer a predição quantitativa, preciso saber para que valor de $\sigma^2$ Welford converge."
- "Nossos testes em `tests/test_welford_variance.py` confirmam: `mc.variance` (atributo do MicroCluster) é o **traço da covariância amostral** — soma das variâncias por dimensão."
- "Para Gaussiana isotrópica $\mathcal{N}(0, \lambda^2 I_d)$, cada dimensão tem variância $\lambda^2$. Soma de $d$ dimensões: $\sigma^2 = d \cdot \lambda^2$."
- "Transição em $\sigma^2 = r_0$ implica $d \lambda^{*2} = r_0$, ou seja: $\lambda^* = \sqrt{r_0/d}$."
- "Para $d = 17$ (CICIoT2023 v1): $\lambda^*_\text{predito}$ varia de 0,008 ($r_0 = 10^{-3}$) a 0,243 ($r_0 = 10^{0}$)."
- "Falsificável: se a estrutura $\sqrt{r_0}$ não emergir empiricamente, ou se $\lambda^*$ não escalar como predito, a hipótese está errada."

**Âncoras:** Welford → trace covariance | $\lambda^* = \sqrt{r_0/d}$ | 3 valores preditos.

**Se perguntarem "por que trace covariance e não variância isotrópica simples?":** Porque é o que o algoritmo realmente acumula, segundo o teste em `test_welford_variance.py::test_matches_numpy_var_2d`. Se eu usar $\lambda^2$ direto, a predição é off por fator $d$ — cobrirá os dados pior, não melhor.

**Se perguntarem "anomalias afetam a predição?":** Sim, parcialmente. Anomalias a $5\sigma$ inflam $\sigma^2$ pré-fragmentação por fator $\sim 2{,}2$ (porque variância da mistura $0{,}95 \cdot \lambda^2 + 0{,}05 \cdot 25 \lambda^2$). Por isso o coeficiente empírico tem deslocamento; a estrutura $\sqrt{r_0}$ permanece exata.

---

## Slide 4 — A Ponte Teoria→Prática: o Fator $(2/d)^2$

**Ponto central:** A predição $\lambda^* = \sqrt{r_0/d}$ é geral. O fator $(2/d)^2$ no V0 é o mecanismo específico que materializa o problema em IoT real.

**Fala:**
- "Slide anterior estabeleceu a teoria. Agora a ponte para o paper original e para IoT real."
- "A predição $\lambda^* = \sqrt{r_0/d}$ assume que $\sigma^2$ é a variância 'real' — trace covariance dos dados, $d \cdot \lambda^2$ para Gaussiana isotrópica."
- "V7 com Welford acumula essa variância real. Em IoT raw, features escala $10^3$–$10^6$, $\sigma^2$ explode para $10^{10}$ ou mais. Sempre data-bounded."
- "V0 — fórmula literal de Maia — calcula $\sigma^2 = (\|\delta\| \cdot 2/d)^2 / (n-1)$. O fator $(2/d)^2$ encolhe a variância em $\sim d^2/4$."
- "Em $d = 2$ (Maia validou), $(2/2)^2 = 1$. NÃO HÁ encolhimento. V0 vê $\sigma^2$ correta. Funciona."
- "Em $d = 17$ (IoT), $(2/17)^2 \approx 0{,}014$. Variância encolhida 72×. V0 'enxerga' $\sigma^2 \approx 0{,}01$ em features que de fato têm $\sigma^2 \approx 10^{10}$."
- "Combinado com $r_0 = 0{,}001$ (default Maia) ou $r_0 = 0{,}1$: $\sigma^2_{V0}$ cruza ou aproxima o floor → V0 entra em **r0-bounded** → fragmenta."
- "Mensagem-chave: $(2/d)^2$ NÃO é o 'bug central'. É o mecanismo por meio do qual o V0 atravessa a fronteira de regime em alta dimensão. Em $d=2$ V0 está OK; em $d=17$ V0 não — mesmo algoritmo, regimes diferentes."

**Âncoras:** $(2/d)^2 = 0{,}014$ em $d=17$ | encolhimento 72× | $\sigma^2_{V0} \approx 0{,}01$ vs $\sigma^2_\text{real} \approx 10^{10}$ | derivação completa em §III-C do paper.

**Se perguntarem "isso é o bug do código do Maia?":**
- **Não exatamente.** É uma fórmula que diverge da publicada $(\|x-\mu\|^2)$. Documentamos a divergência em §III-C como mecanismo de regime-selection, não como erro.
- Em $d \le 6$, o fator $(2/d)^2$ não causa problemas porque $\sigma^2$ ainda fica dentro do regime data-bounded.
- Em $d \ge 10$, o fator passa a determinar o regime — isso é o achado.

**Se perguntarem "por que Maia escolheu essa fórmula?":**
- Não temos resposta autoral. Hipótese: pode ser herança da definição original do TEDA Angelov 2014 (que usa um fator de normalização diferente). Pode ser otimização numérica para $d$ pequeno.
- O paper Maia não explica explicitamente.
- Por isso o framing "delimitação retrospectiva de escopo" — caracterizamos onde a fórmula funciona, não acusamos.

**Se perguntarem "e se o ponto crítico é simplesmente 'não use V0 em alta-d'?":**
- É correto, mas insuficiente. A teoria de phase transition é o que **explica** porque V0 quebra em alta-d e como qualquer detector com mecanismo análogo se comporta. Sem a teoria, "não use em alta-d" é um conselho ad-hoc; com a teoria, tem-se predição quantitativa de QUANDO quebrará e POR QUE.

---

## Slide 5 — Hipóteses Pré-Registradas

**Fala:**
- "Antes de rodar o experimento, registrei três hipóteses falsificáveis no plano (`~/.claude/plans/preciso-que-atualize-todos-lovely-sedgewick.md`)."
- "H1: V7, com mecanismo `max()`, transiciona em $\lambda^*$ predito dentro de 2× — testa o coeficiente."
- "H2: V0, sem `max()` guard, deve diferir qualitativamente — Cohen's $d > 0{,}8$ em alguma métrica."
- "H3: a razão $\lambda^*$ entre diferentes $r_0$'s preserva $\sqrt{r_0_a / r_0_b}$ — testa a estrutura, independente do coeficiente."
- "Critério de robustez: todos os 4 simultaneamente. Se nenhum se confirma, voltamos ao framing operacional. Caso B do plano: estrutura confirma mas coeficiente refina."

**Âncoras:** 3 hipóteses + 1 critério Friedman | falsificáveis | pré-registradas no plano.

**Se perguntarem "isso é p-hacking?":** Não. As 3 hipóteses estão registradas no arquivo de plano timestampado em 04/05 antes do experimento rodar. Posso mostrar `git log` se necessário.

---

## Slide 6 — Setup Experimental

**Fala:**
- "Sweep 2D em $\lambda \times r_0$ com 30 seeds por condição."
- "$\lambda$: 9 valores log-espaçados em $[10^{-3}, 10^1]$ — cobre 4 ordens de grandeza, centrado nos $\lambda^*$ preditos."
- "$r_0$: 3 valores em 3 ordens de grandeza, $\{10^{-3}, 10^{-1}, 10^0\}$. Inclui o default de Maia ($10^{-3}$)."
- "Algoritmos: V0 literal Maia + V7 com 5 adaptações."
- "$d = 17$ fixo (matches IoT real do CICIoT2023 v1 features)."
- "1.000 amostras por run: 950 normais + 50 anomalias a $5\sigma$."
- "Total: 9 × 3 × 2 × 30 = 1.620 runs. Tempo de execução: ~30 min em Mac M2."
- "Estatística: Friedman + Nemenyi (não-paramétrica), ANOVA + Tukey HSD (paramétrica em paralelo para robustez), bootstrap CI 95% com 1.000 amostras (Efron 1993)."

**Âncoras:** 1.620 runs | 30 seeds | $d = 17$ | $5\sigma$ anomalies | Friedman + Nemenyi.

**Se perguntarem "por que não rodar com mais $r_0$'s?":** Cobertura de 3 ordens de grandeza é suficiente para validar a estrutura $\sqrt{r_0}$ — preciso de pelo menos 2 pontos para uma reta em log-log; 3 dá redundância. Mais $r_0$'s aumentariam compute sem ganho marginal de evidência.

**Se perguntarem "tem dados intermediários como $r_0 = 10^{-2}$?":** Não rodei explicitamente, mas a phase diagram (figura 3) mostra interpolação visual com transição gradual.

---

## Slide 7 — Resultado Principal

**Fala:**
- "H3 confirmada exatamente. As três razões empíricas — 10,000, 31,623, 3,162 — batem com $\sqrt{100}, \sqrt{1000}, \sqrt{10}$ até a quarta casa decimal."
- "Isso é evidência **causal** da estrutura $\sqrt{r_0}$. Se fosse coincidência, esperaria check_ratio em torno de 1, não exatamente 1,00."
- "H1 estrutural confirmada (estrutura $\sqrt{r_0}$ existe). Coeficiente diverge: empírico 0,38× predito, **constante** em 3 ordens de grandeza de $r_0$."
- "A **constância** do offset 0,38 é o aspecto crítico: se fosse ruído, varia. Se é offset constante, há um efeito sistemático."
- "O efeito é cluster fragmentation: Welford acumula trace covariance do cluster que aceita o ponto. Quando V7 fragmenta em 5–6 clusters, cada um captura subset espacial — variância por cluster é $<$ trace covariance total. Daí a transição se dá a $\lambda$ menor."
- "Friedman $\chi^2 \ge 160$, $p \le 10^{-30}$ entre $\lambda$ groups: regime separation é estatisticamente massiva."

**Âncoras:** check_ratio = 1,00 exato | offset 0,38 constante | Friedman $p \le 10^{-30}$.

**Se perguntarem "por que o coeficiente diverge e como vamos defender isso?":** É um refinamento, não falha. A predição "ingênua" assume $\sigma^2_\text{cluster} = $ trace covariance do dataset inteiro. Empiricamente, $\sigma^2_\text{cluster} = $ trace covariance do subset capturado pelo cluster, que é menor. Discutimos isso em §V Discussion. Reviewer pode pedir derivação fechada do prefactor; future work.

**Se perguntarem "o experimento é poderoso o suficiente?":** 30 seeds × 9 λ × 3 r0 = 810 medições por algoritmo. Bootstrap CI 95% mostra que erros de estimativa do $\lambda^*$ são da ordem de $10^{-4}$ relativo, muito menor que o offset de 0,62. O sinal é forte.

---

## Slide 8 — V0 vs V7: Diferença Qualitativa Massiva

**Fala:**
- "Cohen's $d = 1.376$ é **mil vezes** o threshold de 'large effect'. Isso não é diferença sutil — é estrutural."
- "Em $r_0 = 10^{-3}$, $\lambda = 0{,}316$: V0 tem FPR 99,8% (alarma quase tudo), V7 tem 0,1%. Não há sobreposição estatística."
- "Visualmente em `fig_regime_v0_vs_v7.pdf`: V0 fragmenta em 1.000 clusters (cap), top-1 fraction colapsa de 1,0 para ~0. V7 mantém ~6 clusters com top-1 estável em ~0,9."
- "**Mensagem:** as 5 adaptações não 'calibram' o algoritmo. Elas mudam o detector estruturalmente. Sem o `max()` guard, V0 não consegue operar no regime data-bounded; ele ou silencia ou hyper-fragmenta."

**Âncoras:** $d = 1.376$ | threshold $> 0{,}8$ | V0 cap 1.000 clusters | V7 estável ~6.

**Se perguntarem "por que V0 não tem o `max()` guard se ele claramente precisa?":** Maia 2020 não previu deployment em alta-d. Em $d \le 6$ + features pequenas, $\sigma^2$ raramente cruza $r_0$ — o algoritmo nunca testou a fronteira. Não é "bug" no sentido tradicional; é **limitação de escopo**.

---

## Slide 9 — Reinterpretação de Maia 2020

**Fala:**
- "Esta é a parte politicamente delicada. Frederico, você é co-autor de Maia 2020. Quero garantir que o framing aqui é **extensão**, não **refutação**."
- "Maia reporta '$r_0 = 0{,}001$ robusto para todos os datasets'. Os datasets do paper têm $d \in \{2, 3\}$ e features escala $\lambda \sim 1$."
- "Substituindo na nossa fórmula: $\lambda^* = \sqrt{0{,}001/2} \approx 0{,}022$. Os datasets do Maia têm $\lambda \approx 1 \gg 0{,}022$. Eles operam **firmemente em regime data-bounded**, longe da fronteira."
- "A 'robustez de $r_0$' do paper é **estabilidade dentro de um regime**, não generalização **entre** regimes. O paper original não testou a fronteira porque não chegou a ela."
- "Em IoT com features raw e $d = 17$: o mesmo $r_0 = 0{,}001$ joga o detector na fronteira. A 'robustez' colapsa porque sai de regime."
- "Esta é a **extensão crítica do escopo de aplicabilidade** do MicroTEDAclus, não correção de erro do paper."

**Âncoras:** Maia op em data-bounded | escopo retrospectivo | $\lambda^*$ Maia $\approx 0{,}022$ | IoT $d = 17$ na fronteira.

**Se perguntarem "vai contatar o Maia?":** Sim, idealmente via você (Frederico). Posso redigir o email com o framing acima — extensão de escopo, não refutação. Pré-submissão.

**Se perguntarem "isso é honest?":** Acho que sim. Não estamos dizendo que Maia errou. Estamos caracterizando onde o método é válido. É exatamente o que o paper de física estatística faz com modelos físicos — define o regime de aplicabilidade.

---

## Slide 10 — IoT Manifestation + Tabela VIII

**Fala:**
- "C04 (per-flow): V0 FPR = 54,4% → V7 FPR = 3,9%. 14× improvement. Já estava no paper."
- "Tabela VIII (C05 baselines): aqui está o ponto sensível."
- "HST F1 = 50,4%, V7 F1 = 9,8%. HST domina F1 em IoT."
- "Não escondo isso. A coluna **Calibration gap** explica: V7 tem gap de 40,6 pp — alarma 4% independentemente do cenário ter 0% ou 63% de ataque. Detector silencioso."
- "Reframe: V7 é caracterização teórica de density-aware streaming clustering, não competidor SOTA de IoT IDS. HST é referência streaming, não competidor."
- "Vou ser claro: se o paper afirmasse 'we built a better IoT IDS', um reviewer competente pergunta 'por que não usar HST'. Não temos resposta. Por isso o paper não afirma isso. Afirma 'we characterize a phase transition; HST is a streaming reference for context'."

**Âncoras:** HST F1 = 50,4% | V7 F1 = 9,8% | calibration gap V7 = 40,6 pp | reframe theoretical.

**Se perguntarem "isso passa pelo reviewer 2?":** Risco médio. Mitigação: o framing em §V Discussion ("Reading the table honestly") expõe o trade-off explicitamente. Reviewer ataca a omissão; não ataca a honestidade. Se ataca: temos defesa pronta no study-guide.md (Q3, Q4).

**Se perguntarem "Campaign-06?":** Em preparação no Linux. Testa se features normalizadas (z-score) colapsam V0 e V7 para mesmo regime em IoT real. Código pronto, script `run_campaign06_normalize.sh` com 16 runs (~3-4h compute). Rodo se sobrar tempo até domingo. Se rodar e confirmar: parágrafo extra na §V. Se não der tempo: "future work" honesto.

---

## Slide 11 — Conclusão e Próximos Passos

**Fala:**
- "Paper escrito, 587 linhas, 22 referências (era 18 + 4 novas relevantes), 0 placeholders, 6 figuras (3 novas Exp 3)."
- "Ainda preciso:"
  1. "Build LaTeX limpo no Overleaf — vou fazer hoje à noite ou amanhã."
  2. "Verificação final dos números da Tabela VIII contra `experiments/results/campaign-05/metrics_summary.csv`."
  3. "Email para Maia, via você."
  4. "Sua revisão deste paper, hoje ou amanhã."
- "Submissão: domingo 10/05. Hard deadline: segunda 11/05."
- "Tenho 4 dias úteis."

**Pergunta direta:**
1. **O framing regime change é defensável?**
2. **Algum ponto da apresentação parece fraco para você?**
3. **Tabela VIII honesta com HST F1 > V7 está OK?**
4. **OK email Maia? Posso redigir, você revisa.**

---

## Encerramento

- Agradecer.
- Anotar todo feedback (lápis e papel).
- Não defender posições; absorver e processar depois.
- Confirmar próximos passos antes de sair.
