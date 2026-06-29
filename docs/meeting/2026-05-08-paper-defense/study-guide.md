# Study Guide — Defesa Paper SoftCom (2026-05-08)

> **Como usar:** Q&A antecipado de 18 perguntas que o orientador (ou um reviewer rigoroso) provavelmente fará. Para cada pergunta: resposta de 30s + âncoras técnicas + Plano B se a defesa não convencer.
>
> **Auto-teste:** antes da reunião, leia cada pergunta sem ver a resposta. Se não souber responder em ~30s, releia o material correspondente.

---

## Bloco A — Framing científico

### Q1. "Por que pivotar para 'phase transition' ao invés de manter '5 adaptations'?"

**Resposta (30s):** O framing antigo é **operacional** — "consertamos um bug, FPR caiu 14×". Isso não tem nome próprio na literatura. O framing novo é **estrutural** — caracterizamos um phenomenon (regime change governado por $\sigma^2$ vs $r_0$) que tem analogia com phase transitions em estatística. Permite predição quantitativa falsificável e estabelece a estrutura $\sqrt{r_0}$ como **universal** entre implementações. Mais forte cientificamente.

**Âncoras:** check_ratio = 1,00 exato em 3 razões | Cohen's d = 1.376 | Friedman $\chi^2 \ge 160$.

**Plano B (se não convencer):** o material existente continua válido como "5 adaptations". Apenas reescrevemos abstract + conclusion mantendo §IV-D (Exp 3 results) como "additional empirical characterization". Custo: 1 dia.

### Q2. "Phase transition é metáfora forte demais. Não é só $\max(\sigma^2, r_0)$ travando em um dos dois?"

**Resposta:** Tecnicamente sim — "phase transition" no sentido **operacional**, não termodinâmico. A função $\max$ é não-suave, então a fronteira é nítida (não gradiente). Em estatística aplicada esse termo é usado para regimes governados por travamentos não-suaves — não estamos invocando Ising/Landau. Se preferir, podemos chamar de "regime transition" no abstract; é equivalente.

**Plano B:** trocar "phase transition" por "regime transition" globalmente. 5 min de busca-e-substitui.

### Q3. "A diferença com 'concept drift' é qual?"

**Resposta:** Concept drift é mudança na **distribuição dos dados** ($P(X, Y)$). Nosso achado é regime change no **detector** dado distribuição fixa. Algo análogo a Page-Hinkley, mas no espaço dos parâmetros do algoritmo, não dos dados. Conexão: um drift extremo em $\lambda$ (escala dos dados) pode levar o detector de um regime a outro — o framework explicaria por que detectores quebram silenciosamente sob drift de escala.

**Âncoras:** Gama et al. 2014 ACM CSUR survey | Bifet & Gavaldà 2007 ADWIN | citados em §VI Discussion.

---

## Bloco B — Validade matemática

### Q4. "Por que o coeficiente predito ($1/\sqrt{d} = 0{,}243$) não bate empiricamente (0,092)?"

**Resposta:** A predição assume $\sigma^2_\text{cluster}$ = trace covariance do dataset = $d \lambda^2$. Empiricamente, V7 fragmenta em ~5–6 clusters, cada um capturando subset espacial. Variância por cluster $<$ trace covariance total. O regime indicator usa **variância por cluster**, não por dataset. Daí a transição é detectada a $\lambda$ menor. **Refinamento quantitativo do prefactor**; estrutura $\sqrt{r_0}$ é exata. Anomalias a $5\sigma$ adicionalmente inflam $\sigma^2$ pré-fragmentação.

**Âncoras:** offset 0,38 constante em 3 ordens de grandeza de $r_0$ → não é ruído | seção §VI Discussion explica isso.

**Plano B (se reviewer pedir derivação fechada):** "Future work: closed-form derivation of cluster-fragmentation prefactor from cluster sizing dynamics". Não é fácil porque depende da geometria dos clusters formados.

### Q5. "A predição $\lambda^* = \sqrt{r_0/d}$ é exata?"

**Resposta:** Não — é a predição "ingênua" assumindo trace covariance ideal. A empírica é $\lambda^* = c \cdot \sqrt{r_0}$ com $c$ constante específico ao detector e setup ($c = 0{,}092$ para V7 com 5σ contamination). A **estrutura** $\sqrt{r_0}$ é exata ($1{,}00$ check_ratio). O **coeficiente** $c$ depende de fragmentation dynamics.

### Q6. "Friedman test entre $\lambda$ groups confirma o quê exatamente?"

**Resposta:** Que existe diferença sistemática de FPR entre os 9 valores de $\lambda$ — **dentro** de cada combinação (algorithm, $r_0$). Confirma que o regime indicator não é constante ao longo de $\lambda$, ou seja, **transição existe**. Não confirma o ponto $\lambda^*$ específico — para isso, a interpolação linear nas curvas FPR vs $\log \lambda$ + bootstrap CI 95% (Efron 1993).

**Âncoras:** $\chi^2 \ge 160$, $p \le 10^{-30}$ | 30 seeds × 9 lambdas × 2 algos × 3 r0 = 1.620 runs.

### Q7. "Cohen's $d = 1.376$ parece absurdo. Não é um sintoma de variances comprimidas?"

**Resposta:** É grande porque V0 FPR ≈ 0,998 com std baixíssimo (todos os seeds saturam em ~1.000 clusters), e V7 FPR ≈ 0,001 com std baixíssimo (estável). Pooled std $\approx 0{,}0007$, diferença de médias $\approx 0{,}997$. $d = 0{,}997 / 0{,}0007 \approx 1.376$. Matemática correta. Significa que as duas distribuições não se sobrepõem — separação total. Cohen 1988 define "large effect" em $|d| = 0{,}8$ como referência humana; nosso fenômeno é puramente computacional, sem ruído humano, por isso $d$ atinge magnitudes extremas. É evidência de regime structurally diferente.

---

## Bloco C — Honestidade da Tabela VIII

### Q8. "HST tem F1 maior que V7. Por que não usar HST e abandonar MicroTEDAclus?"

**Resposta:** Porque a contribuição do paper é **caracterização teórica**, não "novo melhor IoT IDS". HST é referência streaming, não competidor. O paper expõe explicitamente em §V Discussion: V7 tem calibration gap de 40,6 pp (silent collapse), HST tem 7,8 pp. Cada algoritmo tem trade-off. Se o objetivo fosse **best F1 IoT IDS**, escolheríamos HST. Nosso objetivo é **explicar quando MicroTEDAclus opera em qual regime** — V7 é o veículo da explicação, não o produto.

**Âncoras:** Tabela VIII linha "Reading honestly" | calibration_gap em pp | "HST as streaming reference" framing.

**Plano B (se reviewer 2 ataca):** podemos adicionar parágrafo explicitando "HST achieves higher F1 due to its design optimized for axis-aligned anomalies; our framework explains why V7 cannot be calibrated to compete in F1 within the current regime architecture, but provides density-aware semantics absent in HST". Custos: 5 linhas.

### Q9. "Calibration gap é uma métrica que você inventou. Tem precedente?"

**Resposta:** Não exatamente; é nossa formalização. Conceito relacionado: "miscalibration" em probabilistic classifiers (Naeini et al. 2015 reliability diagrams). Aqui aplicamos a streaming AD: |predicted_anomaly_rate − true_attack_rate| em pp. Mostra detectores que "ganham" FPR alarmando muito raramente — silent failure mode. Não é arbitrário; é diagnóstico operacionalmente valioso.

**Plano B:** se reviewer questiona, podemos chamar de "predictive coverage gap" e citar Naeini ou similar. Renomear não muda o argumento.

### Q10. "DDoS-TCP tem F1 = 0% em todos os algoritmos. Não é evidência de que o sinal é fraco demais?"

**Resposta:** Sim, é. DDoS-TCP é fundamentally indistinguishable per-flow — confirmado por Sommer & Paxson 2010 (semantic gap). Não é falha do detector; é falha da representação per-flow. Window aggregation muda a pergunta de "este flow é anômalo" para "este IP está se comportando anomalamente" — única forma de pegar TCP flood. Está em §IV-C texto. Reviewer não deve atacar; é literatura conhecida.

---

## Bloco D — Política e escopo

### Q11. "Você contatou Maia? Como ele vai reagir ao framing?"

**Resposta:** Ainda não. Quero seu input antes de escrever. O framing é **escopo retrospectivo**: Maia validou em $d \le 6$ + features pequenas, logo opera em data-bounded — testou estabilidade dentro de um regime. Em $d = 17$ + features raw IoT, sai de regime — daí precisamos de adaptações. **Não estamos refutando**, estamos **delimitando aplicabilidade**. Posso redigir email; você revisa antes de enviar. Risco se não fizer: Maia descobre via paper publicado — politicamente péssimo.

**Plano B se Maia objetar:** rebaixamos o framing para "extensão para alta dimensão" sem mencionar "limitação de escopo". Custo: 1 frase reformulada no abstract + §III-C.

### Q12. "Apenas um dataset IoT (CICIoT2023). Suficiente?"

**Resposta:** Para SoftCom (paper aplicado), sim — dataset é benchmark padrão (Neto et al. 2023). Para journal de top tier (Future Generation, IEEE TKDE), não — pediria UNSW-NB15 ou TON-IoT também. O paper declara em §VI Limitations explicitamente: "validation is on a single dataset; generalization is future work". Foi escolha consciente para caber no cronograma.

**Plano B:** se reviewer pedir mais datasets, "future work" defendido pela limitação declarada. Não é fatal.

### Q13. "Por que SoftCom e não venue mais teórico?"

**Resposta:** Cronograma — defesa em ago/2026, esta submissão precisa caber em 7 dias (deadline 11/05). Future Generation tem 4–6 meses de revisão; SoftCom é mais rápido (~2 meses). Estratégia: SoftCom estabelece o framing publicamente, journal version vem depois com Phase F (IoT normalization), segundo dataset, derivação fechada do prefactor. O paper já contém todos os elementos para essa expansão.

---

## Bloco E — Detalhes técnicos do experimento

### Q14. "Por que sweep de $\lambda$ até $10^1$ apenas, não maior?"

**Resposta:** Tentativa inicial foi $\lambda \in [10^{-3}, 10^3]$ (13 valores), mas V0 satura em 1.000 clusters em $\lambda \ge 0{,}3$ — explosion combinatória O($N^2$) que faz cada run levar 50–80s. Reduzimos para $[10^{-3}, 10^1]$ (9 valores) cobrindo 4 ordens de grandeza, ainda centrado nos $\lambda^*$ preditos. Compute total reduzido de >2h para 30min. Não perde sinal — todos os $\lambda^*$ relevantes estão dentro do range.

### Q15. "30 seeds suficientes?"

**Resposta:** Padrão do projeto (Exp 1: 1.440 runs com 30 seeds, Exp 2: 240 runs com 30 seeds). Bootstrap CI 95% mostra erros de estimativa de $\lambda^*$ na ordem de $10^{-4}$ relativo — muito menor que o offset 0,62 que estamos discutindo. Sinal é forte mesmo com 30 seeds. Mais seeds aumentariam compute sem mudar conclusão.

### Q16. "$d = 17$ fixo. E em $d = 50$ o framework continua?"

**Resposta:** Pela predição $\lambda^* = \sqrt{r_0/d}$, sim — em $d = 50$, $\lambda^*$ desce por $\sqrt{50/17} \approx 1{,}7$ vs $d = 17$. Estrutura preservada. Exp 1 (já no paper, §IV-A) tem sweep dimensional $d \in \{2, 5, 10, 15, 17, 20, 30, 50\}$ — confirma comportamento monotônico. Não é teste explícito do regime change, mas é consistente.

### Q17. "Welford raw é o jeito 'certo' de fazer variance?"

**Resposta:** Sim, é o padrão numérico estável (Welford 1962, Chan et al. 1983). MicroTEDAclus original usa fórmula recursiva específica que diverge do Welford clássico — daí a discrepância $(2/d)^2$. Welford aqui é a forma neutra; não estamos inventando algoritmo, estamos voltando ao padrão.

---

## Bloco F — Próximos passos

### Q18. "Se o paper for rejeitado, plano B?"

**Resposta:** Resubmeter para Future Generation Computer Systems (mesma revista do Maia 2020) com versão expandida — adicionar Phase F (IoT normalization), segundo dataset (TON-IoT ou similar), derivação fechada do prefactor de fragmentation. Cronograma: 4–6 semanas pós-rejeição. O material já está estruturado para essa expansão.

---

## Bloco G — Sobre o abstract e a "ponte teoria→prática"

### Q19. "O abstract menciona phase transition, $(2/d)^2$ E CICIoT 14×. São três contribuições distintas ou uma?"

**Resposta (30s):** **Uma contribuição com arco completo.** Não três contribuições paralelas. A estrutura é:
- **Teoria** (phase transition): caracterização nova do mecanismo $\max(\sigma^2, r_0)$ que governa o detector.
- **Mecanismo específico** ($(2/d)^2$): a porta pela qual a teoria se materializa em IoT real ($d = 17$). É **explicação**, não contribuição independente.
- **Manifestação empírica** (CICIoT 14×): consequência observada, validação no real.

A narrativa é *theory → mechanism → manifestation*, padrão de papers teórico-aplicados sólidos. Cada elemento depende dos outros.

**Risco:** reviewer pode achar denso. Tabela com Opções A/B/C/D em conversa anterior — se denso for problema, **Opção D** (compressão 7→5 sentenças) é mitigação 30 min de trabalho.

**Plano B (se reviewer pedir foco em uma thread):** preservar phase transition como núcleo (contribuição teórica original), $(2/d)^2$ como subseção §III-C, CICIoT como uma frase de validação. Custos: minutos de reescrita do abstract.

**Âncoras:** abstract atual = 7 sentenças densas | três contribuições conectadas, não paralelas | risco de overload mitigado se for o caso.

---

### Q20. "Como o $(2/d)^2$ se relaciona com a phase transition? Não confundem o leitor?"

**Resposta (30s):** $(2/d)^2$ é o **mecanismo específico** pelo qual a implementação literal do Maia (V0) atravessa a fronteira de regime em alta dimensão. Não é a contribuição central — é a explicação concreta de **por que** o problema aparece em IoT.

**Derivação rápida:**
- A predição $\lambda^* = \sqrt{r_0/d}$ vale para $\sigma^2$ "real" (trace covariance = $d \cdot \lambda^2$).
- V7 com Welford: $\sigma^2 \approx d \cdot \lambda^2$ (correto). Sempre data-bounded em IoT.
- V0 com $(2/d)^2$: $\sigma^2 = (\|\delta\| \cdot 2/d)^2 / (n-1) \approx (4/d) \cdot \lambda^2$. Variância **encolhida em $\sim d^2/4$**.

Em $d = 2$: $(2/2)^2 = 1$. Sem encolhimento. V0 funciona como V7. **Maia validou aqui.**
Em $d = 17$: $(2/17)^2 \approx 0{,}014$. Encolhimento 72×. $\sigma^2_{V0}$ aproxima ou cruza $r_0$ → V0 entra em **r0-bounded**. Quebra.

Conexão: a phase transition é a teoria geral; $(2/d)^2$ é o passaporte específico do V0 para entrar nela em alta-d.

**Não confunde o leitor** se for apresentado nessa ordem (teoria primeiro, mecanismo específico depois) — exatamente como está no paper (§III-B teoria, §III-C mecanismo, §III-D adaptações).

**Plano B (se confundir o leitor):** o paper pode comprimir §III-C de 30 linhas para 15, deixando o derivative completa em apêndice ou em `regime-transition.md`.

**Âncoras:** fator $(2/d)^2 = 0{,}014$ em $d=17$ | encolhimento 72× | $d = 2 \Rightarrow$ V0 OK; $d = 17 \Rightarrow$ V0 quebra | mesma teoria, mecanismos diferentes em V0 e V7.

---

## Auto-teste rápido (5 min antes da reunião)

Tente responder cada item em ~10 segundos sem ver a resposta:

1. Qual a contribuição teórica principal do paper?
2. Qual o ponto de transição predito? E o empírico?
3. Por que H3 confirmar com check_ratio = 1,00 é tão importante?
4. Qual Cohen's d entre V0 e V7? Em que zona?
5. Qual o framing politicamente correto da reinterpretação de Maia?
6. Por que HST tem F1 maior que V7? Como o paper trata isso?
7. Quantas referências o paper tem? Quantas são novas para o framing regime change?
8. Quantas figuras o paper tem? Quais 3 são novas?
9. Quantas runs no Exp 3? Em quanto tempo executou?
10. Qual o deadline de submissão? Qual o deadline hard?
11. Como $(2/d)^2$ se conecta com a phase transition?
12. Em $d = 2$ V0 funciona, em $d = 17$ não. Por quê?
13. As três contribuições do abstract (phase transition, $(2/d)^2$, CICIoT) são paralelas ou um arco?

Se errar mais de 2: releia `presentation.md` e `speaker-notes.md`.

---

## Pasta de referências durante a reunião

Imprimir/abrir lado a lado:

1. `presentation.md` (este diretório)
2. `speaker-notes.md` (este diretório)
3. `research/foundations/regime-transition.md` §1–§3 (mecanismo + derivação)
4. `experiments/teda-high-dim/results/exp03_statistical_tests.txt` (números literais)
5. `experiments/teda-high-dim/results/paper_figures/fig_regime_transition.pdf` + `fig_regime_v0_vs_v7.pdf` + `fig_phase_diagram.pdf`
6. `writing/papers/69da494c7d0d6aa7085e2444/a-sofctom.tex` (paper draft)
