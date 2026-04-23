# Guia de Experimentos: Sensibilidade Dimensional do MicroTEDAclus

**Projeto:** Paper SoftCom 2026 — Dimensional Sensitivity of MicroTEDAclus for IoT Intrusion Detection
**Autor:** Augusto Custodio Vicente (UFMG PPGEE)
**Última atualização:** Abril 2026

> Este documento é um guia acadêmico detalhado para os 3 experimentos críticos que
> sustentam a contribuição do paper. Deve ser lido ANTES de executar os experimentos
> para compreender o que cada um testa, por que testa, e como interpretar os resultados.

---

### ⚠️ Lição da Rodada 1 (v1 → v2): Interação r₀ × Dimensionalidade

A primeira rodada usou r₀=0.001 (default do Maia 2020). Com dados Gaussianos N(0,I_d),
a variância de 2 pontos é ≈ 8/d, muito maior que 0.001. O guard n<3 (`var > r₀`)
rejeitava TUDO antes do Chebyshev agir, causando FPR=99.8% para V0 em TODOS os d —
inclusive d=2 (72%), onde o bug dimensional deveria ser dormente.

**Diagnóstico:** o r₀ do Maia (0.001) foi calibrado para dados industriais com escalas
específicas. Para N(0,I_d) com σ²=1 por dimensão, r₀ deve ser proporcional à escala dos dados.

**Correção v2:**
- **Exp 1:** agora varia r₀ ∈ {0.001, 0.1, 1.0, 10.0} em vez de anomaly scales, mostrando
  a interação r₀ × d explicitamente. Isso é um resultado MELHOR para o paper — mostra que
  a sensibilidade dimensional é mediada pelo r₀.
- **Exp 2:** usa r₀=1.0 (calibrado para N(0,I_d)) para que todas as 8 variantes possam se
  diferenciar. Com r₀=0.001, só V5 (guard n=1) e V7 (full) conseguiam escapar do gargalo.

**Implicação para o paper:** o r₀ NÃO é universalmente robusto como Maia 2020 afirmou —
depende da escala dos dados. Em IoT com 17 features, r₀=0.10 funciona. Em Gaussianos
normalizados, r₀=0.001 é catastrófico. Isso é um achado adicional sobre a sensibilidade
paramétrica do MicroTEDAclus em alta dimensão.

---

---

## Sumário

1. [Introdução: O Problema e as Hipóteses](#1-introdução-o-problema-e-as-hipóteses)
2. [Experimento 1: Sweep Dimensional](#2-experimento-1-sweep-dimensional)
3. [Experimento 2: Estudo de Ablação V0-V7](#3-experimento-2-estudo-de-ablação-v0-v7)
4. [Experimento 3: Baseline IF/OC-SVM](#4-experimento-3-baseline-ifoc-svm)
5. [Validação Cruzada Sintético-Real](#5-validação-cruzada-sintético--real)
6. [Consolidação: Tabelas e Figuras do Paper](#6-consolidação-tabelas-e-figuras-do-paper)
7. [Limitações Conhecidas](#7-limitações-conhecidas)
8. [Referências](#8-referências)

---

## 1. Introdução: O Problema e as Hipóteses

### 1.1. O Bug Dimensional do MicroTEDAclus

O MicroTEDAclus [Maia2020] e uma extensao do framework TEDA (Typicality and Eccentricity
Data Analytics) [Angelov2014] para clustering evolutivo em streaming. O algoritmo mantém
micro-clusters com estimativas online de média e variância, usando o teste de Chebyshev
para decidir se um ponto pertence a um cluster existente ou deve formar um novo cluster.

A implementação original do MicroTEDAclus [Maia2020] calcula a variância acumulada
usando a fórmula:

```
σ² = (‖δ‖ · 2/d)²
```

onde `δ = x - μ` é o vetor diferença entre o ponto e a média do cluster, e `d` é a
dimensionalidade do espaço de features. A fórmula correta, conforme derivada na teoria
TEDA [Angelov2014], deveria ser:

```
σ² = ‖x - μ‖²
```

Em dimensão d=2, o fator `(2/d)² = 1`, e as duas fórmulas são equivalentes. Porém,
para d=17 (o caso real de flows de rede IoT com 17 features), o fator `(2/17)² ≈ 0.0138`,
o que significa que a variância é subestimada em aproximadamente **72 vezes**. Essa
subestimação massiva causa uma cascata de falhas em 3 code paths distintos.

### 1.2. Auto-Cancelamento do Chebyshev e os 3 Code Paths

O teste de Chebyshev em si, utilizado para decidir se um ponto é "típico" de um cluster,
auto-cancela o erro dimensional. Isso ocorre porque tanto o numerador quanto o denominador
do teste usam a mesma fórmula de variância. Consequentemente, o teste de pertinência
retorna resultados corretos independentemente do fator `(2/d)²`.

Entretanto, existem **3 code paths** no MicroTEDAclus que utilizam a variância em escala
absoluta (não em razão), onde a subestimação NÃO se cancela:

1. **Criação de micro-clusters:** A variância inicial determina o raio do cluster.
   Com variância subestimada, clusters nascem com raio muito pequeno, gerando fragmentação
   excessiva e falsos positivos.

2. **Merge de clusters:** O critério de fusão compara distâncias entre centróides com
   a soma dos desvios-padrão. Com σ subestimado, clusters que deveriam fundir permanecem
   separados, mantendo a fragmentação.

3. **Classificação de anomalias:** A decisão final de anomalia usa limiares baseados em
   σ. Com σ subestimado, o limiar é artificialmente baixo, classificando pontos normais
   como anômalos (falsos positivos).

### 1.3. As 5 Adaptações Propostas

Para corrigir o problema, foram implementadas 5 adaptações independentes que podem ser
ativadas individualmente ou em combinação:

| ID  | Adaptação              | Descrição                                              |
|-----|------------------------|--------------------------------------------------------|
| A1  | Welford variance       | Substitui a fórmula `(‖δ‖·2/d)²` pelo algoritmo de Welford [Welford1962], numericamente estável e dimensionalmente correto |
| A2  | Eccentricity corrigida | Recalcula a eccentricidade usando a variância correta   |
| A3  | Update seletivo        | Atualiza apenas o cluster mais próximo (winner-take-all [Kohonen1990]) em vez de todos os clusters |
| A4  | Guard σ mínimo         | Impõe um piso para σ², evitando divisão por zero e instabilidade numérica [Reynolds2009] |
| A5  | Guard N mínimo         | Exige um número mínimo de pontos antes de permitir merge ou split |

### 1.4. Variantes V0-V7

As 5 adaptações geram 8 variantes experimentais (cada flag ON/OFF):

| Variante | A1 (Welford) | A2 (ecc) | A3 (update) | A4 (guard σ) | A5 (guard N) | Descrição                |
|----------|:---:|:---:|:---:|:---:|:---:|--------------------------|
| V0       | -   | -   | -   | -   | -   | Original (baseline)      |
| V1       | x   | -   | -   | -   | -   | Só Welford               |
| V2       | -   | x   | -   | -   | -   | Só ecc corrigida         |
| V3       | x   | x   | -   | -   | -   | Welford + ecc            |
| V4       | x   | x   | x   | -   | -   | + update seletivo        |
| V5       | x   | x   | x   | x   | -   | + guard σ                |
| V6       | x   | x   | x   | -   | x   | + guard N (sem guard σ)  |
| V7       | x   | x   | x   | x   | x   | Full (todas ativas)      |

### 1.5. Hipóteses Experimentais

Cinco hipóteses guiam os 3 experimentos:

- **H1:** A taxa de falsos positivos (FPR) da implementação original (V0) cresce
  monotonicamente com a dimensionalidade d. *(Exp 1)*

- **H2:** A implementação corrigida (V7) mantém FPR estável independentemente de d. *(Exp 1)*

- **H2b:** O padrão de degradação de V0 e estabilidade de V7 se mantém sob diferentes
  intensidades de anomalia (3σ, 5σ, 10σ). *(Exp 1)*

- **H3:** As adaptações A1 (Welford) e A2 (eccentricity corrigida) têm maior impacto
  individual na redução de FPR do que A3, A4 e A5. *(Exp 2)*

- **H4:** O MicroTEDAclus adaptado (V7) tem FPR competitivo em comparação com Isolation
  Forest [Liu2008] e One-Class SVM [Scholkopf2001] em dados IoT reais sob avaliação
  prequential. *(Exp 3)*

- **H5:** O ranking de variantes observado em dados sintéticos (Exp 2) é consistente
  com o ranking observado em dados IoT reais (Exp 3). *(Exp 2 + Exp 3)*

### 1.6. Fundamentação Teórica Geral

O fenômeno de degradação do MicroTEDAclus em alta dimensionalidade se conecta, mas não
se reduz, à conhecida *curse of dimensionality*. Beyer et al. [Beyer1999] demonstraram
que, para distribuições i.i.d., a razão entre distância máxima e mínima converge para 1
quando d tende a infinito, tornando o conceito de "vizinho mais próximo" indistinguível.
Aggarwal et al. [Aggarwal2001] estenderam esse resultado mostrando que o efeito depende
da norma L_k utilizada e que normas fracionárias (k < 1) podem ser mais discriminativas.
Zimek et al. [Zimek2012] fizeram o survey definitivo sobre detecção de outliers em alta
dimensionalidade, catalogando técnicas de mitigação.

No caso do MicroTEDAclus, o colapso NÃO provém da curse of dimensionality genérica. O
teste de Chebyshev opera em razão (numerador/denominador usam a mesma variância), então
auto-cancela o erro. O colapso provém especificamente dos 3 code paths que usam σ² em
escala absoluta. Essa distinção é fundamental: o problema não é inerente à alta
dimensionalidade, e sim um bug de implementação que se amplifica com d.

A estabilidade numérica da estimativa de variância online é tratada extensivamente por
Welford [Welford1962] e Chan et al. [Chan1983]. O algoritmo de Welford mantém uma
estimativa running de variância que é numericamente estável mesmo para n grande e valores
com magnitudes muito diferentes, o que é essencial para streaming.

---

## 2. Experimento 1: Sweep Dimensional

### 2.1. O Que Testa

O Experimento 1 testa como a taxa de falsos positivos (FPR) do MicroTEDAclus degrada
conforme a dimensionalidade d cresce. Compara diretamente a implementação original (V0)
com a implementação completamente corrigida (V7). É o experimento mais importante do
paper porque produz a **curva de colapso dimensional** — a evidência visual central
de que o problema existe e de que a solução proposta o resolve.

Adicionalmente, testa a robustez do fenômeno sob 3 intensidades de anomalia (3σ, 5σ, 10σ),
respondendo preventivamente à pergunta de reviewer "e se a anomalia for mais/menos
intensa?".

**Hipóteses testadas:** H1, H2, H2b.

### 2.2. Base Teórica

A curse of dimensionality [Beyer1999] prevê que métodos baseados em distância perdem
poder discriminativo em alta dimensionalidade. Aggarwal et al. [Aggarwal2001] mostraram
que esse efeito depende da norma utilizada — normas L_k com k alto (como a Euclidiana,
k=2) sofrem mais. Zimek et al. [Zimek2012] catalogaram as consequências para detecção
de outliers: à medida que d cresce, a distinção entre pontos normais e anômalos se
dissolve.

No MicroTEDAclus, porém, o mecanismo de degradação é diferente da curse of dimensionality
genérica. O fator `(2/d)²` na fórmula de variância introduz uma subestimação que cresce
quadraticamente com d. Matematicamente:

```
Subestimação = (2/d)² = 4/d²

d=2:   4/4   = 1.00  (sem erro)
d=5:   4/25  = 0.16  (6.25x subestimação)
d=10:  4/100 = 0.04  (25x subestimação)
d=17:  4/289 ≈ 0.014 (72x subestimação)
d=50:  4/2500= 0.0016 (625x subestimação)
```

Essa subestimação causa um aumento progressivo e previsível na taxa de falsos positivos
à medida que d cresce, porque os 3 code paths que usam σ² absoluto tornam-se
progressivamente mais restritivos com clusters de raio artificialmente pequeno.

### 2.3. Protocolo

**Dados sintéticos:** Distribuição Gaussiana isotrópica N(0, I_d), ou seja, cada dimensão
é independente com média 0 e variância 1. Anomalias são geradas deslocando pontos por
um fator multiplicativo em relação ao desvio padrão.

| Parâmetro          | Valor                              |
|--------------------|------------------------------------|
| Dimensões testadas | d ∈ {2, 5, 10, 15, 17, 20, 30, 50} |
| Algoritmos         | V0 (original), V7 (full corrigido) |
| Escalas de anomalia| 3σ (sutil), 5σ (moderada), 10σ (óbvia) |
| Repetições         | 30 por condição (seeds 0-29)       |
| Amostras por run   | 1000 (950 normais + 50 anomalias)  |
| Parâmetro r0       | {0.001, 0.1, 1.0, 10.0} (sweep)    |

**Total de runs:** 8 dimensões x 2 algoritmos x 3 escalas x 30 seeds = **1440 runs**.

**Justificativa das escolhas:**
- **30 repetições:** Garante intervalo de confiança robusto para IC 95%. É o número mínimo
  convencional para que o Teorema Central do Limite se aplique.
- **3 escalas de anomalia:** Um reviewer pode argumentar que o resultado depende da
  intensidade da anomalia. Testar 3 escalas (sutil, moderada, óbvia) demonstra robustez.
- **950/50 split:** 5% de anomalias é realista para cenários de IDS [SommerPaxson2010].
- **Seeds fixas (0-29):** Reprodutibilidade total. Qualquer pesquisador pode replicar
  exatamente os mesmos resultados com as mesmas seeds.

**Execução:**
```bash
cd ~/mestrado/teda-high-dim
source venv/bin/activate
python experiments/exp01_dimensional_sweep.py
```

**Outputs:**
- CSV: `results/exp01_dimensional_sweep.csv` com colunas: d, variant, anomaly_scale, seed,
  fpr, recall, f1, n_clusters, anomaly_rate
- Plot principal: `results/plots/exp01_fpr_vs_d.png` — FPR (y) vs d (x), V0 vs V7, com
  IC 95%, escala 5σ principal e 3σ/10σ como subplots
- Plot recall: `results/plots/exp01_recall_vs_d.png` — idem para Recall
- Plot validação: `results/plots/exp01_synthetic_vs_real.png` — barplot d=17 sintético
  vs C04 real

### 2.4. Resultado Esperado

O resultado esperado é um gráfico com duas curvas (V0 e V7) que se separam
progressivamente a partir de d ≈ 5-10:

- **V0 (original):** FPR cresce monotonicamente com d. Valores esperados aproximados:
  ~3% em d=2, ~15% em d=10, ~55% em d=17, ~80% em d=50.
- **V7 (corrigido):** FPR permanece estável em ~3-5% para todos os valores de d.

A separação entre as curvas deve ser visualmente clara e estatisticamente significativa
(intervalos de confiança de 95% não sobrepostos) a partir de d ≥ 10.

O padrão qualitativo (V0 degrada, V7 estável) deve ser consistente nas 3 escalas de
anomalia, embora os valores absolutos possam variar (anomalias mais sutis = FPR
potencialmente mais alto para V0).

### 2.5. Critério de Sucesso

1. **Separação estatística:** IC 95% de V0 e V7 não se sobrepõem para d ≥ 10.
2. **Robustez multi-escala:** O padrão qualitativo (V0 degrada, V7 estável) se mantém
   nas 3 escalas (3σ, 5σ, 10σ).
3. **Consistência com dados reais:** Valores de FPR em d=17 do sintético são compatíveis
   com os do C04 real (±10 pontos percentuais). Especificamente, V0 sintético ≈ 54%
   (C04 real: 54.4%) e V7 sintético ≈ 3-5% (C04 real: 3.9%).

### 2.6. Referências do Experimento 1

- [Beyer1999] Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U. (1999). "When Is
  'Nearest Neighbor' Meaningful?" *ICDT 1999*, LNCS 1540, pp. 217-235, Springer.
- [Aggarwal2001] Aggarwal, C.C., Hinneburg, A., Keim, D.A. (2001). "On the Surprising
  Behavior of Distance Metrics in High Dimensional Space." *ICDT 2001*, LNCS 1973,
  pp. 420-434, Springer.
- [Zimek2012] Zimek, A., Schubert, E., Kriegel, H.-P. (2012). "A survey on unsupervised
  outlier detection in high-dimensional numerical data." *Statistical Analysis and Data
  Mining*, 5(5):363-387.
- [Angelov2014] Angelov, P. (2014). "Outside the box: an alternative data analytics
  framework." *JAMRIS*, 8(2):53-68.
- [Maia2020] Maia, J. et al. (2020). "Evolving clustering algorithm based on mixture
  of typicalities." *Future Generation Computer Systems*, 106:13-26.

---

## 3. Experimento 2: Estudo de Ablação V0-V7

### 3.1. O Que Testa

O Experimento 2 isola a contribuição individual de cada uma das 5 adaptações para a
redução de FPR. Com 8 variantes (V0-V7), cada uma ativando/desativando flags
independentes, este experimento responde a uma pergunta fundamental que reviewers farão:
*"Qual adaptação é a mais importante? Todas são necessárias?"*

Ablation studies são o padrão em machine learning para demonstrar que cada componente
de uma proposta contribui significativamente para o resultado final [Arp2022].

**Hipótese testada:** H3.

### 3.2. Base Teórica

O conceito de ablation study vem da neurociência (remoção seletiva de tecido cerebral
para identificar função) e foi adotado em ML para validar contribuições de componentes.
Arp et al. [Arp2022] argumentam que ablation studies são obrigatórios em trabalhos de
ML aplicado a segurança, sendo uma das "boas práticas" que distinguem contribuições
rigorosas de contribuições anedóticas.

Com 5 adaptações (flags), o espaço completo teria 2⁵ = 32 combinações. As 8 variantes
selecionadas (V0-V7) formam uma cadeia incremental que permite avaliar:

- **Efeito isolado:** V1 (só Welford) vs V0 (original) mede o impacto de A1 sozinha.
- **Efeitos cumulativos:** V3 (Welford + ecc) vs V1 e V2 mostra se A1 e A2 são aditivas
  ou se há interação (V3 melhor que V1 + V2 separados sugere sinergia).
- **Contribuições marginais:** V4 vs V3, V5 vs V4, V7 vs V5 medem a contribuição
  marginal de cada adaptação subsequente.

Os testes estatísticos seguem a recomendação de Demšar [Demsar2006] para comparação
de múltiplos classificadores: teste de Friedman para detectar diferença global, seguido
de teste post-hoc de Nemenyi para identificar quais pares diferem significativamente.

A atualização seletiva (A3) tem base no princípio winner-take-all de Kohonen
[Kohonen1990], onde apenas a unidade mais ativada é atualizada. Nos self-organizing maps,
isso promove especialização de cada unidade. No contexto do MicroTEDAclus, a adaptação
evita que clusters distantes incorporem informação espúria de pontos que não lhes
pertencem.

Os guards de variância mínima (A4) seguem a prática padrão de regularização em Gaussian
Mixture Models [Reynolds2009], onde um piso para σ² evita singularidades na verossimilhança
e divisões por zero na avaliação de densidade.

### 3.3. Protocolo

| Parâmetro          | Valor                                   |
|--------------------|-----------------------------------------|
| Dimensão           | d=17 (fixo — o caso real do IoT IDS)    |
| Variantes          | V0, V1, V2, V3, V4, V5, V6, V7 (8 variantes) |
| Repetições         | 30 por variante (seeds 0-29)            |
| Amostras por run   | 1000 (950 normais + 50 anomalias, escala 5σ) |
| Parâmetro r0       | 1.0 (calibrado para N(0,I_d))           |

**Total de runs:** 8 variantes x 30 seeds = **240 runs**.

**Testes estatísticos (duplos):**

O protocolo utiliza dois frameworks de testes estatísticos para robustez:

1. **Paramétrico (ANOVA + Tukey HSD):**
   - ANOVA one-way testa H0: "todas as variantes têm o mesmo FPR médio."
   - Se ANOVA rejeitar H0 (p < 0.05), Tukey HSD post-hoc identifica quais pares de
     variantes diferem significativamente.
   - Pressupõe normalidade da distribuição de FPR e homocedasticidade.

2. **Não-paramétrico (Friedman + Nemenyi):**
   - Teste de Friedman testa H0: "todas as variantes têm o mesmo ranking médio."
   - Se Friedman rejeitar H0 (p < 0.05), teste post-hoc de Nemenyi identifica quais
     pares diferem com significância estatística.
   - Não assume normalidade — é mais robusto quando FPR não segue distribuição normal.
   - Produz um **Critical Difference (CD) diagram** [Demsar2006], que é o padrão
     visual para reportar comparações múltiplas de classificadores.

**Regra de decisão:** Reportar ambos os testes. Se concordam, o resultado é robusto. Se
divergem, priorizar Friedman (FPR não é necessariamente normal).

**Execução:**
```bash
cd ~/mestrado/teda-high-dim
source venv/bin/activate
python experiments/exp02_ablation_study.py
```

**Outputs:**
- CSV: `results/exp02_ablation.csv` com colunas: variant, seed, fpr, recall, f1, n_clusters
- Testes: `results/exp02_statistical_tests.txt` com p-values de ambos os frameworks
- Plot: `results/plots/exp02_ablation_fpr.png` — boxplot FPR por variante
- Plot: `results/plots/exp02_ablation_heatmap.png` — heatmap de p-values Tukey HSD
- Plot: `results/plots/exp02_cd_diagram.png` — Critical Difference diagram [Demsar2006]

### 3.4. Resultado Esperado

A expectativa, baseada nos resultados preliminares do C04, é a seguinte ordenação
de impacto:

1. **V7 (full):** Menor FPR (~3-5%). Todas as adaptações ativas.
2. **V3 (Welford + ecc):** Segundo menor FPR. As duas adaptações com maior impacto
   individual combinadas. Possível interação sinérgica (V3 < V1 + V2 - V0).
3. **V1 (Welford):** Impacto individual significativo. A correção da variância sozinha
   já deve reduzir FPR substancialmente.
4. **V4 (+ update seletivo):** Impacto moderado sobre V3. A atualização seletiva
   reduz contaminação cruzada entre clusters.
5. **V5/V6 (+ guards):** Impacto menor, mas possivelmente significativo estatisticamente.
   Os guards previnem edge cases numéricos.
6. **V2 (só ecc):** Impacto parcial. A eccentricity corrigida sem Welford pode não
   ser suficiente para corrigir o problema fundamental.
7. **V0 (original):** Maior FPR (~55% em d=17). Baseline.

**Interações esperadas:**
- V3 > V1 + V2 (sinergia entre Welford e ecc corrigida)
- V5 ≈ V6 ≈ V4 + ε (guards têm impacto marginal)
- V7 ≈ V5 ∩ V6 (combinar os dois guards não adiciona muito além de cada um sozinho)

### 3.5. Critério de Sucesso

1. **Significância global:** Friedman test rejeita H0 com p < 0.05, confirmando que
   existe diferença significativa de FPR entre as variantes.
2. **Identificação de pares:** Nemenyi post-hoc identifica quais pares específicos
   diferem significativamente e quais não diferem (possibilitando agrupamento).
3. **CD diagram publicável:** O Critical Difference diagram [Demsar2006] deve ser
   claro o suficiente para inclusão direta no paper, mostrando agrupamentos de
   variantes que não diferem significativamente entre si.
4. **Concordância paramétrico/não-paramétrico:** ANOVA + Tukey e Friedman + Nemenyi
   devem concordar na ordenação e nos agrupamentos (se divergirem, documentar e
   priorizar Friedman).

### 3.6. Referências do Experimento 2

- [Demsar2006] Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple
  Data Sets." *Journal of Machine Learning Research*, 7:1-30.
- [Arp2022] Arp, D. et al. (2022). "Dos and Don'ts of Machine Learning in Computer
  Security." *USENIX Security Symposium*.
- [Kohonen1990] Kohonen, T. (1990). "The Self-Organizing Map." *Proceedings of the
  IEEE*, 78(9):1464-1480.
- [Reynolds2009] Reynolds, D.A. (2009). "Gaussian Mixture Models." *Encyclopedia of
  Biometrics*, Springer.

---

## 4. Experimento 3: Baseline IF/OC-SVM

### 4.1. O Que Testa

O Experimento 3 compara o MicroTEDAclus adaptado (V7) com dois algoritmos de detecção
de anomalias amplamente reconhecidos — Isolation Forest (IF) [Liu2008] e One-Class SVM
(OC-SVM) [Scholkopf2001] — nos mesmos dados IoT reais do CICIoT2023, usando o mesmo
protocolo de avaliação prequential.

Este experimento responde à pergunta que qualquer reviewer fará: *"Compararam com quê?"*.
O objetivo NÃO é demonstrar superioridade do MicroTEDAclus sobre IF/OC-SVM — é ter uma
comparação direta, honesta e estatisticamente rigorosa que contextualize os resultados
da proposta.

Adicionalmente, este experimento serve como **validação cruzada** para o Exp 2: além de
IF e OC-SVM, são executadas 5 variantes intermediárias do MicroTEDAclus (V0, V1, V3,
V4, V7) nos dados reais. Isso permite verificar se o ranking observado em dados sintéticos
(Exp 2) se mantém em dados IoT reais (H5).

**Previsão baseada nos Exp 1+2:** O achado mais importante do Exp 2 é que V1 (Welford
sozinho) e V3 (Welford+ecc) PIORAM o FPR em dados sintéticos. Se o mesmo padrão se
confirmar em dados IoT reais, isso valida fortemente a tese do acoplamento entre adaptações.
O r₀ do pipeline real é 0.10 — com var_welford ≈ trace(Σ) >> 0.10 para 17 features de
rede, o guard n<3 deve rejeitar agressivamente para V1/V3. O C04 real já mostrou V0
com FPR=54.4% e V7 com 3.9% — agora vamos ver V1 e V3.

**Hipóteses testadas:** H4, H5.

### 4.2. Base Teórica

**Isolation Forest [Liu2008]:** Isola anomalias utilizando particionamento aleatório
do espaço de features. A intuição é que anomalias, sendo raras e diferentes, são isoladas
por menos partições (menor path length na árvore). A principal vantagem para este estudo
é que IF não depende de métricas de distância Euclidiana, portanto é **menos afetado pela
curse of dimensionality** do que métodos baseados em distância. Isso torna IF um baseline
particularmente relevante: se MicroTEDAclus V7 tem FPR comparável ao IF em d=17, isso
sugere que a correção dimensional eliminou efetivamente a desvantagem de usar distâncias
Euclidianas.

**One-Class SVM [Scholkopf2001]:** Aprende o suporte da distribuição dos dados normais
utilizando um kernel RBF, projetando implicitamente os dados para um espaço de dimensão
infinita onde é possível separar dados do "vazio" com um hiperplano. OC-SVM é sensível
à escolha do parâmetro ν (fração esperada de outliers) e pode sofrer com a curse of
dimensionality através do kernel RBF, mas representa o estado da arte clássico em
one-class classification.

**Adaptação para streaming [Gama2013, Losing2018]:** Nem IF nem OC-SVM são algoritmos
incrementais nativos. A adaptação utiliza retreino periódico em sliding window de N pontos.
Cada ponto é classificado ANTES de ser adicionado ao buffer (prequential — test-then-train).
Esta abordagem "batch-adapted-to-streaming" é aceita na literatura de aprendizado em
streaming [Losing2018] e em avaliação prequential [Gama2013].

**Importante para o paper:** Documentar explicitamente que IF e OC-SVM são batch-adapted
e que a comparação foca em acurácia (FPR, Recall, F1) sob o mesmo protocolo prequential.
O throughput é reportado como informação complementar, mas NÃO como argumento de
superioridade — seria injusto comparar throughput de um algoritmo O(1) incremental com
algoritmos que retreinam periodicamente.

**O gap semântico [SommerPaxson2010]:** Sommer e Paxson argumentam que existe um gap
fundamental entre "anomalia estatística" e "ataque real" em IDS baseados em anomalia.
Altas taxas de falsos positivos são o calcanhar de Aquiles de IDS baseados em anomalia,
independentemente do algoritmo usado. Portanto, FPR é a métrica mais crítica a ser
comparada — um FPR mais baixo tem impacto operacional direto.

### 4.3. Protocolo

| Parâmetro             | Valor                                       |
|-----------------------|---------------------------------------------|
| Dataset               | CICIoT2023 (mesmos PCAPs do C04)            |
| Ataques               | DDoS-ICMP, DDoS-SYN, DDoS-TCP, Mirai, Recon |
| Tráfego benigno       | BenignTraffic.pcap                          |
| Features              | 17 features de flow (mesmo extrator do C04) |
| Avaliação             | Prequential (test-then-train)               |
| Max packets           | 50.000 por PCAP                             |
| Max flows             | 10.000 por experimento                      |
| Seeds                 | 5 por configuração (42, 123, 456, 789, 1024)|

**Algoritmos e configurações:**

| Algoritmo            | Tipo           | Parâmetros                             |
|----------------------|----------------|----------------------------------------|
| MicroTEDAclus V0     | Incremental    | r0=0.10, original (baseline C04)       |
| MicroTEDAclus V1     | Incremental    | r0=0.10, Welford only (previsto: piora) |
| MicroTEDAclus V3     | Incremental    | r0=0.10, Welford + ecc (previsto: piora)|
| MicroTEDAclus V4     | Incremental    | r0=0.10, + update seletivo             |
| MicroTEDAclus V7     | Incremental    | r0=0.10, full (baseline C04)           |
| Isolation Forest     | Batch-adapted  | n_estimators=100, contamination=0.1, buffer=200 |
| One-Class SVM        | Batch-adapted  | nu=0.1, kernel=rbf, gamma=scale, buffer=200 |

**Total de runs:** 7 algoritmos x 6 configurações (5 ataques + benigno) x 5 seeds
= **210 runs**.

**Execução:**
```bash
cd ~/mestrado/final-project/experiments/streaming
source venv/bin/activate
bash scripts/run_baseline_campaign.sh
```

**Nota:** Este experimento requer acesso aos PCAPs do CICIoT2023, que estão
disponíveis apenas na máquina Linux. Não pode ser executado no Mac.

### 4.4. Resultado Esperado

- **FPR:** MicroTEDAclus V7 (esperado ~3.9%, conforme C04) provavelmente terá FPR
  mais baixo que IF e OC-SVM batch-adapted. Algoritmos batch-adapted tendem a FPR mais
  alto porque o retreino em buffer introduz lag na adaptação à distribuição corrente.

- **Recall:** IF possivelmente terá Recall mais alto em alguns tipos de ataque (DDoS
  especificamente) porque não depende de distância Euclidiana. OC-SVM pode ter dificuldade
  com Recall em ataques sutis (Recon) por ser mais conservador.

- **Ranking de variantes (H5):** Espera-se V0 < V1 < V3 < V4 < V7 em dados reais,
  consistente com o ranking de dados sintéticos do Exp 2.

- **Throughput:** MicroTEDAclus O(1) por ponto >> IF O(n_trees * log(n)) >> OC-SVM
  O(n_sv * d). Reportar como informação, NÃO como argumento.

### 4.5. Critério de Sucesso

1. **Tabela comparativa completa:** FPR ± σ, Recall ± σ, F1 ± σ e throughput para
   7 algoritmos x 5 ataques, com intervalos de confiança derivados de 5 seeds.
2. **Ranking consistente (H5):** O ranking de variantes MicroTEDAclus em dados reais
   (V0 < V1 < V3 < V4 < V7) é consistente com o ranking em dados sintéticos do Exp 2.
   Desvios devem ser documentados e explicados.
3. **Intervalos de confiança reportados:** Todas as métricas devem ter ± σ derivado
   de 5 seeds. Uma única seed (como no C04 original) é insuficiente para claims
   estatísticos.

### 4.6. Referências do Experimento 3

- [Liu2008] Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). "Isolation Forest."
  *IEEE International Conference on Data Mining (ICDM)*, pp. 413-422.
- [Scholkopf2001] Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J.,
  Williamson, R.C. (2001). "Estimating the Support of a High-Dimensional Distribution."
  *Neural Computation*, 13(7):1443-1471.
- [Gama2013] Gama, J. et al. (2013). "On evaluating stream learning algorithms."
  *Machine Learning*, 90(3):317-346.
- [Losing2018] Losing, V., Hammer, B., Wersing, H. (2018). "Incremental on-line
  learning: A review and comparison of state of the art algorithms." *Neurocomputing*,
  275:1261-1274.
- [SommerPaxson2010] Sommer, R. & Paxson, V. (2010). "Outside the Closed World: On
  Using Machine Learning for Network Intrusion Detection." *IEEE Symposium on Security
  and Privacy (S&P)*.

### 4.7. Previsoes Baseadas nos Exp 1+2

Based on Exp 1+2 results, we have specific testable predictions for Exp 3:

**Prediction 1 (from Exp 2):** V1 (Welford only) and V3 (Welford+ecc) will have HIGHER FPR than V0 (original) in real IoT data. This is because Welford corrects sigma^2 to the true value (large for 17 network features), but the guard n<3 (var > r0 with r0=0.10) will reject aggressively. This was confirmed in synthetic data (Exp 2: V1 FPR=98.9%, V3 FPR=99.8% vs V0 FPR=0% at r0=1.0).

**Prediction 2 (from Exp 1):** V7 (full corrected) will be robust to r0 calibration, maintaining low FPR regardless of whether r0=0.10 is perfectly calibrated for the IoT features. In Exp 1, V7 was stable at ~0.1% FPR across all r0 values tested.

**Prediction 3:** IF and OC-SVM, being batch-adapted (not incremental), will likely have HIGHER FPR than V7 due to the buffer-retrain mechanism. But IF may have better Recall for some attacks since it doesn't depend on Euclidean distances.

**If predictions hold:** This constitutes strong cross-validation between synthetic and real data, validating the coupled-adaptation thesis.

**If predictions fail:** Document the divergence -- it reveals that real IoT data has characteristics not captured by synthetic Gaussians (multimodal distributions, heterogeneous feature scales, temporal correlations).

---

## 5. Validação Cruzada Sintético -> Real

### 5.1. Por Que É Necessária

Os Experimentos 1 e 2 utilizam dados Gaussianos sintéticos. Embora dados sintéticos
ofereçam controle total sobre a dimensionalidade e a intensidade das anomalias, um
reviewer pode (e vai) argumentar: *"Funciona em Gaussiana, mas dados de rede IoT não
são Gaussianos. Como sabemos que os resultados se generalizam?"*

A validação cruzada sintético-real responde a essa objeção conectando os resultados dos
3 experimentos em uma narrativa coerente. Se os padrões observados em dados sintéticos
são consistentes com os observados em dados IoT reais, isso **não prova** que os dados
sintéticos são representativos em geral, mas demonstra que o fenômeno dimensional
capturado pelos sintéticos é o mesmo fenômeno observado nos dados reais.

### 5.1b. Nota sobre a interacao r0

The r0 interaction discovered in Exp 1 means the synthetic-to-real validation must compare at matching r0 values. Exp 1+2 used r0=1.0 (calibrated for N(0,I_d)), while the IoT pipeline uses r0=0.10 (calibrated for real features). The comparison is QUALITATIVE (ranking of variants should be consistent) not QUANTITATIVE (absolute FPR values will differ due to different data distributions and r0 scales).

### 5.2. Protocolo de Validação

A validação cruzada opera em dois níveis:

**Nível 1 — Comparação de FPR em d=17 (Exp 1 vs C04/Exp 3):**

| Algoritmo | FPR d=17 Sintético (Exp 1) | FPR d=17 Real (C04) | FPR d=17 Real (Exp 3) |
|-----------|---------------------------|---------------------|-----------------------|
| V0        | Resultado ± σ             | 54.4%               | Resultado ± σ         |
| V7        | Resultado ± σ             | 3.9%                | Resultado ± σ         |

**Critério:** Diferença entre sintético e real ≤ 10 pontos percentuais (pp) valida a
representatividade. Se V0 sintético ≈ 54% e V7 sintético ≈ 3-5%, compatível com C04
(54.4% e 3.9%), o modelo Gaussiano captura adequadamente o fenômeno dimensional para
o MicroTEDAclus.

**Nível 2 — Comparação de ranking de variantes (Exp 2 vs Exp 3):**

| Ranking | Sintético (Exp 2)      | Real (Exp 3)           |
|---------|------------------------|------------------------|
| 1 (melhor) | V7 (esperado)       | V7 (esperado)          |
| 2       | V5 ou V6               | V5 ou V6               |
| 3       | V4                     | V4                     |
| ...     | ...                    | ...                    |
| 8 (pior)| V0                     | V0                     |

**Critério:** Se a correlação de ranking (Spearman ρ ou Kendall τ) entre sintético e
real é significativamente > 0 (p < 0.05), valida que o fenômeno dimensional afeta
as variantes da mesma forma independentemente do tipo de dados.

### 5.3. O Que Reportar no Paper

**Se os padrões forem consistentes (cenário esperado):**

> "Os resultados em dados sintéticos são consistentes com os resultados em dados IoT
> reais: V0 apresenta FPR de X ± Y% em dados Gaussianos sintéticos (d=17) vs Z% no
> CICIoT2023, e V7 apresenta FPR de A ± B% vs C%. O ranking de variantes se mantém:
> [V7 < V5 ≈ V4 < V3 < V1 < V0] (ρ = W, p < 0.01). Isso indica que o fenômeno de
> sensibilidade dimensional do MicroTEDAclus, embora identificado em dados Gaussianos
> controlados, manifesta-se de forma comparável em dados de rede IoT reais."

**Se os padrões NÃO forem consistentes:**

Divergências devem ser documentadas honestamente e discutidas. Possíveis explicações:

- **Distribuição multimodal:** Dados IoT reais podem ter múltiplos clusters naturais
  (diferentes tipos de tráfego), enquanto dados Gaussianos são unimodais. Isso afetaria
  a contagem de clusters e, indiretamente, o FPR.

- **Correlação entre features:** Dados IoT reais têm features correlacionadas (e.g.,
  bytes enviados e bytes recebidos). Dados Gaussianos isotrópicos N(0, I_d) assumem
  independência. Correlação reduz a dimensionalidade efetiva, o que poderia explicar
  um efeito dimensional mais atenuado em dados reais.

- **Outliers naturais:** Dados IoT reais podem conter tráfego anômalo mas benigno (e.g.,
  updates de firmware, varreduras legítimas) que não existem nos dados sintéticos.

Qualquer uma dessas divergências é informação científica válida — não é falha do estudo.
Conforme Arp et al. [Arp2022], documentar limitações explicitamente é parte do rigor
científico.

### 5.4. Análise Detalhada por Caso

**Caso 1: V0 sintético ≈ V0 real (ambos ~50-55%)**

Interpretação forte: o modelo Gaussiano captura completamente o efeito do fator (2/d)²
sobre os 3 code paths. A distribuição específica dos dados é irrelevante — o bug é
puramente dimensional e afeta igualmente dados Gaussianos e dados de rede.

**Caso 2: V0 sintético > V0 real (e.g., 55% vs 40%)**

Interpretação: dados reais têm dimensionalidade efetiva menor que 17 (features
correlacionadas). O fenômeno existe, mas é atenuado. O modelo Gaussiano fornece um
upper bound para o efeito dimensional.

**Caso 3: V0 sintético < V0 real (e.g., 55% vs 65%)**

Interpretação: dados reais têm fatores adicionais que agravam o FPR (multimodalidade,
outliers naturais). O modelo Gaussiano fornece um lower bound para o efeito, mas o
problema real é ainda mais severo.

**Caso 4: Rankings divergentes (e.g., V3 melhor que V4 em sintético mas pior em real)**

Interpretação: as adaptações interagem diferentemente com a estrutura dos dados reais.
Isso sugere que a escolha da variante ideal pode depender do domínio — o que é uma
conclusão científica valiosa por si mesma. Investigar quais propriedades dos dados
causam a inversão de ranking.

---

## 6. Consolidação: Tabelas e Figuras do Paper

### 6.1. Tabela Central

A tabela principal do paper consolida os resultados dos 3 experimentos em uma visão
integrada. Esta tabela compara 7 algoritmos (5 variantes MicroTEDAclus + IF + OC-SVM)
nos 5 tipos de ataque, com métricas derivadas de 5 seeds (intervalos de confiança):

| Algoritmo            | Tipo           | FPR ± σ (benigno) | Recall ± σ (Recon) | Recall ± σ (SYN) | Recall ± σ (TCP) | Recall ± σ (ICMP) | Recall ± σ (Mirai) | F1 ± σ (best) | Throughput |
|----------------------|----------------|--------------------|--------------------|-------------------|-------------------|--------------------|--------------------|----------------|------------|
| MicroTEDAclus V0     | Incremental    | Exp 3              | ...                | ...               | 0%                | ...                | ...                | ...            | ...        |
| MicroTEDAclus V1     | Incremental    | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |
| MicroTEDAclus V3     | Incremental    | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |
| MicroTEDAclus V4     | Incremental    | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |
| MicroTEDAclus V7     | Incremental    | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |
| Isolation Forest     | Batch-adapted  | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |
| One-Class SVM        | Batch-adapted  | Exp 3              | ...                | ...               | ...               | ...                | ...                | ...            | ...        |

### 6.2. Resultados Obtidos (Exp 1 + Exp 2)

#### Exp 1 — Achados Principais

A substituição de anomaly_scales por r₀ sweep revelou um resultado **mais rico que o esperado**.
A degradação dimensional NÃO é um efeito simples — é uma **interação r₀ × d**:

| r₀ | V0 FPR (d=2) | V0 FPR (d=17) | V0 FPR (d=50) | V7 FPR (todo d) | Interpretação |
|----|-------------|--------------|--------------|-----------------|---------------|
| 0.001 | 72% | 99.8% | 99.8% | ~0.1% | Guard n<3 domina tudo |
| 0.1 | 10.5% | 6.3% | 0.03% | ~0.1% | Guard + Chebyshev interagem |
| 1.0 | 3.0% | 0% | 0% | ~0.1% | Chebyshev self-cancels (confirmado!) |
| 10.0 | 0% | 0% | 0% | ~0% | Guard totalmente neutralizado |

**Achado 1:** Com r₀≥1.0, V0 tem FPR≈0% em d≥10 — confirmando que o teste de Chebyshev
se auto-cancela perfeitamente. A degradação dimensional vem **inteiramente** do guard n<3,
não do Chebyshev.

**Achado 2:** Com r₀=0.1, o FPR de V0 tem um padrão não-monotônico (pico em d=5-10,
depois cai). Isso ocorre porque var_original = 8/d com a fórmula (2/d)²: para d grande,
var_original fica tão pequena que é menor que r₀, e o guard para de rejeitar.

**Achado 3:** V7 é estável (~0.1%) em TODOS os valores de r₀ e d — completamente
independente da calibração de r₀. Isso é uma vantagem prática: V7 é robusto a
miscalibração de r₀.

**Validação sintético→real:** V0 com r₀=0.1 em d=17 dá FPR=6.3% vs 54.4% real (48pp
de diferença). A divergência indica que em dados IoT reais, outros fatores contribuem
(distribuições multimodais, features heterogêneas, intersecção de macro-clusters e life
decay — os outros 2 code paths que não são testados em dados Gaussianos puros).
V7 sintético=0.1% vs real=3.9% — qualitative agreement (ambos <5%).

#### Exp 2 — Achados Principais

Com r₀=1.0 (calibrado para N(0,I_d)), três grupos claros emergiram:

| Grupo | Variantes | FPR | Recall | Interpretação |
|-------|-----------|-----|--------|---------------|
| **Funciona (FPR~0%)** | V0, V4, V5, V6, V7 | 0-0.1% | 7-37% | Chebyshev domina (self-cancels) |
| **Quebra parcial** | V2 (ecc consistente sozinha) | 50% | 90% | Assimetria: var escalada, ecc real |
| **Quebra total** | V1 (Welford sozinho), V3 (Welford+ecc) | 99% | 99% | Welford sem guards piora! |

**Achado 4 (SURPRESA):** Welford sozinho (V1) ou com ecc consistente (V3) **PIORA**
catastroficamente o FPR. Razão: Welford corrige σ² para o valor real (grande), mas o
guard n<3 usa `var > r₀`. Com σ²_welford >> r₀, o guard rejeita o 2º ponto de todo
cluster, impedindo qualquer cluster de crescer além de n=1. Na fórmula original, σ²
era subestimada (var_original < r₀ com r₀=1.0 em d=17), então o guard NÃO rejeitava,
permitindo clusters crescerem e o Chebyshev (self-consistent) funcionar.

**Achado 5:** V7 (full) funciona porque combina Welford + ecc consistente + guards
recalibrados (n=1 e n=2). Os guards protegem clusters jovens do efeito do Welford nos
primeiros pontos. É um sistema acoplado — não é possível ligar uma correção sem as outras.

**Implicação para o paper:** A narrativa muda de "corrigimos 5 coisas" para "demonstramos
que as 5 adaptações formam um sistema acoplado onde cada peça depende das outras, e
ligar uma correção isoladamente pode PIORAR o resultado."

**Estatística:** Friedman chi²=203.2, p<10⁻⁴⁰. ANOVA F=19381, p=0. Ambos concordam.
Nemenyi post-hoc identifica 3 clusters claros (CD=1.917).

### 6.3. Figuras do Paper (atualizado com resultados)

**Figura 1 — Interação r₀ × dimensionalidade (Exp 1):** ★ NOVA
4 painéis (um por r₀), FPR vs d, V0 vs V7 com IC 95%. Mostra a transição de
"guard domina" (r₀=0.001) para "Chebyshev domina" (r₀=1.0) para "tudo funciona" (r₀=10).
Esta é a figura mais importante — mostra que o colapso é mediado pelo r₀, não pela
dimensionalidade em si.

**Figura 2 — Boxplot de ablação com 3 grupos (Exp 2):** ★ ATUALIZADA
Mostra claramente: V1/V3 catastróficos, V2 parcial, V0/V4/V5/V6/V7 bons.
A mensagem visual é "Welford sozinho piora, precisa do sistema completo."

**Figura 3 — CD diagram (Exp 2):**
3 clusters claros com significância estatística.

**Figura 4 — Validação sintético vs real (Exp 1 + C04):**
Barplot d=17, r₀=0.1. Mostra divergência quantitativa mas acordo qualitativo.

**Figura 5 — Tabela comparativa (Exp 3):**
7 algoritmos × 5 ataques. A ser preenchida após Exp 3.

### 6.4. Narrativa do Paper (REESCRITA com resultados reais)

A narrativa muda fundamentalmente em relação ao planejado:

1. **O Problema:** "O MicroTEDAclus usa (2/d)² na variância e na eccentricidade. O paper
   original descreve ‖x−μ‖² — há uma discrepância entre paper e implementação."

2. **Auto-cancelamento (Exp 1, r₀=1.0/10.0):** "O fator (2/d)² se cancela no teste de
   Chebyshev (n≥3). Com r₀ suficientemente grande para neutralizar o guard, V0 e V7
   produzem FPR idêntico em qualquer d. O bug NÃO está no teste principal."

3. **O guard como gargalo (Exp 1, r₀=0.001/0.1):** "Com r₀ miscalibrado, o guard n<3
   (`var > r₀`) rejeita prematuramente. Var original = 4/d × var_true, então para r₀ fixo,
   o efeito depende de d E da escala dos dados. O FPR de V0 é sensível a r₀; V7 é estável."

4. **Acoplamento das adaptações (Exp 2):** "As 5 adaptações formam um sistema acoplado.
   Welford sozinho (V1) PIORA o FPR porque corrige σ² sem recalibrar os guards. V2 (ecc
   consistente) sozinha cria assimetria. Apenas V7 (todas as 5 juntas) funciona porque
   variância, eccentricidade e guards são internamente consistentes."

5. **Robustez paramétrica (Exp 1+2):** "V7 é robusto a r₀: FPR~0.1% para r₀ ∈ {0.001, 10}.
   V0 varia de 0% a 99.8% no mesmo range. Isso contradiz a afirmação de Maia 2020 de que
   r₀ é universalmente robusto — é robusto apenas quando as fórmulas internas são
   consistentes (V7)."

6. **Comparação com baselines (Exp 3 — pendente):** "Em dados IoT reais, o MicroTEDAclus
   V7 é comparado com IF e OC-SVM para contextualizar os resultados absolutos."

---

## 7. Limitações Conhecidas

Documentar limitações explicitamente é parte integral do rigor científico [Arp2022].
As seguintes limitações devem ser discutidas honestamente no paper:

### 7.1. Dataset Único (CICIoT2023)

Todos os resultados em dados reais (C04 e Exp 3) utilizam exclusivamente o dataset
CICIoT2023. Embora seja um dataset recente e representativo de cenários IoT [referência
do dataset], a generalização para outros datasets de IDS (NSL-KDD, UNSW-NB15, CICIDS2017,
ToN-IoT) NÃO foi validada.

**Mitigação parcial:** O Exp 1 com dados sintéticos demonstra que o fenômeno dimensional
é genérico (não depende do dataset específico). Porém, o desempenho absoluto (valores
de FPR e Recall) pode variar entre datasets.

**Trabalho futuro:** Validar em pelo menos mais 2 datasets de IoT IDS.

### 7.2. Baselines Batch-Adapted

Isolation Forest [Liu2008] e One-Class SVM [Scholkopf2001] não são algoritmos de streaming
genuínos. A adaptação utiliza retreino periódico em sliding window, o que introduz
latência e pode não capturar mudanças rápidas na distribuição dos dados. Algoritmos
incrementais nativos como Half-Space Trees (HS-Trees) ou Streaming Random Patches
seriam baselines mais justos para comparação de eficiência.

**Mitigação parcial:** A comparação foca em acurácia (FPR, Recall, F1), não em
throughput. O throughput é reportado como informação complementar, com a nota explícita
de que IF/OC-SVM são batch-adapted.

**Justificativa da escolha:** IF e OC-SVM são os baselines mais reconhecidos e citados
em detecção de anomalias. Qualquer reviewer os conhece e aceita como referência. Usar
baselines obscuros (mesmo que mais justos) levantaria a pergunta *"por que não usaram
IF/OC-SVM?"*.

### 7.3. Dados Sintéticos Gaussianos

Os Experimentos 1 e 2 utilizam dados Gaussianos isotrópicos N(0, I_d). Dados de rede
IoT reais não seguem distribuição Gaussiana — tipicamente são multimodais, com features
correlacionadas e caudas pesadas.

**Mitigação principal:** A validação cruzada (Seção 5) compara resultados sintéticos
com resultados reais. Se os padrões forem consistentes (FPR em d=17, ranking de
variantes), isso valida que o modelo Gaussiano captura adequadamente o fenômeno
dimensional, mesmo que não represente fielmente a distribuição marginal dos dados.

**Por que Gaussiana é aceitável:** O problema identificado é um bug de implementação
no cálculo de variância, não um fenômeno que depende da distribuição dos dados. O fator
(2/d)² afeta a variância independentemente de a distribuição ser Gaussiana, uniforme,
ou empírica. A Gaussiana foi escolhida por ser a distribuição padrão para testes
controlados e por ter propriedades analíticas bem conhecidas [Chan1983].

### 7.4. Concept Drift Não Testado

O MicroTEDAclus, como algoritmo evolutivo, tem capacidade teórica de se adaptar a concept
drift — mudanças na distribuição dos dados ao longo do tempo. Esta capacidade é uma das
motivações originais do framework TEDA [Angelov2014] e do MicroTEDAclus [Maia2020].

No entanto, nenhum dos 3 experimentos testa concept drift explicitamente. Os dados
sintéticos são estacionários (mesma distribuição do início ao fim) e os PCAPs do
CICIoT2023 representam sessões relativamente curtas sem drift significativo.

**Por que não testar:** Testar concept drift adequadamente requer (1) datasets com drift
controlado ou natural, (2) métricas específicas (e.g., prequential accuracy over time
com detecção de mudança), e (3) baselines incrementais que também lidem com drift. Isso
constituiria um estudo separado e está fora do escopo do paper sobre sensibilidade
dimensional.

**Trabalho futuro:** Avaliar o impacto das adaptações propostas na capacidade de
adaptação a concept drift. Em particular, verificar se a correção Welford afeta a
velocidade de adaptação dos micro-clusters a novas distribuições.

### 7.5. Subconjunto de Ataques (5 de 33)

O CICIoT2023 contém 33 tipos de ataque. Os experimentos utilizam apenas 5: DDoS-ICMP
Flood, DDoS-SYN Flood, DDoS-TCP Flood, Mirai-Greip Flood e Recon-OS Fingerprint. Estes
foram selecionados por serem representativos de diferentes categorias (volumétrico,
protocol-based, botnet, reconhecimento) e por já terem sido usados no C04.

**Mitigação parcial:** Os 5 ataques cobrem as categorias principais de ameaças IoT. O
DDoS-TCP (Recall=0% para V0) é particularmente informativo porque revela um caso onde
a implementação original falha completamente.

**Limitação residual:** Ataques mais sutis (e.g., DNS Tunneling, Man-in-the-Middle) ou
ataques de aplicação (e.g., XSS, SQL Injection adaptados para IoT) podem ter
características dimensionais diferentes e não estão representados.

**Trabalho futuro:** Expandir para os 33 tipos de ataque do CICIoT2023, com análise
por categoria de ataque.

### 7.6. Interacao r0 x dimensionalidade

**L6 -- Interacao r0 x dimensionalidade:** O parametro r0 NAO e universalmente robusto como afirmado por [Maia2020]. A sensibilidade dimensional e mediada pelo r0: com r0 bem calibrado, o Chebyshev se auto-cancela e V0 funciona; com r0 mal calibrado, o guard n<3 domina e o FPR colapsa. V7 e robusto a esta miscalibracao -- mas esta descoberta contradiz a simplicidade parametrica alegada para o MicroTEDAclus. References: [Maia2020], resultados Exp 1.

---

## 8. Referências

Todas as referências citadas neste guia, ordenadas alfabeticamente pelo identificador
de citação:

**[Aggarwal2001]**
Aggarwal, C.C., Hinneburg, A., Keim, D.A. (2001). "On the Surprising Behavior of
Distance Metrics in High Dimensional Space." *International Conference on Database
Theory (ICDT 2001)*, LNCS 1973, pp. 420-434, Springer.

> Demonstra que o efeito da curse of dimensionality depende da norma L_k utilizada.
> Normas com k alto (incluindo Euclidiana, k=2) sofrem mais. Normas fracionárias
> (k < 1) podem manter poder discriminativo. Relevante para entender por que o
> MicroTEDAclus (que usa norma L2) é particularmente vulnerável.

**[Angelov2014]**
Angelov, P. (2014). "Outside the box: an alternative data analytics framework."
*Journal of Automation, Mobile Robotics and Intelligent Systems (JAMRIS)*, 8(2):53-68.

> Framework TEDA (Typicality and Eccentricity Data Analytics) original. Define
> eccentricidade e tipicalidade como medidas não-paramétricas de pertinência a uma
> distribuição. Base teórica para o MicroTEDAclus.

**[Arp2022]**
Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C.,
Rieck, K. (2022). "Dos and Don'ts of Machine Learning in Computer Security." *USENIX
Security Symposium*.

> Catálogo de boas e más práticas em ML aplicado a segurança. Argumenta que ablation
> studies e documentação explícita de limitações são obrigatórios para rigor científico.
> Referência metodológica central deste estudo.

**[Beyer1999]**
Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U. (1999). "When Is 'Nearest
Neighbor' Meaningful?" *International Conference on Database Theory (ICDT 1999)*,
LNCS 1540, pp. 217-235, Springer.

> Resultado teórico fundamental: para distribuições i.i.d., a razão dist_max/dist_min
> converge para 1 quando d tende a infinito. Isso significa que o conceito de "vizinho
> mais próximo" perde significado em alta dimensionalidade. Resultado seminal da curse
> of dimensionality para métodos baseados em distância.

**[Chan1983]**
Chan, T.F., Golub, G.H., LeVeque, R.J. (1983). "Algorithms for Computing the Sample
Variance: Analysis and Recommendations." *The American Statistician*, 37(3):242-247.

> Análise comparativa de algoritmos para computação de variância amostral, incluindo
> análise de estabilidade numérica. Complementa Welford [Welford1962] com análise
> formal de erro de arredondamento e recomendações para implementação.

**[Demsar2006]**
Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets."
*Journal of Machine Learning Research*, 7:1-30.

> Referência padrão para testes estatísticos em comparação de classificadores. Introduz
> o Critical Difference (CD) diagram e recomenda o teste de Friedman + Nemenyi post-hoc
> como alternativa não-paramétrica ao ANOVA + Tukey. Usado no Exp 2 para comparar 8
> variantes com significância estatística.

**[Gama2013]**
Gama, J., Sebastião, R., Rodrigues, P.P. (2013). "On evaluating stream learning
algorithms." *Machine Learning*, 90(3):317-346.

> Define e formaliza a avaliação prequential (test-then-train) para algoritmos de
> aprendizado em streaming. Argumenta que holdout não é adequado para streaming porque
> dados futuros não estão disponíveis no momento da avaliação. Base para o protocolo
> de avaliação dos Exp 3.

**[Kohonen1990]**
Kohonen, T. (1990). "The Self-Organizing Map." *Proceedings of the IEEE*,
78(9):1464-1480.

> Introduz o Self-Organizing Map (SOM) e o princípio winner-take-all, onde apenas a
> unidade mais ativada (e possivelmente suas vizinhas) é atualizada a cada passo. Base
> teórica para a adaptação A3 (update seletivo) do MicroTEDAclus corrected.

**[Liu2008]**
Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). "Isolation Forest." *IEEE International
Conference on Data Mining (ICDM)*, pp. 413-422.

> Propõe o Isolation Forest, que detecta anomalias medindo quão fácil é "isolar" um
> ponto por particionamento aleatório. Não depende de métricas de distância, portanto
> menos afetado pela curse of dimensionality. Baseline principal no Exp 3.

**[Losing2018]**
Losing, V., Hammer, B., Wersing, H. (2018). "Incremental on-line learning: A review
and comparison of state of the art algorithms." *Neurocomputing*, 275:1261-1274.

> Survey de algoritmos incrementais vs batch-adapted para aprendizado online. Distingue
> algoritmos genuinamente incrementais (processam um ponto por vez, O(1)) de algoritmos
> batch-adapted (retreinam periodicamente). Justifica a categorização de IF e OC-SVM
> como batch-adapted no Exp 3.

**[Maia2020]**
Maia, J., Severiano, C.A., Guimarães, F.G., de Castro, C.L., Lemos, A.P., Silva,
J.C., Medeiros, H.R. (2020). "Evolving clustering algorithm based on mixture of
typicalities for stream data mining." *Future Generation Computer Systems*, 106:13-26.

> Propõe o MicroTEDAclus: extensão do TEDA para clustering evolutivo em streaming.
> Mantém micro-clusters com estimativas online de média e variância, usando Chebyshev
> para teste de pertinência. Contém a fórmula de variância (‖δ‖·2/d)² que diverge
> da derivação teórica. Objeto central deste estudo.

**[Reynolds2009]**
Reynolds, D.A. (2009). "Gaussian Mixture Models." *Encyclopedia of Biometrics*,
Springer.

> Referência canônica sobre GMMs, incluindo a prática de regularização da covariância
> (impor piso para σ² para evitar singularidades). Base teórica para a adaptação A4
> (guard σ mínimo) do MicroTEDAclus corrigido.

**[Scholkopf2001]**
Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., Williamson, R.C. (2001).
"Estimating the Support of a High-Dimensional Distribution." *Neural Computation*,
13(7):1443-1471.

> Propõe o One-Class SVM (ν-SVM), que aprende o suporte de uma distribuição usando
> kernel RBF. Permite detecção de anomalias treinando apenas com dados normais. Segundo
> baseline no Exp 3.

**[SommerPaxson2010]**
Sommer, R. & Paxson, V. (2010). "Outside the Closed World: On Using Machine Learning
for Network Intrusion Detection." *IEEE Symposium on Security and Privacy (S&P)*.

> Artigo seminal que identifica o gap semântico entre "anomalia estatística" e "ataque
> real" em IDS baseados em anomalia. Argumenta que altas taxas de falsos positivos são
> o principal obstáculo para adoção prática de IDS baseados em ML. Justifica o foco em
> FPR como métrica principal.

**[Welford1962]**
Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and
Products." *Technometrics*, 4(3):419-420.

> Propõe o algoritmo online para cálculo incremental de variância que é numericamente
> estável (evita catastrophic cancellation). Base para a adaptação A1 do MicroTEDAclus
> corrigido. Mantém running estimates de média e variância atualizadas a cada novo
> ponto, sem necessidade de armazenar pontos anteriores.

**[Zimek2012]**
Zimek, A., Schubert, E., Kriegel, H.-P. (2012). "A survey on unsupervised outlier
detection in high-dimensional numerical data." *Statistical Analysis and Data Mining*,
5(5):363-387.

> Survey definitivo sobre detecção de outliers em alta dimensionalidade. Cataloga os
> efeitos da curse of dimensionality sobre diferentes famílias de algoritmos e técnicas
> de mitigação (subspace methods, ensemble methods, dimensionality reduction). Contexto
> teórico para entender por que alta dimensionalidade é problemática para detecção de
> anomalias em geral.

---

*Este documento foi preparado como guia acadêmico para os 3 experimentos críticos do
paper SoftCom 2026. Deve ser lido integralmente antes de executar os experimentos.*
