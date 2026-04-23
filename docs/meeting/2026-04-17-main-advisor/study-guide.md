# Guia de Estudo — Reuniao com Orientador Principal (17/abr)

> **Proposito:** Guia aprofundado para atingir fluencia na explicacao do bug dimensional do MicroTEDAclus. Material de preparacao — nao e para apresentar.
>
> **Como usar:** Leia secao por secao. Em cada secao, ha fontes locais (com path:linha) e fontes externas (com citacao completa). Apos ler, faca os exercicios da S7.

---

## S1 — A Formula Original vs Welford

### A formula original (Maia 2020)

A formula de atualizacao de variancia no codigo original do MicroTEDAclus e:

```
var = ((n-1)/n) * var_old + ((||delta|| * 2/d)^2) / (n-1)
```

onde `delta = x - mean`, `d` = numero de dimensoes, `n` = numero de amostras no cluster.

O termo critico e `(||delta|| * 2/d)^2`. Essa expressao colapsa a norma do vetor d-dimensional em um escalar, e depois aplica o fator `(2/d)^2`. Para um vetor unitario por dimensao (`delta = [1,1,...,1]`):

```
||delta|| = sqrt(d)
(||delta|| * 2/d)^2 = (sqrt(d) * 2/d)^2 = (2/sqrt(d))^2 = 4/d
```

Portanto a contribuicao de variancia **converge para `4/d`** independente da variancia real dos dados:

| Dimensao d | `(2/d)^2 * ||delta||^2` | Variancia real `sum(delta_i^2)` | Razao |
|------------|-------------------------|-------------------------------|-------|
| 2          | `4*2/4 = 2.0`           | `1^2 + 1^2 = 2.0`            | 1.0x  |
| 5          | `4*5/25 = 0.80`         | `5.0`                         | 6.3x  |
| 10         | `4*10/100 = 0.40`       | `10.0`                        | 25x   |
| **17**     | **`4*17/289 = 0.235`**  | **`17.0`**                    | **72x** |
| 50         | `4*50/2500 = 0.08`      | `50.0`                        | 625x  |

Em d=17 (nosso caso com features de rede), a variancia e **~72x menor** que o valor real. Em vez de `trace(Sigma) ~ 17`, o algoritmo computa `~ 0.235`.

### A formula de Welford (correcao)

```python
n += 1
delta_pre  = x - mean           # antes de atualizar a media
mean       = mean + delta_pre / n
delta_post = x - mean           # depois de atualizar a media
var_sum   += dot(delta_pre, delta_post)
variance   = var_sum / (n - 1)
```

O produto escalar `dot(delta_pre, delta_post)` acumula contribuicoes de **todas as dimensoes** independentemente. Algebricamente:

```
var_sum = sum_k sum_j (x_k_j - mu_{k-1}_j)(x_k_j - mu_k_j)
        = sum_j S_j
```

onde `S_j / (n-1) = var_j` (variancia amostral da dimensao j). Portanto:

```
variance = var_sum / (n-1) = sum_j var_j = trace(Sigma)
```

Para dados isotropicos com variancia unitaria por dimensao: `variance ~ d`. Isso e exatamente o que esperamos.

### Exemplo numerico: `delta = [1,1,...,1]`

**Em d=2:**
- Original: `(sqrt(2) * 2/2)^2 = (sqrt(2))^2 = 2.0`
- Welford: `dot([1,1], [1,1]) = 2.0` (simplificado para primeira atualizacao)
- Ambas corretas! O fator `(2/2)^2 = 1` nao altera nada.

**Em d=17:**
- Original: `(sqrt(17) * 2/17)^2 = 4*17/289 = 0.235`
- Welford: `dot([1,...,1], [1,...,1]) = 17.0` (simplificado)
- Razao: `17.0 / 0.235 = 72x` de subestimacao!

### Fontes locais

- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/original.py:156-166` — formula original (implementacao fiel)
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:33-47` — acumulacao Welford no `CorrectedMicroCluster.update()`
- `~/mestrado/teda-high-dim/tests/test_original_variance.py:36-51` — `test_2d_vs_17d_ratio`: prova que `var = 4/d`
- `~/mestrado/teda-high-dim/tests/test_welford_variance.py:56-68` — `test_variance_scales_linearly_with_d`: prova que `var ~ d`
- `~/mestrado/final-project/experiments/streaming/src/detector/micro_teda.py:305-308` — Welford no pipeline real de streaming
- `~/mestrado/final-project/docs/meeting/advisor-meeting-study-guide.md:218-277` — demonstracao numerica completa

### Fontes externas

- Welford, B. P. (1962) "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419-420.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983) "Algorithms for Computing the Sample Variance: Analysis and Recommendations." The American Statistician 37(3):242-247.

---

## S2 — O Auto-Cancelamento do Chebyshev (Prova Central)

Este e O insight central. Apesar da variancia estar errada em termos absolutos, o **teste principal de anomalia funciona por acidente** porque a variancia aparece **dentro** da formula da eccentricidade, no denominador — e como o numerador tambem carrega o fator `(2/d)^2`, os dois se cancelam na divisao interna. Nao e uma operacao `ecc/var` separada — o cancelamento acontece dentro da propria formula de xi.

### A formula da eccentricity

```
xi = 1/n + (||a|| * 2/d)^2 / (n * var)
```

onde `a = mean - x`, `d = dim(x)`, `var` = variancia acumulada pela formula original.

### A formula da variancia (original)

A variancia tambem usa `(2/d)^2` na sua acumulacao:

```
var ~ (2/d)^2 * var_true    (para n grande)
```

### O cancelamento algebrico

Defina `alpha = (2/d)^2`. Entao:

```
var_original ~ alpha * var_true
```

Agora substitua na eccentricity:

```
xi = 1/n + (||a|| * 2/d)^2 / (n * var_original)
   = 1/n + alpha * ||a||^2 / (n * alpha * var_true)
   = 1/n + ||a||^2 / (n * var_true)
```

**O `alpha` cancela!** A eccentricity resultante e a mesma que seria com variancia correta.

### Passo a passo detalhado

1. O numerador do segundo termo e: `(||a|| * 2/d)^2 = alpha * ||a||^2`
2. O denominador e: `n * var_original ~ n * alpha * var_true`
3. A razao fica: `alpha * ||a||^2 / (n * alpha * var_true) = ||a||^2 / (n * var_true)`
4. Portanto: `xi = 1/n + ||a||^2 / (n * var_true)` — independente de d!

### Consequencia para o teste de Chebyshev (n >= 3)

O teste de outlier para n >= 3 usa:

```
norm_ecc = xi / 2
outlier se norm_ecc > (m(k)^2 + 1) / (2*k)
```

Como `xi` e auto-consistente (o alpha cancelou), o teste de Chebyshev produz decisoes corretas. O limiar `(m(k)^2 + 1) / (2*k)` depende apenas de `k` (tamanho do cluster), nao de `d`.

### Walkthrough numerico: d=17, n=100

Suponha um cluster com n=100 amostras de uma Gaussiana isotropica (var=1.0 por dimensao):

- `var_true = trace(Sigma) = 17.0` (soma das variancias)
- `var_original = alpha * var_true = (4/289) * 17 = 0.235`

Para um ponto teste x com `||a|| = 5.0`:

**Com variancia original:**
```
xi = 1/100 + (5.0 * 2/17)^2 / (100 * 0.235)
   = 0.01 + (0.588)^2 / 23.5
   = 0.01 + 0.346 / 23.5
   = 0.01 + 0.0147
   = 0.0247
```

**Com variancia correta (Welford):**
```
xi = 1/100 + 25.0 / (100 * 17.0)
   = 0.01 + 25.0 / 1700
   = 0.01 + 0.0147
   = 0.0247
```

**Resultado identico!** O alpha se cancela numericamente tambem.

O teste de Chebyshev em n=100:
- `m(100) = 3 / (1 + exp(-0.007*(100-100))) = 3/2 = 1.5`
- `threshold = (1.5^2 + 1) / (2*100) = 3.25/200 = 0.01625`
- `norm_ecc = 0.0247/2 = 0.01235`
- `0.01235 < 0.01625` — **nao e outlier** (decisao correta em ambos os casos)

### Fontes locais

- `~/mestrado/teda-high-dim/tests/test_original_variance.py:86-124` — `test_self_cancellation_at_high_d`: verifica que eccentricity media nao cresce com d
- `~/mestrado/teda-high-dim/README.md:15-22` — resumo do finding ("self-cancels in the eccentricity ratio for n >= 3")
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/original.py:168-182` — formula da eccentricity original

### Fontes externas

- Angelov, P. (2014) "Outside the box: an alternative data analytics framework." JAMRIS 8(2):53-68.
- Maia, J. et al. (2020) "Evolving clustering algorithm based on mixture of typicalities." FGCS 106:13-26.

### Resumo para falar de cabeca:

"A formula original usa (2/d)^2 em dois lugares: na variancia e na eccentricidade. A eccentricidade tem a variancia no denominador e (norm*2/d)^2 no numerador — como ambos carregam o fator (2/d)^2, eles se cancelam dentro da propria formula. Por isso o teste de Chebyshev funciona por acidente. Mas tres code paths usam a variancia sozinha — comparando com distancia real ou com r0 fixo — e ai o (2/d)^2 sobra sem cancelar. E ai que os 70x de subestimacao destroem o algoritmo."

---

## S3 — Os 3 Code Paths Que NAO Cancelam

### Visao unificada: por que d=2 esconde tudo

Em d=2, `(2/d)² = 1` — o fator e neutro. **Todas** as formulas do codigo produzem o mesmo numero que o paper. Nao importa se o Chebyshev cancela ou nao, porque nao ha nada para cancelar. O bug e literalmente invisivel.

Em d>2, os paths divergem. Somente o Chebyshev (n>=3) sobrevive, porque tem simetria interna:

```
                            d=2          d=17
                         (2/d)²=1     (2/d)²=0.014
                         ─────────    ─────────────

  Chebyshev (n>=3)        OK            OK (auto-cancela)

  Interseccao MC          OK            FALHA: raio 12% do real
  Guard n<3               OK            FALHA: var descalibrada vs r0
  Life decay              OK            FALHA: escalas misturadas
```

**Resumo em uma frase:** "Em d=2 o bug e invisivel. Em d=17, o teste principal sobrevive por simetria interna, mas tres mecanismos de gerenciamento de clusters colapsam porque usam a variancia de forma assimetrica."

---

Para cada path: a equacao, POR QUE nao cancela, e demonstracao numerica com d=17.

### 1 — Interseccao de macro-clusters

**Equacao:**
```
interseccao se dist(mu_i, mu_j) < 2 * (sqrt(var_i) + sqrt(var_j))
```

**Por que nao cancela:**
- `dist(mu_i, mu_j)` usa distancia Euclidiana **verdadeira** (sem fator de escala)
- `sqrt(var_i)` usa a variancia **escalada** (subestimada por alpha)
- Nao ha cancelamento: o lado esquerdo esta na escala real, o lado direito esta comprimido

**Demonstracao numerica (d=17):**

Suponha dois clusters com mesma variancia real `var_true = 17.0`, centros separados por `dist = 6.0`:

Com Welford:
```
raio = 2 * (sqrt(17.0) + sqrt(17.0)) = 2 * 2 * 4.12 = 16.49
6.0 < 16.49 → INTERSECTA (correto — clusters sobrepostos)
```

Com formula original:
```
var_original = 0.235
raio = 2 * (sqrt(0.235) + sqrt(0.235)) = 2 * 2 * 0.485 = 1.94
6.0 < 1.94 → NAO INTERSECTA (errado!)
```

O raio original e `1.94 / 16.49 = 12%` do raio correto. Clusters que deveriam ser fundidos **nunca se encontram**. Isso causa proliferacao de micro-clusters redundantes.

**Comentario no codigo:** "CRITICAL: dist uses TRUE Euclidean distance, but sqrt(var) uses the scaled-down variance."

### Fontes locais

- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/original.py:208-219` — `has_intersection()` com comentario CRITICAL

### 2 — Guard n < 3 (cluster jovem)

**Equacao:**
```
outlier se var > r0    (para n < 3)
```

onde `r0 = 0.001` e um threshold fixo.

**Por que nao cancela:**
- `var` e escalada por `alpha = (2/d)^2`
- `r0` e uma constante fixa, nao depende de `d`
- Nao ha fator `(2/d)^2` no lado direito para cancelar

**Demonstracao numerica (d=17):**

Para um cluster jovem (n=2) com variancia real moderada:

Com Welford:
```
var = 5.0 (trace de um cluster com 2 pontos separados)
5.0 > 0.001 → outlier (cria novo cluster)
```

Com formula original:
```
var_original = (4/17) * 5.0 = 1.18... nao, aqui o calculo e direto:
var_original = (||delta|| * 2/17)^2 / (n-1)
```

Na pratica, `var_original = 0.235 * var_factor`. O valor ainda e > r0=0.001 na maioria dos casos (0.235 > 0.001). **Entao o guard geralmente nao falha catastroficamente** — o problema e mais sutil: quando a variancia real ja e pequena (cluster novo com 2 pontos proximos), `var_original = alpha * var_small` pode ser minuscula, levando a aceitacao prematura de pontos que deveriam ser rejeitados, ou rejeicao erratica dependendo da escala.

O efeito principal: a decisao n<3 fica **descalibrada** — o threshold r0 foi sintonizado implicitamente para d=2 e perde sentido em d=17.

### Fontes locais

- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/original.py:191-204` — `_is_outlier()` com guard n<3

### 3 — Life decay (decaimento de vida)

**Equacao:**
```
life += (sqrt(var) - dist) / sqrt(var) * fading_factor
```

**Por que nao cancela:**
- `sqrt(var)` usa variancia escalada → raio comprimido
- `dist = ||x - mean||` usa distancia Euclidiana verdadeira
- A razao `(sqrt(var) - dist) / sqrt(var)` mistura escalas

**Demonstracao numerica (d=17):**

Para um ponto x a distancia `dist = 3.0` do centro de um cluster com `var_true = 17.0`, `fading = 0.01`:

Com Welford:
```
sqrt(17.0) = 4.12
life += (4.12 - 3.0) / 4.12 * 0.01 = 0.272 * 0.01 = +0.00272
→ Ponto dentro do raio, life AUMENTA (correto)
```

Com formula original:
```
sqrt(0.235) = 0.485
life += (0.485 - 3.0) / 0.485 * 0.01 = (-5.18) * 0.01 = -0.0518
→ Ponto "fora" do raio comprimido, life DIMINUI (errado!)
```

O ponto esta claramente dentro da distribuicao real do cluster, mas o raio comprimido faz parecer que esta longe. Resultado: **clusters morrem prematuramente**, acelerando a proliferacao de novos clusters e inflando o FPR.

**Comentario no codigo:** "CRITICAL: Uses sqrt(variance) as radius (scaled-down) but Euclidean distance to mean (true scale). Mixed scales."

### Fontes locais

- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/original.py:251-265` — `_update_life()` com comentario CRITICAL

### Resumo para falar de cabeca:

"O teste principal (Chebyshev) funciona por acidente porque o fator (2/d)^2 se cancela. Mas tres code paths nao tem esse cancelamento: a interseccao de clusters compara distancia real com raio comprimido, o guard n<3 compara variancia escalada com threshold fixo, e o life decay mistura distancia real com raio escalado. Esses tres juntos causam clusters que nunca se fundem, morrem prematuramente, e proliferam sem controle."

---

## S4 — As 5 Adaptacoes em Detalhe

### Adaptacao 1 — Variancia Welford

| Aspecto | Detalhe |
|---------|---------|
| **O que muda** | `(||delta|| * 2/d)^2` substituido por `dot(delta_pre, delta_post)` |
| **Por que** | Elimina subestimacao de 72x em d=17 — computa `trace(Sigma)` correto |
| **Equacao** | `var_sum += dot(x - mean_old, x - mean_new); var = var_sum / (n-1)` |
| **Impacto** | Variancia correta → interseccao, life decay e guard calibrados |
| **Suporte teorico** | Welford (1962), Chan et al. (1983) |

**Codigo:**
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:33-47` — `CorrectedMicroCluster.update()`
- `~/mestrado/final-project/experiments/streaming/src/detector/micro_teda.py:305-308` — Welford no pipeline real

### Adaptacao 2 — Eccentricity consistente

| Aspecto | Detalhe |
|---------|---------|
| **O que muda** | `(||diff|| * 2/d)^2` na eccentricity substituido por `||diff||^2` (`sum(diff**2)`) |
| **Por que** | Quando var usa Welford (escala real), ecc precisa usar escala real tambem |
| **Equacao** | `ecc = 1/n + sum((x-mean)^2) / (n * var_welford)` |
| **Impacto** | Sem isso, ecc ficaria sistematicamente baixa (fator alpha no numerador, sem alpha no denominador) |
| **Suporte teorico** | Consistencia interna: se var = trace(Sigma), ecc deve usar ||diff||^2, nao alpha*||diff||^2 |

**Codigo:**
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:225-231` — `_eccentricity()` com flag `use_consistent_eccentricity`

### Adaptacao 3 — Update seletivo (somente melhor cluster)

| Aspecto | Detalhe |
|---------|---------|
| **O que muda** | Update ALL accepting clusters → Update ONLY best (max typicality) |
| **Por que** | Original atualiza todos os clusters que aceitam o ponto, causando convergencia de centroides para a mesma regiao |
| **Equacao** | `best = argmax_i(tau_i(x))` onde `tau_i = 1 - xi_i` |
| **Impacto** | -20% clusters redundantes, clusters mais estáveis |
| **Suporte teorico** | Winner-take-all: Kohonen (1990) SOM; competitive learning: DenStream Cao et al. (2006); NS-TEDA Chen et al. (2024) |

**Codigo:**
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:147-151` — `if self.use_selective_update: best = max(...)`

### Adaptacao 4 — Guard n=1 (permissivo para singletons)

| Aspecto | Detalhe |
|---------|---------|
| **O que muda** | Sem protecao especial para n=1 → threshold fixo de 13 (equivale a m~5) |
| **Por que** | Clusters com 1 ponto nao tem variancia → qualquer ponto proximo e rejeitado → cluster morre antes de acumular estatisticas |
| **Equacao** | `if n == 1: aceita se norm_ecc <= 13.0` |
| **Impacto** | Seeds sobrevivem o suficiente para crescer, reduz cluster churn |
| **Suporte teorico** | Regularizacao de variancia: Reynolds (2009) GMM — clusters jovens precisam de priores permissivos |

**Codigo:**
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:285-286` — guard n=1

### Adaptacao 5 — Guard n=2 (condicao dual)

| Aspecto | Detalhe |
|---------|---------|
| **O que muda** | Apenas `var > r0` → condicao dual: `zeta > threshold AND var >= r0` |
| **Por que** | Original rejeita qualquer ponto quando var > r0 para n<3, sem considerar eccentricity. Dual condition requer AMBOS criterios. |
| **Equacao** | `outlier se (norm_ecc > threshold) AND (var >= r0)` para n=2 |
| **Impacto** | -20% splits espurios, clusters jovens mais estaveis |
| **Suporte teorico** | Reynolds (2009) regularizacao GMM — nao tomar decisoes definitivas com estatisticas insuficientes |

**Codigo:**
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:289-292` — guard n=2 com dual condition

### Ablation: 8 variantes V0-V7

O arquivo `variants.py` define 8 combinacoes para ablation study:

| Variante | Welford | Ecc consistente | Update seletivo | Guard n=1 | Guard n=2 | Descricao |
|----------|---------|-----------------|-----------------|-----------|-----------|-----------|
| V0 | OFF | OFF | OFF | OFF | OFF | Original (baseline) |
| V1 | **ON** | OFF | OFF | OFF | OFF | Apenas Welford |
| V2 | OFF | **ON** | OFF | OFF | OFF | Apenas ecc consistente |
| V3 | **ON** | **ON** | OFF | OFF | OFF | Welford + ecc |
| V4 | OFF | OFF | **ON** | OFF | OFF | Apenas update seletivo |
| V5 | OFF | OFF | OFF | **ON** | OFF | Apenas guard n=1 |
| V6 | OFF | OFF | OFF | OFF | **ON** | Apenas guard n=2 |
| V7 | **ON** | **ON** | **ON** | **ON** | **ON** | Full corrected |

**Codigo:** `~/mestrado/teda-high-dim/src/teda_hd/algorithms/variants.py:19-76` — definicoes V0-V7

### Resultado C04 (campanha 4)

**FPR 3.9% (adaptado) vs 54.4% (original) — melhoria de 14x.**

### Fontes locais

- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/corrected.py:1-14` — docstring com as 5 adaptacoes listadas
- `~/mestrado/teda-high-dim/src/teda_hd/algorithms/variants.py:1-106` — 8 variantes de ablation

### Fontes externas

- Welford, B. P. (1962) "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419-420.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983) "Algorithms for Computing the Sample Variance: Analysis and Recommendations." The American Statistician 37(3):242-247.
- Kohonen, T. (1990) "The self-organizing map." Proceedings of the IEEE 78(9):1464-1480.
- Cao, F., Ester, M., Qian, W., Zhou, A. (2006) "Density-Based Clustering over an Evolving Data Stream with Noise." SIAM SDM.
- Chen, Y. et al. (2024) "NS-TEDA: Evolving data analytics for streaming data." Applied Soft Computing.
- Reynolds, D. A. (2009) "Gaussian Mixture Models." Encyclopedia of Biometrics, Springer.

### Resumo para falar de cabeca:

"Sao cinco adaptacoes. A primeira (Welford) corrige a variancia de 4/d para trace(Sigma). A segunda faz a eccentricity consistente com Welford. A terceira muda o update para winner-take-all, evitando convergencia de clusters. A quarta e quinta protegem clusters jovens com n=1 e n=2 de serem destruidos antes de acumular estatisticas. A ablation com 8 variantes mostra que Welford + ecc consistente sao as mais impactantes, mas as cinco juntas dao o melhor resultado: FPR de 54% cai para 3.9%."

---

## S5 — Dormencia em d=2

### Por que o bug nunca foi detectado

A raiz e aritmetica simples:

```
Para d=2: (2/d)^2 = (2/2)^2 = 1
```

Quando `(2/d)^2 = 1`, a formula original se reduz a:

```
var = ((n-1)/n) * var_old + ||delta||^2 / (n-1)
```

que e equivalente a acumular `||delta||^2` — o mesmo que o componente escalar de Welford. Portanto, **em d=2 as formulas sao matematicamente identicas**.

### Prova por teste

```python
# test_d2_formula_equals_norm_squared
delta = np.array([3.0, 4.0])  # norm = 5
var = OriginalMicroTEDAclus.update_variance(delta, 2, 0.0)
assert var == pytest.approx(np.sum(delta**2))  # = 25
```

Em d=2, `(norm*2/2)^2 = norm^2 = sum(delta_i^2)`. Identico.

### Contexto historico

Maia et al. (2020) testaram MicroTEDAclus com dados industriais de streaming com **2-6 features**. Nessa faixa:

| d | `(2/d)^2` | Subestimacao |
|---|-----------|--------------|
| 2 | 1.00      | 1x (nenhuma) |
| 3 | 0.44      | 2.3x         |
| 4 | 0.25      | 4x           |
| 5 | 0.16      | 6.3x         |
| 6 | 0.11      | 9x           |

Ate d=6, a subestimacao e de 1 ordem de grandeza — problemática mas nao catastrofica, especialmente com o auto-cancelamento do Chebyshev. O bug **so se torna catastrofico a partir de d~10**.

### O contexto IoT/IDS

Features de rede tipicamente tem **17-40 dimensoes** (CICIoT2023 usa 46 features no CSV; nosso pipeline extrai 17 features por flow). Nenhum paper TEDA/MicroTEDAclus testou com mais de ~6 features antes deste trabalho. A dormencia do bug em baixa dimensao explica por que passou despercebido na literatura.

### Fontes locais

- `~/mestrado/teda-high-dim/tests/test_original_variance.py:70-74` — `test_d2_formula_equals_norm_squared`
- `~/mestrado/teda-high-dim/tests/test_welford_variance.py:71-94` — `TestCorrectedVsOriginalAtD2`: V0 e V7 similares em d=2

### Fontes externas

- Beyer, K. et al. (1999) "When Is 'Nearest Neighbor' Meaningful?" ICDT 1999, LNCS 1540:217-235.
- Aggarwal, C., Hinneburg, A., Keim, D. (2001) "On the Surprising Behavior of Distance Metrics in High Dimensional Space." ICDT 2001, LNCS 1973:420-434.
- Zimek, A., Schubert, E., Kriegel, H.-P. (2012) "A survey on unsupervised outlier detection in high-dimensional numerical data." Stat. Anal. Data Mining 5(5):363-387.

---

## S6 — Pipeline Streaming e o Gap Semantico

### Arquitetura do pipeline

```
PCAP (CICIoT2023)
    |
    v
PCAPProducer --> Kafka (topic: packets)
    |
    v
FlowConsumer --> Kafka (topic: flows)
    |
    |-- Extrai 17 features por flow:
    |   [duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
    |    fwd_pkt_len_mean/std/min/max, bwd_pkt_len_mean/std/min/max,
    |    flow_iat_mean/std, fwd_iat_mean]
    |
    v
StreamingDetector (MicroTEDAclus) --> Kafka (topic: alerts)
    |
    |-- Modo flow: 1 feature vector por flow → anomalia?
    |-- Modo window: WindowAggregator agrupa flows por IP em janelas temporais
    |
    v
WindowAggregator (12-19 features por janela):
    [n_flows, n_fwd, n_bwd, bytes_total, bytes_fwd, bytes_bwd,
     unique_ports, unique_protocols, entropy_ports, fwd_bwd_ratio,
     bytes_per_flow, flow_rate, ...]
    |
    v
MicroTEDAclus --> anomalia por janela de IP
```

### Protocolo prequential

- **Test-then-train:** cada ponto e primeiro testado (anomalia?), depois usado para treinar
- **Sem data leakage:** nenhum dado futuro influencia a decisao
- **O(1) por amostra:** o detector nao armazena dados historicos — requisito para streaming
- **Sem warm-up separado:** as primeiras amostras treinam e testam simultaneamente

Referencia: Gama et al. (2013) — protocolo padrao para avaliacao de streaming learners.

### O gap semantico (Sommer & Paxson 2010)

O problema fundamental: **anomaly detection encontra outliers estatisticos, nao ataques.**

Evidencia empirica da Campaign-01:
- Anomaly rate em trafego benigno: ~3.5%
- Anomaly rate em trafego benigno + ataque: ~3.5%
- O detector encontra a **mesma taxa de outliers** com e sem ataque!

Isso ocorre porque flows individuais de ataque (DDoS, Mirai) sao estatisticamente similares a flows benignos — a anomalia esta no **padrao agregado** (muitos flows similares em pouco tempo), nao no flow individual.

### Windows: mudanca de pergunta

| Granularidade | Pergunta | Recall tipico |
|---------------|----------|---------------|
| Per-flow | "Este flow e anomalo?" | 3-5% |
| Per-window | "Este IP esta se comportando anomalamente?" | 10-54% |

A agregacao temporal muda a pergunta de "flow individual e diferente?" para "o comportamento agregado deste IP e diferente?". Campaign-02 S3 mostrou:

- DDoS-SYN: 3% → 54% Recall com window 60s
- Recon-PortScan: 4% → 45% Recall com window 60s

**Mas FPR tambem explodiu:** 3.5% → 58% com window 60s. O trade-off nao esta resolvido. Isso motiva o S5 Two-Stage (flow + window) como proximo passo.

### Fontes locais

- `~/mestrado/final-project/docs/architecture/CURRENT.md:26-80` — diagrama do pipeline
- `~/mestrado/final-project/experiments/streaming/src/detector/streaming_detector.py` — fluxo de deteccao
- `~/mestrado/final-project/experiments/streaming/src/detector/window_aggregator.py` — agregacao temporal
- `~/mestrado/final-project/experiments/results/campaign-01/ANALYSIS.md` — anomaly rate invariante (~3.5%)
- `~/mestrado/final-project/experiments/results/campaign-02/ANALYSIS.md` — window breakthrough (C02-S3)
- `~/mestrado/final-project/experiments/methodology.md` — protocolo prequential

### Fontes externas

- Sommer, R. & Paxson, V. (2010) "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection." IEEE Symposium on Security and Privacy (S&P):305-316.
- Gama, J., Sebastiao, R., Rodrigues, P. P. (2013) "On evaluating stream learning algorithms." Machine Learning 90(3):317-346.
- Lakhina, A., Crovella, M., Diot, C. (2004) "Diagnosing Network-Wide Traffic Anomalies." ACM SIGCOMM Computer Communication Review 34(4):219-230.
- Li, J. et al. (2023) "Towards real-time ML-based DDoS detection in software-defined IoT networks." Sci China Info Sci 66:112101.
- Goldschmidt, P. & Kucera, S. (2024) "Windower: Feature Extraction for Real-Time DDoS Detection Using Machine Learning." NOMS 2024, IEEE.

---

## S7 — Exercicios de Fluencia

Faca cada exercicio sem consultar as secoes anteriores. Se travar, volte a secao correspondente, releia, e tente novamente.

### Exercicio 1 — Desenhar o pipeline de memoria

Em uma folha em branco, desenhe o pipeline completo:

```
PCAP → [?] → Kafka → [?] → [17 features] → [?] → [janelas por IP] → [?] → anomalia
```

Preencha os componentes: PCAPProducer, FlowConsumer, WindowAggregator, MicroTEDAclus. Indique os topics Kafka (packets, flows, alerts).

**Criterio de sucesso:** desenho correto em < 2 minutos, sem consulta.

### Exercicio 2 — Calcular var_original e var_welford

Para `delta = [1, 1, ..., 1]` (vetor de uns), calcule a contribuicao de variancia de uma unica atualizacao (n=2, var_old=0):

**(a) Em d=2:**
- `var_original = ?`
- `var_welford = ?`
- `razao = ?`

**(b) Em d=17:**
- `var_original = ?`
- `var_welford = ?`
- `razao = ?`

**Respostas esperadas:** (a) ambas = 2.0, razao = 1.0. (b) original = 0.235, Welford = 17.0, razao = 72x.

### Exercicio 3 — Cancelamento algebrico na eccentricity

Partindo de:
```
xi = 1/n + (||a|| * 2/d)^2 / (n * var)
var ~ (2/d)^2 * var_true
```

Mostre que `xi = 1/n + ||a||^2 / (n * var_true)` (3 linhas de algebra).

### Exercicio 4 — Raio de interseccao com var_original vs var_welford

Para d=17, dois clusters com `var_true = 17.0`, centros separados por `dist = 8.0`:

**(a)** Calcule o raio de interseccao `R = 2 * (sqrt(var_i) + sqrt(var_j))` com Welford.

**(b)** Calcule o mesmo raio com formula original (`var = 0.235`).

**(c)** Os clusters intersectam em cada caso?

**Respostas esperadas:** (a) R = 16.49, sim. (b) R = 1.94, nao. Original subestima raio em 88%.

### Exercicio 5 — Listar as 5 adaptacoes de memoria

Sem consultar, liste:

1. Nome da adaptacao
2. O que muda (1 frase)
3. Suporte teorico (autor + ano)

Para cada uma das 5. Depois confira com S4.

### Exercicio 6 — Explicacao oral cronometrada (2 minutos)

Configure um timer de 2 minutos. Explique em voz alta o slide 5 (bug dimensional) sem consultar nenhum material.

**Estrutura sugerida:**
1. (~20s) A formula original usa `(2/d)^2` que subestima variancia em 70x para d=17
2. (~30s) O teste principal (Chebyshev n>=3) funciona por acidente — o alpha cancela
3. (~40s) Mas tres code paths falham: interseccao, guard n<3, life decay — explicar um deles
4. (~20s) Cinco adaptacoes corrigem: Welford, ecc consistente, update seletivo, guards n=1 e n=2
5. (~10s) Resultado: FPR 54% → 3.9%, melhoria de 14x

**Criterio de sucesso:** cobrir todos os 5 pontos dentro de 2 minutos, sem hesitacao em nenhum. A frase-chave que deve aparecer naturalmente: "o teste principal funciona por acidente, mas tres code paths falham."
