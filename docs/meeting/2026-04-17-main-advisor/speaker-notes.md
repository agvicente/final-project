# Speaker Notes — Reunião com Orientador Principal (2026-04-17)

> **Como usar:** Deixe este arquivo aberto lado a lado com os slides. Cada slide tem (1) **ponto central** — a única coisa que você precisa comunicar; (2) **fala** — o que dizer em ~2-3 min; (3) **âncoras** — números e citações para não esquecer; (4) **se perguntarem** — respostas prontas para as perguntas mais prováveis.
>
> **Tempo total estimado:** ~25-30 min apresentação + Q&A
>
> **Princípio:** seja direto. O orientador já conhece o domínio — não gaste tempo introduzindo conceitos básicos. Foco na contribuição técnica (bug dimensional) e nas decisões experimentais.

---

## Slide 1 — Contexto e Objetivo

**Ponto central:** Situar o volume experimental, a contribuição técnica e o que você precisa desta reunião.

**Fala:**
- "Bom dia. Vou apresentar o progresso da Fase 2 do mestrado — detecção de intrusão em IoT com MicroTEDAclus em alta dimensão."
- "São 167 experimentos em 4 campanhas. A defesa está prevista para maio — faltam ~4 semanas."
- "A contribuição técnica principal é um diagnóstico e correção de falhas do MicroTEDAclus em alta dimensionalidade — 5 adaptações validadas empiricamente."
- "Preciso de duas coisas desta reunião: validar se essa contribuição é suficiente para a dissertação, e decidir quais experimentos priorizar no tempo restante."

---

## Slide 2 — TEDA: Framework Base (Angelov 2014)

**Ponto central:** TEDA é a base teórica. Tem três propriedades boas para streaming e uma limitação crítica que motiva o MicroTEDAclus.

**Fala:**
- "TEDA é o framework do Angelov 2014. A ideia central é a eccentricidade: xi(x) = 1/k + ||x - mu||² / (k * sigma²)."
- "Intuição: é um Z-score generalizado que não assume gaussianidade. Usa apenas a estrutura espacial mútua dos dados."
- "Três propriedades que fazem ele adequado para streaming: não-paramétrico — não assume distribuição; O(1) por ponto via atualização incremental de Welford; e teste de Chebyshev que vale para qualquer distribuição."
- "Os parâmetros são: m — número de desvios no teste de Chebyshev; sigma² — variância via Welford; k — contador de amostras."
- "Limitação crítica: um único centro global. Outliers contaminam as estatísticas. Não consegue separar padrões distintos. É o que motiva a extensão para múltiplos micro-clusters."

**Âncoras:** Angelov (2014) JAMRIS 8(2) | Welford (1962) Technometrics | Chebyshev: P(|X - mu| >= m*sigma) <= 1/m², vale para QUALQUER distribuição.

**Se perguntarem "o que é tipicalidade?":**
- tau(x) = 1 - xi(x). Complemento da eccentricidade.
- tau aproximadamente 1 significa muito típico. tau <= 0 significa outlier.

---

## Slide 3 — MicroTEDAclus: Motivação e Mecanismo (Maia 2020)

**Ponto central:** MicroTEDAclus resolve a contaminação com múltiplos micro-clusters isolados. Validado apenas em 2-6 dimensões — nunca testado em alta dimensão.

**Fala:**
- "Maia et al. 2020 estendeu o TEDA para N micro-clusters, cada um com suas estatísticas isoladas: mu_i, sigma²_i, S_i."
- "Fluxo de decisão: ponto chega, calcula zeta para cada micro-cluster i, aplica teste de Chebyshev. Se algum aceita, atualiza SÓ o de maior tipicalidade. Se nenhum aceita, cria novo cluster — e marca como anomalia."
- "O threshold é dinâmico: m(k) = 3 / (1 + e^{-0.007(k-100)}). Clusters jovens são permissivos — k=1 dá m aproximadamente 0.6. Com k tendendo ao infinito, m converge para 3, equivalente a 89% Chebyshev."
- "K é automático. Concept drift é nativo — novos padrões criam novos clusters."
- "Ponto crucial: o algoritmo foi validado em dados de 2 a 6 dimensões. Nunca testado em alta dimensão. É exatamente aí que o problema aparece."

**Se perguntarem "por que atualizar só o melhor, não todos?":**
- Evita convergência de clusters — winner-take-all (Kohonen 1990)
- É o padrão em stream clustering (DenStream 2006, NS-TEDA 2024)
- Update-all causa perda de diversidade

**Se perguntarem "como lida com concept drift?":**
- Novos padrões são rejeitados por todos os clusters existentes e criam novo cluster
- Clusters antigos podem ser desativados se não recebem mais pontos
- Não validei experimentalmente — item para trabalhos futuros

---

## Slide 4 — O Bug: Auto-Cancelamento + 3 Falhas

**Ponto central:** O teste principal de Chebyshev é auto-consistente em qualquer dimensão — o fator (2/d)² cancela. Mas três outros caminhos do código colapsam porque NÃO têm esse cancelamento.

**Fala:**
- "Esse slide é o coração da contribuição técnica. Vou fazer devagar."
- "Primeiro, um ponto importante: o paper de Maia 2020 descreve a atualização de variância com ||x - mu||² — sem nenhum fator de escala. Mas o código publicado no package evolclustering usa uma fórmula diferente: (||delta|| * 2/d)². Essa discrepância entre paper e implementação é a raiz do problema. Não sabemos se foi erro de codificação ou decisão não documentada — o efeito é o mesmo."
- "Agora, o que descobrimos: esse fator (2/d)² é dormente em baixa dimensão e catastrófico em alta. Em d=2, (2/2)² = 1 — neutro, a fórmula do código é equivalente ao paper. Em d=17, (2/17)² = 0.014 — subestima a variância em 70x."
- "Mas a história tem uma reviravolta. O teste principal de Chebyshev funciona corretamente em qualquer dimensão. A explicação: a fórmula da eccentricidade é xi = 1/n + (norm*2/d)² / (n * var). A variância var já carrega o fator (2/d)² da sua própria acumulação. Então dentro da fração, (2/d)² aparece no numerador E no denominador e se cancela. Não é uma operação separada — o cancelamento acontece dentro da própria fórmula de xi."
- "O problema está em três outros caminhos do código que usam sigma² ou sqrt(sigma²) FORA da fórmula de xi — sozinhos, comparando com valores em escala real. Aí não há fração interna para cancelar, e a subestimação de 70x aparece inteira."
- "Falha 1 — intersecção de macro-clusters. A condição de merge compara dist(mu_i, mu_j), que é Euclidiana real, com sqrt(sigma²), que está escalado por (2/d)². Em d=17, o raio é ~12% do real. Clusters que deveriam mergir nunca se encontram."
- "Falha 2 — guard para n<3. Clusters jovens usam sigma² escalado comparado com r0 fixo em escala real. Clusters morrem antes de crescer."
- "Falha 3 — decaimento de vida. sqrt(sigma²) e distância em escalas diferentes. Morte errática."
- "A tabela na parte inferior resume o ponto central. Em d=2, todas as linhas mostram OK — o fator é 1, o bug é invisível. Em d=17, a primeira linha — Chebyshev — continua verde porque tem simetria interna. Mas as três linhas seguintes ficam vermelhas: intersecção com raio 12% do real, guard descalibrado, life decay com escalas misturadas."
- "A cascata desses três: proliferação de clusters, rejeição prematura, FPR catastrófico de 54%."
- "O que fizemos: trouxemos a implementação de volta ao que o paper descreve — usando Welford, que é o algoritmo padrão para variância online estável. Não inventamos uma fórmula nova. Corrigimos a divergência entre paper e código, usando um método numericamente superior."

**Âncoras:** Paper diz ||x-mu||², código usa (||delta||*2/d)² — discrepância | d=2: (2/2)²=1 (dormente) | d=17: (2/17)²=0.014 (70x) | d=32: (2/32)²=0.004 (250x) | FPR original: 54.4% | Welford (1962), Chan (1983).

**Se perguntarem "por que o Maia não encontrou o bug?":**
- Testou com dados de 2 a 6 dimensões. Em d=2, (2/2)² = 1 — a fórmula do código é idêntica à do paper. O bug só se manifesta em d>10 e se torna catastrófico em d=17+.

**Se perguntarem "e vocês confirmaram que é discrepância, não decisão deliberada?":**
- O paper não menciona nenhum fator de escala dimensional. O commit history do evolclustering não documenta a motivação. Independente da intenção, o efeito em alta dimensão é destrutivo — e a correção via Welford é estritamente superior: implementa a equação do paper com estabilidade numérica garantida.

**Se perguntarem "mas por que o código tem esse fator 2/d?":**
- Não há documentação. Minha hipótese: o TEDA original (Angelov 2014) trabalha com eccentricidade escalar — uma feature por vez. Quando Maia adaptou para vetores multidimensionais, precisou colapsar o vetor δ em um escalar. O fator 2/d pode ser uma tentativa de normalização dimensional — uma "distância média por dimensão" — para manter compatibilidade com a formulação escalar. Em d=2 o fator é 1 e funciona perfeitamente. O problema é que essa normalização não preserva a escala absoluta da variância, que é necessária nos 3 code paths que usam var fora da eccentricidade.

**Se perguntarem "então o Welford implementa a mesma coisa que o paper?":**
- Sim, na essência. O paper descreve uma fórmula recursiva para variância. O Welford é o algoritmo padrão para a mesma operação — dot(delta_pre, delta_post) acumula ||x-mu||² de forma numericamente estável. Convergem para o mesmo valor. A diferença é que Welford não sofre cancelamento catastrófico em streams longos (Chan 1983).

---

## Slide 5 — As 5 Adaptações (tabela)

**Ponto central:** 5 adaptações independentes, cada uma com suporte teórico. Togláveis para ablation study (8 variantes V0-V7).

**Fala:**
- "A tabela resume as 5 correções. Cada uma é independente e togável — são 8 variantes possíveis, V0 a V7, para o ablation study."
- "Adaptação 1 — Variância: substitui (norm * 2/d)² por Welford incremental. É a correção do fator 70x. Suporte: Chan 1983 e Welford 1962."
- "Adaptação 2 — Eccentricidade: ||diff||² consistente com Welford. Garante que numerador e denominador do Chebyshev usem a mesma escala."
- "Adaptação 3 — Update seletivo: atualiza só o cluster de maior tipicalidade, não todos. Reduz número de clusters em ~20%. Suporte: Kohonen 1990, NS-TEDA 2024."
- "Adaptação 4 — Guard n=1: threshold permissivo de 13 desvios para cluster recém-criado. Permite que seeds sobrevivam à primeira amostra. Suporte: Reynolds 2009."
- "Adaptação 5 — Guard n=2: condição dupla para segunda amostra. Reduz splits prematuros em ~20%. Suporte: Reynolds 2009."
- "O ablation study com V0-V7 está planejado para isolar a contribuição de cada adaptação individualmente."

**Âncoras:** Welford (1962) | Chan et al. (1983) | Kohonen (1990) | Reynolds (2009) | NS-TEDA (2024) | 8 variantes V0-V7.

---

## Slide 6 — Pipeline Streaming: Por Que Alta Dimensão Importa

**Ponto central:** Agora que o orientador entende o bug e as adaptações, mostrar ONDE isso se aplica — o pipeline de streaming com d=17 features.

**Fala:**
- "Agora que vimos a teoria e o problema, vou mostrar onde isso se aplica na prática."
- "O pipeline: PCAPs do CICIoT2023 entram via Kafka, são reconstruídos em flows pelo FlowConsumer, e cada flow gera um vetor de 17 features. São essas 17 dimensões que expõem o bug que acabamos de ver."
- "Além do per-flow, temos o WindowAggregator que agrega flows por IP em janelas temporais — 12 a 19 features por janela."
- "Avaliação prequential — test-then-train. Sem data leakage, mas exige detector O(1) online."
- "O insight central: o detector encontra outliers estatísticos, não ataques. Um flow DDoS individual é estatisticamente idêntico a um flow benigno — Sommer e Paxson 2010. Janelas temporais mudam a pergunta: de 'este flow é anômalo?' para 'este IP está se comportando de forma anômala?'"

**Âncoras:** Sommer & Paxson (2010) — gap semântico | Lakhina (2004, 2005) — análise agregada | 17 features/flow, 12-19/janela.

**Se perguntarem "por que Kafka?":**
- Reprodutibilidade, desacoplamento, compatibilidade com produção

**Se perguntarem "por que 17 features?":**
- Padrão da literatura. Testamos 25 e 32 — zero impacto (C02-S2)

---

## Slide 7 — Evidência: C04 + Resultados + Gap Semântico

**Ponto central:** As adaptações funcionam (14x melhor FPR), os resultados absolutos são modestos, e o gap semântico é real.

**Fala:**
- "Três blocos de evidência."
- "Primeiro, C04 — 30 runs comparando implementação adaptada vs original. FPR benigno: 3.9% vs 54.4%. Melhoria de 14x. O padrão é consistente em TODAS as granularidades: flow, janela v1, janela v2."
- "Segundo, melhores resultados por ataque nos 167 experimentos. Recon F1=43.7%, DDoS-ICMP 50%, DDoS-SYN 54% mas com FPR de 33%. DDoS-TCP é 0% em todas as configurações — fundamentalmente indetectável."
- "Terceiro, o gap semântico é real e quantificável. Em C01, a taxa de anomalia é ~3.5% com ou sem ataque — o detector encontra outliers estatísticos, não ataques. Janelas temporais melhoram Recall em 10-20x, mas o FPR sobe junto."
- "Lacuna honesta: não há baseline não-supervisionado publicado no CICIoT2023. Kitsune é o candidato mais citado, mas rodou em outros datasets. Posso adaptar Isolation Forest e OC-SVM da Fase 1 em 1-2 dias para produzir comparação direta."

**Âncoras:** FPR 3.9% vs 54.4% (14x) | Recon F1=43.7% | taxa anomalia ~3.5% invariante | Sommer & Paxson (2010).

**Se perguntarem "qual o baseline?":**
- Kitsune (Mirsky 2018 NDSS) é o candidato mais citado, mas nunca rodou no CICIoT2023 — comparação seria indireta
- Tenho IF e OC-SVM da Fase 1 em modo batch — adaptar para prequential leva 1-2 dias
- Arp et al. (2022) argumentam que documentar lacunas é parte do rigor — prefiro admitir a ausência a apresentar extrapolação

---

## Slide 8 — Experimentos a Executar

**Ponto central:** 5 experimentos priorizados. Tempo total ~2 semanas. Preciso decidir o que priorizar em 4 semanas.

**Fala:**
- "Tenho 5 experimentos mapeados, priorizados por impacto na dissertação."
- "Prioridade alta, em vermelho: dimensional sweep com d variando em 2, 5, 10, 17, 20 e 50. Estimativa de 2-3 dias. Produz a curva de colapso para o paper — é a evidência visual mais forte da contribuição."
- "Também prioridade alta: ablation V0-V7. Mais 2-3 dias. Isola a contribuição de cada adaptação individualmente."
- "Prioridade média, em amarelo: S5 Two-Stage — per-flow seguido de concentração por IP. Estimativa de 3-5 dias. Ataca diretamente o trade-off FPR vs Recall."
- "Também prioridade média: baseline com IF e OC-SVM prequential. 1-2 dias. Produz a comparação direta que falta."
- "Prioridade baixa, em verde: KS-test e Spearman para validar o framework de dimensões anômalas. 2-3 dias."
- "Total: ~2 semanas de trabalho experimental. A pergunta é: quais desses cabem nas 4 semanas restantes junto com a escrita?"

**Âncoras:** ~2 semanas total de experimentos | 4 semanas até defesa.

---

## Slide 9 — Decisões para o Orientador

**Ponto central:** Três perguntas concretas que precisam de resposta HOJE. Ser honesto sobre o que é e o que não é contribuição.

**Fala:**
- "Chegando na decisão. Antes das perguntas, quero ser transparente sobre o que considero contribuição e o que não é."
- "A correção da variância em si NÃO é contribuição. O paper descreve ||x-mu||², o código usa (||delta||*2/d)². Corrigir isso é bug fix — é errata, não ciência nova."
- "O que considero contribuição são as camadas ao redor: primeiro, a análise de por que o bug é dormente em baixa dimensão e catastrófico em alta — o auto-cancelamento no Chebyshev e os 3 code paths que falham. Ninguém na comunidade TEDA identificou isso. Segundo, as 4 adaptações adicionais — update seletivo, guards n=1 e n=2 — que não estão no paper original e são necessárias para o contexto de IDS. Terceiro, o primeiro estudo empírico de MicroTEDAclus em dados de rede de alta dimensão — 167 experimentos, 4 campanhas, 5 ataques."
- "Dito isso, as três perguntas."
- "Primeira: essa análise + as 4 adaptações genuinamente novas + o estudo empírico constituem contribuição suficiente? Ou preciso de mais resultados experimentais?"
- "Segunda: prioridade. Caminho A: sweep dimensional d={2,5,10,17,20,50} + ablation V0-V7 — produz a curva de colapso e isola cada adaptação. Fortalece a análise como contribuição. Caminho B: S5 Two-Stage — melhora resultados absolutos. Os dois não cabem junto com a escrita."
- "Terceira: escopo realista em 4 semanas. Paper focado na análise dimensional + dissertação cobrindo o percurso completo. É viável?"
- "Minha preferência é pelo caminho A — porque a curva de colapso com significância estatística é o que transforma a análise de observação em contribuição validada."

**Se perguntarem "então sem o sweep a contribuição é fraca?":**
- Honestamente, sim. Hoje tenho a análise teórica (auto-cancelamento) e a evidência empírica pontual (d=17 em C04). O sweep com d={2,5,10,17,20,50} e testes estatísticos (ANOVA, Tukey) transformaria isso em contribuição robusta — mostraria a curva exata de degradação e em qual d o colapso começa.

**Se perguntarem "e os resultados de detecção? F1=43.7% é suficiente?":**
- Isoladamente, não competitivo com métodos supervisionados (F1>95%). Mas o framing correto é: Recon 43.7% é o melhor resultado não-supervisionado que conseguimos, e documenta com rigor ONDE e POR QUE anomaly detection falha — que é contribuição válida segundo Sommer & Paxson 2010 e Arp et al. 2022.

---

## Cartão de Emergência — Se Esquecer Tudo

**Os 5 fatos que você DEVE ter na ponta da língua:**

1. **167 experimentos, 4 campanhas, 5 adaptações** — 1 é bug fix (variância), 4 são genuinamente novas
2. **A correção da variância é bug fix, não contribuição** — paper diz ||x-mu||², código diverge com (2/d)². A contribuição é a ANÁLISE (auto-cancelamento + 3 code paths) e as 4 adaptações novas
3. **FPR 3.9% (adaptado) vs 54.4% (original) — 14x** — consistente em TODAS as configurações do C04
4. **(2/17)² = 0.014** — Chebyshev auto-cancela, falhas são nos 3 code paths assimétricos
5. **DDoS-TCP = 0%** em tudo — gap semântico real (Sommer & Paxson 2010)

**As 3 citações que você DEVE conseguir mencionar sem pensar:**

1. **Sommer & Paxson (2010)** "Outside the Closed World" IEEE S&P — gap semântico anomalia/ataque
2. **Maia et al. (2020)** "Evolving Clustering Based on Mixture of Typicalities" FGCS 106 — o algoritmo principal
3. **Angelov (2014)** "Autonomous Learning Systems" JAMRIS 8(2) — o framework TEDA

**Se travar em qualquer pergunta, respire e diga:**
> "Boa pergunta. Vou pensar um segundo... [pausa de 3 segundos]... Acho que o caminho seria X, mas confesso que preciso validar. Posso voltar nesse ponto depois de checar os dados?"

Isso é sempre melhor do que inventar uma resposta.
