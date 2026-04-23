# Speaker Notes — Reunião de Orientação

> **Como usar:** Deixe este arquivo aberto lado a lado com o PowerPoint. Cada slide tem (1) **ponto central** — a única coisa que você precisa comunicar; (2) **fala** — o que dizer em ~30-60 segundos; (3) **âncoras** — números e citações para não esquecer; (4) **se perguntarem** — respostas prontas para as perguntas mais prováveis.
>
> **Tempo total estimado:** ~35-40 min apresentação + Q&A
>
> **Princípio:** seja direto. Os orientadores já conhecem o domínio — não gaste tempo introduzindo conceitos básicos. Vá direto aos achados e decisões.

---

## Slide 1 — Capa

**Ponto central:** Situar o volume experimental e o prazo.

**Fala:**
- "Bom dia. Vou apresentar o progresso experimental da Fase 2 do mestrado — detecção de intrusão em IoT com clustering evolutivo em streaming."
- "São 167 experimentos em 4 campanhas, com 6 steps de ablation."
- "A defesa está prevista para maio — faltam ~5 semanas."

---

## Slide 2 — Objetivo da Reunião

**Ponto central:** Estabelecer ANTES dos resultados o que você precisa da reunião.

**Fala:**
- "Antes de entrar nos resultados, quero deixar claro o que preciso desta reunião."
- "Primeiro: validar se o diagnóstico do problema e as 5 adaptações técnicas constituem contribuição da dissertação."
- "Segundo: discutir como enquadrar os resultados — eles ficaram abaixo das metas, mas acredito que há contribuição positiva aqui."
- "Terceiro: decidir o caminho dos próximos 5 semanas. Mais um round experimental ou consolidar e escrever?"
- "Vou ser rápido nos fatos e chegar nessa decisão no final."

**Âncoras:** Sommer & Paxson (2010) e Arp et al. (2022) validam documentar failure modes como contribuição.

---

## Slide 3 — Agenda

**Ponto central:** Dar o mapa em 30 segundos.

**Fala:**
- "São 8 tópicos. Vou ser rápido nos 4 primeiros — pipeline, teoria, metodologia, resultados por campanha."
- "Vou gastar mais tempo no framework de dimensões anômalas e na contribuição técnica do C04."
- "E terminar na decisão dos próximos passos."

---

## Slide 4 — Pipeline Streaming

**Ponto central:** Mostrar que a arquitetura é real e testada end-to-end.

**Fala:**
- "O pipeline tem 6 estágios: PCAP do CICIoT2023 → Kafka producer → broker → reconstrução de flows → MicroTEDAclus → métricas prequential."
- "Kafka desacopla os componentes e preserva a ordem de inserção — garante reprodutibilidade."
- "17 features por flow, timeout de 60 segundos, ground truth por IP."
- "Avaliação prequential (test-then-train) evita data leakage."

**Se perguntarem "por que Kafka?":**
- Reprodutibilidade (ordem de inserção preservada)
- Desacoplamento producer/consumer/detector
- Compatibilidade com cenário real de streaming em produção

**Se perguntarem "por que 17 features?":**
- É o conjunto padrão da literatura para flow-based IDS
- Testamos expansão (v2=25, v3=32) em C02-S2 — zero impacto

---

## Slide 5 — Fundamentação Teórica: TEDA

**Ponto central:** TEDA é a base. Tem uma limitação crítica (centro único) que motiva o MicroTEDAclus.

**Fala:**
- "TEDA é o framework do Angelov 2014. A ideia central é a eccentricidade: ξ = 1/k + ‖x-μ‖² / (k·σ²)."
- "Intuição: é um Z-score generalizado que não assume gaussianidade nem independência. Usa apenas a estrutura espacial mútua dos dados."
- "Três propriedades que fazem ele bom para streaming: não-paramétrico, O(1) por ponto via Welford, teste de Chebyshev que vale para qualquer distribuição."
- "MAS — um único centro global. Outliers contaminam as estatísticas. Não consegue separar padrões distintos. É o que motiva o MicroTEDAclus."

**Âncoras:** Angelov (2014) JAMRIS 8(2) | Welford (1962) Technometrics | Chebyshev: P(|X-μ| ≥ mσ) ≤ 1/m², vale para QUALQUER distribuição.

**Se perguntarem "o que é tipicalidade?":**
- τ(x) = 1 - ξ(x). Complemento da eccentricidade.
- τ ≈ 1 → muito típico. τ ≤ 0 → outlier.

---

## Slide 6 — MicroTEDAclus (Maia 2020)

**Ponto central:** Múltiplos micro-clusters com estatísticas isoladas resolvem a contaminação. É o algoritmo principal da dissertação.

**Fala:**
- "Maia et al. 2020 estendeu o TEDA para N micro-clusters, cada um mantendo suas próprias estatísticas isoladas."
- "Fluxo de decisão: ponto chega, calcula ζ para cada MC_i e testa Chebyshev. Se algum aceita, atualiza SÓ o melhor. Se nenhum aceita, cria novo cluster — isso marca como anomalia."
- "Diferença crucial vs TEDA básico: outliers criam novos clusters em vez de contaminar os existentes. K é automático. Concept drift é nativo."
- "O threshold dinâmico m(k) protege clusters jovens: k=1 é permissivo (m≈0.6), k→∞ é estrito (m→3, equivalente a 89% Chebyshev)."

**Se perguntarem "por que atualizar só o melhor, não todos?":**
- Evita convergência de clusters (winner-take-all, Kohonen 1990)
- É o padrão em stream clustering (DenStream 2006, NS-TEDA 2024)
- Update-all causa perda de diversidade — identifiquei isso como bug crítico da implementação original (próximo assunto no C04)

**Se perguntarem "como lida com concept drift?":**
- Novos padrões são rejeitados por todos os clusters existentes → novo cluster é criado
- Clusters antigos podem ser desativados se não recebem mais pontos
- Foi ponto forte teórico, mas não validei experimentalmente (cenário B do plano original não foi executado — item para futuros)

---

## Slide 7 — Metodologia: Ablation Study

**Ponto central:** Cada campanha testou UMA variável e congelou a decisão. Rigor metodológico.

**Fala:**
- "Ablation study cumulativo: cada step altera uma única variável, congela a melhor decisão e passa para o próximo. Causalidade isolada."
- "C01 decidiu o algoritmo: MicroTEDAclus foi 26x melhor que TEDA básico."
- "C02 testou 3 variáveis em paralelo: ground truth, features, granularidade."
- "C03 testou features comportamentais."
- "C04 validou contra a implementação original do Maia."
- "Total: 167 runs. É o nível de rigor que Arp et al. 2022 argumentam ser essencial em ML para segurança — e que Pendlebury et al. 2019 em TESSERACT defendem para evitar resultados inflados."

---

## Slide 8 — Campaign-01: Baseline

**Ponto central:** Achado impossível de ignorar: a anomaly rate é INVARIANTE. O detector não encontra ataques, encontra outliers estatísticos.

**Fala:**
- "O gráfico mostra o achado principal: anomaly rate constante em ~3.5% com OU sem ataque."
- "FPR em benigno passou no alvo: 3.5% contra alvo de 5%."
- "Mas Recall em ataques ficou em ~3-4%. Reprovado no alvo de 80%."
- "Diagnóstico: o detector encontra outliers estatísticos, não ataques. É exatamente o que Sommer & Paxson 2010 previram — o gap semântico entre 'anomalia' e 'malicioso'."
- "Importante: o problema é de representação, não de algoritmo. Flows DDoS individuais são estatisticamente idênticos a flows IoT benignos."

**Âncoras:** Sommer & Paxson (2010) IEEE S&P — **a citação mais importante da apresentação inteira**.

**Se perguntarem "e se o problema for o r0?":**
- Testamos r0 ∈ {0.05, 0.10, 0.15, 0.20, 0.30}. Não muda o padrão.
- O r0 controla quantas amostras viram anomalia, mas não QUAIS. Ataques e benignos continuam indistinguíveis.

---

## Slide 9 — Campaign-02: 3 Hipóteses

**Ponto central:** Duas hipóteses falharam. A terceira (janelas) mudou o jogo mas trouxe novo problema.

**Fala:**
- "Testamos 3 hipóteses em paralelo para entender o problema."
- "S1 — ground truth por IP: corrigiu medição do DDoS-ICMP, de 4% para 27%. Mas os outros ataques não mudaram. Confirmou que o problema é real."
- "S2 — expansão de features de 17 para 25 e depois 32: ZERO impacto. Features per-flow saturaram. Confirma que mais dimensionalidade não resolve."
- "S3 — janelas temporais: essa foi a mudança importante. SYN passou de 3% para 54%, Recon de 4% para 45%, Mirai de 1% para 33%."
- "MAS o FPR explodiu — 58% em janelas de 60 segundos."
- "Insight central: janelas temporais mudam a PERGUNTA. De 'este flow é anômalo?' para 'este IP está se comportando de forma anômala?'"

---

## Slide 10 — C02-S3: Janelas Temporais

**Ponto central:** Janelas funcionam (Recall 10-20x melhor), mas com trade-off severo vs FPR. A direção está certa, as features precisam melhorar.

**Fala:**
- "O gráfico quantifica a melhoria: Recall cresce monotonicamente com o tamanho da janela para SYN, Recon e Mirai."
- "Picos: SYN 54% em 30s, Recon 45% em 10s, Mirai 33% em 60s."
- "Problema: o FPR acompanha. 58% em 60 segundos é inaceitável para um IDS."
- "Isso replica o que Li et al. 2023 e Goldschmidt & Kucera 2024 reportam: window-based é mais eficaz, mas requer features que capturem comportamento, não só contagens."
- "É o que motivou C03 — adicionar features comportamentais."

**Âncoras:** Lakhina et al. (2004, 2005) ACM SIGCOMM são os papers seminais desse princípio — análise agregada revela anomalias invisíveis per-flow.

---

## Slide 11 — Framework 1/3: Definição

**Ponto central:** ISSO É SÍNTESE MINHA, não está em paper nenhum. Transparência total.

**Fala:**
- "Vou fazer uma pausa aqui. Esse framework NÃO está na literatura — é uma síntese analítica que construí para organizar os resultados dos 167 experimentos."
- "O caveat no topo é explícito. Eu quero deixar muito claro: a contagem de dimensões que vocês vão ver no próximo slide é qualitativa, não foi medida estatisticamente."
- "Definição: uma feature f_j é 'dimensão anômala' para o ataque A quando as distribuições D_benign(f_j) e D_attack(f_j) são qualitativamente separáveis."
- "Exemplo concreto: dst_port_entropy. Tráfego IoT benigno tem ~1.2 bits — 2 ou 3 portas ativas (DNS, HTTPS, MQTT). Recon-PortScan tem ~6.5 bits — 100+ portas probadas. Distribuições claramente separáveis → conta como 1 dimensão anômala para Recon."
- "Hipótese causal: mais dimensões anômalas → maior ‖x-μ‖² → maior ξ → maior rejeição pelo Chebyshev → maior Recall. A conexão com a fórmula é direta: ‖x-μ‖² = Σ(x_j - μ_j)² é aditivo por dimensão."

**Âncoras:** Inspirado em Sommer & Paxson (2010) e Lakhina (2005), mas o framework é meu.

---

## Slide 12 — Framework 2/3: Metodologia + Aplicação

**Ponto central:** Tabela aplicada aos 5 ataques. Cada linha tem citação literária que suporta a descrição.

**Fala:**
- "A metodologia de contagem tem 3 passos: inspeção visual das 19 features v2, cross-reference com a literatura de cada ataque, classificação em forte/fraca/nula. Só 'fortes' entram na contagem."
- "Aplicando aos 5 ataques:"
- "Recon-PortScan: 5+ dimensões (port entropy, destination diversity, flow size, duração, direcionalidade). Recall 49%. Suporte: Nychis et al. 2008."
- "DDoS-ICMP: 3-4 dimensões. Recall 50%. Wang et al. 2022."
- "Mirai: 2-3 dimensões. Recall 46%. Antonakakis et al. 2017."
- "DDoS-SYN: 1-2 dimensões. Recall 38-62%. Bellaiche & Bhunyan 2012."
- "DDoS-TCP: ZERO dimensões anômalas. Recall 0%. Zargar et al. 2013."
- "Padrão monotônico: Recall cresce com d_A. Consistente com a hipótese. Mas N=5 não permite validação estatística rigorosa — é a motivação para o próximo slide."

**Se perguntarem "por que DDoS-TCP é indetectável?":**
- TCP flood usa flags válidos (ACK, PSH+ACK), completa handshakes
- Cada flow individual é indistinguível de uma conexão TCP legítima
- Zargar 2013: "TCP/ACK floods appear to be legitimate acknowledgments"
- Zero dimensões anômalas → zero detectabilidade por anomaly detection

---

## Slide 13 — Framework 3/3: Validação Estatística Proposta

**Ponto central:** Eu sei EXATAMENTE como validar. Só não fiz por tempo. 2-3 dias de trabalho.

**Fala:**
- "Esse slide resolve o problema da contagem qualitativa. É um experimento proposto, não executado."
- "Fase 1: KS-test 2-sample por feature e por ataque. Kolmogorov-Smirnov é não-paramétrico, não assume distribuição — ideal porque TEDA também é não-paramétrico."
- "Para cada feature f_j e cada ataque A, extrair as distribuições, aplicar KS, contar como 'anômala' se p < 0.01 com correção de Bonferroni para 19 testes múltiplos."
- "Fase 2: correlação Spearman entre d_A estatístico e Recall_A. Spearman é não-paramétrico, adequado para N pequeno. H0: sem correlação, H1: correlação positiva."
- "Limitação óbvia: N=5 ataques dá poder estatístico baixo. Extensão futura seria expandir para 10+ ataques do CICIoT2023."
- "Tempo estimado: 2-3 dias. Os scripts de extração já existem das campanhas anteriores."

**Se perguntarem "por que não fez?":**
- Priorizei o diagnóstico do problema fundamental (C02) e a validação da contribuição técnica (C04)
- O framework emergiu como ferramenta analítica depois que os dados já estavam consolidados
- É o próximo passo natural, mas eu queria trazer para discussão antes de fazer

---

## Slide 14 — C03-S4: Features Comportamentais

**Ponto central:** Features comportamentais ajudam em 2/5 ataques mas pioram o FPR. Curse of dimensionality.

**Fala:**
- "C03 adicionou 7 features comportamentais às 12 básicas, totalizando 19: entropia de portas e IPs, flows por segundo, razão de unanswered, unidirecionais."
- "Resultado misto: DDoS-ICMP foi desbloqueado, de 0 para 50%. Recon melhorou de 39 para 45%."
- "MAS Mirai e SYN ficaram piores com v2. E o FPR quintuplicou, de 2.9% para 14.3% em janelas de 10s."
- "Diagnóstico: curse of dimensionality. Com janelas de 10s e min_flows=5, temos apenas ~210 vetores para 19 features. Poucos dados para alta dimensionalidade."
- "Zimek et al. 2012 documentam esse efeito especificamente para outlier detection em alta-D."

**Se perguntarem "por que v2 funciona para ICMP mas não SYN?":**
- ICMP não tem portas nem flags TCP — as 12 features básicas (v1) não capturam ICMP floods
- As features v2 (entropia de IPs, flows/s) são protocol-agnostic → desbloqueiam ICMP
- SYN já era parcialmente detectado pelas v1 (syn_ratio) e o ruído adicional de v2 piora

---

## Slide 15 — C04: 5 Adaptações Técnicas

**Ponto central:** As 5 adaptações são NECESSÁRIAS, não otimizações. Contribuição técnica original validada em 30 runs.

**Fala:**
- "C04 comparou a implementação própria com a original do package evolclustering (Maia 2020). 30 runs, usando exatamente os mesmos dados e configurações."
- "Identifiquei 5 diferenças críticas. A mais impactante é a fórmula de variância — vou mostrar a demonstração numérica no próximo slide."
- "As outras 4: (2) eccentricity consistente com a variância; (3) update seletivo — atualiza só o cluster de maior tipicalidade, não todos; (4) threshold permissivo para n=1 — protege cluster jovem; (5) guard duplo para n=2."
- "Suporte teórico: Welford 1962 e Chan 1983 para a variância. Kohonen 1990 e DenStream 2006 para o update seletivo. NS-TEDA 2024 validou recentemente update seletivo dentro da própria família TEDA."
- "Nenhum paper TEDA testou com mais de 6 features antes. A aplicação em 17 dimensões expõe uma limitação estrutural que Beyer 1999 e Aggarwal 2001 já previram sobre distâncias Euclidianas em alta dimensão."

---

## Slide 16 — Por Que a Fórmula Original Falha

**Ponto central:** (2/dim)² é neutro em 2D (contexto original do Maia) mas catastrófico em 17D — subestima σ² em ~70x.

**Fala:**
- "Essa demonstração é o coração da contribuição técnica. Vou fazer devagar."
- "A fórmula original é σ² = (‖δ‖ · 2/dim)². O fator crítico é (2/dim)²."
- "Em 2 dimensões, contexto original do Maia 2020: (2/2)² = 1. Neutro. A fórmula funciona."
- "Em 17 dimensões, meu caso: (2/17)² = 0.014. A fórmula multiplica σ² por 0.014 — ou seja, subestima em aproximadamente 70 vezes."
- "Em 32 dimensões seria (2/32)² = 0.004 — subestima em 250 vezes."
- "Cascata de falha: σ² subestimada → eccentricity inflada → Chebyshev rejeita quase tudo → cada ponto cria um novo cluster → FPR catastrófico de 54%."
- "Welford corrige: usa o produto escalar dot(δ_old, δ_new), que soma contribuições de TODAS as dimensões corretamente, independente de quantas sejam. Numericamente estável."
- "Isso é contribuição original do trabalho — nenhum paper TEDA expôs esse problema antes."

---

## Slide 17 — C04: Impacto Quantitativo

**Ponto central:** A diferença é dramática e consistente em todas as granularidades. Implementação original é inutilizável como IDS.

**Fala:**
- "O dashboard mostra o impacto: FPR benigno na implementação própria é 3.9%, na original é 54.4%. Diferença de 14x."
- "Window v1 em 10 segundos: 2.9% vs 41.9%, 14x. Em 30 segundos: 5.0% vs 74.5%, 15x."
- "Em todas as granularidades testadas, o original classifica metade ou mais do tráfego benigno como anomalia. É inutilizável como IDS."
- "Os Recalls altos do original (69-100%) são artefato trivial — se você classifica 55% de tudo como anomalia, acerta muitos ataques por sorte."
- "Conclusão: as 5 adaptações não são otimizações incrementais. São necessárias para MicroTEDAclus funcionar em alta dimensionalidade."

---

## Slide 18 — Consolidado: Melhor por Ataque

**Ponto central:** Recon F1=43.7% é o destaque. TCP é 0% em qualquer configuração.

**Fala:**
- "Tabela consolidada com a melhor configuração encontrada para cada ataque."
- "Destaque: Recon-PortScan com F1 43.7%, Recall 49%, Precision 39%, FPR 12.9%. Melhor resultado não-supervisionado da dissertação."
- "Com r0=0.15, Recon muda para Precision 56.7% e FPR de apenas 4.2%, F1 42%. Viável operacionalmente."
- "DDoS-TCP é 0% em TODAS as configurações testadas — fundamentalmente indetectável por anomaly detection."
- "Métodos supervisionados no CICIoT2023 atingem F1 > 95%, mas requerem dados rotulados. Métodos puramente não-supervisionados publicados no mesmo dataset são quase inexistentes."
- "Importante ser transparente aqui: não há baseline não-supervisionado direto para comparação. Essa é uma lacuna metodológica que reconheço explicitamente."

**Se perguntarem "qual o baseline? de qual artigo?":** (PERGUNTA PROVÁVEL E DELICADA)
- **Resposta direta e honesta:** "Não há baseline direto. Kitsune (Mirsky 2018 NDSS) é o candidato mais citado na literatura, mas nunca foi rodado no CICIoT2023 — e os papers sobre CICIoT2023 usam quase exclusivamente métodos supervisionados."
- **Contextualizar:** "O F1=43.7% é o melhor resultado medido nos meus experimentos. Qualquer comparação com Kitsune ou outro método seria indireta — datasets diferentes, protocolos diferentes, métricas diferentes."
- **Ação proativa:** "Tenho Isolation Forest e One-Class SVM já implementados na Fase 1 em modo batch. Adaptá-los para rodar prequential nos mesmos flows do streaming levaria 1-2 dias de trabalho. Isso produziria a comparação direta que falta — quero fazer antes da defesa se houver tempo."
- **Frame metodológico:** "Arp et al. 2022 em 'Dos and Don'ts of ML in Computer Security' argumentam que documentar lacunas explicitamente é parte do rigor científico. Prefiro admitir a ausência de baseline a apresentar extrapolação."

**Se insistirem "mas tem algum número de referência da literatura?":**
- Kitsune reporta F1 70-85% mas em datasets diferentes (câmeras IoT infectadas com Mirai, não CICIoT2023)
- Comparação não seria apples-to-apples — datasets diferentes, ataques diferentes, métricas diferentes
- Por isso prefiro não fazer a comparação e marcar como trabalho futuro

---

## Slide 19 — Contribuições Científicas

**Ponto central:** 4 contribuições distintas. Reframing honesto e positivo dos achados.

**Fala:**
- "Consolidando tudo: quatro contribuições científicas."
- "Primeira, técnica: as 5 adaptações validadas em C04 com 30 runs. Primeiro estudo empírico de TEDA em dados de rede de alta dimensão."
- "Segunda, empírica: 167 experimentos com ablation study rigoroso documentando ONDE e POR QUE anomaly detection falha em IoT IDS. Confirma empiricamente as previsões teóricas de Sommer & Paxson 2010."
- "Terceira, metodológica: o framework de dimensões anômalas como ferramenta analítica proposta. Validação estatística é trabalho futuro, mas a ferramenta organiza os achados de forma reproduzível."
- "Quarta, resultado positivo: Recon com F1 43.7% não-supervisionado demonstra que o pipeline Kafka + MicroTEDAclus funciona para ataques com assinatura estatística distinta."

**Âncoras:** Matosin et al. (2014) e Lipton & Steinhardt (2019) argumentam que resultados negativos documentados e ablation rigoroso são contribuições legítimas.

---

## Slide 20 — 6 Insights Principais

**Ponto central:** Consolidar os aprendizados em formato escaneável. Cada um tem citação.

**Fala (rápido, um insight por frase):**
- "Um: detecção per-flow é fundamentalmente limitada. Sommer & Paxson previram isso em 2010."
- "Dois: janelas temporais mudam a pergunta do detector. Lakhina 2004/2005 já havia mostrado no contexto de anomaly detection em rede."
- "Três: curse of dimensionality com 210 vetores em 19 features — Zimek 2012 documenta exatamente esse regime."
- "Quatro: não existe configuração única ótima. Cada ataque responde diferente às variáveis do detector."
- "Cinco: Recon 43.7% é o resultado positivo demonstrável. Sem baseline não-supervisionado direto no CICIoT2023 — é uma lacuna que pretendo fechar em 1-2 dias adaptando o Isolation Forest da Fase 1."
- "Seis: adaptação ao domínio é contribuição técnica em si — as 5 mudanças validadas em C04."

---

## Slide 21 — Próximos Passos (DECISÃO)

**Ponto central:** Preciso decidir HOJE. Tenho preferência inicial mas quero validação deles.

**Fala:**
- "Chegando na decisão. Três caminhos para os ~5 semanas que restam."
- "Opção A — S5 Two-Stage Detection mais escrita. 1 a 2 semanas de experimento, 3 a 4 de escrita. Stage 1 é per-flow com FPR baixo (3.5%), Stage 2 analisa concentração de anomalias por IP em janela. Ataca diretamente o trade-off FPR/Recall que vimos em todas as campanhas. Base teórica sólida: é ensemble de evidências."
- "Opção B — consolidar e escrever. 5 semanas inteiras de escrita. Os 167 experimentos já sustentam uma dissertação. Menor risco de prazo, mais tempo para revisão."
- "Opção C — S5 mais S6 threshold adaptativo mais escrita. Mais ambicioso, mas margem apertada. Risco alto."
- "Minha preferência inicial é A. Se S5 não funcionar em uma semana, vira B efetivamente — perde só uma semana. Mas quero confirmar com vocês antes de prosseguir."

**Argumento para A:**
- Two-Stage é a hipótese que melhor ataca o trade-off FPR/Recall documentado
- Base teórica sólida (ensemble de evidências)
- Risco contido: fail-fast em 1 semana

**Argumento para B:**
- Zero risco de prazo
- Contribuições atuais já são suficientes para mestrado
- Mais tempo para revisão do orientador

**Se perguntarem "qual tua avaliação honesta do S5?":**
- Acho que reduz FPR significativamente. Stage 1 mantém 3.5% per-flow.
- Incerteza: se a concentração de anomalias por IP é sinal suficiente. Flows de DDoS-TCP são normais individualmente, então Stage 2 dependeria de volume/rate.
- Probabilidade subjetiva: ~60% de melhoria clara, ~30% de melhoria marginal, ~10% de não funcionar.

---

## Slide 22 — Referências Principais

**Ponto central:** Transparência sobre as fontes. Usar para Q&A.

**Fala:**
- "Referências organizadas por tema. Posso responder perguntas sobre qualquer uma delas."
- "As mais importantes para o framing geral são Sommer & Paxson 2010 e Arp et al. 2022."
- "Obrigado. Abro para perguntas."

---

## Cartão de Emergência — Se Esquecer Tudo

**Os 5 fatos que você DEVE ter na ponta da língua:**

1. **167 experimentos, 4 campanhas, 5 steps de ablation**
2. **Recon F1 = 43.7%** é o melhor resultado **interno** da dissertação (C03-S4, v2/w10s/r0=0.05). **Sem baseline não-supervisionado direto no CICIoT2023 — lacuna reconhecida; trabalho futuro (Isolation Forest, 1-2 dias).**
3. **Implementação própria tem FPR ~1 ordem de magnitude menor** em TODAS as 5 configurações do C04. Exemplo direto: flow-level r0=0.10 → **3.9% (próprio) vs 54.4% (original)**, razão 14x. O padrão se replica em window v1 w=10s (14x), w=30s (15x) e window v2 w=10s (3x), w=30s (11x).
4. **(2/17)² = 0.014** é o fator que subestima σ² em ~70x — causa raiz do FPR catastrófico da implementação original
5. **DDoS-TCP = 0%** em todas as configurações — fundamentalmente indetectável por anomaly detection

**As 3 citações que você DEVE conseguir mencionar sem pensar:**

1. **Sommer & Paxson (2010)** "Outside the Closed World" IEEE S&P — gap semântico anomalia/ataque
2. **Maia et al. (2020)** "Evolving Clustering Based on Mixture of Typicalities" FGCS 106 — o algoritmo principal
3. **Arp et al. (2022)** "Dos and Don'ts of ML in Computer Security" USENIX Security — valida a metodologia

**Se travar em qualquer pergunta, respire e diga:**
> "Boa pergunta. Vou pensar um segundo... [pausa de 3 segundos]... Acho que o caminho seria X, mas confesso que preciso validar. Posso voltar nesse ponto depois de checar os dados?"

Isso é sempre melhor do que inventar uma resposta.
