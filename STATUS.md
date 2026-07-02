# STATUS — IoT IDS Research
<!-- STATUS.md é um snapshot. Substituir seções dinâmicas a cada sessão. -->
<!-- Histórico em docs/progress/ (gerado automaticamente) -->

**Atualizado:** 2026-07-01 | **Branch:** main (código) + repo Overleaf (dissertação) | **Prazo defesa:** ~ago/2026 | **Submit SoftCom:** feito

---

**Sessão 01/07 — P3: INTEGRAÇÃO DO RE-RUN NA DISSERTAÇÃO (§5.7 corrigida, gate duplo OK).**
Recuperação de contexto após desligamento + fechamento do P3 (o re-run do Exp A, P2, já estava
240/240 commitado; faltava levar os números corrigidos à §5.7, que ainda tinha os do detector com bug).
- **Cross-validação dos números (VM voltou a ficar acessível):** rodei `plot_matrix.py` canônico nos
  crus dos 2 braços (`/tmp/matrixA-v7corr|v7forget` na VM) — os 32 p-values Wilcoxon batem EXATO com
  o recompute do `matrix_summary.csv` local. Counts de cluster extraídos dos crus: V7_corrected 14,9
  (med 15); V7_forgetting 2,8 (med 3); bug ~204. Tabela de revisão em
  `experiments/results/matrixA-drift-rerun-revisao.csv`.
- **§5.7 reescrita (repo Overleaf):** Tabela `tab:res-drift` agora tem coluna "Variante" (V7 vs +poda),
  16 linhas × 7 col — leads+sig do re-run (V7_corrected: **PH 7/8, ADWIN 7/8**; exceção única DoS-SYN
  r0=0,10). Parágrafo novo "Efeito do esquecimento (poda)": poda material p/ topologia (14,9→2,8),
  benéfica p/ ρ, **recupera** a 8ª célula (DoS-SYN r0=0,10 PH −9,4→+117,3) → **+poda dá PH 8/8**; e
  eleva recall (~13→~48 anom). Figuras A-3/A-4 substituídas pelas do V7_corrected. chp4/síntese H4
  alinhados. chp1/chp6 (só qualitativos) intactos.
- **Gate duplo PASSOU:** build Docker OK (78 págs), inspect 0 fatais / 0 ref-cit undefined / 0 `??`
  (único overfull >20pt é legenda pré-existente do Apêndice C). Leitura visual (pdftoppm): tabela cabe
  na margem, célula recuperada em negrito, Figuras 10/11 nítidas, síntese coerente.
- ⚠️ **NÃO-COMMITADO:** edições vivem no repo Overleaf separado (`writing/dissertation/...`, gitignored
  no final-project), MODIFICADAS mas não commitadas. 2 CSVs untracked no final-project.

**Próxima sessão (P4 — fechar e revisar):**
1. Commitar a §5.7 no repo Overleaf (`git -C writing/dissertation/67c73... commit` + `push`) e os
   CSVs/figuras .bug-ref no final-project.
2. Enviar a §5.7 corrigida para revisão do orientador (mudança de números + veredito reforçado).
3. Blocos `[EXPERIMENTO PENDENTE]` ainda abertos: normalização (Campaign-06), multi-semente baselines,
   cenários drift D/E. (Opcional) regerar figuras A-3/A-4 com rótulos em PT (editar plot_matrix.py na VM).

---

**Sessão 28-29/06 — CONSOLIDAÇÃO GIT (3 cópias → main) + AUDITORIA DO FORGETTING + P1.**
Recuperação de contexto após desligamento abrupto (26-27/06) + consolidação de 3 cópias divergentes
do repo em `main`, e verificação do ambiente para o re-run do Exp A na linhagem correta.
- **Auditoria do forgetting (recuperada de `study-wiki/experimentos/14-auditoria-implementacao.md`):**
  a pergunta era se a ausência do esquecimento compromete o Exp A. Veredito: forgetting IMATERIAL no
  sintético; o achado de 1ª ordem é que o **Exp A rodou com `micro_teda.py` (bug: variância atual vs
  hipotética) — detector fragmentado, NÃO o V7 caracterizado em E1-E3**. Blast radius: artigo1 imune,
  E1-E3/Campaign-05 limpos, só Exp A + Campaigns 01-02 afetados. Decisão: **migrar streaming →
  corrected.py** (via wrapper `variant_micro_teda`, já existente) e re-rodar.
- **Consolidação git:** VM tinha branch `experimentos-drift` com 13 commits exclusivos (pipeline Exp A)
  nunca sincronizados. Tudo mergeado em `main` (commit `1c3e7de`), push GitHub, VM alinhada. Crus
  (597 MB) seguem gitignored na VM. Conflitos resolvidos (base=VM que rodou o Exp A; normalização
  FeatureNormalizer/mode/warmup do Mac volta na Campaign-06).
- **Overleaf DESTRAVADO:** push da dissertação voltou a funcionar (commit `6255819`, master). 4 capítulos
  (incl. chp5/E5) preservados e sincronizados.
- **P1 verificado:** testes 52/52 verdes; smoke do wrapper reproduz o laudo (V0→139/2500 ncl,
  V7→9 ncl/6 anom, V7_forgetting→2 ncl/6 anom). Migração confirmada como troca de flag.

**Próxima sessão (P2 — re-run do Exp A na linhagem correta, na VM):**
1. Re-rodar a matriz A (4 ataques × 2 r₀ × 30 comp) via pipeline streaming com
   `--algorithm variant_micro_teda --variant-name V7_full_corrected` (substitui o detector com bug)
   E braço `V7_forgetting` (Fase 3b: forgetting em IoT real). Usar tmux+checkpoint (resiliente).
2. Comparação pareada (Wilcoxon) vs os 240 runs antigos: num_clusters, F1/FPR/recall, leads ρ-vs-erro.
   Critérios: regime muda de fragmentado→estabilizado? forgetting altera ρ/lead? H4 se mantém?
3. Atualizar §5.7 (e §5.1) da dissertação com os números corretos + gate (build + leitura visual).

---

**Sessão 25/06 — INTEGRAÇÃO DO EXP A NA DISSERTAÇÃO (E5 executado, H4 com veredito).**
Os resultados da matriz de drift (240 runs, Exp A) foram integrados ao texto da dissertação,
reposicionados como a execução do **E5** no cenário realista benign→ataque (decisão de
enquadramento do usuário). Honestidade preservada: H4 ganha veredito; H5/cenários D/E seguem futuros.
- **chp5 — nova §5.7 "Comportamento sob Concept Drift (E5)"** (antes da Discussão): protocolo
  (matriz 4 ataques × 2 r₀ × 30 comp, normalizado), comparação ρ-vs-erro como entrada de ADWIN/PH,
  **Tabela 5 (Wilcoxon, 8 células)**, **Figuras 10/11 (A-3 latência, A-4 lead×r₀)**, síntese.
  **Veredito: H4 confirmada** (ρ antecipa PH sempre; ADWIN só em r₀=0,001 — liga a λ*=√(r₀/d)).
- **chp4** — E5 movido de "planejado" para executado; H4 com veredito (ref §5.7); bloco
  `exp:drift-pendente` reescrito para cobrir só **D/E**. **chp1/chp6** — frases de "futuro/hipótese"
  do monitor ρ atualizadas para "confirmado/reportado".
- **Build:** 74→**77 págs**, 0 refs/citações indefinidas, 0 `??`, tabela nova sem overfull (único
  overfull >20pt é legenda pré-existente do Apêndice C). **Gate completo:** Docker build + leitura
  visual do PDF (§5.7, tabela, 2 figuras, chp4) — tudo renderiza correto.
- **Preservação:** commit `095d74e` na VM (matrix_summary.csv + figuras A-3/A-4); `.gitignore`
  endurecido (240 .json crus, 597 MB, ficam fora do git, no disco da VM).
- ⚠️ **Pendência:** push Overleaf (token sem git access); revisão do orientador sobre a nova §5.7.

---

**Sessão 22/06 — RECUPERAÇÃO + DESFECHO DO EXPERIMENTO A (concept drift, matriz cheia 240/240).**
Sessão anterior (20/06) fechou abruptamente em 119/240 na máquina remota (`192.168.222.209`, VM-Augusto).
Ao reconectar: o run rodava em `tmux` desacoplado com checkpoint por-run → **completou sozinho**
(`MATRIX_DONE 240/240`, 0 FAILs, fim 21/06 04:55; VM up 75 dias → `/tmp` intacto). Ações: (1) dados
597 MB preservados de `/tmp/matrixA` → `experiments/results/matrixA-drift/`; (2) `plot_matrix.py`
re-rodado sobre os 240 → `matrix_summary.csv` final (240 linhas) + figuras A-3/A-4; (3) CSV+figuras
trazidos p/ o Mac e lidos visualmente; (4) log de execução e STATUS atualizados.
- **Resultado DEFINITIVO (4 ataques × 2 r₀ × 30 comp, Wilcoxon pareado erro-vs-ρ):**
  - **r₀=0,001 (fragmentação): ρ antecipa AMBOS detectores nos 4 ataques, p<2e-5. Sem exceção** (central).
  - **vs Page-Hinkley: ρ antecipa SEMPRE** (8/8 células, p≤0,02; lead até +738 fluxos).
  - **vs ADWIN: depende do regime** — vence em r₀=0,001 (p≈0); empata em r₀=0,10 (n.s. p/ dos/mirai/recon).
    **HA4 confirmada:** lead de ρ cresce quando r₀ cai (λ*=√(r₀/d)).
- **Contribuição consolidada:** monitorar drift sobre o sinal de regime intrínseco ρ antecipa o erro
  prequencial — robusto vs PH (sempre), supera ADWIN no regime r₀-bounded — SEM detector externo;
  generaliza entre 4 famílias. **Experimento A encerrado.** 📄 `study-wiki/experimentos/13-execucao-log.md`.
- ⚠️ **Pendência:** alimentar os blocos `[EXPERIMENTO PENDENTE]`/§4.6 (cenários drift) da dissertação com
  estes resultados; commitar `matrixA-drift/` (resultados + figuras) na branch.

---

**Sessão 11/06 (cont.) — FUNDAMENTAR EXPERIMENTOS DE CONCEPT DRIFT (cap. não-resultados).**
Pedido do orientador: aprofundar concept drift e seu uso na detecção. Expandidos chp2/3/4/1/6
para fundamentar A PRIORI os experimentos finais de drift (cenários B/C/D/E), **sem resultados
fabricados** (Resultados intocado). **71→74 págs**, 3 equações novas, 2 símbolos. Loop E1→E6,
cada passo com gate de 8 dimensões (contextual, raciocínio, coerência, rigor matemático, figuras,
honestidade-sem-resultado, compilação, estrutura PDF) + leitura visual do PDF.
- **E1 §2.1:** decomposição formal P(X,y)=P(y)P(x|y)=P(x)P(y|x) (eq:fund-drift-joint), 4 fontes
  de drift mapeadas a IoT.
- **E2 §2.4:** métricas prequenciais de drift (MTTD, tempo de adaptação, fading factor α=0,01,
  Page-Hinkley via gama2013 — sem ref nova).
- **E3 §2.7.1 (NOVA):** ponte regime↔drift formalizada — ρ_i(t)=σ²_i/r0 (eq:fund-drift-monitor),
  transição ⟺ ρ cruza 1; hipótese falsificável (σ²/r0 como monitor de drift). Subjuntivo.
- **E4 §4.6 (centerpiece):** experimento E5 "planejado" com 4 cenários B/C/D/E (C reaberto);
  H4/H5 pré-registradas SEM veredito; marcador [EXPERIMENTO PENDENTE] exp:drift-pendente.
  Reescrito o parágrafo que rebaixava B/D/E a futuro (evita contradição).
- **E5 §3.4.3 + símbolos:** posicionado σ²/r0-monitor como ângulo original vs INSOMNIA/PWPAE;
  ρ_i(t) e α adicionados a simbolos.tex; ρ agregado (§4.7) unificado a ρ_i (eq 2.12).
- **E6 chp1/chp6:** frase forward no §Organização; item 4 de Futuros elevado a "executar E5".
- **Auditoria de honestidade (doc inteiro):** 0 cenários descritos como executados; H4/H5 sem
  veredito; todo \ref de drift aponta a design/teoria (exp:drift-pendente/sec:fund-regime-drift/
  eq:fund-drift-monitor), NENHUM a Resultados. Build final limpo (0 ??, 0 undefined, 48 refs).
- 📄 Logs: `BUILD_LOG.md`. ⚠️ Pendência: push Overleaf (token sem git access; commit local).

---

**Sessão 11/06 — BENCHMARK PPGEE + EXPANSÃO CALIBRADA (3 ciclos convergidos).** Calibração contra
8 dissertações REAIS do PPGEE-UFMG (4 do orientador). Expansão SÓ com substância real, cada
iteração com gate completo (build + log + **leitura visual do PDF**). **63→71 págs, 7→12 figuras,
9→10 tabelas, 40→48 referências.** Zero fabricação: toda figura existe no repo, toda citação é
verificável.
- **Ciclo 1 (Estrutura):** +6 figuras reais migradas p/ Resultados+Apêndice (jornada IoT:
  recall_by_attack, anomaly_invariant, recall_evolution, fpr_benign; baseline: acc_vs_balanced,
  bayesian_matrix), tríade texto→figura→discussão. +Tabela das 17 features de fluxo (reprodutib.).
  2 figuras deferidas por critério anti-volume-morto (redundante / 6-painéis ilegível em coluna).
- **Ciclo 2 (Profundidade):** +8 referências reais verificadas (Crossref/IEEE/ACM/Springer) —
  linhagem TEDA (Angelov 2014 EALS, Bezerra 2016/2020) + **concept drift em IDS** (nova §3.4.3:
  INSOMNIA, Andresini 2021, Shyaa 2023/2024, PWPAE), revisão trabalho-a-trabalho.
- **Ciclo 3 (Estilo):** 4 listas descritivas do Cap. 3 → prosa conectiva; coerência global
  verificada (LoF=12, LoT=10, 0 refs/citações undefined, 0 `??`).
- **GAPS_FINAIS.md (entregue):** gaps de substância não-preenchíveis (G1-G6) vs. dissertações
  reais + roteiro de expansão priorizado + **seção dedicada ao concept drift na detecção**
  (ponto do orientador): ponte σ²/r0↔drift, cenários B/D/E (já no repo), implementação faltante.
- 📄 Logs: `BENCHMARK_PPGEE.md`, `GAPS_FINAIS.md`, `BUILD_LOG.md`.
- ⚠️ **Pendência:** push p/ Overleaf bloqueado (token sem git access; commit local 646885e
  preservado). Usuário precisa restaurar git access no Overleaf web.

**Estado anterior (10/06):** LACUNAS DE CITAÇÃO AUDITADAS. 12 afirmações sem suporte endereçadas com 8 fontes verificadas. .bib 32→40, todas citadas. 63 págs, compila limpo, 0 ??.

Sessão 10/06 — auditoria de under-citation (`CITGAP_AUDIT.md`):
- ✅ Boas práticas pesquisadas (árvore "precisa citar?" + qualidade da fonte; Ohio State/BU/NeurIPS/NBR 10520).
- ✅ 8 fontes novas buscadas e VERIFICADAS (DOI/venue; Kafka e river sem DOI corretamente — não inventados).
- ✅ Cap.3 (5 lacunas ALTA): seções de abertura antes sem nenhuma citação agora ancoradas.
- ✅ Honestidade: Kafka 2011 não cobre replicação → reformulado (evita citação que não sustenta o claim); "bilhões" sem fonte acadêmica → reformulado qualitativo.
- 📄 Logs: `CITGAP_AUDIT.md`, `BUILD_LOG.md`.

**Estado anterior (08/06):** referências existentes auditadas (metadados verificados, venues/autores corrigidos, órfãs removidas).

Sessão 08/06 — auditoria de referências (`REFS_AUDIT.md`, backup `references.bib.pre-refaudit-backup`):
- ✅ **3 verificações web independentes:** nenhuma referência inventada, nenhum DOI falso (3 SoftCOM suspeitos são REAIS).
- ✅ **Erros graves corrigidos:** venue/tipo/autor/páginas de 5 refs (incl. a do dataset, neto2023ciciot).
- ✅ **DOIs:** 9 ausentes adicionados (verificados) + formato consistente `note={DOI:}` (o .bst não suporta campo `doi=`); underscores escapados.
- ✅ **Limpeza:** 10 órfãs não citadas removidas (incl. duplicata ciciot2023dataset); .bib = 32 entradas citadas.
- ✅ **build.sh endurecido:** limpa .aux/.bbl (evita .bbl corrompido mascarar bibliografia faltante).
- ⚠️ Erro benigno pré-existente do .bst (`format.doi` indefinida) — não afeta o PDF.
- 📄 Logs: `REFS_AUDIT.md`, `BUILD_LOG.md`.

**Estado:** 63 págs · 6 caps + 3 apêndices · 32 referências auditadas (19 com DOI) · compila limpo (0 `??`).

Sessão 07/06 (cont.) — auditoria científica em 3 frentes (`INSIGHTS.md`, `CLAIMS_AUDIT.md`, `COMPLETUDE_GAP.md`):
- ✅ **Completude:** Apêndice C com resultados bayesianos da Fase 1 (10 algos, F1±dp + balanced-acc; não-superv. competitivo).
- ✅ **Insights (loop c/ crítica adversarial):** I3 granularidade ataque-específica; I2 trade-off FPR/recall = manifestação dos 2 regimes; I1+I4+I7 bloco de recalibração de escala; I5 não-monotonicidade=fronteira dimensional; I6 topologia subordinada a σ²/r₀; I8 throughput=implementação. Crítica preveniu 2 auto-gols.
- ✅ **Apresentação (boas práticas Arp/Axelsson):** base-rate fallacy invocada; incerteza já presente nas tabelas sintéticas; trade-offs honestos.
- ✅ **Claims:** 4 factuais sem citação corrigidos com FONTES REAIS (Guo 2023 zero-day; Neto 2023 47 features p/ d≥17; Gama 2014 drift). DOIs verificados.
- 📄 Logs: `INSIGHTS.md`, `CLAIMS_AUDIT.md`, `COMPLETUDE_GAP.md`, `BUILD_LOG.md`.

**Estado:** 63 págs · 6 caps + 3 apêndices · compila limpo · 32 referências (2 novas reais).

Sessão 07/06 — auditoria de gap + expansão (`writing/dissertation/.../GAP_ANALYSIS.md`, `BUILD_LOG.md`):
- ✅ **Pesquisa de boas práticas + tamanho** (AUT mediana CS 96p; DFKI ML 60-80p; Cambridge/UNSW/CAPES): alvo ~70-110p corpo, profundidade > volume.
- ✅ **GAP_ANALYSIS** (G1-G10): mapeou conteúdo de pesquisa relevante ausente do texto.
- ✅ **Passe A** (conteúdo): Fase 1 baseline supervisionado (§4.1: 10 algos, 705 exps, F1=0,9964, bayesiano Brodersen); jornada experimental IoT (§5.1: C01 falha→C02 refinamentos→C04/C05 descoberta do regime); fundamentação aprofundada (proximidade acumulada, métricas, drift DDM/ADWIN); arquitetura Kafka detalhada; cenários de drift A-E (honesto: C cortado, B/D/E futuro); **Apêndice A** (tabela C05 completa) + **Apêndice B** (reprodutibilidade); baselines aprofundados; interpretação integradora.
- ✅ **Passe B** (revisão): corrigida ambiguidade V0/V7 na jornada, transições, notação, redundância; 2ª revisão independente confirmou convergência (números batem em todo o documento).
- ✅ **build.sh endurecido**: pré-flight `docker info` + `rm` do PDF antigo (pega Docker down).
- 📄 Logs: `GAP_ANALYSIS.md`, `BUILD_LOG.md`.

**Estado da dissertação:** 59 págs · 6 caps + 2 apêndices · 6 figuras · 6 tabelas · narrativa progressiva Fase 1→jornada→sintético→discussão→conclusão · compila limpo (0 refs/citações/`??`/overfull).

Sessão 06/06 — escrita+revisão da dissertação (`writing/dissertation/67c73a7a961a07e62dace94d/`):
- ✅ **Gate de build:** Docker TeX Live (`build.sh`/`inspect.sh`); corrigido `abntex2.cls` quebrado (v1.9.6→oficial v1.9.7). Baseline da proposta compilava em 43 págs.
- ✅ **Re-ancoragem:** estrutura de proposta→dissertação (6 caps). Resumo/abstract reescritos (transição de fase). Símbolos (13) e siglas (19) reais. Placeholders de template removidos.
- ✅ **6 capítulos preenchidos:** Intro (4 contribuições), Fundamentação (TEDA/MicroTEDAclus/regime, 9 eqs), Trabalhos Relacionados, Metodologia (CICIoT2023/pipeline/V0–V7/E1–E4), Resultados (E1–E4, baselines, 2 blocos `[EXPERIMENTO PENDENTE]`), Conclusão.
- ✅ **Bibliografia:** `.bib` 12→39 entradas (29 citadas, ABNT). **Figuras:** 5 paper-quality migradas + diagrama de arquitetura gerado (matplotlib). **Tabelas:** 4 com dados reais.
- ✅ **Loop de revisão (5 iter, parada por convergência):** 3 críticos numéricos corrigidos (totais 3.780; FPR E4 vs baselines), redundância Cap.2↔3 removida, notação padronizada, aritmética da Tab. 5.3/Eq. 5.1/legendas corrigida. 2ª revisão sem achados críticos.
- 📄 Log completo: `writing/dissertation/.../BUILD_LOG.md`.

**Estado da dissertação:** 51 págs · 6 caps · 6 figuras · 5 tabelas · 9 equações · 29 refs · compila limpo (0 refs/citações/`??`/overfull>20pt).

Sessão 06/05 (Fases 1A–3C do plano `~/.claude/plans/preciso-que-atualize-todos-lovely-sedgewick.md`):
- ✅ **Fase 1A** Linux exp prep: `src/utils/feature_normalizer.py` + 11 testes passando + integração em `streaming_detector.py` (config, init, _process_flow, _process_window_vectors) + flags CLI em `run_experiment.py` (`--normalize-features`, `--normalize-mode`, `--normalize-warmup-size`) + `scripts/run_campaign06_normalize.sh` + `experiments/results/campaign-06/ANALYSIS.md` skeleton. **142 testes passam.**
- ✅ **Fase 1B** Paper §III-C "Phase Transition Mechanism" + 6 entries .bib novas (Gama 2014, Bifet 2007, Efron 1993, Cao 2006, Aggarwal 2003, Cohen 1988).
- ✅ **Fase 2A** Paper §IV-C "Regime Transition Characterization (Experiment 3)" com 3 figuras de Exp 3 integradas (`fig_regime_transition.pdf`, `fig_regime_comparison.pdf`, `fig_phase_diagram.pdf`).
- ✅ **Fase 2B** Tabela VIII honesta com HST/V7/V0/LOF + calibration_gap column. Abstract reescrito (regime change). §I Introduction com 4 contribuições reformuladas. §V Discussion + §VI Conclusion reescritas.
- ✅ **Fase 3A** Material defesa orientador: `presentation.md` (10 slides) + `speaker-notes.md` (matemática profunda) + `study-guide.md` (18 Q&A antecipado).
- ✅ **Fase 3B** `generate_pptx.py` + `2026-05-08-paper-defense.pptx` (46 KB, 11 slides).
- ✅ **Fase 3C** `regime-transition.md` §11 (bibliografia citável mapping claim→bib) + `TIMELINE.md` §10 atualizado + STATUS.md atualizado.

**Paper estado final:** 587 linhas, 22 referências únicas, 0 placeholders, 6 figuras (3 novas + 3 antigas), 8 tabelas, IEEE conference format.

**Sessão anterior (04/05):**
- ✅ Fundamentação teórica registrada: TIMELINE.md, regime-transition.md (250+ linhas), teda-framework.md §12, maia-2020 §14, methodology.md §9.
- ✅ Fase A: extensão de `GaussianStreamGenerator`, módulo `metrics/regime.py`, `exp03_regime_transition.py`. 21 testes unitários novos passando.
- ✅ Fase B: full run **1.620 runs** (9 λ × 3 r₀ × {V0,V7} × 30 seeds) executado em ~30 min. Range λ ∈ [10⁻³, 10¹] (4 ordens de grandeza).
- ✅ Fase C análise estatística:
  - **H1** parcial — estrutura $\lambda^* \propto \sqrt{r_0}$ confirmada, mas coeficiente difere de $1/\sqrt{d}$ (V7 emp = 0,092·√r₀ vs predito 0,243·√r₀; fator 0,38 constante para os 3 r₀'s).
  - **H2** ✅ massivo — Cohen's $d$ V0 vs V7 atinge 1.376 na zona de transição.
  - **H3** ✅ exato — razão $\lambda^*(r_0_a)/\lambda^*(r_0_b)$ bate $\sqrt{r_0_a/r_0_b}$ com check_ratio = **1,00** em todas as comparações.
  - Friedman $p \le 10^{-30}$ em todos os groupings; ANOVA $F \ge 9$, $p \le 10^{-11}$.
- ✅ Fase D: 3 figuras paper-quality (`fig_regime_transition_v7`, `fig_regime_v0_vs_v7`, `fig_regime_phase_diagram`) em `paper_figures/`.
- ✅ Fase E: §8 de `regime-transition.md` preenchido com resultado empírico, TIMELINE §5.7 atualizado.

**Achado central confirmado para o paper:** A lei de escala $\lambda^* \propto \sqrt{r_0}$ é **universal** entre V0 e V7 — ambos transicionam, mas em pontos diferentes via mecanismos diferentes (V0: n<3 guard `var > r₀`; V7: `max(var, r₀)`). Confirma que o regime change é **propriedade fundamental** do MicroTEDAclus comparando $\sigma^2$ vs $r_0$, não peculiaridade de uma implementação. Cohen's $d$ massivo (>1000) entre V0 e V7 valida que as 5 adaptações **transformam estruturalmente** o detector (não apenas calibram).

**Próxima sessão:**
1. **Rodar os experimentos finais** que alimentam os 2 blocos `[EXPERIMENTO PENDENTE]` da dissertação:
   - Validação multi-semente dos baselines (§5.5) — substituir a tabela de semente única.
   - Campaign-06 normalização de \textit{features} (§5.6.3) — `experiments/streaming/scripts/run_campaign06_normalize.sh` no Linux.
2. **Preencher os blocos sinalizados** na dissertação com os resultados acima (estrutura de tabela/figura já montada e marcada no `chp5_results.tex`).
3. **Sincronizar com o Overleaf:** subir os capítulos reescritos + figuras + `abntex2.cls` corrigido. Verificar build no Overleaf (deve casar com o build Docker local).
4. **Revisão do orientador** sobre a dissertação re-ancorada.
5. **Ficha catalográfica** (biblioteca) e **folha de aprovação** (pós-defesa) — substituir os PDFs placeholder em `pre-textuais/`.

---

## Critérios de Sucesso

| Critério | Status |
|----------|--------|
| Campaign-01 completa (17 runs) | ✅ |
| Campaign-02 completa (72 runs) + ANALYSIS.md | ✅ |
| Campaign-03 S4 completa (48 runs) + ANALYSIS.md | ✅ |
| Campaign-04 completa (30 runs) + ANALYSIS.md | ✅ |
| Campaign-05 completa (37 runs streaming baselines) + ANALYSIS.md | ✅ |
| Exp 1: Sweep dimensional (1440 runs sintéticos) | ✅ |
| Exp 2: Ablation V0–V7 (240 runs, Friedman p<10⁻⁴⁰) | ✅ |
| Exp 3: Regime transition (1620 runs sintéticos, λ × r₀ × algo × seed) | ✅ Caso B (estrutura confirmada, coef. ajustado) |
| `regime-transition.md` (fundamentação teórica permanente) | ✅ Criado + §8 preenchido |
| Predições H1/H2/H3 confirmadas em Exp 3 | ✅ H2/H3 fortes; H1 estrutural (off em coef constante) |
| 3 figuras paper-quality `fig_regime_*.pdf` | ✅ Geradas |
| Paper SoftCom reformulado com framing regime change | ⬜ Próxima fase |
| Tabela VIII honesta (HST presente, calibration column) | ⬜ Pendente |
| Validação IoT com features normalizadas (Fase F) | ⬜ Opcional, paralelo |
| Submissão SoftCom até 2026-05-11 | ⬜ Em andamento |
| 64+ testes passando (21 novos + 43 existentes) | ✅ |

---

## Código Relevante Agora

| O quê | Onde |
|-------|------|
| **Documento de re-entrada** | `TIMELINE.md` (raiz, gitignored) |
| **Fundamentação regime change** | `research/foundations/regime-transition.md` (novo) |
| **TEDA framework expandido** | `research/foundations/teda-framework.md` §12 |
| **Maia 2020 com escopo retrospectivo** | `research/summaries/maia-2020-microtedaclus.md` §14 |
| **Protocolo metodológico para phase transition** | `experiments/methodology.md` §9 |
| Algoritmo corrigido (V7) | `experiments/teda-high-dim/src/teda_hd/algorithms/corrected.py` |
| Algoritmo original (V0) | `experiments/teda-high-dim/src/teda_hd/algorithms/original.py` |
| **Gerador sintético estendido** | `experiments/teda-high-dim/src/teda_hd/generators/gaussian.py` |
| **Métricas de regime** | `experiments/teda-high-dim/src/teda_hd/metrics/regime.py` (novo) |
| **Exp 3 — script principal** | `experiments/teda-high-dim/experiments/exp03_regime_transition.py` (novo) |
| Análise Campaign-05 (com cluster topology) | `experiments/results/campaign-05/ANALYSIS.md` §7 |
| Paper SoftCom (a reformular) | `writing/papers/69da494c7d0d6aa7085e2444/a-sofctom.tex` |
| Plano executável atual | `~/.claude/plans/preciso-que-atualize-todos-lovely-sedgewick.md` |
| Arquitetura v0.6.0 | `docs/architecture/CURRENT.md` |

---

## Roadmap operacional (próximos 7 dias até deadline)

| Dia | Foco |
|---|---|
| **D-7 (hoje, 04/05)** | Fase A ✅ + Full run iniciado + docs teóricos atualizados |
| **D-6 (05/05)** | Fase C análise (Friedman, ANOVA, transition fit, bootstrap CI) → decisão Caso A/B/C/D |
| **D-5 (06/05)** | Fase D figuras paper-quality + Fase F (Linux IoT) em background |
| **D-4 (07/05)** | Reescrita do paper: abstract, intro, nova seção §III "Regime Characterization", conclusion |
| **D-3 (08/05)** | Tabela VIII honesta (com HST) + revisão completa de coerência |
| **D-2 (09/05)** | Revisão com orientador (Frederico) |
| **D-1 (10/05)** | Revisões finais |
| **D-0 (11/05)** | Submissão SoftCom |

---

## Referências Rápidas

- Como usar o repositório: `USAGE.md`
- Plano experimental: `experiments/campaign-plan.md`
- Metodologia: `experiments/methodology.md`
- **Plano corrente (executável):** `~/.claude/plans/preciso-que-atualize-todos-lovely-sedgewick.md`
- Arquitetura do sistema: `docs/architecture/CURRENT.md`
- Leituras e lacunas: `research/reading-log.md`
