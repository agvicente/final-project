# Plano de Retomada — Progresso de Estudo

> **Objetivo:** recuperar contexto matemático e técnico do projeto com segurança para
> discutir **todos os pontos da apresentação pronta** na reunião de sexta (17/abr).
>
> **Ritmo:** ~2h/dia, sem subdivisão manhã/tarde/noite. Sem cronograma rígido —
> avanço sequencial até sexta. Se um item ficar obscuro, aprofundo nele antes de seguir.
>
> **Como usar:** marcar `[x]` conforme concluo. Cada bloco = uma sessão de ~2h.
> Dúvidas vão para a seção "Dúvidas" ao final; traço para a próxima sessão com o Claude.

---

## Princípio Central

**A apresentação (`docs/meeting/2026-03-19-advisor-meeting.pptx`, 14 slides) é o eixo.**
Cada sessão cobre um grupo de slides e entra fundo só nos pontos que eu não sei defender.
Se eu já sei, avanço. Se travo, faço drill-down (matemática, código, paper).

**Critério de sucesso por slide:** consigo explicar em voz alta em ~1 min, sem consultar,
respondendo "o que é, por que importa, qual o número, qual a citação que sustenta".

---

## Sessão 1 — Panorama + Slides 1-3 (Introdução, Fase 1, Fase 2) — ~2h

**Ler:**
- [x] `STATUS.md` (snapshot atual, 10 min)
- [x] `USAGE.md` (organização do repo, 10 min)
- [x] `docs/meeting/2026-03-19-advisor-meeting.md` — seções 1 e 2 (Situação Atual + Metodologia) (40 min)
- [ ] `docs/meeting/advisor-meeting-speaker-notes.md` — slides 1-3 (30 min)

**Auto-teste (sem consultar):**
- [ ] Consigo descrever Fase 1 em 3 frases? (dataset, algoritmos, resultado F1>0.99)
- [ ] Consigo descrever Fase 2 em 3 frases? (streaming, não-supervisionado, TEDA/MicroTEDAclus)
- [ ] O que é prequential (test-then-train) e por que usamos?
- [ ] Por que ground truth por IP > ground truth por fase?

**Drill-down se travar:** `experiments/methodology.md` (primeiras 3-4 seções).

---

## Sessão 2 — Matemática do TEDA + Slides 4-5 (Detector) — ~2h

**Ler:**
- [ ] `docs/meeting/advisor-meeting-study-guide.md` — seção 2 (Domínio Matemático) completa (60 min)
- [ ] `docs/meeting/advisor-meeting-speaker-notes.md` — slides 4-5 (detector, equações) (30 min)
- [ ] `research/foundations/teda-framework.md` (30 min)

**Exercício com papel:**
- [ ] Escrever de cabeça: ξ(x) = fórmula da eccentricidade
- [ ] Refazer exemplo numérico 2D do study guide §2.1
- [ ] Traçar cadeia: Welford → μ,σ² → eccentricidade → Chebyshev → threshold → decisão

**Auto-teste:**
- [ ] Por que Chebyshev e não Gaussiana? (não assume distribuição)
- [ ] O que é "tipicalidade" e por que é dual da eccentricidade?
- [ ] Por que Welford em vez de calcular média/variância do zero a cada ponto?

**Drill-down se travar:** `research/summaries/angelov-2014-teda.md`.

---

## Sessão 3 — Slides 6-8 (Campaigns 01–02: baseline + 3 hipóteses) — ~2h

**Ler:**
- [ ] `experiments/results/campaign-01/ANALYSIS.md` (45 min)
- [ ] `experiments/results/campaign-02/ANALYSIS.md` (45 min)
- [ ] `docs/meeting/advisor-meeting-speaker-notes.md` — slides 6-8 (30 min)

**Anotar números-chave:**
- [ ] FPR benigno (flow / window): ___ / ___
- [ ] Melhor Recall per-flow (C01/C02): ___ em qual ataque?
- [ ] Ganho de janela vs flow (SYN, Recon, Mirai): ___x, ___x, ___x
- [ ] Por que DDoS-TCP tem 0% de detecção?

**Auto-teste:**
- [ ] Por que expandir de 17→32 features não ajudou? (C02-S2)
- [ ] Qual o "breakthrough" das janelas temporais? (C02-S3)
- [ ] Anomaly rate invariante: o que significa e por que é um achado?

**Drill-down se travar:** `research/reading-log.md` seções sobre window-based detection.

---

## Sessão 4 — Slides 9-11 (Campaigns 03–04: features comportamentais + 5 adaptações) — ~2h

**Ler:**
- [ ] `experiments/results/campaign-03/ANALYSIS.md` (40 min)
- [ ] `experiments/results/campaign-04/ANALYSIS.md` (40 min)
- [ ] `docs/meeting/advisor-meeting-speaker-notes.md` — slides 9-11 (40 min)

**Anotar números-chave:**
- [ ] F1 máximo alcançado (C03): ___% em qual config?
- [ ] FPR original vs próprio (C04): ___% vs ___%
- [ ] Fator (2/17)² = ___ (e o que significa)
- [ ] As 5 adaptações (listar de cabeça)

**Auto-teste:**
- [ ] Por que features comportamentais ajudam em 2/5 ataques mas pioram em 2/5?
- [ ] Qual a causa-raiz do FPR catastrófico do original em 17D?
- [ ] As 5 adaptações são contribuição técnica ou apenas engenharia?

**Drill-down se travar:**
- `experiments/streaming/src/detector/micro_teda.py` — mapear as 5 adaptações no código
- `research/summaries/maia-2020-microtedaclus.md`

---

## Sessão 5 — Slides 12-14 (Conclusões + Decisões) + Revisão Geral — ~2h

**Ler:**
- [ ] `docs/meeting/advisor-meeting-speaker-notes.md` — slides 12-14 (30 min)
- [ ] `docs/meeting/advisor-meeting-study-guide.md` — seção 3 (Perguntas Prováveis) (60 min)
- [ ] `docs/meeting/2026-03-19-advisor-meeting.md` — seções finais (decisões, próximos passos) (30 min)

**Preparar para a reunião:**
- [ ] Escrever em 1 parágrafo: "Rodamos 167 experimentos em 4 campanhas e descobrimos que..."
- [ ] Listar 3 decisões para levar ao orientador (S5 Two-Stage vs escrita; escopo do artigo; validação estatística)
- [ ] Memorizar 3 citações críticas: Sommer & Paxson 2010; Kopmann 2022; Lakhina 2004/2005

**Auto-teste final (dry-run mental da apresentação):**
- [ ] Consigo falar cada slide em ~1 min?
- [ ] Quais são meus 2-3 slides mais fracos? → reler speaker notes deles

---

## Sessão 6 (Quinta à noite ou sexta de manhã) — Revisão Leve — ~1h

- [ ] Releitura rápida do study-guide seções 1 (citações) e 3 (Q&A)
- [ ] Cartão de emergência na cabeça: 5 fatos + 3 citações
- [ ] Anotar as 3 perguntas/decisões numa folha para levar

---

## Buffer / Aprofundamento Opcional

Se sobrar tempo em qualquer sessão, ou se quiser ir além:

- [ ] `research/foundations/concept-drift.md`
- [ ] `research/summaries/gama-2013-prequential-evaluation.md`
- [ ] `research/summaries/angelov-2014-teda.md` (paper fundador)
- [ ] `experiments/streaming/src/detector/teda.py` (TEDA base)
- [ ] `experiments/streaming/src/detector/streaming_detector.py` (fluxo Kafka)
- [ ] `experiments/streaming/src/detector/window_aggregator.py` (features v2)
- [ ] `docs/architecture/CURRENT.md` (arquitetura sistema)
- [ ] Derivação de Welford (study guide §2.3)
- [ ] Curse of dimensionality (Beyer 1999, Aggarwal 2001)

---

## Dúvidas

> Anotar qualquer dúvida que surgir durante o estudo. Trazer para a próxima sessão com o Claude.

1.
2.
3.

---

## Notas de Estudo

> Espaço livre para anotações durante a leitura.

### Sessão 1 — Panorama


### Sessão 2 — Matemática TEDA


### Sessão 3 — C01 + C02 (baseline + 3 hipóteses)


### Sessão 4 — C03 + C04 (comportamentais + 5 adaptações)


### Sessão 5 — Conclusões + Revisão


### Outros

