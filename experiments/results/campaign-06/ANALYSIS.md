# Campaign-06 — IoT Validation of Regime Transition Hypothesis

**Status:** ⏳ Skeleton (pre-execution). To be filled after `run_campaign06_normalize.sh` finishes on the Linux machine.

**Hypothesis under test (from `research/foundations/regime-transition.md`):**
> The regime transition observed synthetically in Exp 3 (1.620 runs) — where V0 and V7 transition between r0-bounded and data-bounded regimes governed by $\sigma^2$ vs $r_0$ — should also manifest in real CICIoT2023 IoT data. Specifically, *normalizing features* projects raw IoT data into unit-variance space; if the regime is governed solely by feature scale, V0 and V7 should converge into the same operational regime under normalization.

**Predictions:**
- **P1 (raw, baseline):** V0 fragments catastrophically (~1.000 clusters, FPR 50–75%) ; `micro_teda` collapses to long-tail (~130–700 clusters, FPR 1–4%). Reproduces Campaign-04/05 results.
- **P2 (normalized):** V0 and `micro_teda` converge to similar regime — comparable cluster counts, comparable FPRs. Either both data-bounded or both r0-bounded depending on warmup statistics.
- **P3 (Cohen's d):** V0 vs V7 effect size on FPR drops dramatically under normalization (from $|d| \gg 1$ raw to $|d| < 0{,}5$ normalized).

**Falsifiability:** if V0 still fragments under normalization, the regime transition isn't governed solely by feature scale — there's an additional mechanism specific to V0's `(2/d)^2` formula or the n<3 guard.

---

## 1. Setup

- **Algorithms:** V0 (`variant_V0_original`), V7 (`micro_teda`).
- **Scenarios:** benign, ddos (DDoS-ICMP), mirai, recon (Recon-PortScan).
- **Conditions:** raw vs normalized (z-score, warmup_size=200).
- **Seed:** 42 (single seed; statistical comparison via the synthetic Exp 3 already established).
- **Total runs:** 16.
- **Hardware:** Linux machine.
- **Estimated time:** 3–4h.

Configuration:
- `r0 = 0{,}10`
- `max_packets = 50.000`
- `max_flows = 10.000`
- `granularity = flow` (per-flow detection)
- `feature_set = v1` (17 features)
- `ground_truth = ip` (IP-based)

Script: `experiments/streaming/scripts/run_campaign06_normalize.sh`.

## 2. Resultados — TODO

### 2.1 Raw features (baseline, replicate Campaign-04/05)

To fill from `experiments/results/campaign-06/<algo>-<scenario>-raw-seed42/detection_results.json`:

| Algo | Scenario | FPR | Recall | F1 | n_clusters | top1_frac |
|---|---|---|---|---|---|---|
| V0 | benign | TBD | TBD | TBD | TBD | TBD |
| V0 | ddos | TBD | TBD | TBD | TBD | TBD |
| V0 | mirai | TBD | TBD | TBD | TBD | TBD |
| V0 | recon | TBD | TBD | TBD | TBD | TBD |
| V7 | benign | TBD | TBD | TBD | TBD | TBD |
| V7 | ddos | TBD | TBD | TBD | TBD | TBD |
| V7 | mirai | TBD | TBD | TBD | TBD | TBD |
| V7 | recon | TBD | TBD | TBD | TBD | TBD |

### 2.2 Normalized features (z-score)

| Algo | Scenario | FPR | Recall | F1 | n_clusters | top1_frac |
|---|---|---|---|---|---|---|
| V0 | benign | TBD | TBD | TBD | TBD | TBD |
| V0 | ddos | TBD | TBD | TBD | TBD | TBD |
| V0 | mirai | TBD | TBD | TBD | TBD | TBD |
| V0 | recon | TBD | TBD | TBD | TBD | TBD |
| V7 | benign | TBD | TBD | TBD | TBD | TBD |
| V7 | ddos | TBD | TBD | TBD | TBD | TBD |
| V7 | mirai | TBD | TBD | TBD | TBD | TBD |
| V7 | recon | TBD | TBD | TBD | TBD | TBD |

## 3. Verdict — TODO

| Prediction | Confirmed? | Notes |
|---|---|---|
| P1 (raw replicates earlier) | ⏳ | ... |
| P2 (norm convergence) | ⏳ | ... |
| P3 (Cohen's d collapses) | ⏳ | ... |

## 4. Implication for the paper

- If P1+P2+P3 all confirm: §V "IoT Manifestation" gets a paragraph showing the regime hypothesis transfers from synthetic to real data. Strong contribution.
- If only P1 confirms: paper notes "preliminary IoT normalization results inconclusive; full validation deferred to future work". Acceptable.
- If P1 fails: re-examine experimental setup; data integrity check on Campaign-06 first.
