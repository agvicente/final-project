#!/usr/bin/env python3
"""
Baselines de detecção de drift EXTERNOS (ADWIN, Page-Hinkley) vs sinal INTRÍNSECO (rho).

Esta é a comparação central da tese (HA3): o estado interno do MicroTEDAclus
(rho = sigma^2/r0) detecta o drift de ataque ANTES/melhor que os detectores de
drift externos padrão, que operam sobre o erro prequencial.

ADWIN e Page-Hinkley (river) consomem o `fading_error` POR FLUXO (logado no
detection_results pelo commit 18b45d6). O instante de detecção de cada um é o
1º fluxo após t_a em que `drift_detected` dispara. RLT vs externo:
    RLT_adwin = t_adwin - t_rho     (>0  => rho antecipa o ADWIN)
    RLT_ph    = t_ph    - t_rho

t_rho aqui = 1º fluxo após t_a em que rho_mean (por fluxo) cruza mu+3sigma da
base benigna. (Fase 0 mostrou rho_mean normalizado como o sinal válido.)

Uso:
    python drift_baselines.py <run_dir> [--signal rho_mean] [--out baselines.json]
"""
import argparse
import json
from pathlib import Path

from river import drift


def first_attack_index(recs):
    for i, r in enumerate(recs):
        if r.get("y_true"):
            return i
    return None


def detect_external(fading, t_a, make):
    """1º índice de fluxo após t_a em que o detector externo dispara."""
    det = make()
    for i, e in enumerate(fading):
        det.update(float(e))
        if det.drift_detected and i >= t_a:
            return i
    return None


def detect_rho(recs, t_a, signal):
    """
    1º fluxo após t_a em que `signal` (ex.: rho_mean) cruza mu+3sigma da base
    benigna (fluxos antes de t_a). Retorna (t_rho_3sigma, t_rho_eq1, mu, sd).
    """
    base = [r.get(signal, 0.0) for r in recs[:t_a]]
    if not base:
        return None, None, None, None
    n = len(base)
    mu = sum(base) / n
    sd = (sum((x - mu) ** 2 for x in base) / (n - 1)) ** 0.5 if n > 1 else 0.0
    thr = mu + 3 * sd
    t3 = teq1 = None
    for i in range(t_a, len(recs)):
        v = recs[i].get(signal, 0.0)
        if t3 is None and v > thr:
            t3 = i
        if teq1 is None and v > 1.0:
            teq1 = i
    return t3, teq1, mu, sd


def analyze(run_dir: Path, signal: str):
    d = json.loads((run_dir / "detection_results.json").read_text())
    recs = d.get("detection_results", [])
    if not recs:
        raise SystemExit(f"{run_dir}: detection_results vazio")
    t_a = first_attack_index(recs)
    if t_a is None:
        raise SystemExit(f"{run_dir}: sem ataque (y_true) — RLT não se aplica")

    fading = [r.get("fading_error", 0.0) for r in recs]
    t_adwin = detect_external(fading, t_a, lambda: drift.ADWIN())
    t_ph = detect_external(fading, t_a, lambda: drift.PageHinkley())
    t_rho3, t_rhoeq1, mu, sd = detect_rho(recs, t_a, signal)

    def rlt(t_ext, t_rho):
        return (t_ext - t_rho) if (t_ext is not None and t_rho is not None) else None

    out = {
        "run": run_dir.name,
        "signal": signal,
        "t_a": t_a,
        "n_flows": len(recs),
        "t_rho_3sigma": t_rho3,
        "t_rho_eq1": t_rhoeq1,
        "t_adwin": t_adwin,
        "t_page_hinkley": t_ph,
        # latencias absolutas (instante - t_a): quanto cada sinal demora a detectar
        "lat_rho_3sigma": (t_rho3 - t_a) if t_rho3 is not None else None,
        "lat_adwin": (t_adwin - t_a) if t_adwin is not None else None,
        "lat_page_hinkley": (t_ph - t_a) if t_ph is not None else None,
        # RLT vs externo: >0 => rho antecipa o detector externo
        "RLT_vs_adwin": rlt(t_adwin, t_rho3),
        "RLT_vs_page_hinkley": rlt(t_ph, t_rho3),
        "rho_base_mu": mu,
        "rho_base_sd": sd,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Baselines externos (ADWIN/PH) vs rho intrínseco")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--signal", default="rho_mean",
                    help="sinal intrínseco por-fluxo (default rho_mean; normalizado na Fase 0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = [analyze(Path(rd), args.signal) for rd in args.run_dirs]
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nsalvo: {args.out}")


if __name__ == "__main__":
    main()
