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


def detect_on(series, t_a, make):
    """
    1º índice de fluxo após t_a em que o detector `make()` dispara, alimentado
    com `series` (qualquer série por-fluxo: erro fading OU o sinal rho).
    Generaliza detect_external — o ponto da tese é comparar a MESMA família de
    detector (ADWIN/PH) com ENTRADAS diferentes (erro prequencial vs rho).
    """
    det = make()
    for i, v in enumerate(series):
        det.update(float(v))
        if det.drift_detected and i >= t_a:
            return i
    return None


# alias retrocompat
def detect_external(fading, t_a, make):
    return detect_on(fading, t_a, make)


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
    rho_series = [r.get(signal, 0.0) for r in recs]

    # MESMO detector, ENTRADAS distintas (a comparação justa central da tese):
    #   *_err = detector sobre o erro prequencial (prática padrão / baseline)
    #   *_rho = detector sobre o sinal intrínseco rho (nossa proposta)
    t_adwin_err = detect_on(fading, t_a, lambda: drift.ADWIN())
    t_ph_err = detect_on(fading, t_a, lambda: drift.PageHinkley())
    t_adwin_rho = detect_on(rho_series, t_a, lambda: drift.ADWIN())
    t_ph_rho = detect_on(rho_series, t_a, lambda: drift.PageHinkley())

    # sinal 3sigma-instantaneo (secundario; Fase 0 mostrou fragil em base ruidosa)
    t_rho3, t_rhoeq1, mu, sd = detect_rho(recs, t_a, signal)

    def lat(t):
        return (t - t_a) if t is not None else None

    def lead(t_err, t_rho):  # >0 => rho-input antecipa erro-input (MESMO detector)
        return (t_err - t_rho) if (t_err is not None and t_rho is not None) else None

    out = {
        "run": run_dir.name,
        "signal": signal,
        "t_a": t_a,
        "n_flows": len(recs),
        # detector sobre ERRO (baseline) vs sobre RHO (proposta) — latencias
        "lat_adwin_err": lat(t_adwin_err),
        "lat_adwin_rho": lat(t_adwin_rho),
        "lat_ph_err": lat(t_ph_err),
        "lat_ph_rho": lat(t_ph_rho),
        # LEAD = quanto rho-input antecipa erro-input (MESMO detector) — a metrica chave
        "lead_adwin": lead(t_adwin_err, t_adwin_rho),
        "lead_ph": lead(t_ph_err, t_ph_rho),
        # 3sigma instantaneo (secundario)
        "t_rho_3sigma": t_rho3,
        "t_rho_eq1": t_rhoeq1,
        "lat_rho_3sigma": lat(t_rho3),
        "rho_base_mu": mu,
        "rho_base_sd": sd,
    }
    return out


def compare_signals(run_dir: Path, signals):
    """
    Para um run, mede t_externo (ADWIN/PH) UMA vez e t_rho_3sigma para cada sinal
    candidato. Retorna dict com latencias e RLT por sinal — para escolher o sinal
    intrinseco mais robusto (detecta sempre? menor latencia vs externo?).
    """
    d = json.loads((run_dir / "detection_results.json").read_text())
    recs = d.get("detection_results", [])
    t_a = first_attack_index(recs)
    if t_a is None:
        return {"run": run_dir.name, "error": "sem ataque"}
    fading = [r.get("fading_error", 0.0) for r in recs]
    t_adwin = detect_external(fading, t_a, lambda: drift.ADWIN())
    t_ph = detect_external(fading, t_a, lambda: drift.PageHinkley())
    row = {"run": run_dir.name, "t_a": t_a, "t_adwin": t_adwin,
           "t_ph": t_ph, "lat_adwin": (t_adwin - t_a) if t_adwin else None,
           "lat_ph": (t_ph - t_a) if t_ph else None, "signals": {}}
    for sig in signals:
        t3, teq1, mu, sd = detect_rho(recs, t_a, sig)
        row["signals"][sig] = {
            "t_rho_3sigma": t3,
            "lat": (t3 - t_a) if t3 is not None else None,
            "detectou": t3 is not None,
            "RLT_vs_adwin": (t_adwin - t3) if (t_adwin and t3) else None,
            "RLT_vs_ph": (t_ph - t3) if (t_ph and t3) else None,
            "base_cv": (sd / mu) if (mu and abs(mu) > 1e-9) else None,  # coef. de variacao da base
        }
    return row


def main():
    ap = argparse.ArgumentParser(description="Baselines externos (ADWIN/PH) vs rho intrínseco")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--signal", default="rho_mean",
                    help="sinal intrínseco por-fluxo (default rho_mean; normalizado na Fase 0)")
    ap.add_argument("--compare-signals", default=None,
                    help="lista de sinais separada por virgula p/ comparar robustez "
                         "(ex: rho_mean,rho_p90,rho_max,rho_frac_above_1,num_clusters)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare_signals:
        sigs = [s.strip() for s in args.compare_signals.split(",")]
        results = [compare_signals(Path(rd), sigs) for rd in args.run_dirs]
        # resumo legivel: por sinal, em quantos runs detectou + RLT medio vs PH
        print(json.dumps(results, indent=2))
        print("\n=== RESUMO por sinal (robustez) ===")
        print(f"{'sinal':18s} {'detectou':10s} {'lat_media':10s} {'RLT_vs_PH_med':14s} {'RLT_vs_ADWIN_med'}")
        for sig in sigs:
            dets = [r["signals"][sig] for r in results if "signals" in r]
            ndet = sum(1 for s in dets if s["detectou"])
            lats = [s["lat"] for s in dets if s["lat"] is not None]
            rph = [s["RLT_vs_ph"] for s in dets if s["RLT_vs_ph"] is not None]
            radw = [s["RLT_vs_adwin"] for s in dets if s["RLT_vs_adwin"] is not None]
            am = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
            print(f"{sig:18s} {ndet}/{len(dets):8d} {str(am(lats)):10s} {str(am(rph)):14s} {am(radw)}")
    else:
        results = [analyze(Path(rd), args.signal) for rd in args.run_dirs]
        print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nsalvo: {args.out}")


if __name__ == "__main__":
    main()
