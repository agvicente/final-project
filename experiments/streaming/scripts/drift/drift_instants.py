#!/usr/bin/env python3
"""
Calculo dos instantes de drift (t_rho, t_c, t_F) e do RLT sobre serie_temporal.csv.

Implementa o algoritmo de study-wiki/experimentos/11-outputs-figuras.md §3.1.
Reutilizado pelo plotter (generate_drift_plots.py) e pelo extrator de latencias.

Convencao de instantes: todos em INDICE DE FLUXO (flow_start da janela), nao em
indice de janela — para casar com t_a (1o IP atacante, em fluxos) e medir RLT em
fluxos, como definido na metrica M-A1.

t_rho tem DOIS criterios reportados SEPARADAMENTE (resolve o achado A2 da auditoria):
  - t_rho_3sigma: rho_mean > mu_base + 3*sd_base   (operacional, funciona em qualquer r0)
  - t_rho_eq1:    rho_mean > 1.0                    (teorico, fronteira de regime)
"""
import csv
import math
from pathlib import Path


def read_serie(path: Path):
    """Le serie_temporal.csv. Retorna (rows, t_a_flow, window)."""
    t_a_flow = None
    window = None
    data_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith("# t_a_flow="):
                v = line.strip().split("=", 1)[1]
                t_a_flow = None if v == "None" else int(v)
            elif line.startswith("# window="):
                window = int(line.strip().split("=", 1)[1])
            else:
                data_lines.append(line)
    reader = csv.DictReader(data_lines)
    rows = []
    for r in reader:
        rows.append({
            "win": int(r["win"]),
            "flow_start": int(r["flow_start"]),
            "flow_end": int(r["flow_end"]),
            "rho_mean": float(r["rho_mean"]),
            # rho_median e o sinal robusto principal (Fase 0 mostrou rho_mean instavel);
            # fallback para rho_mean se a coluna nao existir (CSVs antigos).
            "rho_median": float(r.get("rho_median", r["rho_mean"])),
            "rho_p90": float(r.get("rho_p90", r["rho_mean"])),
            "rho_max": float(r["rho_max"]),
            "c_rate": float(r["c_rate"]),
            "num_clusters": float(r["num_clusters"]),
            "f1_w": float(r["f1_w"]),
            "fpr_w": float(r["fpr_w"]),
            "recall_w": float(r["recall_w"]),
            "attacks": int(r["attacks"]),
            "n": int(r["n"]),
        })
    return rows, t_a_flow, window


def _mean_sd(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def compute_instants(rows, t_a_flow):
    """
    Calcula t_rho_3sigma, t_rho_eq1, t_c, t_F e os dois RLTs.

    Base (warm-up benigno) = janelas cujo flow_end < t_a_flow. Se nao houver base
    benigna (t_a no comeco), os limiares 3sigma ficam indefinidos e os instantes
    relativos retornam None — sinalizando que a composicao precisa de warm-up.

    Retorna dict com instantes (em fluxo) e diagnostico da base.
    """
    out = {
        "t_a": t_a_flow,
        "t_rho_3sigma": None, "t_rho_eq1": None, "t_c": None, "t_F": None,
        "rlt_3sigma": None, "rlt_eq1": None,
        "f1_pre": None, "mu_rho": None, "sd_rho": None,
        "signal": "rho_median",  # sinal robusto principal (Fase 0)
        "n_base_windows": 0, "warning": None,
    }
    if t_a_flow is None:
        out["warning"] = "sem ataque na serie (benign-only) — instantes nao se aplicam"
        return out

    base = [r for r in rows if r["flow_end"] < t_a_flow]
    post = [r for r in rows if r["flow_start"] >= t_a_flow]
    out["n_base_windows"] = len(base)
    if not base:
        out["warning"] = ("sem janela benigna de warm-up antes de t_a "
                          "(t_a no comeco) — limiares 3sigma indefinidos")
        return out

    # sinal de regime = rho_median (robusto a outliers de escala; Fase 0 mostrou
    # rho_mean dominado por 1 cluster com variancia astronomica).
    mu_rho, sd_rho = _mean_sd([r["rho_median"] for r in base])
    mu_c, sd_c = _mean_sd([r["c_rate"] for r in base])
    f1_pre = sum(r["f1_w"] for r in base) / len(base)
    out.update(mu_rho=mu_rho, sd_rho=sd_rho, f1_pre=f1_pre)

    thr_rho = mu_rho + 3 * sd_rho
    thr_c = mu_c + 3 * sd_c

    def first(seq, cond):
        for r in seq:
            if cond(r):
                return r["flow_start"]
        return None

    out["t_rho_3sigma"] = first(post, lambda r: r["rho_median"] > thr_rho)
    out["t_rho_eq1"] = first(post, lambda r: r["rho_median"] > 1.0)
    out["t_c"] = first(post, lambda r: r["c_rate"] > thr_c)

    # t_F: queda relativa de F1. Se F1_pre ~ 0 (detector mudo), t_F indefinido.
    if f1_pre > 1e-6:
        out["t_F"] = first(post, lambda r: r["f1_w"] < 0.8 * f1_pre)
    else:
        out["warning"] = ("F1_pre ~ 0 (detector mudo neste r0) — t_F/RLT nao se "
                          "aplicam (regime documentado, nao refutacao)")

    if out["t_F"] is not None and out["t_rho_3sigma"] is not None:
        out["rlt_3sigma"] = out["t_F"] - out["t_rho_3sigma"]
    if out["t_F"] is not None and out["t_rho_eq1"] is not None:
        out["rlt_eq1"] = out["t_F"] - out["t_rho_eq1"]

    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Calcula instantes de drift de uma serie")
    ap.add_argument("serie_csv", help="caminho do serie_temporal.csv")
    args = ap.parse_args()
    rows, t_a, window = read_serie(Path(args.serie_csv))
    inst = compute_instants(rows, t_a)
    print(json.dumps(inst, indent=2))
