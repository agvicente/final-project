#!/usr/bin/env python3
"""
Compara runs de r0 distintos (HA4: RLT/sinal variam com r0 = fenomeno de regime).

Para cada run_dir, gera serie_temporal.csv (se ausente), calcula os instantes de
drift (drift_instants) e tabula F1_pre, t_rho, t_F, RLT + diagnostico de regime
(detector mudo? clusters data-bounded?). Saida: tabela no stdout + r0_summary.csv.

Uso:
    python compare_r0.py <run_dir1> <run_dir2> ... [--window 200] [--out r0_summary.csv]
    # cada run_dir deve ter run_meta.json (para ler r0) e detection_results.json
"""
import argparse
import csv
import json
from pathlib import Path

import build_serie_temporal
from drift_instants import read_serie, compute_instants


def r0_of(run_dir: Path):
    try:
        m = json.loads((run_dir / "run_meta.json").read_text())
        return m.get("params", {}).get("r0")
    except Exception:
        return None


def regime_diag(rows, t_a):
    """Fracao media de clusters data-bounded (rho>1) na base e no ataque."""
    base = [r for r in rows if r["flow_end"] < t_a] if t_a else rows
    post = [r for r in rows if r["flow_start"] >= t_a] if t_a else []
    def avg(g, k):
        return sum(r[k] for r in g) / len(g) if g else 0.0
    return {
        "frac_above_1_base": round(avg(base, "rho_frac_above_1"), 4),
        "frac_above_1_post": round(avg(post, "rho_frac_above_1"), 4),
        "rho_mean_base": round(avg(base, "rho_mean"), 4),
        "rho_mean_post": round(avg(post, "rho_mean"), 4),
        "nclusters_base": round(avg(base, "num_clusters"), 1),
        "nclusters_post": round(avg(post, "num_clusters"), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Compara runs de r0 distintos (HA4)")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table = []
    for rd in args.run_dirs:
        rd = Path(rd)
        serie = rd / "serie_temporal.csv"
        if not serie.exists():
            build_serie_temporal.build(rd, args.window)
        rows, t_a, _ = read_serie(serie)
        inst = compute_instants(rows, t_a)
        diag = regime_diag(rows, t_a)
        row = {
            "run": rd.name,
            "r0": r0_of(rd),
            "t_a": t_a,
            "f1_pre": round(inst["f1_pre"], 4) if inst["f1_pre"] is not None else None,
            "mudo": (inst["f1_pre"] is not None and inst["f1_pre"] < 1e-6),
            "t_rho_3sigma": inst["t_rho_3sigma"],
            "t_rho_eq1": inst["t_rho_eq1"],
            "t_F": inst["t_F"],
            "RLT_3sigma": inst["rlt_3sigma"],
            "RLT_eq1": inst["rlt_eq1"],
            **diag,
            "warning": inst["warning"],
        }
        table.append(row)

    # ordenar por r0 desc
    table.sort(key=lambda r: (r["r0"] is None, -(r["r0"] or 0)))

    cols = ["run", "r0", "t_a", "f1_pre", "mudo", "t_rho_3sigma", "t_rho_eq1",
            "t_F", "RLT_3sigma", "RLT_eq1", "frac_above_1_base", "frac_above_1_post",
            "rho_mean_base", "rho_mean_post", "nclusters_base", "nclusters_post"]
    print("\n=== HA4: sinal e RLT vs r0 ===")
    print(" | ".join(f"{c}" for c in ["r0", "f1_pre", "mudo", "t_rho_3s", "t_F", "RLT_3s",
                                       "frac1_base", "frac1_post", "rho_base", "rho_post"]))
    for r in table:
        print(" | ".join(str(r[k]) for k in ["r0", "f1_pre", "mudo", "t_rho_3sigma",
              "t_F", "RLT_3sigma", "frac_above_1_base", "frac_above_1_post",
              "rho_mean_base", "rho_mean_post"]))
        if r["warning"]:
            print(f"     ^ {r['r0']}: {r['warning']}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in table:
                w.writerow({k: r.get(k) for k in cols})
        print(f"\nsalvo: {args.out}")


if __name__ == "__main__":
    main()
