#!/usr/bin/env python3
"""
Figuras agregadas da matriz do Experimento A (A-3 latencia por detector, A-4 lead x r0).

Le os runs da matriz (dirs <atk>_r<r0>_c<i>/), roda drift_baselines.analyze em
cada um, e gera:
  A-3  latencia de deteccao do drift por detector (ADWIN/PH) e por entrada (erro vs rho),
       barras com IC bootstrap 95%, agregado sobre composicoes. Por ataque.
  A-4  lead (rho antecipa erro) x r0, por ataque — testa HA4 (dependencia de regime).
Tambem salva matrix_summary.csv (1 linha por run) e imprime os Wilcoxon por celula.

Uso:
    python plot_matrix.py <matrix_dir> [--signal rho_mean] [--outdir <dir>]
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

import drift_baselines as db

CELL_RE = re.compile(r"^(?P<atk>.+)_r(?P<r0>[0-9.]+)_c(?P<c>\d+)$")


def parse_cell(name):
    m = CELL_RE.match(name)
    if not m:
        return None
    return m.group("atk"), float(m.group("r0")), int(m.group("c"))


def bootstrap_ci(xs, n=2000):
    """IC 95% da media por bootstrap (sem dependencia externa de seed)."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (None, None)
    arr = np.array(xs, dtype=float)
    # numpy default_rng com seed fixo (reprodutivel; nao usa Math.random global)
    rng = np.random.default_rng(12345)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def collect(matrix_dir: Path, signal: str):
    rows = []
    for d in sorted(matrix_dir.iterdir()):
        if not d.is_dir():
            continue
        cell = parse_cell(d.name)
        if not cell or not (d / "detection_results.json").exists():
            continue
        atk, r0, c = cell
        try:
            res = db.analyze(d, signal)
        except SystemExit:
            continue
        res.update(atk=atk, r0=r0, comp=c)
        rows.append(res)
    return rows


def fig_a3(rows, outdir: Path):
    """A-3: latencia por detector e entrada (erro vs rho), barras+IC, por ataque."""
    atks = sorted({r["atk"] for r in rows})
    metrics = [("ADWIN-erro", "lat_adwin_err"), ("ADWIN-rho", "lat_adwin_rho"),
               ("PH-erro", "lat_ph_err"), ("PH-rho", "lat_ph_rho")]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(atks))
    w = 0.2
    colors = ["#bbbbbb", "#1f77b4", "#dddddd", "#ff7f0e"]
    for j, (label, key) in enumerate(metrics):
        means, los, his = [], [], []
        for atk in atks:
            vals = [r[key] for r in rows if r["atk"] == atk and r[key] is not None]
            m = np.mean(vals) if vals else 0
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            los.append(m - lo if lo is not None else 0)
            his.append(hi - m if hi is not None else 0)
        ax.bar(x + (j - 1.5) * w, means, w, label=label, color=colors[j],
               yerr=[los, his], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(atks)
    ax.set_ylabel("latencia de deteccao do drift (fluxos apos t_a)")
    ax.set_title("A-3 | latencia: detector sobre ERRO vs sobre RHO (menor = melhor)")
    ax.legend()
    fig.tight_layout()
    p = outdir / "A-3_latencia_por_detector.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_a4(rows, outdir: Path):
    """A-4: lead (rho antecipa erro) x r0, por ataque — HA4."""
    atks = sorted({r["atk"] for r in rows})
    r0s = sorted({r["r0"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (leadkey, title) in zip(axes, [("lead_adwin", "ADWIN"), ("lead_ph", "Page-Hinkley")]):
        for atk in atks:
            ys, yerr = [], []
            for r0 in r0s:
                vals = [r[leadkey] for r in rows
                        if r["atk"] == atk and r["r0"] == r0 and r[leadkey] is not None]
                ys.append(np.mean(vals) if vals else np.nan)
                yerr.append(np.std(vals) if len(vals) > 1 else 0)
            ax.errorbar([str(r) for r in r0s], ys, yerr=yerr, marker="o", label=atk, capsize=3)
        ax.axhline(0, color="red", ls="--", alpha=0.5)
        ax.set_xlabel("r0 (regime)"); ax.set_ylabel(f"lead = t_erro - t_rho ({title})")
        ax.set_title(f"A-4 | lead x r0 ({title})  [>0: rho antecipa]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "A-4_lead_vs_r0.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def per_cell_tests(rows):
    """Wilcoxon pareado (lat_err vs lat_rho) por (ataque x r0 x detector)."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["atk"], r["r0"])
        for det, ek, rk in [("adwin", "lat_adwin_err", "lat_adwin_rho"),
                            ("ph", "lat_ph_err", "lat_ph_rho")]:
            if r[ek] is not None and r[rk] is not None:
                cells[key][det].append((r[ek], r[rk]))
    out = []
    for (atk, r0), dets in sorted(cells.items()):
        for det, pairs in dets.items():
            e = [a for a, b in pairs]; rr = [b for a, b in pairs]
            row = {"atk": atk, "r0": r0, "detector": det, "n": len(pairs),
                   "lat_err_mean": round(np.mean(e), 1) if e else None,
                   "lat_rho_mean": round(np.mean(rr), 1) if rr else None,
                   "lead_mean": round(np.mean([a - b for a, b in pairs]), 1) if pairs else None}
            if len(pairs) >= 6 and any(a != b for a, b in pairs):
                try:
                    w = sstats.wilcoxon(e, rr)
                    row["wilcoxon_p"] = round(float(w.pvalue), 5)
                except Exception as ex:
                    row["wilcoxon_p"] = f"err:{ex}"
            else:
                row["wilcoxon_p"] = None
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description="Figuras agregadas da matriz A")
    ap.add_argument("matrix_dir")
    ap.add_argument("--signal", default="rho_mean")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    md = Path(args.matrix_dir)
    outdir = Path(args.outdir) if args.outdir else md / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect(md, args.signal)
    print(f"runs analisados: {len(rows)}")
    if not rows:
        raise SystemExit("nenhum run valido na matriz ainda")

    # CSV tabular
    csv_path = md / "matrix_summary.csv"
    cols = ["atk", "r0", "comp", "t_a", "lat_adwin_err", "lat_adwin_rho",
            "lat_ph_err", "lat_ph_rho", "lead_adwin", "lead_ph"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    p3 = fig_a3(rows, outdir)
    p4 = fig_a4(rows, outdir)

    print("\n=== Wilcoxon por celula (ataque x r0 x detector) ===")
    tests = per_cell_tests(rows)
    for t in tests:
        sig = ""
        if isinstance(t["wilcoxon_p"], float):
            sig = " ***" if t["wilcoxon_p"] < 0.05 else ""
        print(f"  {t['atk']:9s} r0={t['r0']:<6} {t['detector']:5s} n={t['n']:2d} "
              f"lead={t['lead_mean']} (err={t['lat_err_mean']} rho={t['lat_rho_mean']}) "
              f"p={t['wilcoxon_p']}{sig}")
    print(f"\nfiguras: {p3}\n         {p4}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
