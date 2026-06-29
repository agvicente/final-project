#!/usr/bin/env python3
"""
Plotter de figuras de drift (serie temporal) — A-0, A-1, A-2.

Estende o padrao dos generate_plots.py das campanhas (que so plotam escalar
agregado) para ler serie_temporal.csv e produzir figuras de SERIE TEMPORAL com
axvline nos instantes de drift, conforme o catalogo de
study-wiki/experimentos/11-outputs-figuras.md §5.

Figuras geradas (as que dependem so de serie_temporal.csv):
  A-0  Fase 0: rho_mean e c_rate ao longo do tempo, marcador t_a (verificacao)
  A-1  rho(t) vs F1(t), eixo-y duplo, verticais t_a / t_rho(3sigma) / t_F  -> RLT visual
  A-2  burst de c_rate com banda mu+-3sigma benigna, marcador t_a e pico

(A-3, A-4, B-*, C-* dependem de latencias.csv/zeroday.csv e vem depois do piloto.)

Uso:
    python generate_drift_plots.py <run_dir> [--outdir <dir>] [--window 200]
    # le <run_dir>/serie_temporal.csv (gera se ausente via build_serie_temporal)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (VM sem display)
import matplotlib.pyplot as plt

from drift_instants import read_serie, compute_instants


def _ensure_serie(run_dir: Path, window: int) -> Path:
    serie = run_dir / "serie_temporal.csv"
    if not serie.exists():
        import build_serie_temporal
        build_serie_temporal.build(run_dir, window)
    return serie


def _x_flow(rows):
    """Eixo x em indice de fluxo (flow_start de cada janela)."""
    return [r["flow_start"] for r in rows]


def fig_a0(rows, inst, outdir: Path):
    """A-0: rho_mean e c_rate ao longo do tempo (verificacao de instrumentacao)."""
    x = _x_flow(rows)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, [r["rho_mean"] for r in rows], color="tab:blue", label="rho_mean")
    ax1.set_xlabel("fluxo")
    ax1.set_ylabel("rho_mean = mean(sigma^2/r0)", color="tab:blue")
    ax1.set_yscale("symlog")  # rho tem escala enorme (feature crua nao-normalizada)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, [r["c_rate"] for r in rows], color="tab:orange",
             alpha=0.6, label="c_rate")
    ax2.set_ylabel("c_rate (novos clusters/janela)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    if inst["t_a"] is not None:
        ax1.axvline(inst["t_a"], color="red", ls="--", label="t_a (1o IP atacante)")
    ax1.set_title("A-0 | Fase 0: rho e c_rate ao longo do tempo")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    p = outdir / "A-0_fase0_rho_crate.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_a1(rows, inst, outdir: Path):
    """A-1: rho(t) vs F1(t), eixo-y duplo, com t_a/t_rho/t_F -> RLT visual."""
    x = _x_flow(rows)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, [r["rho_mean"] for r in rows], color="tab:blue", label="rho_mean")
    ax1.set_xlabel("fluxo")
    ax1.set_ylabel("rho_mean", color="tab:blue")
    ax1.set_yscale("symlog")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, [r["f1_w"] for r in rows], color="tab:green", label="F1 (janela)")
    ax2.set_ylabel("F1 de janela", color="tab:green")
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    marks = [
        ("t_a", inst["t_a"], "red", "-"),
        ("t_rho(3sig)", inst["t_rho_3sigma"], "purple", "--"),
        ("t_F", inst["t_F"], "black", ":"),
    ]
    for label, val, color, ls in marks:
        if val is not None:
            ax1.axvline(val, color=color, ls=ls, label=label)

    rlt = inst.get("rlt_3sigma")
    title = "A-1 | rho(t) antecipa F1(t)"
    if rlt is not None:
        title += f"  (RLT_3sigma = {rlt} fluxos)"
    elif inst.get("warning"):
        title += f"  [{inst['warning'][:40]}]"
    ax1.set_title(title)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    p = outdir / "A-1_rho_vs_f1.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_a2(rows, inst, outdir: Path):
    """A-2: burst de c_rate com banda mu+-3sigma benigna."""
    x = _x_flow(rows)
    c = [r["c_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, c, color="tab:orange", label="c_rate")

    # banda benigna (base = janelas pre-t_a)
    if inst["t_a"] is not None:
        base = [r["c_rate"] for r in rows if r["flow_end"] < inst["t_a"]]
        if base:
            mu = sum(base) / len(base)
            sd = (sum((v - mu) ** 2 for v in base) / max(1, len(base) - 1)) ** 0.5
            ax.axhline(mu, color="gray", ls="-", alpha=0.5, label="mu benigno")
            ax.axhspan(mu - 3 * sd, mu + 3 * sd, color="gray", alpha=0.15,
                       label="mu +- 3sigma")
        ax.axvline(inst["t_a"], color="red", ls="--", label="t_a")

    if c:
        peak_i = max(range(len(c)), key=lambda i: c[i])
        ax.plot(x[peak_i], c[peak_i], "rv", markersize=10, label="pico")
    ax.set_xlabel("fluxo")
    ax.set_ylabel("c_rate (novos clusters/janela)")
    ax.set_title("A-2 | burst de criacao de clusters")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p = outdir / "A-2_burst_crate.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Plotter de figuras de drift (serie temporal)")
    ap.add_argument("run_dir", help="diretorio do run")
    ap.add_argument("--outdir", default=None, help="dir de saida (default: <run_dir>/plots)")
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir) if args.outdir else run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    serie = _ensure_serie(run_dir, args.window)
    rows, t_a, window = read_serie(serie)
    inst = compute_instants(rows, t_a)

    produced = [
        fig_a0(rows, inst, outdir),
        fig_a1(rows, inst, outdir),
        fig_a2(rows, inst, outdir),
    ]
    print(f"t_a_flow={t_a}  RLT_3sigma={inst.get('rlt_3sigma')}  "
          f"t_rho_3sigma={inst.get('t_rho_3sigma')}  t_F={inst.get('t_F')}")
    if inst.get("warning"):
        print(f"AVISO: {inst['warning']}")
    for p in produced:
        print(f"  figura: {p}")


if __name__ == "__main__":
    main()
