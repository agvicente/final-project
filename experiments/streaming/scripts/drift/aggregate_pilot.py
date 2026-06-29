#!/usr/bin/env python3
"""
Agrega o piloto de A sobre N composicoes: latencia de deteccao do drift com o
MESMO detector (ADWIN/Page-Hinkley) alimentado por rho vs pelo erro prequencial.

Saida: por composicao, lat_adwin_err/rho, lat_ph_err/rho e o lead; e um RESUMO
com media/desvio do lead + teste de Wilcoxon pareado (lat_err vs lat_rho) — a
estatistica que sustenta HA3 (sinal intrinseco detecta o drift antes do erro).

Uso:
    python aggregate_pilot.py <run_dir>... [--signal rho_mean] [--out pilot_summary.json]
"""
import argparse
import json
import statistics as st
from pathlib import Path

from scipy import stats as sstats

import drift_baselines as db


def main():
    ap = argparse.ArgumentParser(description="Agrega piloto de A (lead rho vs erro)")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--signal", default="rho_mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for rd in args.run_dirs:
        try:
            rows.append(db.analyze(Path(rd), args.signal))
        except SystemExit as e:
            print(f"  skip {rd}: {e}")

    # vetores pareados de latencia (so composicoes onde ambos detectaram)
    def paired(err_key, rho_key):
        e, r = [], []
        for row in rows:
            if row.get(err_key) is not None and row.get(rho_key) is not None:
                e.append(row[err_key]); r.append(row[rho_key])
        return e, r

    summary = {"n_comps": len(rows), "signal": args.signal, "per_comp": rows, "tests": {}}

    for name, ek, rk, leadk in [
        ("adwin", "lat_adwin_err", "lat_adwin_rho", "lead_adwin"),
        ("page_hinkley", "lat_ph_err", "lat_ph_rho", "lead_ph"),
    ]:
        e, r = paired(ek, rk)
        leads = [row[leadk] for row in rows if row.get(leadk) is not None]
        block = {
            "n_paired": len(e),
            "lat_err_mean": round(st.mean(e), 1) if e else None,
            "lat_rho_mean": round(st.mean(r), 1) if r else None,
            "lead_mean": round(st.mean(leads), 1) if leads else None,
            "lead_sd": round(st.pstdev(leads), 1) if len(leads) > 1 else 0.0,
            "lead_min": min(leads) if leads else None,
            "lead_max": max(leads) if leads else None,
            "lead_positive_frac": round(sum(1 for x in leads if x > 0) / len(leads), 2) if leads else None,
        }
        # Wilcoxon pareado: lat_err vs lat_rho (H0: mesma latencia)
        if len(e) >= 5 and any(a != b for a, b in zip(e, r)):
            try:
                w = sstats.wilcoxon(e, r)
                block["wilcoxon_stat"] = float(w.statistic)
                block["wilcoxon_p"] = float(w.pvalue)
            except Exception as ex:
                block["wilcoxon_error"] = str(ex)
        else:
            block["wilcoxon_note"] = f"n={len(e)} < 5 (piloto): reportar descritivo, nao inferencial"
        summary["tests"][name] = block

    print(json.dumps(summary, indent=2))
    print("\n=== RESUMO PILOTO (lead = quanto rho antecipa o erro, MESMO detector) ===")
    for name, block in summary["tests"].items():
        print(f"\n{name.upper()}: n_pareado={block['n_paired']}")
        print(f"  latencia media: erro={block['lat_err_mean']}  rho={block['lat_rho_mean']} fluxos")
        print(f"  lead: media={block['lead_mean']} +-{block['lead_sd']}  "
              f"(min {block['lead_min']}, max {block['lead_max']}, "
              f"positivo em {block['lead_positive_frac']} das comps)")
        if "wilcoxon_p" in block:
            print(f"  Wilcoxon pareado (erro vs rho): p={block['wilcoxon_p']:.4g}")
        elif "wilcoxon_note" in block:
            print(f"  {block['wilcoxon_note']}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nsalvo: {args.out}")


if __name__ == "__main__":
    main()
