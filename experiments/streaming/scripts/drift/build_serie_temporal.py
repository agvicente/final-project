#!/usr/bin/env python3
"""
Agregador de serie temporal de drift (offline).

Le `detection_results.json` (serie por-fluxo, enriquecida pelo G3 com
rho_mean/rho_max/rho_frac_above_1/num_clusters/new_cluster_created/y_true/
fading_error) e `run_meta.json` (phase_markers), agrupa os fluxos em janelas
de tamanho fixo (default 200) e emite `serie_temporal.csv` no esquema de
study-wiki/experimentos/11-outputs-figuras.md §2.1.

Decisao de design: o G3 ja loga rho POR FLUXO no detection_results, entao nao
e preciso reconstruir a partir de snapshots periodicos de clusters_state — basta
agrupar os registros por-fluxo em janelas. f1/fpr/recall sao RECOMPUTADOS na
janela de 200 a partir de is_anomaly + y_true (granularidade fina), separados da
janela de desempenho de 1000 do metrics_windowed.csv.

Uso:
    python build_serie_temporal.py <run_dir> [--window 200]
    # le <run_dir>/detection_results.json e run_meta.json
    # escreve <run_dir>/serie_temporal.csv

O instante de drift t_a (1o fluxo de IP atacante) e gravado no cabecalho do CSV
(comentario `# t_a_flow=<idx>`) e tambem derivavel da coluna `attacks`.
"""
import argparse
import csv
import json
from pathlib import Path


def load_per_flow(run_dir: Path):
    """Carrega a lista por-fluxo do detection_results.json."""
    d = json.loads((run_dir / "detection_results.json").read_text())
    recs = d.get("detection_results", [])
    if not recs:
        raise SystemExit(
            f"detection_results.json em {run_dir} nao contem a serie por-fluxo "
            "(chave 'detection_results' vazia). Rode o experimento com a versao "
            "do run_experiment.py que reanexa stats['detection_results']."
        )
    return recs


def first_attack_index(recs):
    """t_a = indice do 1o fluxo com y_true=True (1o IP atacante visto)."""
    for i, r in enumerate(recs):
        if r.get("y_true"):
            return i
    return None  # nenhum ataque na serie (ex.: run benign-only)


def safe_div(a, b):
    return a / b if b else 0.0


def window_stats(chunk):
    """
    Estatisticas de uma janela de fluxos. f1/fpr/recall recomputados na janela
    a partir de is_anomaly (y_pred) vs y_true.
    """
    n = len(chunk)
    rho_mean = [r.get("rho_mean", 0.0) for r in chunk]
    rho_max = [r.get("rho_max", 0.0) for r in chunk]
    rho_fa1 = [r.get("rho_frac_above_1", 0.0) for r in chunk]
    nclust = [r.get("num_clusters", 0) for r in chunk]
    c_rate = sum(1 for r in chunk if r.get("new_cluster_created"))

    # matriz de confusao na janela (y_pred = is_anomaly, y_true = y_true)
    tp = fp = tn = fn = 0
    for r in chunk:
        yp = bool(r.get("is_anomaly"))
        yt = bool(r.get("y_true"))
        if yp and yt:
            tp += 1
        elif yp and not yt:
            fp += 1
        elif (not yp) and yt:
            fn += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    attacks = tp + fn  # fluxos de ataque na janela

    # fading_error: ultimo da janela (e um acumulador suave)
    fading = chunk[-1].get("fading_error", 0.0)

    return {
        "rho_mean": sum(rho_mean) / n,
        "rho_max": max(rho_max) if rho_max else 0.0,
        "rho_frac_above_1": sum(rho_fa1) / n,
        "c_rate": c_rate,
        "num_clusters": sum(nclust) / n,
        "f1_w": f1,
        "fpr_w": fpr,
        "recall_w": recall,
        "fading_error": fading,
        "attacks": attacks,
        "n": n,
    }


def build(run_dir: Path, window: int):
    recs = load_per_flow(run_dir)
    t_a = first_attack_index(recs)

    rows = []
    for w, start in enumerate(range(0, len(recs), window)):
        chunk = recs[start:start + window]
        if not chunk:
            continue
        stats = window_stats(chunk)
        row = {
            "win": w,
            "flow_start": start,
            "flow_end": start + len(chunk) - 1,
            **stats,
        }
        rows.append(row)

    out = run_dir / "serie_temporal.csv"
    fields = [
        "win", "flow_start", "flow_end",
        "rho_mean", "rho_max", "rho_frac_above_1",
        "c_rate", "num_clusters",
        "f1_w", "fpr_w", "recall_w",
        "fading_error", "attacks", "n",
    ]
    with open(out, "w", newline="") as f:
        f.write(f"# t_a_flow={t_a}\n")  # instante de drift (None se benign-only)
        f.write(f"# window={window}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    print(f"serie_temporal.csv: {len(rows)} janelas (w={window}), "
          f"t_a_flow={t_a} -> {out}")
    return out, t_a, rows


def main():
    ap = argparse.ArgumentParser(description="Agregador de serie temporal de drift")
    ap.add_argument("run_dir", help="Diretorio do run (com detection_results.json)")
    ap.add_argument("--window", type=int, default=200,
                    help="Tamanho da janela da serie (default 200, fecha A3)")
    args = ap.parse_args()
    build(Path(args.run_dir), args.window)


if __name__ == "__main__":
    main()
