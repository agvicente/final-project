"""
Comparador de Experimentos - Versão Básica (Semana 5 Fase B)

Lê resultados de múltiplas execuções e gera tabela comparativa.

Uso:
    python compare_experiments.py results/week5/scenarioA/

Output:
    Tabela no terminal + arquivo comparison_report.md
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_experiment(exp_dir: Path) -> Dict[str, Any]:
    """
    Carrega resultados de um experimento.

    Args:
        exp_dir: Diretório do experimento

    Returns:
        Dict com dados consolidados
    """
    run_meta_path = exp_dir / "run_meta.json"
    detection_path = exp_dir / "detection_results.json"

    if not run_meta_path.exists() or not detection_path.exists():
        return None

    with open(run_meta_path) as f:
        meta = json.load(f)

    with open(detection_path) as f:
        results = json.load(f)

    # Extrai métricas principais
    metrics = results.get("prequential_metrics", {})

    return {
        "experiment": exp_dir.name,
        "algorithm": meta.get("algorithm", "unknown"),
        "r0": meta.get("params", {}).get("r0", 0),
        "min_samples": meta.get("params", {}).get("min_samples", 0),
        "flows_processed": results.get("flows_processed", 0),
        "anomalies_detected": results.get("anomalies_detected", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
        "fpr": metrics.get("fpr", 0),
        "mttd": metrics.get("mttd_seconds", None),
        "num_clusters": results.get("detector_stats", {}).get("num_clusters", 0),
        "duration_sec": meta.get("execution", {}).get("duration_seconds", 0),
    }


def compare_experiments(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Compara todos os experimentos em um diretório.

    Args:
        results_dir: Diretório contendo subdiretórios de experimentos

    Returns:
        Lista de dicts com resultados
    """
    experiments = []

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            exp_data = load_experiment(exp_dir)
            if exp_data:
                experiments.append(exp_data)

    # Ordena por F1 (melhor primeiro)
    experiments.sort(key=lambda x: x["f1"], reverse=True)

    return experiments


def print_comparison_table(experiments: List[Dict[str, Any]]) -> None:
    """Imprime tabela comparativa no terminal."""
    if not experiments:
        print("Nenhum experimento encontrado.")
        return

    print("\n" + "=" * 120)
    print("COMPARAÇÃO DE EXPERIMENTOS")
    print("=" * 120)

    # Header
    print(f"{'Experimento':<30} {'Algo':<12} {'r0':<6} {'Flows':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'FPR':<8} {'MTTD':<8} {'#Clust':<8}")
    print("-" * 120)

    # Dados
    for exp in experiments:
        mttd_str = f"{exp['mttd']:.2f}" if exp['mttd'] is not None else "N/A"
        print(
            f"{exp['experiment']:<30} "
            f"{exp['algorithm']:<12} "
            f"{exp['r0']:<6.2f} "
            f"{exp['flows_processed']:<8} "
            f"{exp['f1']:<8.4f} "
            f"{exp['precision']:<8.4f} "
            f"{exp['recall']:<8.4f} "
            f"{exp['fpr']:<8.4f} "
            f"{mttd_str:<8} "
            f"{exp['num_clusters']:<8}"
        )

    print("=" * 120 + "\n")


def save_comparison_report(experiments: List[Dict[str, Any]], output_path: Path) -> None:
    """Salva relatório em Markdown."""
    with open(output_path, 'w') as f:
        f.write("# Relatório de Comparação de Experimentos\n\n")
        f.write("## Resumo\n\n")
        f.write(f"Total de experimentos: {len(experiments)}\n\n")

        f.write("## Resultados\n\n")
        f.write("| Experimento | Algoritmo | r0 | Flows | F1 | Precision | Recall | FPR | MTTD | #Clusters |\n")
        f.write("|-------------|-----------|-------|-------|--------|-----------|--------|-----|------|----------|\n")

        for exp in experiments:
            mttd_str = f"{exp['mttd']:.2f}" if exp['mttd'] is not None else "N/A"
            f.write(
                f"| {exp['experiment']} | {exp['algorithm']} | {exp['r0']:.2f} | "
                f"{exp['flows_processed']} | {exp['f1']:.4f} | {exp['precision']:.4f} | "
                f"{exp['recall']:.4f} | {exp['fpr']:.4f} | {mttd_str} | {exp['num_clusters']} |\n"
            )

        f.write("\n## Observações\n\n")
        f.write("- Experimentos ordenados por F1-Score (melhor primeiro)\n")
        f.write("- FPR = False Positive Rate (menor é melhor)\n")
        f.write("- MTTD = Mean Time To Detection em segundos (menor é melhor)\n")

    print(f"✅ Relatório salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparador de Experimentos")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Diretório contendo experimentos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Arquivo de saída (default: <results_dir>/comparison_report.md)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Erro: Diretório não encontrado: {results_dir}")
        sys.exit(1)

    # Carregar e comparar experimentos
    experiments = compare_experiments(results_dir)

    # Imprimir tabela
    print_comparison_table(experiments)

    # Salvar relatório
    output_path = Path(args.output) if args.output else (results_dir / "comparison_report.md")
    save_comparison_report(experiments, output_path)


if __name__ == "__main__":
    main()
