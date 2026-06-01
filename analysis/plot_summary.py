"""
Scatter plots from aggregated summary.csv.
One point per model per dataset subplot.

Output: analysis/images/{F1_vs_query_rep,F1_vs_search_turns,rep_vs_search_turns}/summary.png

Usage:
    python analysis/plot_summary.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUMMARY_CSV = Path("analysis/summary.csv")
OUT_DIR     = Path("analysis/images")

EXCLUDE_MODELS   = {"ReSearch/7b_e5", "Search_o1/7b_e5"}
EXCLUDE_DATASETS = {"browsecomp"}

DATASET_ORDER = [
    "2wikimultihopqa",
    "hotpotqa",
    "musique",
    "bamboogle",
    "totalrecall-wiki1",
    "totalrecall-wiki2",
    "totalrecall-e-commerce",
    "trec_rag_2024",
]

MODEL_STYLES = {
    "ASearcher/7b_e5":          {"color": "#1f77b4", "marker": "o", "label": "ASearcher 7B E5"},
    "Search_r1/3b_e5":          {"color": "#ff7f0e", "marker": "s", "label": "Search-R1 3B E5"},
    "Search_r1/7b_e5":          {"color": "#2ca02c", "marker": "^", "label": "Search-R1 7B E5"},
    "Search_r1/7b_bm25":        {"color": "#d62728", "marker": "D", "label": "Search-R1 7B BM25"},
    "Search_r1/7b_bm25_rerank": {"color": "#9467bd", "marker": "P", "label": "Search-R1 7B BM25+Rerank"},
    "Search_r1/14b_e5":         {"color": "#8c564b", "marker": "*", "label": "Search-R1 14B E5"},
}

SCATTER_TYPES = {
    "F1_vs_query_rep": (
        "avg_query_repetition", "avg_answer_f1",
        "Query Repetition (avg pairwise token-F1)", "Answer F1",
    ),
    "F1_vs_search_turns": (
        "avg_search_turns", "avg_answer_f1",
        "Avg # Search Turns", "Answer F1",
    ),
    "rep_vs_search_turns": (
        "avg_search_turns", "avg_query_repetition",
        "Avg # Search Turns", "Query Repetition",
    ),
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> dict:
    """Returns {(model, dataset): {metric: float}}"""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            model   = row["model"]
            dataset = row["dataset"]
            if model in EXCLUDE_MODELS or dataset in EXCLUDE_DATASETS:
                continue
            data[(model, dataset)] = {
                "avg_search_turns":     float(row["avg_search_turns"]),
                "avg_query_repetition": float(row["avg_query_repetition"]),
                "avg_answer_f1":        float(row["avg_answer_f1"]),
            }
    return data

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_scatter_type(data: dict, folder: str, x_key: str, y_key: str,
                      xlabel: str, ylabel: str):
    out_dir = OUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in DATASET_ORDER
                if any((m, d) in data for m in MODEL_STYLES)]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    fig.suptitle(f"{ylabel} vs {xlabel}", fontsize=12, fontweight="bold")

    handles = []
    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, 4)
        ax = axes[row][col]

        for model, style in MODEL_STYLES.items():
            key = (model, ds)
            if key not in data:
                continue
            x = data[key][x_key]
            y = data[key][y_key]
            sc = ax.scatter(x, y, s=80,
                            color=style["color"],
                            marker=style["marker"],
                            label=style["label"],
                            zorder=3)

        ax.set_title(ds.replace("totalrecall-", "tr-"), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(datasets), 8):
        row, col = divmod(idx, 4)
        axes[row][col].set_visible(False)

    # shared legend below
    handles = [
        plt.scatter([], [], s=80, color=s["color"], marker=s["marker"], label=s["label"])
        for s in MODEL_STYLES.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(MODEL_STYLES),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_summary(SUMMARY_CSV)
    print(f"Loaded {len(data)} (model, dataset) entries.")

    for folder, (x_key, y_key, xlabel, ylabel) in SCATTER_TYPES.items():
        print(f"\n{folder}")
        plot_scatter_type(data, folder, x_key, y_key, xlabel, ylabel)

    print("\nDone.")


if __name__ == "__main__":
    main()
