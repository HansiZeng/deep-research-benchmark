"""
Binned version of plot_rollouts.py.

Per-query scatter is mostly visual noise because n_search_turns is a small
integer and answer_f1 is heavily 0/1. Here we bin queries by the x-axis
variable, then plot mean(y) per bin (line+marker) with bin count as a
gray bar on a secondary y-axis.

For each (model, dataset, scatter type), one PNG per model with 8 subplots
(2x4) — one per dataset (browsecomp excluded).

Output:
    analysis/images_binned/F1_vs_search_turns/<model>.png
    analysis/images_binned/rep_vs_search_turns/<model>.png
    analysis/images_binned/F1_vs_query_rep/<model>.png

Original analysis/images/ folder is untouched.

Usage:
    python analysis/plot_rollouts_binned.py
    python analysis/plot_rollouts_binned.py --rollouts_dir /path/to/rollouts
"""

import argparse
import json
import string
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLLOUTS_DIR = Path("/scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/")
OUT_DIR      = Path("analysis/images_binned")

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

# folder -> (x_key, y_key, x_is_continuous, xlabel, ylabel)
SCATTER_TYPES = {
    "F1_vs_search_turns": (
        "n_search_turns", "answer_f1", False,
        "# Search Turns", "Mean Answer F1",
    ),
    "rep_vs_search_turns": (
        "n_search_turns", "query_repetition", False,
        "# Search Turns", "Mean Query Repetition",
    ),
    "F1_vs_query_rep": (
        "query_repetition", "answer_f1", True,
        "Query Repetition bin", "Mean Answer F1",
    ),
}

N_BINS_CONTINUOUS = 10


def normalize(text) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(t for t in text.split() if t not in {"a", "an", "the"})


def token_f1(s1, s2) -> float:
    t1 = Counter(normalize(s1).split())
    t2 = Counter(normalize(s2).split())
    if not t1 or not t2:
        return float(t1 == t2)
    overlap   = sum((t1 & t2).values())
    precision = overlap / sum(t2.values())
    recall    = overlap / sum(t1.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def query_repetition(queries: list) -> float:
    if len(queries) <= 1:
        return 0.0
    scores = [token_f1(a, b) for a, b in combinations(queries, 2)]
    return sum(scores) / len(scores)


def answer_f1(prediction: str, golden_answers: list) -> float:
    if not golden_answers:
        return 0.0
    return max(token_f1(prediction, g) for g in golden_answers)


def parse_record(record: dict) -> dict:
    turns          = record.get("turns", [])
    search_queries = [t["content"] for t in turns if t.get("type") == "search"]
    answer_turns   = [t["content"] for t in turns if t.get("type") == "answer"]
    prediction     = answer_turns[-1] if answer_turns else ""
    golden         = record.get("golden_answers", [])
    return {
        "n_search_turns":   len(search_queries),
        "query_repetition": query_repetition(search_queries),
        "answer_f1":        answer_f1(prediction, golden),
    }


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(parse_record(json.loads(line)))
    return records


def discover(rollouts_dir: Path) -> dict:
    """Returns {model_tag: {dataset: Path}}"""
    result = {}
    for model_dir in sorted(rollouts_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"out", "__pycache__", "venv"}:
            continue
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name == "raw":
                continue
            tag = f"{model_dir.name}/{variant_dir.name}"
            if tag in EXCLUDE_MODELS:
                continue
            datasets = {
                jsonl.stem: jsonl
                for jsonl in variant_dir.glob("*.jsonl")
                if jsonl.stem not in EXCLUDE_DATASETS
            }
            if datasets:
                result[tag] = datasets
    return result


def bin_records(records, x_key, y_key, x_is_continuous):
    """Returns list of (x_center, mean_y, count) sorted by x_center."""
    if not records:
        return []

    if not x_is_continuous:
        # discrete int bins (n_search_turns)
        buckets = defaultdict(list)
        for r in records:
            buckets[int(r[x_key])].append(r[y_key])
        return [(k, sum(v) / len(v), len(v)) for k, v in sorted(buckets.items())]

    # continuous: equal-width bins between min and max
    xs = [r[x_key] for r in records]
    x_min, x_max = min(xs), max(xs)
    if x_max - x_min < 1e-9:
        ys = [r[y_key] for r in records]
        return [(x_min, sum(ys) / len(ys), len(ys))]

    width = (x_max - x_min) / N_BINS_CONTINUOUS
    buckets = defaultdict(list)
    for r in records:
        bi = min(int((r[x_key] - x_min) / width), N_BINS_CONTINUOUS - 1)
        buckets[bi].append(r[y_key])
    out = []
    for bi in sorted(buckets):
        center = x_min + (bi + 0.5) * width
        ys = buckets[bi]
        out.append((center, sum(ys) / len(ys), len(ys)))
    return out


def plot_one(model_tag, records_per_ds, x_key, y_key, x_is_continuous,
             xlabel, ylabel, out_dir):
    datasets = [d for d in DATASET_ORDER if d in records_per_ds]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    fig.suptitle(f"{model_tag}  —  {ylabel} vs {xlabel}",
                 fontsize=11, fontweight="bold")

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        ax2 = ax.twinx()

        bins = bin_records(records_per_ds[ds], x_key, y_key, x_is_continuous)
        if not bins:
            ax.set_visible(False)
            ax2.set_visible(False)
            continue

        xs   = [b[0] for b in bins]
        ys   = [b[1] for b in bins]
        cnts = [b[2] for b in bins]

        if x_is_continuous and len(xs) > 1:
            bar_w = (xs[-1] - xs[0]) / max(len(xs), 1) * 0.9
        else:
            bar_w = 0.7

        ax2.bar(xs, cnts, width=bar_w,
                color="lightgray", alpha=0.6, edgecolor="none", zorder=1)
        ax2.tick_params(axis="y", labelsize=6, colors="gray")
        ax2.set_ylabel("count", fontsize=7, color="gray")

        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.3,
                color="#1f77b4", zorder=3)

        # primary axis on top of secondary so the line isn't hidden by bars
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.set_title(ds.replace("totalrecall-", "tr-"), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(len(datasets), 8):
        row, col = divmod(idx, 4)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    filename = model_tag.replace("/", "_") + ".png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_dir", default=str(ROLLOUTS_DIR))
    parser.add_argument("--out_dir",      default=str(OUT_DIR))
    args = parser.parse_args()

    index = discover(Path(args.rollouts_dir))
    base  = Path(args.out_dir)

    for folder in SCATTER_TYPES:
        (base / folder).mkdir(parents=True, exist_ok=True)

    for model_tag, dataset_paths in index.items():
        print(f"\n{model_tag}")
        records_per_ds = {ds: load_records(p) for ds, p in dataset_paths.items()}

        for folder, (x_key, y_key, x_is_continuous, xlabel, ylabel) in SCATTER_TYPES.items():
            plot_one(model_tag, records_per_ds,
                     x_key, y_key, x_is_continuous,
                     xlabel, ylabel, base / folder)

    print("\nDone.")


if __name__ == "__main__":
    main()
