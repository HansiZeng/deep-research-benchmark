"""
For each scatter relationship, create a folder under images/:
    F1_vs_query_rep/
    F1_vs_search_turns/
    rep_vs_search_turns/

Inside each folder, one PNG per model (e.g. search_r1_3b_e5.png).
Each PNG has 8 subplots in a 2×4 grid — one per dataset (browsecomp excluded).

Usage:
    python analysis/plot_rollouts.py
    python analysis/plot_rollouts.py --rollouts_dir /path/to/rollouts
"""

import argparse
import json
import string
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROLLOUTS_DIR = Path("/scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/")
OUT_DIR      = Path("analysis/images")

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

# folder_name -> (x_key, y_key, xlabel, ylabel)
SCATTER_TYPES = {
    "F1_vs_query_rep": (
        "query_repetition", "answer_f1",
        "Query Repetition (mean pairwise token-F1)", "Answer F1",
    ),
    "F1_vs_search_turns": (
        "n_search_turns", "answer_f1",
        "# Search Turns", "Answer F1",
    ),
    "rep_vs_search_turns": (
        "n_search_turns", "query_repetition",
        "# Search Turns", "Query Repetition",
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(parse_record(json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one(model_tag: str, dataset_paths: dict,
             x_key: str, y_key: str, xlabel: str, ylabel: str,
             out_dir: Path):
    datasets = [d for d in DATASET_ORDER if d in dataset_paths]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    fig.suptitle(f"{model_tag}  —  {ylabel} vs {xlabel}", fontsize=11, fontweight="bold")

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, 4)
        ax = axes[row][col]

        records = load_records(dataset_paths[ds])
        xs = [r[x_key] for r in records]
        ys = [r[y_key] for r in records]

        ax.scatter(xs, ys, s=4, alpha=0.3, linewidths=0)
        ax.set_title(ds.replace("totalrecall-", "tr-"), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    # hide any unused subplots
    for idx in range(len(datasets), 8):
        row, col = divmod(idx, 4)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    filename = model_tag.replace("/", "_") + ".png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_dir", default=str(ROLLOUTS_DIR))
    parser.add_argument("--out_dir",      default=str(OUT_DIR))
    args = parser.parse_args()

    index = discover(Path(args.rollouts_dir))
    base  = Path(args.out_dir)

    for folder, (x_key, y_key, xlabel, ylabel) in SCATTER_TYPES.items():
        out_dir = base / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {folder} ===")
        for model_tag, dataset_paths in index.items():
            print(f"  {model_tag}")
            plot_one(model_tag, dataset_paths, x_key, y_key, xlabel, ylabel, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
