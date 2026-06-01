"""
Analyze rollout JSONL files.

Two modes:

1. Single file / directory (per-record output + aggregate):
       python eval_rollouts.py --input <file_or_dir> [--output records.jsonl]

2. Summary table across all (model, dataset) pairs:
       python eval_rollouts.py --summary --rollouts_dir /scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/
       python eval_rollouts.py --summary --rollouts_dir ... --output summary.csv

Metrics per trajectory:
  - query_repetition: mean pairwise token-F1 across all search queries issued
  - answer_f1:        max token-F1 between model answer and any golden answer

The summary table shows per (model+retriever, dataset):
  avg_query_repetition, avg_answer_f1, avg_n_search_turns
"""

import argparse
import json
import string
from collections import Counter
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def normalize(text) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(s1: str, s2: str) -> float:
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


# ---------------------------------------------------------------------------
# Per-trajectory metrics
# ---------------------------------------------------------------------------

def query_repetition(queries: list[str]) -> float:
    """Mean pairwise token-F1 over all (i<j) pairs of search queries."""
    if len(queries) <= 1:
        return 0.0
    scores = [token_f1(a, b) for a, b in combinations(queries, 2)]
    return sum(scores) / len(scores)


def answer_f1(prediction: str, golden_answers: list[str]) -> float:
    """Max token-F1 between prediction and any golden answer."""
    if not golden_answers:
        return 0.0
    return max(token_f1(prediction, g) for g in golden_answers)


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def parse_record(record: dict) -> dict:
    turns          = record.get("turns", [])
    search_queries = [t["content"] for t in turns if t.get("type") == "search"]
    answer_turns   = [t["content"] for t in turns if t.get("type") == "answer"]
    prediction     = answer_turns[-1] if answer_turns else ""
    golden         = record.get("golden_answers", [])

    return {
        "qid":              record.get("qid", ""),
        "n_search_turns":   len(search_queries),
        "query_repetition": query_repetition(search_queries),
        "answer_f1":        answer_f1(prediction, golden),
        "prediction":       prediction,
        "golden_answers":   golden,
        "search_queries":   search_queries,
    }


def process_file(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(parse_record(json.loads(line)))
    return results


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n":                   n,
        "avg_search_turns":    sum(r["n_search_turns"]   for r in results) / n,
        "avg_query_repetition": sum(r["query_repetition"] for r in results) / n,
        "avg_answer_f1":       sum(r["answer_f1"]        for r in results) / n,
    }


# ---------------------------------------------------------------------------
# Summary mode: scan rollouts_dir for all (model, dataset) .jsonl files
# ---------------------------------------------------------------------------

def discover_rollouts(rollouts_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Returns list of (model_tag, dataset, path).
    Expects structure: rollouts_dir/<Model>/<size_retriever>/raw?/<dataset>.jsonl
    We use the processed files (not raw/), directly under <size_retriever>/.
    """
    entries = []
    for model_dir in sorted(rollouts_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"out", "__pycache__", "venv"}:
            continue
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name == "raw":
                continue
            model_tag = f"{model_dir.name}/{variant_dir.name}"
            for jsonl in sorted(variant_dir.glob("*.jsonl")):
                dataset = jsonl.stem
                entries.append((model_tag, dataset, jsonl))
    return entries


def build_summary_table(rollouts_dir: Path) -> list[dict]:
    entries = discover_rollouts(rollouts_dir)
    rows = []
    for model_tag, dataset, path in entries:
        print(f"  processing {model_tag} / {dataset} ...", flush=True)
        records = process_file(path)
        agg = aggregate(records)
        rows.append({
            "model":                model_tag,
            "dataset":              dataset,
            "n":                    agg["n"],
            "avg_search_turns":     round(agg["avg_search_turns"],    3),
            "avg_query_repetition": round(agg["avg_query_repetition"], 4),
            "avg_answer_f1":        round(agg["avg_answer_f1"],        4),
        })
    return rows


def print_summary_table(rows: list[dict]):
    # Group by dataset for readable output
    datasets = sorted({r["dataset"] for r in rows})
    models   = sorted({r["model"]   for r in rows})

    # Index by (model, dataset)
    idx = {(r["model"], r["dataset"]): r for r in rows}

    col_w = 10
    header_w = 36

    for metric, key in [
        ("avg_answer_f1",        "avg_answer_f1"),
        ("avg_query_repetition", "avg_query_repetition"),
        ("avg_search_turns",     "avg_search_turns"),
    ]:
        print(f"\n{'─'*120}")
        print(f"  {metric}")
        print(f"{'─'*120}")
        header = f"{'model':<{header_w}}" + "".join(f"{d[:col_w]:>{col_w}}" for d in datasets)
        print(header)
        print("─" * len(header))
        for model in models:
            row_str = f"{model:<{header_w}}"
            for dataset in datasets:
                val = idx.get((model, dataset))
                cell = f"{val[key]:.4f}" if val else "    —   "
                row_str += f"{cell:>{col_w}}"
            print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",       help="Path to .jsonl file or directory (single-mode)")
    group.add_argument("--summary",     action="store_true", help="Build cross-model summary table")

    parser.add_argument("--rollouts_dir",
                        default="/scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/",
                        help="Base rollouts directory (used with --summary)")
    parser.add_argument("--output", default=None,
                        help="Save results: .jsonl (single-mode) or .csv (summary mode)")
    args = parser.parse_args()

    # ── Summary mode ──────────────────────────────────────────────────────
    if args.summary:
        print(f"Scanning {args.rollouts_dir} ...")
        rows = build_summary_table(Path(args.rollouts_dir))
        print_summary_table(rows)

        if args.output:
            out = Path(args.output)
            import csv
            fieldnames = ["model", "dataset", "n", "avg_search_turns",
                          "avg_query_repetition", "avg_answer_f1"]
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"\nSaved to {out}")
        return

    # ── Single file / directory mode ──────────────────────────────────────
    input_path = Path(args.input)
    files = sorted(input_path.rglob("*.jsonl")) if input_path.is_dir() else [input_path]

    all_records = []
    for f in files:
        records = process_file(f)
        agg = aggregate(records)
        print(f"\n{f}")
        print(f"  n={agg['n']}  search_turns={agg['avg_search_turns']:.2f}"
              f"  rep={agg['avg_query_repetition']:.4f}  f1={agg['avg_answer_f1']:.4f}")
        for r in records:
            r["source_file"] = str(f)
        all_records.extend(records)

    if len(files) > 1:
        agg = aggregate(all_records)
        print(f"\nCOMBINED  n={agg['n']}  search_turns={agg['avg_search_turns']:.2f}"
              f"  rep={agg['avg_query_repetition']:.4f}  f1={agg['avg_answer_f1']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved per-record results to {args.output}")


if __name__ == "__main__":
    main()
