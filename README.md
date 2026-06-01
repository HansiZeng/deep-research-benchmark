# Search-Agent Trajectory Analysis

Code for analyzing **search-agent trajectories** from the CS791U deep-research benchmark.
Given the rollout logs of several search agents (Search-R1, ASearcher, ReSearch, Search-o1) across
eight QA / retrieval datasets, this repo computes three per-trajectory quantities and studies how they
relate to the final answer:

- **search turns** — how many searches the agent issued,
- **query repetition** — how much the agent re-asked the same thing, and
- **answer F1** — how good the final answer was.

Everything lives in [`analysis/`](analysis/). The rest of this README explains how to use it.

---

## Layout

```
analysis/
├── eval_rollouts.py          # compute per-trajectory metrics + aggregate summary tables
├── plot_rollouts.py          # raw per-query scatter plots
├── plot_rollouts_binned.py   # binned mean-F1 line plots (the readable ones)
├── plot_summary.py           # cross-model aggregate scatter (one point per config × dataset)
│
├── summary.csv / summary.md  # precomputed aggregate metrics per (model, dataset)
├── rollouts_inventory.md     # which models/datasets exist, file sizes, coverage gaps
├── rollouts_schema.md        # the trajectory JSONL schema + per-retriever format quirks
│
├── images/                   # scatter outputs (+ cross-model summary.png)
│   ├── F1_vs_query_rep/
│   ├── F1_vs_search_turns/
│   └── rep_vs_search_turns/
└── images_binned/            # binned line-plot outputs (same three relationships)
    ├── F1_vs_query_rep/
    ├── F1_vs_search_turns/
    └── rep_vs_search_turns/
```

---

## Input data

The scripts read **rollout JSONL files**, one trajectory per line. Layout expected by the discovery code:

```
<rollouts_dir>/<Model>/<size_retriever>/<dataset>.jsonl
e.g.  .../Search_r1/7b_bm25/hotpotqa.jsonl
```

Each line is a single trajectory:

```jsonc
{
  "qid": "dev_13",
  "query": "Where was the director of film Thomas Jefferson born?",
  "golden_answers": ["Brooklyn, New York", "Brooklyn"],
  "turns": [ {"type": "query", ...}, {"type": "think", ...},
             {"type": "search", "content": "..."},
             {"type": "information", "content": ...},
             {"type": "answer", "content": "..."} ]
}
```

The sequence is always `query → (think → search → information)* → answer`. See
[`analysis/rollouts_schema.md`](analysis/rollouts_schema.md) for the full field reference and the
per-retriever differences in how `information.content` is formatted, and
[`analysis/rollouts_inventory.md`](analysis/rollouts_inventory.md) for which (model, dataset) pairs exist.

The default rollouts path is the course scratch directory
(`/scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/`); override it on any script with
`--rollouts_dir /your/path`.

---

## Metrics

All three metrics are built on one **token-level F1** between two strings (lowercase, strip
punctuation, drop `a/an/the`). Defined in `eval_rollouts.py`:

| Metric | Definition |
|--------|------------|
| **search turns** | number of `search` actions in the trajectory |
| **query repetition** | mean pairwise token-F1 over all `C(n,2)` pairs of search queries (0 if ≤1 query). High ⇒ the agent keeps issuing near-identical searches; low ⇒ diverse sub-queries |
| **answer F1** | max token-F1 between the model's final answer and any gold answer |

---

## Usage

No special dependencies for the metrics (pure standard library); plotting needs `matplotlib`.

```bash
pip install matplotlib
```

### 1. Compute metrics for one file or directory

```bash
# single file — prints n, avg search turns, avg repetition, avg F1
python analysis/eval_rollouts.py --input path/to/hotpotqa.jsonl

# a directory — aggregates over every .jsonl found, and can dump per-query rows
python analysis/eval_rollouts.py --input path/to/Search_r1/7b_bm25/ --output records.jsonl
```

### 2. Build the cross-model summary table

```bash
python analysis/eval_rollouts.py --summary \
    --rollouts_dir /scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/ \
    --output analysis/summary.csv
```

This scans every `(model, dataset)` pair and writes `avg_search_turns`, `avg_query_repetition`,
`avg_answer_f1` per row. A precomputed copy is checked in at
[`analysis/summary.csv`](analysis/summary.csv) / [`analysis/summary.md`](analysis/summary.md).

### 3. Generate the plots

```bash
# cross-model aggregate scatter (reads summary.csv) -> images/<rel>/summary.png
python analysis/plot_summary.py

# per-query raw scatter, one PNG per model (2x4 grid of datasets) -> images/
python analysis/plot_rollouts.py        --rollouts_dir <dir>

# binned mean-F1 line plots, one PNG per model -> images_binned/
python analysis/plot_rollouts_binned.py --rollouts_dir <dir>
```

Each plot script emits three relationships: **F1 vs query repetition**, **F1 vs search turns**, and
**repetition vs search turns**. `browsecomp` (near-zero accuracy everywhere) and the search-free
Search-o1 / partial-coverage ReSearch configs are excluded from the plots by default — edit
`EXCLUDE_DATASETS` / `EXCLUDE_MODELS` at the top of each script to change this.

> **Why binned?** Per-query scatter is mostly noise — `search_turns` is a small integer and `answer_F1`
> is heavily 0/1. `plot_rollouts_binned.py` bins each query by the x-axis variable and plots the
> **mean F1 per bin** (line) with the bin count as a gray bar, which is what the report figures use.

---

## What the analysis shows

Two levels of analysis give the relationship **opposite signs** — the headline result:

- **Cross-model** (`plot_summary.py`, `images/.../summary.png`): across configurations, search turns and
  query repetition are almost perfectly coupled (corr ≈ 0.83–0.99) and both correlate **positively** with
  answer F1 on multi-hop QA — the configs that search the most (Search-R1 7B + BM25 / BM25+rerank) score
  the highest.
- **In-model** (`plot_rollouts_binned.py`, `images_binned/`): holding one configuration fixed, F1
  **decreases** as a trajectory takes more turns or repeats more — for a single model, heavy searching marks
  a hard, unresolved, looping question (selection by difficulty), not a path to a better answer.
