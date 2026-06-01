# Rollout JSONL Schema Analysis

Each `.jsonl` file has one JSON object per line. All models share the same top-level schema but differ in how `information.content` is formatted.

---

## Top-level Schema (all models)

```json
{
  "qid": "dev_13",
  "query": "Where was the director of film Thomas Jefferson born?",
  "golden_answers": ["Brooklyn, New York", "Brooklyn"],
  "turns": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `qid` | string | Question ID (e.g. `dev_13`, `train_42`) |
| `query` | string | The original question |
| `golden_answers` | list[str] | All acceptable answers (for EM/F1 eval) |
| `turns` | list[dict] | The multi-turn trajectory |

---

## Turn Structure

Each turn has `{"type": ..., "content": ...}`. The sequence always starts with `query` and ends with `answer`.

```
query → [think → search → information]* → answer
```

| Turn type | `content` type | Description |
|-----------|----------------|-------------|
| `query` | string | Repeats the question (always first) |
| `think` | string | Model's chain-of-thought reasoning |
| `search` | string | The search query issued to the retriever |
| `information` | string or list[str] | Retrieved documents (format varies — see below) |
| `answer` | string | Final answer (always last) |

---

## Key Difference: `information.content` Format by Retriever

This is the main structural difference across model configs.

### E5 — list of strings, `"Doc N(Title: ...) text"` format

Used by: **ASearcher 7b_e5**, **Search-R1 3b/7b/14b E5**, **Search-R1 7b_bm25_rerank**

```json
{
  "type": "information",
  "content": [
    "Doc 1(Title: Thomas Jefferson (film)) Thomas Jefferson is a 1997 two-part American documentary...",
    "Doc 2(Title: Lincoln Schatz) strategist Karl Rove, and liberal heavyweight...",
    "Doc 3(Title: Thomas Jefferson (film)) did have a long-term relationship with Hemings..."
  ]
}
```

- `content` is a **list of strings**
- Each doc prefixed with `Doc N(Title: X)`

---

### ReSearch E5 — single string, `"Title: X\ntext"` format

Used by: **ReSearch 7b_e5**

```json
{
  "type": "information",
  "content": "Title: Polish-Russian War (film)\nfilm was shot between May 6 and 18 June 2008...\n\nTitle: Jerzy Hoffman\nfilm \"Colonel Wolodyjowski\" was entered into..."
}
```

- `content` is a **single string**
- Docs separated by `\n\n`, each prefixed with `Title: X\n`

---

### BM25 — single string, `<doc>` XML blocks

Used by: **Search-R1 7b_bm25**

```json
{
  "type": "information",
  "content": "<doc>\nPolish-Russian War (film)\nPolish-Russian War (Wojna polsko-ruska) is a 2009 Polish film...\n</doc>\n<doc>\nJacek Rotmil\nwas later arrested and executed in Pawiak prison...\n</doc>"
}
```

- `content` is a **single string**
- Each doc wrapped in `<doc>...</doc>` tags
- No explicit title field — title is the first line inside `<doc>`

---

## Model-specific Notes

### Search-o1 7b_e5
- Sometimes answers **without issuing any search** — the `think` turn contains full reasoning and the answer is derived internally:
  ```
  query → think (with embedded answer) → answer
  ```
- When it does search, the format is the same E5 list-of-strings format.

### Search-R1 7b_e5
- Occasionally gets stuck in a **search loop** — repeats the same search query multiple times before answering (observed in 2wikimultihopqa example: 4× `David Lynch birthplace`).

### BM25+Rerank (Search-R1 7b_bm25_rerank)
- Despite using BM25+Reranker for retrieval, the `information.content` format is the **same E5 list format** (`Doc N(Title: ...)`), not the BM25 XML format.

---

## Summary Table

| Model | `information.content` type | Doc format |
|-------|---------------------------|------------|
| ASearcher 7b_e5 | `list[str]` | `Doc N(Title: X) text` |
| ReSearch 7b_e5 | `str` | `Title: X\ntext\n\nTitle: X\ntext` |
| Search-o1 7b_e5 | `list[str]` | `Doc N(Title: X) text` |
| Search-R1 3b_e5 | `list[str]` | `Doc N(Title: X) text` |
| Search-R1 7b_e5 | `list[str]` | `Doc N(Title: X) text` |
| Search-R1 7b_bm25 | `str` | `<doc>\nTitle\ntext\n</doc>` |
| Search-R1 7b_bm25_rerank | `list[str]` | `Doc N(Title: X) text` |
| Search-R1 14b_e5 | `list[str]` | `Doc N(Title: X) text` |
