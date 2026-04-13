# Deep Research Benchmark — Retrieval Servers

Retrieval servers for the deep research course project. For each corpus we host three retriever types: **E5-base dense**, **BM25**, and **BM25 + Reranker** — all exposing the same `POST /retrieve` endpoint so you can swap them without changing your code.

---

## Server Status

| Type | Corpus | Dataset | Server URL | Status |
|------|--------|---------|------------|--------|
| E5 (dense) | trqa-wiki | Total Recall QA — wiki | `http://gpu018:8001/retrieve` | ✅ Ready |
| E5 (dense) | trqa-ecommerce | Total Recall QA — ecommerce | `http://gpu016:8011/retrieve` | ✅ Ready |
| E5 (dense) | browsecomp-plus | BrowseComp-Plus | `http://gpu021:8021/retrieve` | ✅ Ready |
| E5 (dense) | wiki-18 | MultiHop QA | `http://gypsum-gpu176:8031/retrieve` | ✅ Ready |
| E5 (dense) | trec-rag | TREC RAG 2024 | — | ⏳ Pending |
| BM25 | trqa-wiki | Total Recall QA — wiki | `http://gypsum-gpu126:8002/retrieve` | ✅ Ready |
| BM25 | trqa-ecommerce | Total Recall QA — ecommerce | `http://gypsum-gpu126:8012/retrieve` | ✅ Ready |
| BM25 | browsecomp-plus | BrowseComp-Plus | `http://gypsum-gpu127:8022/retrieve` | ✅ Ready |
| BM25 | wiki-18 | MultiHop QA | `http://cpu068:8032/retrieve` | ✅ Ready |
| BM25 | trec-rag | TREC RAG 2024 | `http://gypsum-gpu144:8042/retrieve` | ✅ Ready |
| BM25+Reranker | trqa-wiki | Total Recall QA — wiki | — | ⏳ Down (restarting) |
| BM25+Reranker | trqa-ecommerce | Total Recall QA — ecommerce | — | ⏳ Down (restarting) |
| BM25+Reranker | browsecomp-plus | BrowseComp-Plus | — | ⏳ Down (restarting) |
| BM25+Reranker | wiki-18 | MultiHop QA | — | ⏳ Down (restarting) |
| BM25+Reranker | trec-rag | TREC RAG 2024 | — | ⏳ Down (restarting) |

> **Note:** All servers are on the cluster internal network. You must run your code from a cluster node (login node or compute job) — these URLs are not accessible from your laptop.

---

## API Reference

All retrieval servers share the same interface:

### `POST /retrieve`

**Request:**
```json
{
  "queries": ["your search query here"],
  "topk": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `queries` | `list[str]` | required | One or more query strings (batch supported) |
| `topk` | `int` | `5` | Number of documents to return per query. Max: **100** |

**Response:**
```json
{
  "result": [
    [
      {
        "id": "doc_123",
        "title": "Machine Learning",
        "text": "Machine learning is a subset of artificial intelligence...",
        "contents": "Machine Learning\nMachine learning is a subset of artificial intelligence..."
      }
    ]
  ]
}
```

`result` is a list of lists: `result[i]` contains the top-k documents for `queries[i]`.

---

### BM25+Reranker Pipeline

Same `POST /retrieve` interface, with two additional optional parameters:

```json
{
  "queries": ["your search query here"],
  "topk": 5,
  "bm25_topk": 50
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topk` | `int` | `5` | Number of documents to return after reranking |
| `bm25_topk` | `int` | `50` | How many BM25 candidates to retrieve before reranking. Max: **100** |

Internally: BM25 retrieves `bm25_topk` candidates → Qwen3-Reranker-0.6B scores all of them → returns top `topk`.

The response includes an extra `reranker_score` field on each document.

---

## Quick Start

See [`call_retriever.py`](call_retriever.py) for a ready-to-run example. The minimal usage:

```python
import requests

url = "http://gpu016:8011/retrieve"   # pick any server above
resp = requests.post(url, json={"queries": ["What is machine learning?"], "topk": 5})
docs = resp.json()["result"][0]       # list of top-5 docs for the first query

for doc in docs:
    print(doc["title"])
    print(doc["text"][:200])
    print()
```

---

## Batch Queries

All servers support batching — send multiple queries in one request for efficiency:

```python
queries = ["What is BERT?", "How does RAG work?", "Define transformer architecture"]
resp = requests.post(url, json={"queries": queries, "topk": 5})
all_results = resp.json()["result"]  # all_results[i] = top-5 docs for queries[i]
```

Batching is especially helpful when running evaluations over large question sets.
