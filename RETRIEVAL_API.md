# Retrieval Server API Guide

This document describes how to call the retrieval servers for each dataset in the benchmark.

---

## Overview

There are **3 retrieval methods** available per corpus:

| Method | Description |
|--------|-------------|
| **E5** | Dense retrieval (intfloat/e5-base, FAISS) |
| **BM25** | Sparse retrieval (Pyserini / Lucene) |
| **BM25+Reranker** | BM25 top-N → Qwen3-Reranker-0.6B → top-k |

All servers share the same endpoint:

```
POST /retrieve
Content-Type: application/json

{
  "queries": ["your question here"],   // list of queries (batch supported)
  "topk": 5,                           // number of results to return (default: 5)
  "return_scores": false               // set true to include retrieval scores
}
```

Response:
```json
{
  "result": [
    [
      {"id": "doc123", "title": "...", "text": "...", "contents": "title\ntext"},
      ...
    ]
  ]
}
```

**Limits:**
- `topk` max: **100** for E5 and BM25
- `topk` max: **100** for BM25+Reranker (but only top-k after reranking N=50 BM25 candidates)
- BM25+Reranker: `bm25_topk` (N) max = **50**, `topk` (k) default = **5**

---

## Dataset Servers

### 1. Total Recall QA (TRQA)

TRQA has two splits with separate corpora and servers.

#### 1a. TRQA — Wiki split

| Method | Host | Port |
|--------|------|------|
| E5 | gypsum-gpu117 | **8001** |
| BM25 | gypsum-gpu126 | **8002** |
| BM25+Reranker | gypsum-gpu145 | **8003** |

```python
import requests

# E5
resp = requests.post("http://gypsum-gpu117:8001/retrieve",
    json={"queries": ["What is the capital of Japan?"], "topk": 5})

# BM25
resp = requests.post("http://gypsum-gpu126:8002/retrieve",
    json={"queries": ["What is the capital of Japan?"], "topk": 5})

# BM25 + Reranker (N=50 BM25 candidates → reranked top-5)
resp = requests.post("http://gypsum-gpu145:8003/retrieve",
    json={"queries": ["What is the capital of Japan?"], "topk": 5})

docs = resp.json()["result"][0]
```

#### 1b. TRQA — Ecommerce split

| Method | Host | Port |
|--------|------|------|
| E5 | gpu016 | **8011** |
| BM25 | gypsum-gpu126 | **8012** |
| BM25+Reranker | gypsum-gpu094 | **8013** |

```python
# E5
resp = requests.post("http://gpu016:8011/retrieve",
    json={"queries": ["wireless headphones under $50"], "topk": 5})

# BM25
resp = requests.post("http://gypsum-gpu126:8012/retrieve",
    json={"queries": ["wireless headphones under $50"], "topk": 5})

# BM25 + Reranker
resp = requests.post("http://gypsum-gpu094:8013/retrieve",
    json={"queries": ["wireless headphones under $50"], "topk": 5})
```

---

### 2. BrowseComp-Plus

| Method | Host | Port |
|--------|------|------|
| E5 | gpu021 | **8021** |
| BM25 | gypsum-gpu127 | **8022** |
| BM25+Reranker | gypsum-gpu146 | **8023** |

```python
# E5
resp = requests.post("http://gpu021:8021/retrieve",
    json={"queries": ["your query"], "topk": 5})

# BM25
resp = requests.post("http://gypsum-gpu127:8022/retrieve",
    json={"queries": ["your query"], "topk": 5})

# BM25 + Reranker
resp = requests.post("http://gypsum-gpu146:8023/retrieve",
    json={"queries": ["your query"], "topk": 5})
```

---

### 3. MultiHop QA (HotpotQA, Musique, Wiki2, Bamboogle)

These datasets use **Wikipedia 2018** as the corpus.

| Method | Host | Port |
|--------|------|------|
| E5 | gypsum-gpu176 | **8031** |
| BM25 | cpu068 | **8032** |
| BM25+Reranker | gypsum-gpu146 | **8033** |

```python
# E5
resp = requests.post("http://gypsum-gpu176:8031/retrieve",
    json={"queries": ["Who is the wife of the actor who played Iron Man?"], "topk": 5})

# BM25
resp = requests.post("http://cpu068:8032/retrieve",
    json={"queries": ["Who is the wife of the actor who played Iron Man?"], "topk": 5})

# BM25 + Reranker
resp = requests.post("http://gypsum-gpu146:8033/retrieve",
    json={"queries": ["Who is the wife of the actor who played Iron Man?"], "topk": 5})
```

---

### 4. TREC RAG 2024

Uses the MS MARCO V2.1 segmented corpus (113.5M segments).

| Method | Host | Port |
|--------|------|------|
| E5 | gpu022 | **8041** |
| BM25 | gypsum-gpu144 | **8042** |
| BM25+Reranker | gypsum-gpu147 | **8043** |

```python
# E5
resp = requests.post("http://gpu022:8041/retrieve",
    json={"queries": ["your query"], "topk": 5})

# BM25
resp = requests.post("http://gypsum-gpu144:8042/retrieve",
    json={"queries": ["your query"], "topk": 5})

# BM25 + Reranker
resp = requests.post("http://gypsum-gpu147:8043/retrieve",
    json={"queries": ["your query"], "topk": 5})
```

---

## Batch Queries

All servers support batching — send multiple queries in one request:

```python
resp = requests.post("http://gypsum-gpu126:8002/retrieve", json={
    "queries": ["question one", "question two", "question three"],
    "topk": 5
})
results = resp.json()["result"]  # list of 3 lists, one per query
for i, docs in enumerate(results):
    print(f"Query {i}: {len(docs)} docs")
```

---

## Return Scores

```python
resp = requests.post("http://gpu016:8011/retrieve", json={
    "queries": ["your query"],
    "topk": 5,
    "return_scores": True
})
# For E5/BM25: result contains {"document": {...}, "score": 0.87}
# For BM25+Reranker: result contains doc fields + "reranker_score": 0.95
```

---

## Port Reference

| Corpus | E5 | BM25 | BM25+Reranker |
|--------|---:|-----:|--------------:|
| TRQA wiki | 8001 | 8002 | 8003 |
| TRQA ecommerce | 8011 | 8012 | 8013 |
| BrowseComp-Plus | 8021 | 8022 | 8023 |
| Wiki-18 (MultiHop) | 8031 | 8032 | 8033 |
| TREC RAG 2024 | 8041 | 8042 | 8043 |

---

## Notes

- Servers run on SLURM compute nodes — hostnames may change if a job is resubmitted. Check `SERVER_STATUS.md` for the latest host assignments.
- E5 servers use GPU for encoding; under high concurrency requests are serialized (one GPU at a time). Expect higher latency under load.
- BM25 servers are highly concurrent — tested at 100+ req/s.
- TRQA wiki E5 (port 8001) uses a CPU-based SQ8 index (57.7M vectors) — single-request latency is ~1-2s; avoid high concurrency.
