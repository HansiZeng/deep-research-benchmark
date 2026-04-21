# Retrieval Server Status

Last updated: 2026-04-21

---

## Active Servers

| Type | Corpus | Host | Port | Status |
|------|--------|------|------|--------|
| E5 | trqa-wiki | gpu022 | 8001 | ✅ Running (job 55875182) |
| E5 | trqa-ecommerce | gpu016 | 8011 | ❌ Offline |
| E5 | browsecomp-plus | gpu021 | 8021 | ❌ Offline |
| E5 | wiki-18 | gypsum-gpu176 | 8031 | ❌ Offline |
| E5 | trec-rag | gpu015 | 8041 | ✅ Running (job 55875169) |
| BM25 | trqa-wiki | gypsum-gpu126 | 8002 | ❌ Offline |
| BM25 | trqa-ecommerce | gypsum-gpu126 | 8012 | ❌ Offline |
| BM25 | browsecomp-plus | gypsum-gpu127 | 8022 | ❌ Offline |
| BM25 | trec-rag | gypsum-gpu144 | 8042 | ❌ Offline |
| BM25 | wiki-18 | cpu068 | 8032 | ❌ Offline |
| BM25+Reranker | trqa-wiki | gypsum-gpu153 | 8003 | ❌ Offline |
| BM25+Reranker | trqa-ecommerce | gypsum-gpu094 | 8013 | ❌ Offline |
| BM25+Reranker | browsecomp-plus | gypsum-gpu131 | 8023 | ❌ Offline |
| BM25+Reranker | wiki-18 | gypsum-gpu132 | 8033 | ❌ Offline |
| BM25+Reranker | trec-rag | gypsum-gpu132 | 8043 | ❌ Offline |
| Reranker | (corpus-agnostic) | gpu020 | 9001 | ❌ Offline |

---

## Stress Test Results (100 concurrent requests, 2026-04-10)

Query: `"What is machine learning?"` (and 9 others, round-robin)

| Server | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Success |
|--------|---------|---------|---------|-------------------|---------|
| E5 trqa-ecommerce | 2849 | 4941 | 5106 | 19.1 | 100/100 |
| E5 browsecomp-plus | 1764 | 2706 | 2730 | 34.5 | 100/100 |
| E5 wiki-18 | 3658 | 4857 | 4942 | 19.6 | 100/100 |
| BM25 trqa-wiki | 576 | 747 | 771 | 113.6 | 100/100 |
| BM25 trqa-ecommerce | 311 | 355 | 391 | 206.3 | 100/100 |
| BM25 browsecomp | 2954 | 4052 | 4161 | 23.3 | 100/100 |
| BM25 trec-rag | 1412 | 1735 | 1783 | 54.4 | 100/100 |
| BM25 wiki-18 | 209 | 287 | 327 | 230.2 | 100/100 |

### Notes
- **E5 servers**: GPU lock serializes encoder + FAISS search. Single-request latency ~50-100ms; high concurrency queues up.
- **BM25 browsecomp** unexpectedly slow (p50=2954ms) — Lucene I/O on this corpus, worth investigating.
- **E5 trqa-wiki** (SQ8, CPU, 57.7M vectors): ~1-2s per search, avoid high concurrency.
- **BM25 wiki-18 & trqa-ecommerce**: fastest, both >200 req/s.

---

## FAISS Index Types

| Corpus | Index Type | faiss_gpu | Notes |
|--------|-----------|-----------|-------|
| trqa-wiki | **IVF65536_HNSW32,PQ128** | Yes | Rebuilding — was SQ8 (CPU-only, slow) |
| trqa-ecommerce | Flat | Yes | 3.28M, fits on A100 |
| browsecomp-plus | Flat | Yes | 100K, fits on A100 |
| wiki-18 | Flat (pre-built) | Yes | 6× 2080ti sharded |
| trec-rag | **IVF65536_HNSW32,PQ128** | Yes | Rebuilding — was SQ8 (CPU-only, slow) |

**IVF65536_HNSW32,PQ128**: GPU-compatible, PQ128 = 128 bytes/vector (6× smaller than SQ8).
trqa-wiki: ~7.4GB, trec-rag: ~14.5GB — both fit on a single A100.

---

## BM25+Reranker Pipeline Config

- BM25 retrieve top-N: **50** (capped at 100)
- Reranker return top-k: **5**
- Reranker model: `Qwen/Qwen3-Reranker-0.6B`
- Hardware: 6× 2080ti with `torch.nn.DataParallel`
- Load balancing: round-robin across Reranker-1 (`gypsum-gpu177:9001`) and Reranker-2 (`gypsum-gpu171:9002`)
- Thread safety: `threading.Lock` on GPU forward pass (both reranker and E5 servers)
