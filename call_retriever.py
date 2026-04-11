"""
call_retriever.py — Example script for querying retrieval servers.

Usage:
    python call_retriever.py

All servers expose POST /retrieve with the same interface.
Pick the server URL for the corpus + retriever type you want.
See README.md for the full server list and status.
"""

import requests

# ── Server URLs ───────────────────────────────────────────────────────────────
# Pick one of these for your experiment. See README.md for full list.

SERVERS = {
    # E5 dense retrieval
    "e5_trqa_ecommerce":  "http://gpu016:8011/retrieve",
    "e5_browsecomp_plus": "http://gpu021:8021/retrieve",
    "e5_wiki18":          "http://gypsum-gpu176:8031/retrieve",

    # BM25
    "bm25_trqa_wiki":      "http://gypsum-gpu126:8002/retrieve",
    "bm25_trqa_ecommerce": "http://gypsum-gpu126:8012/retrieve",
    "bm25_browsecomp":     "http://gypsum-gpu127:8022/retrieve",
    "bm25_wiki18":         "http://cpu068:8032/retrieve",
    "bm25_trec_rag":       "http://gypsum-gpu144:8042/retrieve",

    # BM25 + Reranker
    "reranker_trqa_wiki":      "http://gypsum-gpu153:8003/retrieve",
    "reranker_trqa_ecommerce": "http://gypsum-gpu146:8013/retrieve",
    "reranker_browsecomp":     "http://gypsum-gpu116:8023/retrieve",
    "reranker_wiki18":         "http://gypsum-gpu127:8033/retrieve",
    "reranker_trec_rag":       "http://gypsum-gpu131:8043/retrieve",
}


# ── Core function ─────────────────────────────────────────────────────────────

def retrieve(server_url: str, queries: list[str], topk: int = 5) -> list[list[dict]]:
    """
    Query a retrieval server.

    Args:
        server_url: Full URL of the server, e.g. "http://gpu016:8011/retrieve"
        queries:    List of query strings (supports batch)
        topk:       Number of documents to return per query (max 100)

    Returns:
        List of lists: result[i] = top-k docs for queries[i]
        Each doc has: id, title, text, contents
    """
    resp = requests.post(
        server_url,
        json={"queries": queries, "topk": topk},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["result"]


# ── Example 1: single query ───────────────────────────────────────────────────

def example_single_query():
    url = SERVERS["bm25_wiki18"]
    query = "What is retrieval-augmented generation?"

    print(f"Server : {url}")
    print(f"Query  : {query!r}")
    print()

    docs = retrieve(url, [query], topk=5)[0]   # [0] = results for first query

    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc['title']}")
        print(f"     {doc['text'][:200].strip()}...")
        print()


# ── Example 2: batch queries ──────────────────────────────────────────────────

def example_batch_queries():
    url = SERVERS["e5_wiki18"]
    queries = [
        "How does BERT work?",
        "What causes inflation?",
        "History of the Roman Empire",
    ]

    print(f"Server : {url}")
    print(f"Batch  : {len(queries)} queries, topk=3")
    print()

    all_docs = retrieve(url, queries, topk=3)

    for q, docs in zip(queries, all_docs):
        print(f"Q: {q!r}")
        for doc in docs:
            print(f"  - [{doc['id']}] {doc['title']}")
        print()


# ── Example 3: compare BM25 vs E5 on the same corpus ─────────────────────────

def example_compare_retrievers():
    query = "neural networks for image classification"

    servers = {
        "BM25": SERVERS["bm25_wiki18"],
        "E5":   SERVERS["e5_wiki18"],
    }

    print(f"Query: {query!r}")
    print()

    for name, url in servers.items():
        docs = retrieve(url, [query], topk=5)[0]
        print(f"=== {name} ({url}) ===")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc['title']}")
        print()


# ── Run examples ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Single query (BM25 wiki-18)")
    print("=" * 60)
    example_single_query()

    print("=" * 60)
    print("Example 2: Batch queries (E5 wiki-18)")
    print("=" * 60)
    example_batch_queries()

    print("=" * 60)
    print("Example 3: BM25 vs E5 comparison (wiki-18)")
    print("=" * 60)
    example_compare_retrievers()
