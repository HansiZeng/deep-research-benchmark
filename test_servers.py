"""
test_servers.py — Retrieval server smoke test + stress test

Update SERVERS / PENDING_SERVERS as jobs start:
  - Run `squeue -u $USER` to find the node for each job
  - Move entries from PENDING_SERVERS → SERVERS with the real host
"""

import time
import concurrent.futures
import requests

# ── Server registry ──────────────────────────────────────────────────────────
# Format: {"label": (host, port)}
# Update host once you see "running on: <hostname>" in the job's .out file.

SERVERS = {
    "E5 trqa-ecommerce":   ("gpu016",          8011),
    "E5 browsecomp-plus":  ("gpu021",          8021),
    "E5 trqa-wiki":        ("gpu018",          8001),
    "E5 wiki-18":          ("gypsum-gpu176",   8031),
    "BM25 trqa-wiki":      ("gypsum-gpu126",   8002),
    "BM25 trqa-ecommerce": ("gypsum-gpu126",   8012),
    "BM25 browsecomp":     ("gypsum-gpu127",   8022),
    "BM25 trec-rag":       ("gypsum-gpu144",   8042),
    "BM25 wiki-18":        ("cpu068",          8032),
}

PENDING_SERVERS = {
    "E5 trec-rag":         ("???",             8041),  # rebuilding IVF65536_HNSW32,PQ96 index
}

# BM25+Reranker proxy servers (POST /retrieve, same interface)
PROXY_SERVERS = {
    "Proxy trqa-wiki":      ("gypsum-gpu153",  8003),
    "Proxy trqa-ecommerce": ("gypsum-gpu153",  8013),
    "Proxy browsecomp":     ("gypsum-gpu131",  8023),
    "Proxy wiki-18":        ("gypsum-gpu132",  8033),
    "Proxy trec-rag":       ("gypsum-gpu132",  8043),
}

# ── Config ───────────────────────────────────────────────────────────────────
TOPK = 5
TIMEOUT = 120
STRESS_CONCURRENCY = 10
STRESS_QUERIES = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "How does BERT work?",
    "What causes inflation?",
    "History of the Roman Empire",
    "How to treat a fever at home?",
    "What is quantum entanglement?",
    "Best practices for machine learning?",
    "Who wrote Hamlet?",
    "How does photosynthesis work?",
]


def retrieve(host: str, port: int, query: str, topk: int = TOPK) -> dict:
    url = f"http://{host}:{port}/retrieve"
    resp = requests.post(url, json={"queries": [query], "topk": topk}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()



def print_docs(docs: list[dict], max_text: int = 200):
    for i, doc in enumerate(docs):
        title = doc.get("title", "(no title)")
        text  = (doc.get("text") or doc.get("contents", ""))[:max_text]
        docid = doc.get("id", "?")
        print(f"  [{i+1}] id={docid}")
        print(f"       title : {title}")
        print(f"       text  : {text!r}{'...' if len(doc.get('text',''))>max_text else ''}")


# ── Test 1: smoke test (retrieve 5 docs) ─────────────────────────────────────
def test_smoke():
    print("=" * 60)
    print("TEST 1 — Smoke test (topk=5)")
    print("=" * 60)
    query = "What is machine learning?"

    for label, (host, port) in SERVERS.items():
        print(f"\n>>> {label}  ({host}:{port})")
        print(f"    query: {query!r}")
        try:
            t0 = time.perf_counter()
            result = retrieve(host, port, query, topk=TOPK)
            elapsed = time.perf_counter() - t0
            docs = result["result"][0]
            print(f"    latency: {elapsed*1000:.0f} ms   returned: {len(docs)} docs")
            print_docs(docs)
        except Exception as e:
            print(f"    ERROR: {e}")

    if PENDING_SERVERS:
        print(f"\n  (skipping {len(PENDING_SERVERS)} pending retrieval servers)")

    # Proxy smoke test
    print("\n--- BM25+Reranker proxy servers ---")
    for label, (host, port) in PROXY_SERVERS.items():
        if host == "???":
            print(f"  [PEND] {label} (port {port})")
            continue
        print(f"\n>>> {label}  ({host}:{port})")
        try:
            t0 = time.perf_counter()
            result = retrieve(host, port, query, topk=TOPK)
            elapsed = time.perf_counter() - t0
            docs = result["result"][0]
            print(f"    latency: {elapsed*1000:.0f} ms   returned: {len(docs)} docs")
            print_docs(docs)
        except Exception as e:
            print(f"    ERROR: {e}")


# ── Test 2: stress test ───────────────────────────────────────────────────────
DUMMY_DOCS_STRESS = [
    {"id": str(i), "title": f"Doc {i}", "text": q, "contents": f"Doc {i}\n{q}"}
    for i, q in enumerate(STRESS_QUERIES)
]

def _stress_worker(args):
    host, port, query = args
    try:
        t0 = time.perf_counter()
        result = retrieve(host, port, query, topk=3)
        elapsed = time.perf_counter() - t0
        n = len(result["result"][0])
        return ("ok", elapsed, n)
    except Exception as e:
        return ("err", 0.0, str(e))



def _run_stress(label, host, port, worker_fn):
    print(f"\n>>> {label}  ({host}:{port})")
    queries = [STRESS_QUERIES[i % len(STRESS_QUERIES)] for i in range(STRESS_CONCURRENCY)]
    worker_args = [(host, port, q) for q in queries]

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=STRESS_CONCURRENCY) as pool:
        results = list(pool.map(worker_fn, worker_args))
    wall = time.perf_counter() - t_start

    ok   = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    latencies = sorted([r[1] for r in ok])

    print(f"    total wall time : {wall:.2f}s")
    print(f"    success / total : {len(ok)} / {STRESS_CONCURRENCY}")
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"    latency p50/p95/p99 : {p50*1000:.0f}/{p95*1000:.0f}/{p99*1000:.0f} ms")
        print(f"    throughput : {len(ok)/wall:.1f} req/s")
    if errs:
        print(f"    ERRORS ({len(errs)}):")
        for _, _, msg in errs[:5]:
            print(f"      {msg}")


def test_stress():
    print("\n" + "=" * 60)
    print(f"TEST 2 — Stress test ({STRESS_CONCURRENCY} concurrent requests)")
    print("=" * 60)

    print("\n--- Retrieval servers ---")
    for label, (host, port) in SERVERS.items():
        _run_stress(label, host, port, _stress_worker)

    print("\n--- BM25+Reranker proxy servers ---")
    for label, (host, port) in PROXY_SERVERS.items():
        if host == "???":
            print(f"  [PEND] {label} (port {port})")
            continue
        _run_stress(label, host, port, _stress_worker)


if __name__ == "__main__":
    print(f"\nRetrieval servers   : {list(SERVERS.keys())}")
    print(f"Pending servers     : {list(PENDING_SERVERS.keys())}")
    print(f"Proxy servers       : {list(PROXY_SERVERS.keys())}\n")
    test_smoke()
    test_stress()
