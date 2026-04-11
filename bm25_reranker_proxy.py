"""
BM25 + Reranker Proxy Server (one per corpus).

Pipeline:
  1. Call local BM25 server → top-bm25_topk docs
  2. Round-robin to one of N reranker servers → reranked top-k
  3. Return to caller

Callers only see one endpoint and don't need to know about BM25 or reranker internals.

Usage:
  python bm25_reranker_proxy.py \
      --bm25_url http://localhost:8002 \
      --reranker_urls http://NODE1:9001 http://NODE2:9002 \
      --port 8003 \
      --bm25_topk 100 \
      --topk 10
"""

import argparse
import itertools
import threading
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--bm25_url", type=str, required=True,
                    help="URL of the BM25 retrieval server for this corpus, e.g. http://localhost:8002")
parser.add_argument("--reranker_urls", type=str, nargs="+", required=True,
                    help="URLs of reranker servers, e.g. http://node1:9001 http://node2:9002")
parser.add_argument("--port", type=int, default=8003)
parser.add_argument("--bm25_topk", type=int, default=100,
                    help="Number of BM25 candidates to fetch before reranking.")
parser.add_argument("--topk", type=int, default=10,
                    help="Number of final results to return.")
parser.add_argument("--timeout", type=float, default=60.0,
                    help="HTTP timeout in seconds for upstream calls.")
args = parser.parse_args()

# ── Round-robin reranker selector ──────────────────────────────────────────────
_reranker_cycle = itertools.cycle(args.reranker_urls)
_lock = threading.Lock()

def next_reranker() -> str:
    with _lock:
        return next(_reranker_cycle)


app = FastAPI()
client = httpx.Client(timeout=args.timeout)


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    POST /retrieve  (same interface as bm25/e5 servers)
    Input:  {"queries": [...], "topk": 10, "return_scores": false}
    Output: {"result": [[{doc...}, ...], ...]}
    """
    topk = min(request.topk or args.topk, 100)
    bm25_topk = min(args.bm25_topk, 100)
    resp_data = []

    for query in request.queries:
        # Step 1: BM25 retrieval
        bm25_resp = client.post(
            f"{args.bm25_url}/retrieve",
            json={"queries": [query], "topk": bm25_topk, "return_scores": False},
        )
        if bm25_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"BM25 server error: {bm25_resp.text}")
        candidates = bm25_resp.json()["result"][0]

        if not candidates:
            resp_data.append([])
            continue

        # Step 2: Rerank (round-robin)
        reranker_url = next_reranker()
        rerank_resp = client.post(
            f"{reranker_url}/rerank",
            json={"query": query, "documents": candidates, "topk": topk},
        )
        if rerank_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Reranker error: {rerank_resp.text}")
        reranked = rerank_resp.json()["result"]

        if request.return_scores:
            resp_data.append(reranked)  # already has reranker_score field
        else:
            resp_data.append([{k: v for k, v in doc.items() if k != "reranker_score"}
                               for doc in reranked])

    return {"result": resp_data}


@app.get("/health")
def health():
    return {"bm25_url": args.bm25_url, "reranker_urls": args.reranker_urls}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
