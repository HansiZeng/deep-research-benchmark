"""
Retrieval server for deep-research-benchmark.
Supports BM25 and E5 dense retrieval via FastAPI.

Based on Search-R1/search_r1/search/retrieval_server.py
"""

import json
import os
import threading
import warnings
from typing import List, Optional
import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Launch a retrieval server.")
parser.add_argument("--retrieval_method", type=str, required=True,
                    choices=["bm25", "e5"],
                    help="Retrieval method: bm25 or e5 (dense).")
parser.add_argument("--index_path", type=str, required=True,
                    help="Path to FAISS index file (dense) or Lucene index dir (bm25).")
parser.add_argument("--corpus_path", type=str, required=True,
                    help="Path to corpus JSONL file.")
parser.add_argument("--topk", type=int, default=10,
                    help="Default number of retrieved passages per query.")
parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base",
                    help="HuggingFace model name/path for dense retrieval.")
parser.add_argument("--port", type=int, default=8000,
                    help="Port to run the FastAPI server on.")
parser.add_argument("--max_length", type=int, default=256,
                    help="Max token length for dense encoding.")
parser.add_argument("--batch_size", type=int, default=512,
                    help="Encode batch size for dense retrieval.")
parser.add_argument("--faiss_gpu", action="store_true", default=False,
                    help="Load FAISS index to GPU (dense only).")
parser.add_argument("--use_fp16", action="store_true", default=False,
                    help="Use fp16 for the dense encoder.")
args = parser.parse_args()


# ── shared utilities (copied from Search-R1 to avoid import path issues) ──────
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=4,
    )
    return corpus


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def load_model(model_path: str, use_fp16: bool = False):
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                               trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None,
            pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError(f"Pooling method '{pooling_method}' not implemented!")


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length,
                 use_fp16):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model, self.tokenizer = load_model(model_path, use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [prefix + q for q in query_list]
        elif "bge" in self.model_name.lower() and is_query:
            query_list = [
                f"Represent this sentence for searching relevant passages: {q}"
                for q in query_list
            ]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output, output.last_hidden_state,
                            inputs["attention_mask"], self.pooling_method)
        if "dpr" not in self.model_name.lower():
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32,
                                                             order="C")
        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


# ── Retrievers ─────────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, index_path: str, corpus_path: str, topk: int):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_path)
        self.topk = topk
        # check whether raw docs are stored in the index
        self.contain_doc = self.searcher.doc(0) is not None and \
                           self.searcher.doc(0).raw() is not None
        if not self.contain_doc:
            self.corpus = load_corpus(corpus_path)

    def _search(self, query: str, num: int, return_score: bool = False):
        hits = self.searcher.search(query, num)
        if not hits:
            return ([], []) if return_score else []

        scores = [hit.score for hit in hits[:num]]
        if self.contain_doc:
            results = []
            for hit in hits[:num]:
                raw = json.loads(self.searcher.doc(hit.docid).raw())
                contents = raw.get("contents", "")
                lines = contents.split("\n")
                results.append({
                    "id": hit.docid,
                    "title": lines[0].strip('"') if lines else "",
                    "text": "\n".join(lines[1:]) if len(lines) > 1 else contents,
                    "contents": contents,
                })
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits[:num]])

        return (results, scores) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None,
                     return_score: bool = False):
        if num is None:
            num = self.topk
        all_results, all_scores = [], []
        for query in query_list:
            r, s = self._search(query, num, return_score=True)
            all_results.append(r)
            all_scores.append(s)
        return (all_results, all_scores) if return_score else all_results


class DenseRetriever:
    def __init__(self, index_path: str, corpus_path: str, topk: int,
                 model_name: str, model_path: str, pooling_method: str,
                 max_length: int, use_fp16: bool, batch_size: int,
                 faiss_gpu: bool):
        self.topk = topk
        self.batch_size = batch_size

        self.index = faiss.read_index(index_path)
        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        # FAISS GPU StackDeviceMemory is not thread-safe; serialize all searches
        self._lock = threading.Lock()

        self.corpus = load_corpus(corpus_path)
        self.encoder = Encoder(model_name, model_path, pooling_method,
                               max_length, use_fp16)

    def batch_search(self, query_list: List[str], num: int = None,
                     return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        all_results, all_scores = [], []
        for start in tqdm(range(0, len(query_list), self.batch_size),
                          desc="Dense retrieval"):
            batch = query_list[start:start + self.batch_size]
            with self._lock:
                emb = self.encoder.encode(batch, is_query=True)
                scores, idxs = self.index.search(emb, k=num)

            for i in range(len(batch)):
                docs = load_docs(self.corpus, idxs[i])
                all_results.append(docs)
                all_scores.append(scores[i].tolist())

            del emb, scores, idxs
            torch.cuda.empty_cache()

        return (all_results, all_scores) if return_score else all_results


# ── FastAPI app ────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# Build retriever at startup
if args.retrieval_method == "bm25":
    retriever = BM25Retriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
    )
else:
    retriever = DenseRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
        model_name=args.retriever_model,
        model_path=args.retriever_model,
        pooling_method="mean",
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        batch_size=args.batch_size,
        faiss_gpu=args.faiss_gpu,
    )


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    POST /retrieve
    Body: {"queries": ["q1", "q2"], "topk": 10, "return_scores": false}
    Returns: {"result": [[{id, title, text, contents}, ...], ...]}
    """
    topk = request.topk or args.topk

    if request.return_scores:
        results, scores = retriever.batch_search(
            request.queries, num=topk, return_score=True)
        resp = [
            [{"document": doc, "score": score}
             for doc, score in zip(res, sc)]
            for res, sc in zip(results, scores)
        ]
    else:
        results = retriever.batch_search(
            request.queries, num=topk, return_score=False)
        resp = results

    return {"result": resp}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
