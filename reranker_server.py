"""
Pure Reranker Server (corpus-agnostic).

Takes a query + list of documents, returns them reranked by Qwen3-Reranker-0.6B.
Run 2 instances on 2 separate A100s for round-robin load balancing.

Usage:
  CUDA_VISIBLE_DEVICES=0 python reranker_server.py --port 9001
  CUDA_VISIBLE_DEVICES=1 python reranker_server.py --port 9002

Endpoint: POST /rerank
  Input:  {"query": "...", "documents": [{id, title, text, ...}, ...], "topk": 10}
  Output: {"result": [{doc..., "reranker_score": 0.95}, ...]}
"""

import argparse
import threading
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-Reranker-0.6B")
parser.add_argument("--port", type=int, default=9001)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()


# ── Qwen3 Reranker ─────────────────────────────────────────────────────────────
class Qwen3Reranker:
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct. Note that the answer can only be \"yes\" or \"no\"."
    )
    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, model_name, max_length, batch_size):
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"Loading reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16,
        ).eval().cuda()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        self._lock = threading.Lock()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id  = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id == self.tokenizer.unk_token_id:
            self.yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
            self.no_id  = self.tokenizer.convert_tokens_to_ids("No")

    def _build_prompt(self, query: str, document: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"<Instruct>: {self.INSTRUCT}\n"
                f"<Query>: {query}\n"
                f"<Document>: {document}"
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return prompt + "<think>\n\n</think>\n\n"

    @torch.no_grad()
    def score(self, query: str, documents: List[str]) -> List[float]:
        prompts = [self._build_prompt(query, doc) for doc in documents]
        scores = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            with self._lock:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=self.max_length)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                out = self.model(**inputs)
                # .clone() breaks the view so out (the full ~20GB logits tensor) can be
                # freed immediately — without this, logits holds a view reference to
                # out.logits and 20GB stays allocated for the lifetime of this thread
                logits = (out.logits if hasattr(out, "logits") else out[0])[:, -1, :].clone()
                del out, inputs
                torch.cuda.empty_cache()
                yn = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=-1)
                del logits
                batch_scores = torch.softmax(yn, dim=-1)[:, 0].cpu().tolist()
            scores.extend(batch_scores)
        return scores


reranker = Qwen3Reranker(args.reranker_model, args.max_length, args.batch_size)
app = FastAPI()


class RerankRequest(BaseModel):
    query: str
    documents: List[dict]
    topk: Optional[int] = 10


@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """
    POST /rerank
    Input:  {"query": "...", "documents": [{id, title, text, ...}, ...], "topk": 10}
    Output: {"result": [{doc..., "reranker_score": 0.95}, ...]}
    """
    topk = min(request.topk or 10, len(request.documents))
    doc_texts = [
        f"{d.get('title','')}\n{d.get('text', d.get('contents',''))}"
        for d in request.documents
    ]
    scores = reranker.score(request.query, doc_texts)
    ranked = sorted(zip(scores, request.documents), key=lambda x: x[0], reverse=True)
    result = [{"reranker_score": s, **doc} for s, doc in ranked[:topk]]
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
