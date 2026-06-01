# Rollouts Inventory

Base path: `/scratch/workspace/oyilmazel_umass_edu-cs791/rollouts/`

---

## Models Overview

| Model | Size | Retriever | Folder |
|-------|------|-----------|--------|
| ASearcher | 7B | E5 | `ASearcher/7b_e5/` |
| ReSearch | 7B | E5 | `ReSearch/7b_e5/` |
| Search-o1 | 7B | E5 | `Search_o1/7b_e5/` |
| Search-R1 | 3B | E5 | `Search_r1/3b_e5/` |
| Search-R1 | 7B | E5 | `Search_r1/7b_e5/` |
| Search-R1 | 7B | BM25 | `Search_r1/7b_bm25/` |
| Search-R1 | 7B | BM25+Rerank | `Search_r1/7b_bm25_rerank/` |
| Search-R1 | 14B | E5 | `Search_r1/14b_e5/` |

---

## Dataset Coverage per Model

✅ = present  ❌ = absent

| Dataset | ASearcher 7B E5 | ReSearch 7B E5 | Search-o1 7B E5 | Search-R1 3B E5 | Search-R1 7B E5 | Search-R1 7B BM25 | Search-R1 7B BM25+Rerank | Search-R1 14B E5 |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2wikimultihopqa | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| hotpotqa | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| musique | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| bamboogle | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| trec_rag_2024 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| browsecomp | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| totalrecall-e-commerce | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| totalrecall-wiki1 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| totalrecall-wiki2 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## File Sizes

### ASearcher/7b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 77M |
| bamboogle.jsonl | 585K |
| browsecomp.jsonl | 29M |
| hotpotqa.jsonl | 41M |
| musique.jsonl | 15M |
| totalrecall-e-commerce.jsonl | 8.6M |
| totalrecall-wiki1.jsonl | 2.0M |
| totalrecall-wiki2.jsonl | 14M |
| trec_rag_2024.jsonl | 2.1M |

### ReSearch/7b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 79M |
| bamboogle.jsonl | 675K |
| hotpotqa.jsonl | 43M |
| musique.jsonl | 33M |
| trec_rag_2024.jsonl | 2.4M |

### Search_o1/7b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 75M |
| bamboogle.jsonl | 1.1M |
| browsecomp.jsonl | 4.7M |
| hotpotqa.jsonl | 44M |
| musique.jsonl | 19M |
| totalrecall-e-commerce.jsonl | 3.9M |
| totalrecall-wiki1.jsonl | 750K |
| totalrecall-wiki2.jsonl | 5.0M |
| trec_rag_2024.jsonl | 2.0M |

### Search_r1/3b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 75M |
| bamboogle.jsonl | 703K |
| browsecomp.jsonl | 35M |
| hotpotqa.jsonl | 43M |
| musique.jsonl | 33M |
| totalrecall-e-commerce.jsonl | 5.7M |
| totalrecall-wiki1.jsonl | 1.2M |
| totalrecall-wiki2.jsonl | 6.9M |
| trec_rag_2024.jsonl | 3.1M |

### Search_r1/7b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 122M |
| bamboogle.jsonl | 1.3M |
| hotpotqa.jsonl | 71M |
| musique.jsonl | 24M |
| totalrecall-e-commerce.jsonl | 8.3M |
| totalrecall-wiki1.jsonl | 1.5M |
| totalrecall-wiki2.jsonl | 11M |
| trec_rag_2024.jsonl | 5.7M |

### Search_r1/7b_bm25/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 130M |
| bamboogle.jsonl | 1.4M |
| browsecomp.jsonl | 21M |
| hotpotqa.jsonl | 77M |
| musique.jsonl | 26M |
| totalrecall-e-commerce.jsonl | 8.9M |
| totalrecall-wiki1.jsonl | 1.8M |
| totalrecall-wiki2.jsonl | 13M |
| trec_rag_2024.jsonl | 5.5M |

### Search_r1/7b_bm25_rerank/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 114M |
| bamboogle.jsonl | 1.2M |
| browsecomp.jsonl | 36M |
| hotpotqa.jsonl | 68M |
| musique.jsonl | 44M |
| totalrecall-e-commerce.jsonl | 13M |
| totalrecall-wiki1.jsonl | 2.5M |
| totalrecall-wiki2.jsonl | 18M |
| trec_rag_2024.jsonl | 5.8M |

### Search_r1/14b_e5/
| File | Size |
|------|------|
| 2wikimultihopqa.jsonl | 86M |
| bamboogle.jsonl | 741K |
| browsecomp.jsonl | 18M |
| hotpotqa.jsonl | 44M |
| musique.jsonl | 35M |
| totalrecall-e-commerce.jsonl | 5.6M |
| totalrecall-wiki1.jsonl | 1.2M |
| totalrecall-wiki2.jsonl | 7.3M |
| trec_rag_2024.jsonl | 3.1M |

---

## Notes

- ReSearch only has 5 datasets (missing browsecomp and all totalrecall splits).
- Search-R1 7b_e5 is missing browsecomp.
- `raw/` subdirectories contain the original submitted files (named by submitter); the parent folder contains the cleaned/renamed versions used for analysis.
- BM25+Rerank files are noticeably larger (e.g. musique 44M vs 24M for E5) likely due to longer retrieved passages from the reranker pipeline.
