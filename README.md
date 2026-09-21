# Compliance Assistant — RAG over Regulatory & Policy Documents

![CI](https://github.com/baddy1411/compliance-assistant/actions/workflows/python-ci.yml/badge.svg)

A working **retrieval-augmented generation (RAG)** service that answers compliance
questions grounded in your own regulatory and policy documents. Ask a question,
get an answer with **cited sources** — and an honest *"I don't know"* when the
indexed documents don't cover it.

This is a portfolio project demonstrating production-style LLM engineering:
document ingestion, chunking strategy, local embeddings, a persistent vector
store, grounded prompting, evals, and a LoRA fine-tuning path.

> **Scope note:** the bundled documents are short *sample* summaries written for
> the demo (GDPR, EU AI Act, a fictional company policy). They are not legal
> advice — replace `data/` with your own corpus.

## Architecture

```
                        ┌──────────────┐
                        │  data/*.md   │
                        │  data/*.pdf  │
                        └──────┬───────┘
                               │ ingest.py  (load → chunk 500/50)
                               ▼
┌──────────┐   ┌─────────────────────────────────────┐
│ Question │──▶│  ComplianceAssistant (rag.py)        │
└──────────┘   │   1. embed query (all-MiniLM-L6-v2) │
               │   2. ChromaDB top-k retrieval       │
               │   3. grounded LLM call w/ citations │──▶ POST /ask
               │      (or extractive fallback)       │    POST /ingest
               └─────────────────────────────────────┘    GET /health
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        vectorstore.py    embeddings.py      llm.py
        (ChromaDB,        (sentence-         (OpenAI-compatible:
         persistent)       transformers)      DeepSeek / OpenAI)
```

## Quickstart — local mode, no Docker (< 5 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU torch: see note below

# 1. Ingest the sample documents
python ingest.py --folder data --chroma-dir chroma_db

# 2. Ask questions (works without any API key — extractive fallback)
python ask.py "Within how many hours must a data breach be reported?"

# 3. Or run the API
uvicorn api:app --reload
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long does Northwind retain customer data?", "top_k": 3}'
```

For grounded LLM answers, set `LLM_API_KEY` (and optionally `LLM_BASE_URL` /
`LLM_MODEL`, see `.env.example`). Without a key the service still works: it
returns the most relevant passages instead of a generated answer.

> **torch:** `requirements.txt` pins the default (CUDA-bundled) wheel. On a
> CPU-only machine install the CPU build first:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`,
> then `pip install -r requirements.txt`.

## Components

| File | What it does |
|---|---|
| `ingest.py` | Loads `.md`/`.txt`/`.pdf` (`pypdf`), recursive character chunking (500 chars, 50 overlap — see below), CLI + `ingest_folder()` |
| `embeddings.py` | Lazy `sentence-transformers` wrapper (`all-MiniLM-L6-v2` by default, local, no API key) |
| `vectorstore.py` | Persistent ChromaDB wrapper: add/query/count/reset, cosine similarity scores |
| `llm.py` | OpenAI-compatible client; system prompt forces citations as `[doc_id]` and refusal when context is insufficient |
| `rag.py` | Orchestrator: retrieve → generate → cite; extractive fallback when no LLM key; timed answers |
| `api.py` | FastAPI: `POST /ask`, `POST /ingest`, `GET /health` (pydantic schemas, dependency-injectable) |
| `ask.py` | CLI demo client |
| `producer.py` / `streaming_pipeline.py` | Kafka path: publish docs → consume & index (needs `docker compose up`) |
| `train.py` | Real LoRA fine-tuning (transformers + PEFT) on `data/train_sample.jsonl` — needs a GPU |
| `evals/` | Eval harness: gold Q&A set, recall@k / MRR, LLM-as-judge faithfulness |

### Chunking choice

Compliance text is dense, so chunks are **500 characters with 50 characters of
overlap** (~100–130 tokens). Small chunks keep retrieval precise (one
obligation per chunk instead of a whole article); the overlap preserves
sentence continuity across boundaries. Both are configurable
(`CHUNK_SIZE` / `CHUNK_OVERLAP`).

## Evals — measured, not claimed

`python -m evals.run_evals` ingests `data/` into a throwaway index and runs a
10-question gold set (`evals/eval_set.json`: 9 in-scope, 1 out-of-scope):

<!--EVAL_RESULTS-->

```
Compliance RAG evals (top_k=5, chunks=30)        # measured 2026-09-21
recall@5: 1.000   (9/9 in-scope questions, expected doc in top-5)
MRR:      1.000   (expected doc ranked #1 for all 9)
faithfulness: skipped (no LLM_API_KEY in this run)
abstention (out-of-scope refused): True
```

Small honest caveat: this is a 10-question set over 3 short sample documents,
so treat it as a regression harness, not a benchmark. Re-run any time with
`python -m evals.run_evals` after changing chunking, embeddings, or prompts.

Retrieval metrics are recall@k and MRR against the expected source document;
faithfulness is an LLM-as-judge check (skipped without `LLM_API_KEY`); the
out-of-scope question verifies the abstention path.

## LoRA fine-tuning

```bash
python train.py --data data/train_sample.jsonl --epochs 3        # GPU
python train.py --data data/train_sample.jsonl --max-steps 2     # CPU smoke test
```

Fine-tunes `HuggingFaceTB/SmolLM2-135M` with LoRA (r=8, α=16) on 12
instruction examples. Saves the adapter to `adapters/compliance-lora/`.
Practical training needs a GPU — CPU is only for pipeline smoke tests.

Smoke test (verified 2026-09-21, CPU): `--max-steps 2` completed,
921,600 trainable params (0.68% of 135M), train loss 2.877, adapter +
tokenizer saved successfully.

## Kafka streaming path

```bash
docker compose up -d        # single-node Kafka (KRaft, no Zookeeper)
python producer.py          # publish data/ documents to the topic
python streaming_pipeline.py  # consume + index into ChromaDB
```

## Tech stack

Python 3.11 · FastAPI · Pydantic · ChromaDB · sentence-transformers ·
OpenAI-compatible LLMs (DeepSeek default) · pypdf · Kafka · transformers +
PEFT/LoRA · pytest · ruff · Docker

## Roadmap

- Hybrid retrieval (BM25 + dense) and a cross-encoder reranker
- Citation-level faithfulness evals with a fixed judge model
- Incremental ingestion (file watcher) and document versioning
- Wiring the LoRA adapter into the serving path (vLLM)
- Auth for the API; Postgres-backed metadata

## Disclaimer

Sample documents are simplified demo content, not legal advice. The EU AI Act
and GDPR summaries are abbreviated and may be outdated — always consult the
official texts and qualified counsel for real compliance work.
