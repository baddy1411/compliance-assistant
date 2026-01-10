# Compliance Assistant with RAG and LoRA
![CI](https://github.com/baddy1411/compliance-assistant/actions/workflows/python-ci.yml/badge.svg)

This repository showcases a domain‑specific compliance engine that retrieves the latest regulations and internal policies using a retrieval‑augmented generation (RAG) pipeline and generates tailored documents with a fine‑tuned model. RAG pulls documents from databases, knowledge bases or the web to ground outputs and reduce hallucinations【736712269151221†L420-L425】. Parameter‑efficient techniques like LoRA allow the model to be tuned to company‑specific guidelines at low cost【736712269151221†L420-L425】.

## Components

- **Producer:** Streams regulatory updates and company policy documents into Kafka topics.
- **Train model:** Placeholder script demonstrating how to fine‑tune a base language model with LoRA on domain data.
- **Streaming pipeline:** Reads from Kafka, performs vector search over indexed documents, and generates compliance summaries or risk assessments.
- **API:** FastAPI endpoint that accepts compliance questions and returns model responses.
- **Docker Compose:** Defines the Kafka stack.

## Setup

1. Start the Kafka cluster via `docker compose up`.
2. Fine‑tune your model: `python train_model.py`.
3. Populate the vector store and index documents.
4. Run the producer to stream new documents: `python producer.py`.
5. Start the streaming pipeline: `python streaming_pipeline.py`.
6. Serve the API: `uvicorn api:app --reload`.

## Disclaimer

This repository provides scaffolding for building a compliance assistant and is not a complete implementation. You must integrate a vector database, retrieval libraries, and a generative model to make it functional.
