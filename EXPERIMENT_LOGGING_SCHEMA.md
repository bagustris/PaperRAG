
# PaperRAG Experiment Logging Schema

All experiments must produce a JSON log file.

## Required Fields

{
  "experiment_id": "string",
  "timestamp": "ISO8601",
  "embedding_model": "string",
  "embedding_dimension": int,
  "index_version": "string",
  "chunk_size": int,
  "chunk_overlap": int,
  "top_k": int,
  "precision_at_k": float,
  "recall_at_k": float,
  "mrr": float,
  "avg_latency_ms": float,
  "num_documents_indexed": int,
  "num_chunks": int,
  "hardware": {
      "cpu": "string",
      "gpu": "string or null",
      "ram_gb": int
  }
}

## Determinism Requirements

- temperature = 0.0
- fixed random seed
- fixed embedding model version
- index snapshot stored

Logs must be stored in:

experiments/logs/<experiment_id>.json
