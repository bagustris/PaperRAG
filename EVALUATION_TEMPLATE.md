# PaperRAG Retrieval Benchmark Template

This template defines a reproducible evaluation format for retrieval
experiments.

------------------------------------------------------------------------

## Benchmark File Format (JSONL)

Each line:

{ "question": "...", "relevant_documents": \[ {"paper_title": "...",
"section": "..."}, {"paper_title": "...", "section": "..."} \] }

------------------------------------------------------------------------

## Metrics

Required metrics:

-   Precision@k
-   Recall@k
-   Mean Reciprocal Rank (MRR)
-   Retrieval latency
-   Context length

------------------------------------------------------------------------

## Example Benchmark Entry

{ "question": "What dataset is used for cross-database evaluation?",
"relevant_documents": \[ {"paper_title": "Cross-DB Deepfake Study",
"section": "Experiments"} \] }

------------------------------------------------------------------------

## Evaluation Command

    paper_rag evaluate --benchmark-file benchmark.jsonl --top-k 5

------------------------------------------------------------------------

## Output Format

{ "precision@5": 0.80, "recall@5": 0.75, "mrr": 0.83, "avg_latency_ms":
42 }

------------------------------------------------------------------------

All evaluations must be deterministic. Set embedding model and index
version explicitly.
