
import json
import os
import time
from collections import defaultdict


def _normalize(path: str) -> str:
    """Compare by filename only so benchmark files stay portable."""
    return os.path.basename(path)


def precision_at_k(retrieved, relevant, k):
    retrieved_k = {_normalize(d) for d in retrieved[:k]}  # deduplicate
    relevant_norm = {_normalize(d) for d in relevant}
    hits = len(retrieved_k & relevant_norm)
    return hits / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = {_normalize(d) for d in retrieved[:k]}  # deduplicate
    relevant_norm = {_normalize(d) for d in relevant}
    hits = len(retrieved_k & relevant_norm)
    return hits / len(relevant_norm) if relevant_norm else 0.0

def reciprocal_rank(retrieved, relevant):
    relevant_norm = {_normalize(d) for d in relevant}
    for i, doc in enumerate(retrieved, start=1):
        if _normalize(doc) in relevant_norm:
            return 1 / i
    return 0.0

def evaluate(benchmark_file, retriever_fn, k=5):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    latencies = []

    with open(benchmark_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            question = sample["question"]
            relevant = sample["relevant_documents"]

            start = time.time()
            retrieved = retriever_fn(question)
            latency = (time.time() - start) * 1000

            precision_scores.append(precision_at_k(retrieved, relevant, k))
            recall_scores.append(recall_at_k(retrieved, relevant, k))
            mrr_scores.append(reciprocal_rank(retrieved, relevant))
            latencies.append(latency)

    results = {
        "precision@{}".format(k): sum(precision_scores)/len(precision_scores),
        "recall@{}".format(k): sum(recall_scores)/len(recall_scores),
        "mrr": sum(mrr_scores)/len(mrr_scores),
        "avg_latency_ms": sum(latencies)/len(latencies)
    }

    return results


if __name__ == "__main__":
    print("This script must be integrated with PaperRAG retriever function.")
