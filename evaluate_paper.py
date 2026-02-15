
import json
import time
from collections import defaultdict

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved, relevant):
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
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
