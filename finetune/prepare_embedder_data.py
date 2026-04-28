#!/usr/bin/env python3
"""
Prepare training data for SentenceTransformer fine-tuning from a PaperRAG index.

Reads chunks from an existing index and produces (query, positive, negative) triplets.

Two query generation modes:
  - heuristic (default): uses paper title + section name — fast, no LLM needed
  - llm: asks Ollama to write a question for each chunk — slower, higher quality

Usage:
    python finetune/prepare_embedder_data.py
    python finetune/prepare_embedder_data.py --index-dir ~/papers/.paperrag-index
    python finetune/prepare_embedder_data.py --mode llm --model qwen2.5:1.5b
    python finetune/prepare_embedder_data.py --max-chunks 2000

Output:
    finetune/data/embedder_train.jsonl
    finetune/data/embedder_val.jsonl
Each line: {"query": "...", "pos": "...", "neg": "..."}
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_chunks(index_dir: Path) -> list[dict]:
    from paperrag.vectorstore import VectorStore
    if not VectorStore.exists(index_dir):
        print(f"Error: no PaperRAG index at {index_dir}")
        print("Run: paperrag index --input-dir <your-pdf-dir>")
        sys.exit(1)
    store = VectorStore.load(index_dir)
    print(f"Loaded {len(store.chunks)} chunks from {len(store.file_hashes)} papers")
    return store.chunks


def heuristic_queries(chunks: list[dict]) -> list[dict]:
    """Build queries from paper title and section name — no LLM required."""
    skip_sections = {"references", "bibliography", "acknowledgments", "acknowledgements", ""}
    pairs = []
    for c in chunks:
        section = (c.get("section_name") or "").strip()
        title = (c.get("paper_title") or "").strip()
        text = (c.get("text") or "").strip()
        if not text or len(text) < 100:
            continue
        if section.lower() in skip_sections:
            continue
        if section:
            query = f"{title}: {section}" if title else section
        elif title:
            query = title
        else:
            continue
        pairs.append({"query": query, "pos": text, "_file": c.get("file_path", "")})
    return pairs


def llm_queries(chunks: list[dict], model: str) -> list[dict]:
    """Generate a synthetic question per chunk using Ollama."""
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    skip_sections = {"references", "bibliography", "acknowledgments", "acknowledgements", ""}
    pairs = []
    total = len(chunks)
    for i, c in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Generating questions {i}/{total}...")
        section = (c.get("section_name") or "").strip()
        text = (c.get("text") or "").strip()
        if not text or len(text) < 100 or section.lower() in skip_sections:
            continue
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Write one specific research question that the following academic "
                        "text directly answers. Output only the question.\n\n"
                        f"{text[:600]}"
                    ),
                }],
                max_tokens=64,
                temperature=0.7,
            )
            question = resp.choices[0].message.content.strip()
            question = re.sub(r'^["\']+|["\']+$', "", question).strip()
            if question and len(question) > 10:
                pairs.append({"query": question, "pos": text, "_file": c.get("file_path", "")})
        except Exception:
            continue
    return pairs


def add_negatives(pairs: list[dict], chunks: list[dict]) -> list[dict]:
    """Attach a hard negative to each pair using BM25, falling back to random."""
    try:
        from rank_bm25 import BM25Okapi
        corpus_tokens = [c["text"].split() for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        for pair in pairs:
            scores = bm25.get_scores(pair["query"].split())
            # Pick highest-scoring chunk from a *different* paper
            for idx in sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True):
                if chunks[idx].get("file_path") != pair["_file"]:
                    pair["neg"] = chunks[idx]["text"]
                    break
        print("Hard negatives added via BM25")
    except ImportError:
        print("rank-bm25 not found — using random negatives (pip install rank-bm25 for better results)")
        by_file: dict[str, list[dict]] = {}
        for c in chunks:
            by_file.setdefault(c.get("file_path", ""), []).append(c)
        files = list(by_file.keys())
        for pair in pairs:
            others = [f for f in files if f != pair["_file"]]
            if others:
                pair["neg"] = random.choice(by_file[random.choice(others)])["text"]
    return [p for p in pairs if "neg" in p]


def save_splits(pairs: list[dict], output: Path, val_ratio: float) -> None:
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    output.parent.mkdir(parents=True, exist_ok=True)
    val_path = output.with_name(output.stem + "_val" + output.suffix)

    def _write(path: Path, data: list[dict]) -> None:
        with open(path, "w") as f:
            for item in data:
                # Strip internal routing field before saving
                item.pop("_file", None)
                f.write(json.dumps(item) + "\n")

    _write(output, train_pairs)
    _write(val_path, val_pairs)
    print(f"Saved {len(train_pairs)} train → {output}")
    print(f"Saved {len(val_pairs)} val   → {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", default=None, help="Path to .paperrag-index (auto-detected if omitted)")
    parser.add_argument("--output", default="finetune/data/embedder_train.jsonl")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic",
                        help="Query generation mode (default: heuristic)")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model for --mode llm")
    parser.add_argument("--max-chunks", type=int, default=None, help="Process at most N chunks")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    # Resolve index directory
    if args.index_dir:
        index_dir = Path(args.index_dir).expanduser()
    else:
        from paperrag.config import PaperRAGConfig, load_rc, apply_rc
        cfg = PaperRAGConfig()
        apply_rc(cfg, load_rc(Path.home() / ".paperragrc"))
        apply_rc(cfg, load_rc(Path.cwd() / ".paperragrc"))
        index_dir = Path(cfg.index_dir)

    chunks = load_chunks(index_dir)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    print(f"Generating queries (mode={args.mode})...")
    pairs = llm_queries(chunks, args.model) if args.mode == "llm" else heuristic_queries(chunks)
    print(f"Generated {len(pairs)} pairs before negatives")

    pairs = add_negatives(pairs, chunks)
    print(f"{len(pairs)} pairs with negatives")

    save_splits(pairs, Path(args.output), args.val_split)


if __name__ == "__main__":
    main()
