#!/usr/bin/env python3
"""
Generate Q&A training data for LLM fine-tuning from a PaperRAG index.

For each paper chunk, asks Ollama to produce a question and a reference answer.
Output is alpaca-format JSONL, ready for Unsloth / TRL fine-tuning.

Usage:
    python finetune/prepare_llm_data.py
    python finetune/prepare_llm_data.py --index-dir ~/papers/.paperrag-index
    python finetune/prepare_llm_data.py --model qwen2.5:7b --max-chunks 3000

Output: finetune/data/llm_train.jsonl
Each line:
    {
      "instruction": "Answer the question based on the provided context.",
      "input": "Context: ...\n\nQuestion: ...",
      "output": "..."
    }
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer based on the provided context. "
    "Be concise and cite relevant parts of the context."
)

GENERATION_PROMPT = """\
Given the following excerpt from an academic paper, write:
1. A specific research question that this text answers (one sentence)
2. A concise, accurate answer to that question based only on the text

Output in this exact format:
QUESTION: <question>
ANSWER: <answer>

Text:
{text}"""


def load_chunks(index_dir: Path) -> list[dict]:
    from paperrag.vectorstore import VectorStore
    if not VectorStore.exists(index_dir):
        print(f"Error: no index at {index_dir}")
        print("Run: paperrag index --input-dir <your-pdf-dir>")
        sys.exit(1)
    store = VectorStore.load(index_dir)
    print(f"Loaded {len(store.chunks)} chunks from {len(store.file_hashes)} papers")
    return store.chunks


def generate_qa(chunks: list[dict], model: str, max_tokens: int) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    skip_sections = {"references", "bibliography", "acknowledgments", "acknowledgements", ""}
    records = []
    total = len(chunks)

    for i, c in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Processing chunk {i}/{total}  (generated {len(records)} so far)...")

        section = (c.get("section_name") or "").strip().lower()
        text = (c.get("text") or "").strip()
        if not text or len(text) < 150 or section in skip_sections:
            continue

        # Trim very long chunks to fit context
        truncated = text[:1200] if len(text) > 1200 else text

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": GENERATION_PROMPT.format(text=truncated)},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            output = resp.choices[0].message.content.strip()
        except Exception as e:
            continue

        # Parse structured output
        question, answer = None, None
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("QUESTION:"):
                question = line[len("QUESTION:"):].strip()
            elif line.startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()

        if not question or not answer or len(answer) < 20:
            continue

        records.append({
            "instruction": "Answer the question based on the provided context.",
            "input": f"Context:\n{truncated}\n\nQuestion: {question}",
            "output": answer,
        })

    return records


def save_splits(records: list[dict], output: Path, val_ratio: float) -> None:
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    output.parent.mkdir(parents=True, exist_ok=True)
    val_path = output.with_name(output.stem + "_val" + output.suffix)

    def _write(path: Path, data: list[dict]) -> None:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    _write(output, train_records)
    _write(val_path, val_records)
    print(f"Saved {len(train_records)} train → {output}")
    print(f"Saved {len(val_records)} val   → {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--output", default="finetune/data/llm_train.jsonl")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model to generate Q&A pairs (use a capable model for quality)")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

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

    print(f"\nGenerating Q&A pairs using {args.model} (this may take a while)...")
    records = generate_qa(chunks, args.model, args.max_tokens)
    print(f"\nGenerated {len(records)} Q&A pairs")

    if not records:
        print("No records generated. Check that Ollama is running: ollama serve")
        sys.exit(1)

    save_splits(records, Path(args.output), args.val_split)
    print("\nNext step: python finetune/finetune_llm.py")


if __name__ == "__main__":
    main()
