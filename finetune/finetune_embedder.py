#!/usr/bin/env python3
"""
Fine-tune a SentenceTransformer embedding model on your paper collection.

Reads JSONL produced by prepare_embedder_data.py and trains with
MultipleNegativesRankingLoss (state-of-the-art for dense retrieval).

After training, update your PaperRAG config to use the new model:
    paperrag index --input-dir <dir> --embed-model ./finetune/models/embedder
or in ~/.paperragrc:
    embed-model = "/absolute/path/to/finetune/models/embedder"

Usage:
    python finetune/finetune_embedder.py
    python finetune/finetune_embedder.py --base-model sentence-transformers/all-mpnet-base-v2
    python finetune/finetune_embedder.py --epochs 3 --batch-size 32
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="finetune/data/embedder_train.jsonl")
    parser.add_argument("--val-data", default="finetune/data/embedder_val.jsonl")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Starting checkpoint (HuggingFace model ID or local path)")
    parser.add_argument("--output-dir", default="finetune/models/embedder")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    args = parser.parse_args()

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from sentence_transformers.evaluation import TripletEvaluator
        from torch.utils.data import DataLoader
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)

    if not train_path.exists():
        print(f"Error: training data not found at {train_path}")
        print("Run: python finetune/prepare_embedder_data.py")
        sys.exit(1)

    train_data = load_jsonl(train_path)
    print(f"Train: {len(train_data)} pairs from {train_path}")

    # Build InputExamples — triplets (anchor, positive, negative)
    train_examples = [
        InputExample(texts=[d["query"], d["pos"], d["neg"]])
        for d in train_data
        if "query" in d and "pos" in d and "neg" in d
    ]
    print(f"Built {len(train_examples)} training triplets")

    model = SentenceTransformer(args.base_model)
    print(f"Base model: {args.base_model}  (dim={model.get_sentence_embedding_dimension()})")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = None
    if val_path.exists():
        val_data = load_jsonl(val_path)
        evaluator = TripletEvaluator(
            anchors=[d["query"] for d in val_data],
            positives=[d["pos"] for d in val_data],
            negatives=[d["neg"] for d in val_data],
            name="val",
        )
        print(f"Val: {len(val_data)} triplets from {val_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning for {args.epochs} epoch(s)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.eval_steps if evaluator else 0,
        output_path=str(output_dir),
        save_best_model=evaluator is not None,
        show_progress_bar=True,
    )

    if evaluator is None:
        model.save(str(output_dir))

    print(f"\nModel saved to {output_dir}")
    print("\nNext steps:")
    print(f"  1. Re-index your papers with the new model:")
    print(f"     paperrag index --input-dir <your-pdf-dir> --embed-model {output_dir}")
    print(f"  2. Or set it permanently in ~/.paperragrc:")
    print(f"     embed-model = \"{output_dir.resolve()}\"")


if __name__ == "__main__":
    main()
