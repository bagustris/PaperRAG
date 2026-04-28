#!/usr/bin/env python3
"""
Fine-tune an LLM on your paper Q&A data using Unsloth + LoRA.

Reads alpaca-format JSONL from prepare_llm_data.py, fine-tunes efficiently
with LoRA, and optionally exports the result as a GGUF file for llama.cpp.

Requirements (GPU strongly recommended):
    pip install "unsloth[colab-new]" trl transformers datasets accelerate

Usage:
    python finetune/finetune_llm.py
    python finetune/finetune_llm.py --base-model unsloth/Qwen2.5-1.5B-Instruct
    python finetune/finetune_llm.py --export-gguf  # also save as GGUF for llama.cpp

After training, use the model in PaperRAG:
    paperrag query "..." --model finetune/models/llm-gguf/model-q4_k_m.gguf
or via REPL:
    /model finetune/models/llm-gguf/model-q4_k_m.gguf
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ALPACA_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """\
Below is an instruction that describes a task. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_alpaca(sample: dict) -> dict:
    if sample.get("input"):
        text = ALPACA_TEMPLATE.format(**sample) + "<|end_of_text|>"
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(**sample) + "<|end_of_text|>"
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="finetune/data/llm_train.jsonl")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct",
                        help="Unsloth-compatible base model (see https://huggingface.co/unsloth)")
    parser.add_argument("--output-dir", default="finetune/models/llm")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--export-gguf", action="store_true",
                        help="Export fine-tuned model as GGUF (requires llama.cpp)")
    parser.add_argument("--gguf-quant", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization method")
    args = parser.parse_args()

    train_path = Path(args.train_data)
    if not train_path.exists():
        print(f"Error: training data not found at {train_path}")
        print("Run: python finetune/prepare_llm_data.py")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        print("Error: unsloth not installed.")
        print('Run: pip install "unsloth[colab-new]" trl')
        sys.exit(1)

    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("Error: trl/transformers/datasets not installed.")
        print("Run: pip install trl transformers datasets accelerate")
        sys.exit(1)

    # Load model with Unsloth (4-bit quantized for memory efficiency)
    print(f"Loading {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,        # auto-detect float16 or bfloat16
        load_in_4bit=True,
    )

    # Wrap with LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Prepare dataset
    raw = load_jsonl(train_path)
    print(f"Loaded {len(raw)} training examples")
    formatted = [format_alpaca(s) for s in raw]
    dataset = Dataset.from_list(formatted)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=50,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(output_dir / "checkpoints"),
        ),
    )

    print(f"\nTraining for {args.epochs} epoch(s)...")
    trainer.train()

    # Save LoRA adapter
    lora_dir = output_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"\nLoRA adapter saved to {lora_dir}")

    if args.export_gguf:
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        print(f"\nExporting to GGUF ({args.gguf_quant})...")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=args.gguf_quant,
        )
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            gguf_path = gguf_files[0]
            print(f"GGUF saved to {gguf_path}")
            print(f"\nUse in PaperRAG:")
            print(f"  paperrag query \"...\" --model {gguf_path}")
            print(f"  # or in REPL: /model {gguf_path}")
    else:
        print(f"\nTo export as GGUF later, re-run with --export-gguf")
        print(f"Or load the LoRA adapter directly in Ollama with:")
        print(f"  ollama create my-paperrag-llm -f <Modelfile>")


if __name__ == "__main__":
    main()
