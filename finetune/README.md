# PaperRAG Fine-Tuning

Fine-tune either (or both) of PaperRAG's two model components to improve performance on your specific paper collection:

| Component | Default model | Fine-tune for… |
|-----------|--------------|----------------|
| **Embedding model** | `all-MiniLM-L6-v2` | Better retrieval relevance for your domain |
| **LLM** | `qwen2.5:1.5b` (Ollama) | Better answer generation for academic Q&A |

**Which one matters more?**
Embedding model fine-tuning typically gives a larger gain because it directly controls *which chunks* are retrieved. The LLM can only answer well if the right context is retrieved first.

---

## Directory Layout

```
finetune/
├── prepare_embedder_data.py   # Extract training pairs from your PaperRAG index
├── finetune_embedder.py       # Fine-tune the SentenceTransformer embedding model
├── prepare_llm_data.py        # Generate Q&A pairs from your papers (uses Ollama)
├── finetune_llm.py            # Fine-tune an LLM with Unsloth + LoRA → GGUF
├── requirements.txt           # Extra dependencies for fine-tuning
└── data/                      # Generated training data (created by scripts)
```

---

## Prerequisites

### 1. A working PaperRAG index

All scripts read from an existing index. If you have not indexed your PDFs yet:

```bash
paperrag index --input-dir ~/Documents/Mendeley\ Desktop
```

### 2. Install fine-tuning dependencies

```bash
pip install -r finetune/requirements.txt
```

For LLM fine-tuning, install Unsloth according to your CUDA version:

```bash
# CUDA 12.1+
pip install "unsloth[cu121-torch240]"

# CUDA 11.8
pip install "unsloth[cu118-torch240]"

# CPU only / Google Colab
pip install "unsloth[colab-new]"

# Then install the rest
pip install trl transformers datasets accelerate bitsandbytes
```

> **Note:** LLM fine-tuning (`finetune_llm.py`) requires a CUDA-capable GPU.
> Embedding fine-tuning (`finetune_embedder.py`) runs on CPU but is faster with a GPU.

---

## Embedding Model Fine-Tuning

Fine-tuning the embedding model teaches it to rank chunks from *your* papers as more relevant than generic web text. This directly improves retrieval quality.

### Step 1 — Prepare training data

```bash
python finetune/prepare_embedder_data.py
```

This loads all chunks from your index and produces `(query, positive_chunk, hard_negative_chunk)` triplets.

**Options:**

```bash
# Use LLM to generate synthetic questions per chunk (slower, higher quality)
python finetune/prepare_embedder_data.py --mode llm --model qwen2.5:1.5b

# Limit chunks processed (useful for a quick test run)
python finetune/prepare_embedder_data.py --max-chunks 2000

# Custom index or output location
python finetune/prepare_embedder_data.py \
  --index-dir ~/papers/.paperrag-index \
  --output finetune/data/embedder_train.jsonl
```

**Query generation modes:**

| Mode | How queries are built | Speed | Quality |
|------|-----------------------|-------|---------|
| `heuristic` (default) | `"{paper_title}: {section_name}"` | Fast, no LLM | Good baseline |
| `llm` | Ollama generates a specific research question per chunk | Slower | Higher quality |

**Hard negatives** are added automatically via BM25 (install `rank-bm25` for this; falls back to random sampling otherwise).

Output files:
- `finetune/data/embedder_train.jsonl`
- `finetune/data/embedder_val.jsonl`

Each line: `{"query": "...", "pos": "...", "neg": "..."}`

### Step 2 — Fine-tune

```bash
python finetune/finetune_embedder.py
```

Uses `TripletLoss` with the generated triplets. Evaluates on the val split after each epoch.

**Options:**

```bash
# Start from a larger base model for higher accuracy
python finetune/finetune_embedder.py \
  --base-model sentence-transformers/all-mpnet-base-v2

# Adjust training length
python finetune/finetune_embedder.py --epochs 3 --batch-size 64

# Custom output location
python finetune/finetune_embedder.py --output-dir finetune/models/my-embedder
```

Output: `finetune/models/embedder/` (a standard SentenceTransformers model directory)

### Step 3 — Re-index and use

```bash
# Re-index your PDFs with the fine-tuned model
paperrag index --input-dir ~/Documents/Mendeley\ Desktop \
  --embed-model finetune/models/embedder

# Or set permanently in ~/.paperragrc
echo 'embed-model = "/absolute/path/to/finetune/models/embedder"' >> ~/.paperragrc
paperrag index --input-dir ~/Documents/Mendeley\ Desktop
```

> **Important:** Changing the embedding model requires a full re-index.
> The old index is incompatible because vector dimensions may differ.

---

## LLM Fine-Tuning

Fine-tuning the LLM teaches it to answer questions in an academic style, cite sources, and handle domain-specific terminology from your papers.

> The LLM is already fully swappable in PaperRAG via `--model`, `.paperragrc`, or
> the `/model` REPL command. Fine-tuning is optional — you can also use any
> off-the-shelf model from Ollama or HuggingFace.

### Step 1 — Generate Q&A training data

```bash
python finetune/prepare_llm_data.py
```

For each paper chunk, asks an Ollama model to produce a question and a reference answer. Uses the same index as the rest of PaperRAG.

**Options:**

```bash
# Use a more capable model for better training data quality
python finetune/prepare_llm_data.py --model qwen2.5:7b

# Limit chunks (faster test run)
python finetune/prepare_llm_data.py --max-chunks 3000

# Custom index location
python finetune/prepare_llm_data.py --index-dir ~/papers/.paperrag-index
```

> Ollama must be running: `ollama serve`

Output files:
- `finetune/data/llm_train.jsonl`
- `finetune/data/llm_train_val.jsonl`

Each line (alpaca format):
```json
{
  "instruction": "Answer the question based on the provided context.",
  "input": "Context: ...\n\nQuestion: ...",
  "output": "..."
}
```

### Step 2 — Fine-tune with Unsloth + LoRA

```bash
python finetune/finetune_llm.py --export-gguf
```

Uses [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient 4-bit LoRA fine-tuning. With `--export-gguf`, the result is converted to a GGUF file ready for llama.cpp (PaperRAG's second LLM backend).

**Options:**

```bash
# Choose a different base model (must be Unsloth-compatible)
python finetune/finetune_llm.py \
  --base-model unsloth/Llama-3.2-1B-Instruct \
  --export-gguf

# Adjust LoRA rank and training length
python finetune/finetune_llm.py \
  --lora-r 32 \
  --epochs 3 \
  --export-gguf

# Choose GGUF quantization (q4_k_m is the best size/quality tradeoff)
python finetune/finetune_llm.py \
  --export-gguf \
  --gguf-quant q5_k_m
```

Recommended base models (Unsloth hub):

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Qwen2.5-1.5B-Instruct` | ~4 GB | Fast, good for academic Q&A |
| `unsloth/Qwen2.5-7B-Instruct` | ~8 GB | Higher quality |
| `unsloth/Llama-3.2-3B-Instruct` | ~6 GB | Strong general reasoning |
| `unsloth/mistral-7b-instruct-v0.3` | ~8 GB | Good instruction following |

Output:
- `finetune/models/llm/lora/` — LoRA adapter (merge with base model for full weights)
- `finetune/models/llm/gguf/model-q4_k_m.gguf` — quantized GGUF (with `--export-gguf`)

### Step 3 — Use in PaperRAG

```bash
# One-shot query
paperrag query "What is the main contribution?" \
  --model finetune/models/llm/gguf/model-q4_k_m.gguf

# Interactive REPL (switch model at runtime)
paperrag
/model finetune/models/llm/gguf/model-q4_k_m.gguf

# Persistent setting in ~/.paperragrc
model = "/absolute/path/to/finetune/models/llm/gguf/model-q4_k_m.gguf"
n-gpu-layers = 35   # offload layers to GPU for faster inference
```

---

## Full Workflow (Both Models)

```bash
# 0. Index your papers (if not done yet)
paperrag index --input-dir ~/Documents/Mendeley\ Desktop

# --- Embedding model ---
# 1a. Prepare embedding training data
python finetune/prepare_embedder_data.py --mode llm --model qwen2.5:1.5b

# 2a. Fine-tune
python finetune/finetune_embedder.py --epochs 3

# 3a. Re-index with the new embedding model
paperrag index --input-dir ~/Documents/Mendeley\ Desktop \
  --embed-model finetune/models/embedder

# --- LLM ---
# 1b. Generate Q&A pairs
python finetune/prepare_llm_data.py --model qwen2.5:7b

# 2b. Fine-tune and export GGUF
python finetune/finetune_llm.py --export-gguf

# 3b. Use the fine-tuned LLM
paperrag query "Summarize the attention mechanism" \
  --model finetune/models/llm/gguf/model-q4_k_m.gguf
```

---

## Troubleshooting

**`VectorStore not found` / `no index at ...`**
Run `paperrag index --input-dir <your-pdf-dir>` first, or pass `--index-dir` explicitly.

**`ollama: connection refused`**
Start Ollama with `ollama serve` before running `prepare_embedder_data.py --mode llm` or `prepare_llm_data.py`.

**`rank-bm25 not installed` warning during data prep**
Install it for better hard negatives: `pip install rank-bm25`. Without it, random negatives are used (acceptable but less effective).

**`unsloth not installed`**
Follow the Unsloth installation instructions for your CUDA version above. CPU-only machines cannot run `finetune_llm.py`.

**Out-of-memory during LLM fine-tuning**
- Reduce `--batch-size` (try 2 or 1)
- Increase `--grad-accum` proportionally (e.g., `--batch-size 1 --grad-accum 16`)
- Use a smaller base model (e.g., `unsloth/Qwen2.5-1.5B-Instruct`)
- Reduce `--lora-r` to 8

**Dimension mismatch error after re-indexing**
This happens if you query an old index with a new embedding model. Always re-run `paperrag index` after changing `--embed-model`.
