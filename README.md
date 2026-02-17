# PaperRAG

Local-first Retrieval-Augmented Generation (RAG) system for querying academic PDF collections.

Demo:  https://asciinema.org/a/788900  

------------------------------------------------------------------------

## Features

- DOCLING-based structured PDF parsing
- Adaptive OCR (auto-detects per-PDF whether OCR is needed)
- Section-aware chunking
- Deterministic indexing with SHA256 hashing
- FAISS vector store
- Local LLM via Ollama
- Interactive REPL with command history
- Fully offline-capable
- Reproducible experiments
- CLI interface (Typer)

------------------------------------------------------------------------

## Installation

### 1. Install Python 3.11

### 2. Create environment

    uv venv --python 3.11
    source .venv/bin/activate

### 3. Install dependencies

    uv pip install -e ".[llm,test]"

------------------------------------------------------------------------

## Usage

### Quick Start

```bash
# 1. Index your PDFs (one time)
paperrag index -i test_pdfs

# 2. Start REPL with local LLM (best for multiple queries)
paperrag -i test_pdfs -m qwen2.5:1.5b  

# OR use one-off query command (for single questions)
paperrag query "your question" -i test_pdfs
```

**REPL vs Query Command:**

| Mode | Use When | Pros |
|------|----------|------|
| **REPL** `paperrag -i test_pdfs` | Multiple questions, exploring papers | Fast (index loaded once), interactive, can adjust settings |
| **Query** `paperrag query "..." -i test_pdfs` | Single question, scripting | Simple, good for automation |

------------------------------------------------------------------------

### Index PDFs

Standard indexing (with adaptive OCR):

    paperrag index

Custom directory:

    paperrag index --input-dir /path/to/pdfs

**Adaptive OCR (Intelligent Speed Optimization):**

PaperRAG automatically detects if each PDF needs OCR:

    # Default mode - automatic detection (recommended!)
    paperrag index

    # Behavior:
    # Text-based PDFs -> OCR skipped (2-3x faster)
    # Scanned PDFs -> OCR enabled (accurate extraction)
    # Mixed collection -> Optimized per file!

**Manual OCR Control:**

    # Force disable OCR for all PDFs (fastest, text-only)
    paperrag index --ocr never

    # Force enable OCR for all PDFs (slowest, scanned documents)
    paperrag index --ocr always

**Speed Optimization:**

    # Combine adaptive OCR with more workers (4-5x faster!)
    paperrag index --workers 10

    # Conservative for low RAM
    paperrag index --ocr never --workers 4

**Other index options:**

    # Force full re-index (ignore previous index)
    paperrag index --force

    # Custom checkpoint interval
    paperrag index --checkpoint-interval 20

    # Use a CSV manifest with paper metadata
    paperrag index --manifest papers.csv

------------------------------------------------------------------------

### Query Command (One-off queries)

**Use this for:** Single questions, scripting, or batch processing

Query with local LLM (Ollama):

    paperrag query "what is speech chain?" -i test_pdfs

Query with custom top-k:

    paperrag query "what is speech chain?" -i test_pdfs --topk 5

**Note:** Each query reloads the index. For multiple questions, use REPL mode.

------------------------------------------------------------------------

### REPL Mode (Interactive)

**Best practice - start REPL with LLM configured:**

```bash
# With index directory
paperrag --index-dir /path/to/index

# With custom model
paperrag -i /path/to/index --model qwen3:1.7b

# With custom top-k
paperrag -i /path/to/index --topk 5
```

**CLI Options for REPL:**
- `--model, -m <name>` - Set LLM model (default: `qwen2.5:1.5b`)
- `--index-dir, -i <path>` - Custom index directory (required)
- `--input-dir, -d <path>` - PDF directory
- `--topk, -k <int>` - Number of chunks to retrieve (default: 1)
- `--threshold, -t <float>` - Similarity score threshold (default: 0.1)
- `--temperature <float>` - LLM temperature (default: 0.0)
- `--max-tokens <int>` - Max output tokens (default: 512)
- `--version` - Show version info

**Why REPL mode is recommended:**
- **Much faster** - Index loaded once, reused for all queries
- **Interactive** - Ask follow-up questions naturally
- **Flexible** - Change settings (model, top-k, threshold, temperature) on the fly
- **Efficient** - Perfect for exploratory research sessions

**Example session:**

    $ paperrag -i test_pdfs

    PaperRAG version 0.2.0
    Found 10 PDFs - all indexed
    LLM: qwen2.5:1.5b
    Top-k: 1 (retrieve 1 chunks)
    Threshold: 0.1 (minimum similarity score)
    Temperature: 0.0 (0.0=deterministic, higher=creative)
    Max tokens: 512 (max output length)
    Type help for commands.

    paperrag> What is a Python package to predict speaker characteristics called?

    Answer:
    Nkululeko is a tool for rapid speaker characteristics detection.

    References:
      [1] example_paper.pdf
    Retrieval: 0.02s | LLM: 2.15s | Total: 2.17s

    paperrag> exit
    Bye!

**REPL Commands:**

    <any text>           Query the indexed papers
    index                Re-index the PDF directory
    topk <n>             Set top-k for retrieval (default: 1)
    threshold <n>        Set similarity threshold (default: 0.1)
    temperature <n>      Set LLM temperature (default: 0.0)
    max-tokens <n>       Set max output tokens (default: 512)
    model <name>         Set LLM model name (default: qwen2.5:1.5b)
    config               Show current configuration
    help                 Show help message
    exit / quit          Exit the REPL

------------------------------------------------------------------------

## LLM Configuration

PaperRAG uses a local LLM via [Ollama](https://ollama.com).

**Default model:** `qwen2.5:1.5b` (lightweight, fast inference)

Install Ollama, then pull the default model:

    ollama pull qwen2.5:1.5b

Use with PaperRAG (uses default model):

    paperrag -i test_pdfs

Use with a different model:

    paperrag -i test_pdfs --model llama2:7b
    paperrag query "your question" -i test_pdfs --model neural-chat

------------------------------------------------------------------------

## Project Structure

```
paperrag/
    __init__.py
    cli.py           # Typer CLI entrypoint
    repl.py          # Interactive REPL
    config.py        # Pydantic configuration models
    parser.py        # DOCLING-based PDF parsing + adaptive OCR
    chunker.py       # Section-aware text chunking
    embedder.py      # Sentence-transformer embeddings
    vectorstore.py   # FAISS vector store
    retriever.py     # Retrieval + re-ranking
    llm.py           # LLM integration (Ollama)
    parallel.py      # Parallel indexing with worker management
```

------------------------------------------------------------------------

## Troubleshooting

### Memory Issues During Indexing

**Symptoms:**
- Terminal disappears/crashes during indexing
- System becomes unresponsive
- "Killed" message appears
- Process stops after indexing only a few files

**Root Cause:** Out-Of-Memory (OOM) killer terminating the process due to excessive RAM usage.

**RAM Requirements:**
- Base system: ~2GB
- Per worker: ~2GB during peak Docling usage
- Example: 8 workers = 2 + (8 x 2) = **18GB minimum**

**Solutions:**

1. **Reduce worker count** (recommended):
   ```bash
   # For 16GB RAM, use 4-6 workers
   paperrag index --workers 4

   # For 8GB RAM, use 2-3 workers
   paperrag index --workers 2

   # For testing/debugging, use 1 worker
   paperrag index --workers 1
   ```

2. **Check if OOM killer was triggered**:
   ```bash
   dmesg | grep -i "killed process"
   ```

3. **Monitor memory during indexing**:
   ```bash
   watch -n 2 free -h
   ```

**Recommended Worker Counts by RAM:**
- **8GB RAM**: `--workers 2`
- **16GB RAM**: `--workers 4-6` (auto-detected: ~6-7)
- **32GB RAM**: `--workers 8-12` (auto-detected: ~10-15)
- **64GB+ RAM**: Auto-detection works well

**How Auto-Detection Works:**
- PaperRAG automatically calculates safe worker count based on available RAM
- Formula: `min(cpu_cores - 1, available_ram_gb // 2)`
- Manual `--workers` override always available if needed

**Progress Tracking:**
- Index is checkpointed periodically
- If process crashes, restart with same command to resume
- Previously indexed files are skipped automatically

### Hanging/Slow PDFs

**Symptoms:**
- Indexing appears frozen on specific PDFs
- One batch takes extremely long

**Solution:**
- Timeout is set to 300s (5 minutes) per PDF by default
- Problematic PDFs will be skipped automatically
- Check `failed_pdfs.log` in index directory for details

### Finding Failed PDFs

After indexing completes, check for failures:
```bash
cat ~/Documents/Mendeley\ Desktop/.paperrag-index/failed_pdfs.log
```

This log shows which PDFs failed and why (parse errors, timeouts, corrupted files).

------------------------------------------------------------------------

## Reproducibility

- Embedding model version stored in index
- File hash tracking
- Deterministic chunk ordering
- Temperature = 0.0 by default
- Seed-based embedding reproducibility

------------------------------------------------------------------------

## License

MIT
