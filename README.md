# PaperRAG

Local-first Retrieval-Augmented Generation (RAG) system for querying academic PDF collections.

Default PDF directory: `/home/$USER/Documents/Mendeley Desktop/`

------------------------------------------------------------------------

## Features

- DOCLING-based structured PDF parsing
- Adaptive OCR (auto-detects per-PDF whether OCR is needed)
- Section-aware chunking
- Deterministic indexing with SHA256 hashing
- FAISS vector store
- Local (Ollama) or OpenAI-compatible LLM support
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
paperrag index

# 2. Start REPL with local LLM (best for multiple queries)
paperrag -m qwen3:1.7b

# OR use one-off query command (for single questions)
paperrag query "your question" -m qwen3:1.7b
```

**REPL vs Query Command:**

| Mode | Use When | Pros |
|------|----------|------|
| **REPL** `paperrag -m qwen3:1.7b` | Multiple questions, exploring papers | Fast (index loaded once), interactive, can adjust settings |
| **Query** `paperrag query "..." -m qwen3:1.7b` | Single question, scripting | Simple, good for automation |

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

    paperrag query "what is speech chain?" -m qwen3:1.7b

Query with OpenAI:

    export OPENAI_API_KEY='your-key-here'
    paperrag query "what is speech chain?"

Query with similarity threshold:

    paperrag query "what is speech chain?" -m qwen3:1.7b --threshold 0.3

Retrieval only (no LLM):

    paperrag query "what is speech chain?" --no-llm

**Note:** Each query reloads the index. For multiple questions, use REPL mode.

------------------------------------------------------------------------

### REPL Mode (Interactive)

**Best practice - start REPL with LLM configured:**

```bash
# With local Ollama (recommended for privacy & offline use)
paperrag -m qwen3:1.7b

# With OpenAI (requires OPENAI_API_KEY env var)
paperrag --model gpt-4 --api-base https://api.openai.com/v1

# With custom index directory
paperrag --index-dir /path/to/index -m qwen3:1.7b

# With similarity threshold
paperrag -m qwen3:1.7b --threshold 0.3

# Without LLM configured (will work but show tip to enable LLM)
paperrag
```

**CLI Options for REPL:**
- `--model, -m <name>` - Set LLM model (e.g., `qwen3:1.7b` for local, `gpt-4` for OpenAI)
- `--api-base, -a <url>` - Custom API endpoint (implies OpenAI mode)
- `--index-dir, -i <path>` - Custom index directory
- `--input-dir, -d <path>` - PDF directory
- `--threshold, -t <float>` - Similarity score threshold (0.0-1.0)
- `--temperature <float>` - LLM temperature (0.0-2.0)
- `--version` - Show version info

**Why REPL mode is recommended:**
- **Much faster** - Index loaded once, reused for all queries
- **Interactive** - Ask follow-up questions naturally
- **Flexible** - Change settings (model, top-k, threshold, temperature) on the fly
- **Efficient** - Perfect for exploratory research sessions

**Example session:**

    $ paperrag -m qwen3:1.7b

    PaperRAG version 0.1.0
    PDF directory: /home/bagus/Documents/Mendeley Desktop
    Found 2949 PDFs - 347 unindexed
    LLM: local / qwen3:1.7b
    Type help for commands.

    paperrag> what is speech chain?
    [retrieval results + answer with numbered citations]

    References:
      [1] Author et al. - 2020 - Paper Title.pdf
      [2] Author et al. - 2019 - Another Paper.pdf

    paperrag> topk 10
    top-k set to 10

    paperrag> threshold 0.3
    Threshold set to 0.30

    paperrag> exit
    Bye!

**REPL Commands:**

    <any text>           Query the indexed papers
    index                Re-index the PDF directory
    topk <n>             Set top-k for retrieval (default: 5)
    threshold <n>        Set similarity threshold (0.0-1.0)
    temperature <n>      Set LLM temperature (0.0-2.0)
    no-llm               Toggle LLM off/on (retrieval-only mode)
    model <name>         Set LLM model name
    config               Show current LLM configuration
    help                 Show help message
    exit / quit          Exit the REPL

------------------------------------------------------------------------

## LLM Configuration

**Option 1: Local LLM with Ollama (Recommended)**

Install Ollama from https://ollama.com, then pull a model:

    ollama pull qwen3:1.7b

Use with PaperRAG:

    paperrag -m qwen3:1.7b
    paperrag query "your question" -m qwen3:1.7b

**Option 2: OpenAI API**

Set your API key:

    export OPENAI_API_KEY='sk-...'

Use with PaperRAG (openai mode is default):

    paperrag
    paperrag query "your question"

Or specify model:

    paperrag --model gpt-4 --api-base https://api.openai.com/v1

**Option 3: No LLM (Retrieval Only)**

LLM is optional. Use `--no-llm` for retrieval-only mode:

    paperrag query "your question" --no-llm

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
    llm.py           # LLM integration (Ollama / OpenAI)
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
