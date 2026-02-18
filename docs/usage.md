# PaperRAG Usage Guide

## Indexing PDFs

Standard indexing with adaptive OCR (recommended):

    paperrag index -d /path/to/pdfs

OCR options:

    # Auto-detect per PDF (default — text PDFs skip OCR, scanned PDFs use OCR)
    paperrag index -d /path/to/pdfs --ocr auto

    # Disable OCR for all (fastest, text-based PDFs only)
    paperrag index -d /path/to/pdfs --ocr never

    # Force OCR for all (slowest, for scanned documents)
    paperrag index -d /path/to/pdfs --ocr always

Speed and memory options:

    # More workers = faster indexing (each worker needs ~2GB RAM)
    paperrag index -d /path/to/pdfs --workers 4

    # Force full re-index
    paperrag index -d /path/to/pdfs --force

    # Use a CSV manifest with paper metadata (filename,title,authors,abstract,doi)
    paperrag index -d /path/to/pdfs --manifest papers.csv

------------------------------------------------------------------------

## REPL Mode (Interactive)

Best for multiple queries — index is loaded once and reused.

    paperrag -i /path/to/pdfs

REPL commands:

| Command | Description |
| --- | --- |
| `<any text>` | Query the indexed papers |
| `topk <n>` | Set number of retrieved chunks |
| `threshold <n>` | Set minimum similarity score (0.0–1.0) |
| `temperature <n>` | Set LLM temperature (0.0–2.0) |
| `max-tokens <n>` | Set max output tokens |
| `model <name>` | Switch Ollama model |
| `config` | Show current settings |
| `rc` | Show loaded .paperragrc files |
| `index` | Re-index the PDF directory |
| `help` | Show all commands |
| `exit` / `quit` | Exit |

------------------------------------------------------------------------

## Query Command (Single Questions)

    paperrag query "what is speech chain?" -i /path/to/pdfs
    paperrag query "what is speech chain?" -i /path/to/pdfs --topk 5

Note: each invocation reloads the index. Use REPL for multiple queries.

------------------------------------------------------------------------

## Persistent Configuration (.paperragrc)

Create `~/.paperragrc` (global) or `.paperragrc` (per-project) in TOML format:

```toml
index-dir = "/path/to/my-index"
input-dir = "/path/to/pdfs"
model = "qwen2.5:1.5b"
topk = 2
max-tokens = 128
temperature = 0.0
threshold = 0.1
```

Priority order: CLI args > local `.paperragrc` > global `~/.paperragrc` > defaults.

------------------------------------------------------------------------

## Troubleshooting

### Out of Memory During Indexing

Each Docling worker uses ~2GB RAM at peak. Auto-detection uses:
`min(cpu_cores - 1, available_ram_gb // 2)`

Recommended worker counts:
- **8GB RAM**: `--workers 2`
- **16GB RAM**: `--workers 4`
- **32GB RAM**: `--workers 8`

Check if OOM killer was triggered:

    dmesg | grep -i "killed process"

### Failed PDFs

After indexing, check the failure log:

    cat /path/to/index/.paperrag-index/failed_pdfs.log

Shows which PDFs failed and why (parse errors, timeouts, corrupted files).
The index checkpoints periodically, so restart with the same command to resume.

------------------------------------------------------------------------

## Project Structure

```
paperrag/
    cli.py           # Typer CLI entrypoint
    repl.py          # Interactive REPL
    config.py        # Pydantic configuration models
    parser.py        # Docling-based PDF parsing + adaptive OCR
    chunker.py       # Section-aware text chunking
    embedder.py      # Sentence-transformer embeddings
    vectorstore.py   # FAISS vector store
    retriever.py     # Retrieval + re-ranking
    llm.py           # LLM integration (Ollama)
    parallel.py      # Parallel indexing with worker management
```

------------------------------------------------------------------------

## Reproducibility

- Embedding model version stored in index
- File hash tracking (SHA256) — unchanged files are skipped
- Deterministic chunk ordering
- Temperature = 0.0 by default
- Seed-based embedding reproducibility
