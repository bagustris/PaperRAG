# Architecture

## Pipeline Overview

PaperRAG processes academic PDFs through a six-stage pipeline:

```
PDF Files
  |
  v
[Parse] -- Docling extraction with adaptive OCR
  |
  v
[Chunk] -- Section-aware text splitting
  |
  v
[Embed] -- Sentence-transformer vectorization
  |
  v
[Store] -- FAISS index with metadata
  |
  v
[Retrieve] -- Similarity search + re-ranking
  |
  v
[Generate] -- LLM answer with citations
```

## Module Responsibilities

### `parser.py` -- PDF Parsing

- Uses Docling for structured document parsing
- **Adaptive OCR**: inspects each PDF with PyMuPDF to detect whether it contains extractable text. Text-based PDFs skip OCR entirely (2-3x speedup), while scanned PDFs enable OCR automatically.
- Supports CSV manifests for pre-supplied metadata (title, authors, abstract, DOI)
- Falls back to raw text extraction on parse failure

### `chunker.py` -- Text Chunking

- Section-aware splitting that respects document structure
- Configurable chunk size (default: 1000 chars) and overlap (default: 200 chars)
- Preserves metadata (file path, section title) on each chunk

### `embedder.py` -- Embedding

- Uses `sentence-transformers/all-MiniLM-L6-v2` by default
- Batch embedding with configurable batch size
- L2 normalization for cosine similarity
- Deterministic seeding for reproducibility

### `vectorstore.py` -- Vector Storage

- FAISS-based vector index
- Tracks file hashes (SHA-256) for incremental indexing
- Versioned index with config snapshot persistence
- Supports per-file removal for handling deleted/updated PDFs

### `retriever.py` -- Retrieval

- Top-k similarity search with configurable score threshold
- Optional Maximal Marginal Relevance (MMR) for result diversity
- Per-paper result limiting to avoid over-representation

### `llm.py` -- LLM Integration

- OpenAI-compatible API (works with Ollama and OpenAI)
- Streaming responses with citation support
- Descriptive error messages for common LLM failures

### `parallel.py` -- Parallel Indexing

- Multiprocessing-based parallel PDF processing
- Uses `spawn` start method (avoids deadlocks with PyTorch/CUDA)
- Per-PDF timeout for hanging documents
- Batch checkpointing for crash recovery

### `cli.py` -- CLI

- Typer-based command-line interface
- Commands: `index`, `query`, `evaluate`, plus the default REPL entrypoint
- Rich console output with progress bars (tqdm)

### `repl.py` -- Interactive REPL

- prompt-toolkit-based interactive session
- Command history persistence
- Live settings adjustment (top-k, threshold, temperature, model)

### `config.py` -- Configuration

- Pydantic v2 models with validation
- Config snapshot save/load for index reproducibility
- RAM-aware worker auto-detection

## Key Design Decisions

### Deterministic Hashing

Every PDF is identified by its SHA-256 hash. Re-running `paperrag index` only processes new or changed files, making incremental indexing fast.

### Adaptive OCR

Rather than a global OCR toggle, PaperRAG inspects each PDF individually using PyMuPDF. This gives optimal speed on mixed collections containing both text-based and scanned documents.

### Crash Recovery

The index is checkpointed after every batch during indexing. If the process crashes (e.g. OOM), restarting the same command resumes from where it left off.

### Spawn Multiprocessing

PaperRAG forces the `spawn` multiprocessing start method to avoid deadlocks caused by forking processes that use PyTorch, CUDA, or OpenMP.
