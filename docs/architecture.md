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

- Supports two local backends:
  - Ollama for standard model names such as `qwen2.5:1.5b`
  - `llama.cpp` via `llama-server` for local `.gguf` files and HuggingFace GGUF repos
- Streaming responses with citation support
- Descriptive error messages for common LLM failures

### `parallel.py` -- Parallel Indexing

- Multiprocessing-based parallel PDF processing
- Uses `spawn` start method (avoids deadlocks with PyTorch/CUDA)
- Per-PDF timeout for hanging documents
- Batch checkpointing for crash recovery

### `cli.py` -- CLI

- Typer-based command-line interface
- Commands: `index`, `review`, `query`, `evaluate`, plus the default REPL entrypoint
- Rich console output with progress bars (tqdm)

### `repl.py` -- Interactive REPL

- prompt-toolkit-based interactive session
- Command history persistence
- Live settings adjustment via slash commands:
  - `/index`, `/topk`, `/threshold`, `/temperature`
  - `/max-tokens`, `/ctx-size`, `/prompt`, `/model`
  - `/config`, `/rc`, `/help`, `/exit`, `/quit`

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

## Design Principles

These principles guide decisions about UX, API shape, and feature scope.

### Progressive Disclosure

Default behaviour is minimal; complexity surfaces only when the user needs it. Running `paperrag review paper.pdf` requires zero flags. The `/focus`, `/topk`, and `/config` commands exist for users who want more control, but they are never required. New features should follow the same pattern: sensible default first, opt-in complexity second.

### Convention over Configuration

The index location auto-derives from the input path (`<input-dir>/.paperrag-index`). Workers auto-detect from available RAM. The LLM backend is inferred from the model name format (Ollama name vs. `.gguf` path vs. HuggingFace repo ID). `.paperragrc` files are optional overrides, not required setup. A new user should be able to run a useful command without reading the configuration docs.

### Incremental by Default

Re-running any command is always safe and cheap. SHA-256 hashes mean unchanged PDFs are skipped automatically during indexing — no `--skip-cached` flag needed. This principle extends to index saves (atomic writes via `.tmp` + move) and REPL state (re-indexing only resets what changed).

### Ownership of Output

Each layer is responsible for its own console output. `_handle_index` prints its own progress; the `review` command does not pre-announce what `_handle_index` is about to say. Functions that call other functions do not narrate on their behalf. This keeps output coherent and prevents duplicate or contradictory messages as the codebase grows.

### Local-First, No Cloud Dependencies

Everything runs on-device: FAISS (embedded library, no server), Ollama or llama-server (local inference), sentence-transformers (downloaded once, cached). The design treats offline use as the default, not an edge case. Features that require external services should be clearly opt-in.

### Atomic Persistence

The vector store writes to a `.tmp` file and then uses `shutil.move()` to replace the live index. A crash mid-save leaves the previous index intact. The same principle applies to batch checkpointing during indexing: partial progress is always recoverable.
