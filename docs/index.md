# PaperRAG

**Local-first Retrieval-Augmented Generation for academic PDF collections.**

 PaperRAG lets you index academic PDFs and query them using natural language, powered by local LLM backends. It runs entirely offline once models are available locally.

## Key Features

- **Structured PDF parsing** via Docling with adaptive OCR
- **Section-aware chunking** that respects document structure
- **FAISS vector store** with deterministic SHA-256 hashing
- **Local LLM backends** via Ollama or `llama.cpp`
- **Interactive REPL** with command history and live settings
- **Focused `review` command** for index-and-open workflows
- **Parallel indexing** with RAM-aware worker auto-detection
- **Fully offline-capable** and reproducible

## Quick Example

```bash
# Index your PDFs
paperrag index --input-dir ~/papers

# Start an interactive session from the generated index
paperrag --index-dir ~/papers/.paperrag-index -m qwen2.5:1.5b

# Or review one paper directly
paperrag review ~/papers/paper.pdf
```

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
cli
configuration
usage  
architecture
api
```
