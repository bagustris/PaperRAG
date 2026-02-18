# PaperRAG

**Local-first Retrieval-Augmented Generation for academic PDF collections.**

PaperRAG lets you index a directory of academic PDFs and query them using natural language, powered by a local LLM via Ollama. It runs entirely offline.

## Key Features

- **Structured PDF parsing** via Docling with adaptive OCR
- **Section-aware chunking** that respects document structure
- **FAISS vector store** with deterministic SHA-256 hashing
- **Local LLM** via Ollama
- **Interactive REPL** with command history and live settings
- **Parallel indexing** with RAM-aware worker auto-detection
- **Fully offline-capable** and reproducible

## Quick Example

```bash
# Index your PDFs
paperrag index --input-dir ~/papers --index-dir ~/papers

# Start interactive session
paperrag --index-dir ~/papers -m qwen3:1.7b
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
