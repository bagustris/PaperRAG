# PaperRAG

Local-first Retrieval-Augmented Generation (RAG) system for querying academic PDF collections.

Demo: https://asciinema.org/a/789531

------------------------------------------------------------------------

## Features

- Structured PDF parsing with adaptive OCR (auto-detects per file)
- Section-aware chunking and FAISS vector store
- Local LLM via Ollama — fully offline
- Interactive REPL for fast multi-query sessions
- Deterministic, reproducible indexing

------------------------------------------------------------------------

## Installation

### 1. Install Ollama

Download from https://ollama.com, then pull the recommended model:

    ollama pull qwen2.5:1.5b

### 2. Install Python 3.11 and uv

    uv venv --python 3.11
    source .venv/bin/activate

### 3. Install PaperRAG

CPU-only (default, recommended):

    uv pip install -e .

With GPU/CUDA support (NVIDIA GPU):

    uv pip install -e ".[gpu]"

------------------------------------------------------------------------

## Usage

### Index your PDFs (first time)

    paperrag index -d /path/to/pdfs

### Query (interactive REPL — recommended)

    paperrag -i /path/to/pdfs

### Query (single question)

    paperrag query "what is speech chain?" -i /path/to/pdfs

#### Example  

```bash 
$ paperrag index -d test_pdfs  
$ paperrag -i test_pdfs -m qwen2.5:1.5b  
paperag> What is speech chain?  

Sources (0.05s)
  [1] Rabiner, Schafer - 2007 - Introduction to digital speech processing.pdf (0.68)
  [2] Tjandra, Sakti, Nakamura - 2018 - Listening while speaking Speech chain by deep learning.pdf (0.66)

Answer:  
The speech chain, introduced by Denes et al. in  their classic 
introduction to speech science (Fig. 1), describes the basic 
mechanism involved in speech communication when a spoken message
travels from the speaker's mind to the listener's mind. It 
consists of a speech production process where words are
generated and speech sound waves are produced, transmitted 
through a medium like air, and then perceived by the 
listener's auditory system to understand what was said (Fig. 
1).

Retrieval: 0.05s | LLM: 8.65s | Total: 8.69s

```
------------------------------------------------------------------------

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index-dir, -i` | — | Index directory (required) |
| `--input-dir, -d` | — | PDF directory |
| `--model, -m` | `qwen2.5:1.5b` | Ollama model name |
| `--topk, -k` | `2` | Chunks to retrieve |
| `--threshold, -t` | `0.1` | Minimum similarity score |
| `--temperature` | `0.0` | LLM temperature (0=deterministic) |
| `--max-tokens` | `128` | Max output tokens |

------------------------------------------------------------------------

## Persistent Configuration

Create `~/.paperragrc` (global) or `.paperragrc` (per-project) in TOML format to avoid repeating CLI flags:

```toml
index-dir = "/path/to/my-index"
model = "qwen2.5:1.5b"
topk = 2
```

------------------------------------------------------------------------

## Documentation

Full usage guide, troubleshooting, and project structure: [paperrag.readthedocs.org](https://paperrag.readthedocs.org)

------------------------------------------------------------------------

## License

MIT

## Note
- This project was born as answer to [my own question](https://academia.stackexchange.com/questions/203900/local-rag-like-chatgpt-for-abundant-of-pdfs/225980#225980). 
- On my laptop (Dell XPS 9320, i7-1260P 16GB RAM), it takes about **20 hours** to index ~2000 PDFs with default configuration.

## Citation  
TBA 
