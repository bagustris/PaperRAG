# PaperRAG

Local-first Retrieval-Augmented Generation (RAG) system for querying academic PDF collections.

Demo: https://asciinema.org/a/789531  
Deepwiki:  https://deepwiki.com/bagustris/PaperRAG

------------------------------------------------------------------------

## Features

- Structured PDF parsing with adaptive OCR (auto-detects per file)
- Section-aware chunking and FAISS vector store
- Local LLM via Ollama — fully offline
- Interactive REPL with prompt presets, session export, and config inspection
- Single-PDF review mode with autofocus on the active paper
- Deterministic, reproducible indexing with auto-discovered `.paperrag-index`
- Default input directory at `~/Documents/Mendeley Desktop/` for local collections

------------------------------------------------------------------------

## Installation

### 1. Install a Local LLM Backend

For Ollama models, download from https://ollama.com, then pull the default recommended model:

    ollama pull qwen2.5:1.5b

For GGUF models, install `llama-server` from `llama.cpp`, then point `paperrag` at a local `.gguf` file path:

    brew install llama-cpp

### 2. Install Python 3.11 and uv

    uv venv --python 3.11
    source .venv/bin/activate

### 3. Install PaperRAG

CPU-only (default, recommended):

    uv sync

CPU-only editable install:

    uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e .

GPU/CUDA support for pure uv:

    uv sync
    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124

pip CPU editable install:

    pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e .

uv pip CPU editable install:

    uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e .

pip GPU editable install (CUDA 12.4):

    pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple -e .

uv pip GPU editable install (CUDA 12.4):

    uv pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple -e .

Note: `pip install -e ".[gpu]"` cannot by itself switch PyTorch from CPU to CUDA
reliably, because PyTorch CPU and CUDA wheels come from different indexes, and
`pip` does not read `tool.uv.sources`. The index has to be selected on the
command line for `pip` and `uv pip`. Pure `uv sync` remains CPU-only by design.

------------------------------------------------------------------------

## Suggested Models

| Device (CPU/GPU) | Backend | Models | Size |
|--------|---------|-------------|------|
| CPU | Ollama | `qwen2.5:1.5b` (CPU recommended) | 986 MB |
| CPU | Ollama | `qwen3:1.7b` | 1.4 GB |
| GPU | Ollama | `llama3.2:3b` | 2 GB |
| CPU/GPU | Ollama | `phi4-mini-reasoning:latest` | 3.2 GB |
| CPU/GPU | Ollama | `gemma3:1b` | 815 MB |
| GPU | Ollama | `gemma3:12b` | 8.1 GB |
| GPU | Ollama | `phi4-mini:latest` | 2.5 GB |
| CPU/GPU | llama.cpp) | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | 731 MB |
| CPU/GPU | llama.cpp) | `LFM2-1.2B-RAG-Q4_K_M.gguf` | 731 MB |
| CPU/GPU | llama.cpp) | `hf.co/Qwen/Qwen3-8B-GGUF:Q8_0` (GPU recommended)| 8.7 GB |

For other Hugging Face GGUF models, prefer quantizations such as `Q4_K_M`, `Q8_0`, or similar variants available in the same repo. Example:

    Qwen/Qwen3-1.7B-GGUF:Q8_0

Use Ollama model names directly:

    paperrag query "what is speech chain?" --index-dir /path/to/.paperrag-index -m qwen2.5:1.5b

Use a local GGUF file through `llama.cpp`:

    paperrag query "what is speech chain?" --index-dir /path/to/.paperrag-index -m /path/to/LFM2.5-1.2B-Instruct-Q4_K_M.gguf

------------------------------------------------------------------------

## Usage

### Index a PDF directory or single PDF

    paperrag index -d /path/to/pdfs
    paperrag index -d /path/to/paper.pdf

### Open the interactive REPL

If you already have an index in the current project, you can run:

    paperrag

Or point to the index directory explicitly:

    paperrag -i /path/to/.paperrag-index

### Query a single question

    paperrag query "what is speech chain?" -i /path/to/.paperrag-index

### Review a single paper

    paperrag review /path/to/paper.pdf --preset reviewer --output review.md

### Export query results

    paperrag export -q "what is speech chain?" -i /path/to/.paperrag-index -o results.md
    paperrag export -q "what is speech chain?" -i /path/to/.paperrag-index -o results.csv --format csv

#### Example  

```bash 
$ paperrag index -d test_pdfs
$ paperrag -i test_pdfs/.paperrag-index -m qwen2.5:1.5b
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
| `--index-dir, -i` | — | Index directory (required for query/export; auto-discovered for REPL when possible) |
| `--input-dir, -d` | `~/Documents/Mendeley Desktop/` | PDF directory or single PDF file |
| `--model, -m` | `qwen2.5:1.5b` | Ollama model name |
| `--top-k, --topk, -k` | `3` | Chunks to retrieve |
| `--threshold, -t` | `0.1` | Minimum similarity score |
| `--temperature` | `0.0` | LLM temperature (0=deterministic) |
| `--ctx-size` | `2048` | LLM context window size |
| `--max-tokens` | `256` | Max output tokens |

The REPL also accepts `/preset`, `/prompt`, `/ctx-size`, `/export`, `/config`, and `/rc` for interactive tuning and session export.

------------------------------------------------------------------------

## Benchmarking

Compare how different small LLMs perform on RAG questions:

    python -m paperrag.llm_benchmark --index-dir /path/to/index

With specific models:

    python -m paperrag.llm_benchmark --index-dir /path/to/index --models qwen2.5:1.5b,llama3.2:3b

With LLM-as-judge scoring (requires a larger model):

    python -m paperrag.llm_benchmark --index-dir /path/to/index --judge-model llama3.3:70b

Outputs:
- Per-model comparison table in `paperrag/results.md`
- Experiment logs in `experiments/logs/`

------------------------------------------------------------------------

## Persistent Configuration

Create `~/.paperragrc` (global) or `.paperragrc` (per-project) in TOML format to avoid repeating CLI flags:

```toml
index-dir = "/path/to/my-index"
model = "qwen2.5:1.5b"
topk = 2
ctx-size = 2048
```

------------------------------------------------------------------------

## Documentation

Full usage guide, troubleshooting, and project structure: [paperrag.readthedocs.org](https://paperrag.readthedocs.org) (use deepwiki to ask question regarding paperrag).

------------------------------------------------------------------------

## License

MIT

## Note
- This project was born as an answer to [my own question](https://academia.stackexchange.com/questions/203900/local-rag-like-chatgpt-for-abundant-of-pdfs/225980#225980). 
- On my laptop (Dell XPS 9320, i7-1260P 16GB RAM), it takes about **20 hours** to index ~2000 PDFs with default configuration.

## Citation  
TBA 
