# Getting Started

## Installation

### Prerequisites

- Python 3.11 or later
- (Optional) [Ollama](https://ollama.com) for local LLM inference

### Install with uv (recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[llm]"
```

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[llm]"
```

The `llm` extra includes `transformers`, `accelerate`, and `openai` for LLM support. Omit it if you only need retrieval.

## Indexing PDFs

Before querying, you must index your PDF collection:

```bash
paperrag index --input-dir /path/to/pdfs
```

This will:

1. Discover all PDFs in the directory
2. Parse each PDF using Docling (with adaptive OCR)
3. Chunk the text into sections
4. Embed chunks and store them in a FAISS index

The index is saved to `<input-dir>/.paperrag-index/` by default. Use `--index-dir` to specify a different location.

### Adaptive OCR

PaperRAG automatically detects whether each PDF needs OCR:

- **Text-based PDFs** skip OCR (2-3x faster)
- **Scanned PDFs** enable OCR for accurate extraction
- Override with `--ocr always` or `--ocr never`

### Parallel Workers

Worker count is auto-detected based on available RAM (~2 GB per worker). Override with `--workers N`:

```bash
# For low-RAM systems
paperrag index --workers 2

# Speed up on high-RAM systems
paperrag index --workers 10
```

## Running Queries

### Interactive REPL (recommended)

The REPL loads the index once and lets you ask multiple questions:

```bash
# With local Ollama model
paperrag --index-dir /path/to/pdfs -m qwen3:1.7b

# With OpenAI
export OPENAI_API_KEY='sk-...'
paperrag --index-dir /path/to/pdfs
```

Inside the REPL, type any question to query your papers. Use `help` to see all available commands.

### One-off Query

For single questions or scripting:

```bash
paperrag query "what is speech chain?" --index-dir /path/to/pdfs -m qwen3:1.7b
```

### Retrieval Only (no LLM)

```bash
paperrag query "what is speech chain?" --index-dir /path/to/pdfs --no-llm
```

## REPL Commands

Once inside the REPL, these commands are available:

| Command | Description |
|---------|-------------|
| `<any text>` | Query the indexed papers |
| `index` | Re-index the PDF directory |
| `topk <n>` | Set top-k for retrieval (default: 5) |
| `threshold <n>` | Set similarity threshold (0.0-1.0) |
| `temperature <n>` | Set LLM temperature (0.0-2.0) |
| `no-llm` | Toggle LLM off/on |
| `model <name>` | Switch LLM model |
| `config` | Show current configuration |
| `help` | Show help |
| `exit` / `quit` | Exit the REPL |

## LLM Setup

### Option 1: Ollama (local, recommended)

Install Ollama from <https://ollama.com>, then:

```bash
ollama pull qwen3:1.7b
paperrag --index-dir /path/to/index -m qwen3:1.7b
```

### Option 2: OpenAI API

```bash
export OPENAI_API_KEY='sk-...'
paperrag --index-dir /path/to/index --model gpt-4 --api-base https://api.openai.com/v1
```

### Option 3: No LLM

LLM is optional. Use `--no-llm` for retrieval-only mode.
