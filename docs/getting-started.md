# Getting Started

## Installation

### Prerequisites

- Python 3.11 or later
- `uv` for environment management
- One local LLM backend:
  - [Ollama](https://ollama.com) for model names like `qwen2.5:1.5b`
  - `llama-server` from `llama.cpp` for local `.gguf` models or HuggingFace GGUF repos

### Install with uv (recommended, CPU-only default)

```bash
uv sync
```

To run commands without activating the environment:

```bash
uv run paperrag --help
```

If you prefer an editable install with `uv pip`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e .
```

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -e .
```

### Optional LLM extras

If you want HuggingFace GGUF download support, install:

```bash
uv pip install huggingface-hub
```

If you want additional Transformers-based tooling outside the default PaperRAG flow:

```bash
uv pip install transformers accelerate
```

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
paperrag --index-dir /path/to/index -m qwen2.5:1.5b
```

Inside the REPL, type any question to query your papers. Use `/help` to see all available commands.

### Focused Review Mode

For one paper or one directory, `review` runs indexing first and then opens the REPL:

```bash
paperrag review /path/to/paper.pdf
paperrag review /path/to/papers --max-tokens 512
```

If the file hash is unchanged in the target index, PaperRAG skips re-indexing automatically.

### One-off Query

For single questions or scripting:

```bash
paperrag query "what is speech chain?" --index-dir /path/to/index -m qwen2.5:1.5b
```

## REPL Commands

Once inside the REPL, these commands are available:

| Command | Description |
|---------|-------------|
| `<any text>` | Query the indexed papers |
| `/index` | Re-index the current PDF directory or file |
| `/index <path>` | Re-index a specific PDF file or directory |
| `/topk <n>` | Set top-k for retrieval |
| `/threshold <n>` | Set similarity threshold (0.0-1.0) |
| `/temperature <n>` | Set LLM temperature (0.0-2.0) |
| `/max-tokens <n>` | Set max output tokens |
| `/ctx-size <n>` | Set LLM context window size |
| `/prompt <text>` | Set the system prompt |
| `/model <name>` | Switch the active model/backend |
| `/config` | Show current configuration |
| `/rc` | Show loaded `.paperragrc` files and values |
| `/help` | Show help |
| `/exit` / `/quit` | Exit the REPL |

## LLM Setup

### Ollama backend

Install Ollama from <https://ollama.com>, then:

```bash
ollama pull qwen2.5:1.5b
paperrag --index-dir /path/to/index -m qwen2.5:1.5b
```

### llama.cpp backend

Install `llama-server` from `llama.cpp`:

```bash
brew install llama-cpp
```

Then either use a local GGUF file:

```bash
paperrag --index-dir /path/to/index -m /path/to/model.gguf
```

Or a HuggingFace GGUF repo ID:

```bash
paperrag --index-dir /path/to/index -m Qwen/Qwen3-1.7B-GGUF
```
