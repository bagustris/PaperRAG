# CLI Reference

PaperRAG is invoked via the `paperrag` command. Running it without a subcommand starts the interactive REPL.

## Global Options (REPL mode)

```
paperrag [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--index-dir PATH` | `-i` | Index directory (required unless set in `.paperragrc`) |
| `--input-dir PATH` | `-d` | PDF directory |
| `--model NAME` | `-m` | LLM model name. Ollama names (for example `qwen2.5:1.5b`) use Ollama. Local `.gguf` paths and HuggingFace repo IDs use `llama.cpp` via `llama-server`. |
| `--topk N` | `-k` | Number of chunks to retrieve (default: 3) |
| `--threshold FLOAT` | `-t` | Similarity score threshold (0.0-1.0) |
| `--temperature FLOAT` | | LLM temperature (0.0-2.0) |
| `--max-tokens N` | | Maximum response tokens (default: 256) |
| `--ctx-size N` | | LLM context window size (default: 2048) |
| `--system-prompt TEXT` | | Override the default system prompt |
| `--version` | | Show version and license |
| `--help` | `-h` | Show help |

## `paperrag index`

Index PDF files into the FAISS vector store.

```
paperrag index [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-dir PATH` | `-d` | PDF directory (default: `~/Documents/Mendeley Desktop`) |
| `--index-dir PATH` | `-i` | Index directory (default: `<input-dir>/.paperrag-index`) |
| `--force` | `-f` | Force full re-index |
| `--checkpoint-interval N` | `-c` | Save index every N PDFs (0 to disable) |
| `--workers N` | `-w` | Number of parallel workers (0 = auto-detect) |
| `--ocr MODE` | | OCR mode: `auto` (default), `always`, `never` |
| `--manifest PATH` | | CSV manifest file with paper metadata |

### Examples

```bash
# Index with default settings
paperrag index --input-dir ~/papers

# Index a single PDF into ~/papers/.paperrag-index
paperrag index --input-dir ~/papers/paper.pdf

# Force re-index with 4 workers, OCR disabled
paperrag index --input-dir ~/papers --force --workers 4 --ocr never

# Index with a metadata manifest
paperrag index --input-dir ~/papers --manifest papers.csv
```

## `paperrag review`

Index a PDF file or directory, then drop directly into the interactive review session.

```
paperrag review INPUT_PATH [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--index-dir PATH` | `-i` | Index directory. If omitted, auto-derived from the input path. |
| `--topk N` | `-k` | Number of chunks to retrieve (default: current config) |
| `--threshold FLOAT` | `-t` | Similarity score threshold (0.0-1.0) |
| `--temperature FLOAT` | | LLM temperature |
| `--max-tokens N` | | Maximum response tokens |
| `--ctx-size N` | | LLM context window size |
| `--system-prompt TEXT` | | Override the default system prompt |
| `--model NAME` | `-m` | LLM model name |

Default index location:

- Directory input: `<input-dir>/.paperrag-index`
- Single PDF input: `<pdf-parent-dir>/.paperrag-index`

Examples:

```bash
# Review one paper using the parent directory index
paperrag review ~/papers/paper.pdf

# Review a whole directory and increase answer length
paperrag review ~/papers --max-tokens 512

# Review using a custom index location
paperrag review ~/papers/paper.pdf --index-dir /tmp/paperrag-review
```

## `paperrag query`

Run a one-off query against the indexed papers.

```
paperrag query QUESTION [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--index-dir PATH` | `-i` | Index directory (required) |
| `--input-dir PATH` | `-d` | PDF directory |
| `--top-k N` | `-k` | Number of results (default: 3) |
| `--threshold FLOAT` | `-t` | Similarity score threshold (0.0-1.0) |
| `--temperature FLOAT` | | LLM temperature (0.0-2.0) |
| `--max-tokens N` | | Maximum response tokens (default: 256) |
| `--ctx-size N` | | LLM context window size (default: 2048) |
| `--system-prompt TEXT` | | Override the default system prompt |
| `--model NAME` | `-m` | LLM model name |

### Examples

```bash
# Query with Ollama
paperrag query "what is speech chain?" --index-dir ~/papers -m qwen3:1.7b

# Query with a local GGUF model through llama.cpp
paperrag query "summarize the paper" --index-dir ~/papers -m ./models/qwen3-1.7b.gguf

# Query with a HuggingFace GGUF repo through llama.cpp
paperrag query "summarize the paper" --index-dir ~/papers -m Qwen/Qwen3-1.7B-GGUF

# Adjust retrieval parameters
paperrag query "attention mechanism" --index-dir ~/papers -k 10 -t 0.3 --max-tokens 512
```

## `paperrag evaluate`

Evaluate retrieval quality using a JSONL benchmark file.

```
paperrag evaluate BENCHMARK_FILE [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--top-k N` | `-k` | Number of results (default: 3) |
| `--input-dir PATH` | `-d` | PDF directory |
| `--index-dir PATH` | `-i` | Index directory |

The benchmark file should be JSONL with each line containing:

```json
{"question": "...", "relevant_documents": ["path1.pdf", "path2.pdf"]}
```

## REPL Slash Commands

Inside the REPL, all control commands are slash-prefixed:

| Command | Description |
|--------|-------------|
| `<any text>` | Query the indexed papers |
| `/index` | Re-index the current PDF directory or file |
| `/index <path>` | Re-index a different PDF file or directory |
| `/topk <n>` | Set top-k retrieval |
| `/threshold <n>` | Set minimum similarity score |
| `/temperature <n>` | Set LLM temperature |
| `/max-tokens <n>` | Set maximum response tokens |
| `/ctx-size <n>` | Set LLM context window size |
| `/prompt <text>` | Set the system prompt |
| `/model <name>` | Switch the active LLM backend/model |
| `/config` | Show the current effective configuration |
| `/rc` | Show loaded `.paperragrc` files and values |
| `/help` | Show REPL help |
| `/exit` or `/quit` | Exit the REPL |
