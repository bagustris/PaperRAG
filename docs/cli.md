# CLI Reference

PaperRAG is invoked via the `paperrag` command. Running it without a subcommand starts the interactive REPL.

## Global Options (REPL mode)

```
paperrag [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--index-dir PATH` | `-i` | Index directory (required) |
| `--input-dir PATH` | `-d` | PDF directory |
| `--model NAME` | `-m` | LLM model name (e.g. `qwen3:1.7b`) |
| `--api-base URL` | `-a` | API base URL (default: `http://localhost:11434/v1`) |
| `--threshold FLOAT` | `-t` | Similarity score threshold (0.0-1.0) |
| `--temperature FLOAT` | | LLM temperature (0.0-2.0) |
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

# Force re-index with 4 workers, OCR disabled
paperrag index --input-dir ~/papers --force --workers 4 --ocr never

# Index with a metadata manifest
paperrag index --input-dir ~/papers --manifest papers.csv
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
| `--top-k N` | `-k` | Number of results (default: 5) |
| `--threshold FLOAT` | `-t` | Similarity score threshold (0.0-1.0) |
| `--temperature FLOAT` | | LLM temperature (0.0-2.0) |
| `--no-llm` | | Retrieval only, skip LLM |
| `--model NAME` | `-m` | LLM model name |
| `--api-base URL` | | API base URL |

### Examples

```bash
# Query with local LLM
paperrag query "what is speech chain?" --index-dir ~/papers -m qwen3:1.7b

# Retrieval only
paperrag query "what is speech chain?" --index-dir ~/papers --no-llm

# Adjust retrieval parameters
paperrag query "attention mechanism" --index-dir ~/papers -k 10 -t 0.3
```

## `paperrag evaluate`

Evaluate retrieval quality using a JSONL benchmark file.

```
paperrag evaluate BENCHMARK_FILE [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--top-k N` | `-k` | Number of results (default: 5) |
| `--input-dir PATH` | `-d` | PDF directory |
| `--index-dir PATH` | `-i` | Index directory |

The benchmark file should be JSONL with each line containing:

```json
{"question": "...", "relevant_documents": ["path1.pdf", "path2.pdf"]}
```
