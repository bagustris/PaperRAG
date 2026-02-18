# Configuration

PaperRAG uses Pydantic models for configuration. Settings can be provided via CLI options, `.paperragrc` run control files, config snapshots (saved with the index), or code defaults.

## Config Hierarchy

Settings are applied in this order (later overrides earlier):

1. **Defaults** -- Pydantic model defaults
2. **Global `.paperragrc`** -- `~/.paperragrc`
3. **Local `.paperragrc`** -- `.paperragrc` in current directory
4. **Config snapshot** -- loaded from `<index-dir>/config_snapshot.json` when an existing index is opened
5. **CLI options** -- highest priority

## `.paperragrc` Run Control File

Create a `.paperragrc` file to set persistent defaults so you don't have to pass CLI arguments every time. The file uses [TOML](https://toml.io) format (parsed with Python's built-in `tomllib` -- no extra dependencies).

**Scopes:**

- **Global**: `~/.paperragrc` -- applies to all PaperRAG invocations
- **Local**: `.paperragrc` in the current working directory -- project-specific overrides

Local values override global values. CLI arguments override both.

**Example `~/.paperragrc`:**

```toml
# PaperRAG defaults
model = "qwen2.5:1.5b"
topk = 3
max-tokens = 256
temperature = 0.0
threshold = 0.1
index-dir = "/home/user/papers/.paperrag-index"
input-dir = "/home/user/papers"
```

**Supported keys:**

| RC Key | Maps To | Type |
|--------|---------|------|
| `model` | LLM model name | `str` |
| `topk` | Number of chunks to retrieve | `int` |
| `max-tokens` | LLM max output tokens | `int` |
| `temperature` | LLM temperature | `float` |
| `threshold` | Minimum similarity score | `float` |
| `index-dir` | Index directory path | `str` |
| `input-dir` | PDF input directory path | `str` |

Unknown keys produce a warning but do not cause errors. Invalid TOML files are skipped with a warning.

Use `rc` in the REPL to see which RC files are loaded and their values.

## Top-level: `PaperRAGConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dir` | `str` | `~/Documents/Mendeley Desktop` | PDF directory |
| `index_dir` | `str` | `<input_dir>/.paperrag-index` | Index storage directory (property) |
| `parser` | `ParserConfig` | | PDF parsing settings |
| `chunker` | `ChunkerConfig` | | Chunking settings |
| `embedder` | `EmbedderConfig` | | Embedding model settings |
| `retriever` | `RetrieverConfig` | | Retrieval settings |
| `indexing` | `IndexingConfig` | | Indexing pipeline settings |
| `llm` | `LLMConfig` | | LLM settings |

## `ParserConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extract_tables` | `bool` | `False` | Extract tables from PDFs |
| `fallback_to_raw` | `bool` | `True` | Fall back to raw text extraction on parse failure |
| `ocr_mode` | `"auto" \| "always" \| "never"` | `"auto"` | OCR strategy per PDF |
| `manifest_file` | `str \| None` | `None` | CSV manifest path with paper metadata |

## `ChunkerConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunk_size` | `int` | `1000` | Maximum chunk size in characters (min: 100) |
| `chunk_overlap` | `int` | `200` | Overlap between consecutive chunks (min: 0) |

## `EmbedderConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer model |
| `batch_size` | `int` | `64` | Embedding batch size |
| `device` | `str \| None` | `None` | Device (`cuda`, `cpu`, or auto-detect) |
| `normalize` | `bool` | `True` | L2-normalize embeddings |
| `seed` | `int` | `42` | Random seed for reproducibility |

## `RetrieverConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | `int` | `3` | Number of results to return |
| `score_threshold` | `float` | `0.1` | Minimum similarity score (0.0 = no filtering) |
| `use_mmr` | `bool` | `False` | Use Maximal Marginal Relevance for diversity |
| `mmr_lambda` | `float` | `0.5` | MMR lambda (0 = max diversity, 1 = max relevance) |
| `max_results_per_paper` | `int` | `2` | Maximum results from the same paper |

## `IndexingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_interval` | `int` | `50` | Save index every N PDFs (0 = disabled) |
| `n_workers` | `int` | `0` | Parallel workers (0 = auto-detect) |
| `pdf_timeout` | `int` | `300` | Timeout per PDF in seconds (0 = no timeout) |
| `enable_gc_per_batch` | `bool` | `True` | Run garbage collection after each batch |
| `log_memory_usage` | `bool` | `False` | Log memory usage during indexing |
| `continue_on_error` | `bool` | `True` | Continue if individual PDFs fail |
| `max_failures` | `int` | `-1` | Stop after N failures (-1 = unlimited) |

### Worker Auto-detection

When `n_workers` is 0, PaperRAG calculates a safe worker count:

```
workers = min(cpu_cores - 1, (available_ram_gb - 2) / 2)
```

Each worker requires approximately 2 GB of RAM during peak Docling usage.

## `LLMConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `qwen3:1.7b` | LLM model name |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `256` | Maximum response tokens |
