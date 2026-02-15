# PaperRAG

Research-oriented (hopefully someday will be production-grade), local-first Retrieval-Augmented Generation (RAG)
system for querying academic PDF collections.

Default PDF directory: `/home/$USER/Documents/Mendeley Desktop/`

------------------------------------------------------------------------

## Features

-   DOCLING-based structured PDF parsing
-   Section-aware chunking
-   Deterministic indexing with SHA256 hashing
-   FAISS vector store
-   Local or API-based LLM support
-   Fully offline-capable
-   Reproducible experiments
-   CLI interface (Typer)

------------------------------------------------------------------------

## Installation

### 1. Install Python 3.11

### 2. Create environment

    uv venv --python 3.11
    source .venv/bin/activate

### 3. Install dependencies

    uv pip install -e .

------------------------------------------------------------------------

## Usage

### Quick Start

```bash
# 1. Index your PDFs (one time)
paperrag index

# 2. Start REPL with local LLM (best for multiple queries)
paperrag --llm-mode local --llm-model llama3.2:3b

# OR use one-off query command (for single questions)
paperrag query "your question" --llm-mode local --llm-model llama3.2:3b
```

**REPL vs Query Command:**

| Mode | Use When | Pros |
|------|----------|------|
| **REPL** `paperrag --llm-mode local --llm-model llama3.2:3b` | Multiple questions, exploring papers | Fast (index loaded once), interactive, can adjust settings |
| **Query** `paperrag query "..." --llm-mode local --llm-model llama3.2:3b` | Single question, scripting | Simple, good for automation |

------------------------------------------------------------------------

### Index PDFs

    paperrag index

Custom directory:

    paperrag index --input-dir /path/to/pdfs

------------------------------------------------------------------------

### Query Command (One-off queries)

**Use this for:** Single questions, scripting, or batch processing

Query with local LLM (Ollama):

    paperrag query "what is speech chain?" --llm-mode local --llm-model llama3.2:3b

Query with OpenAI:

    export OPENAI_API_KEY='your-key-here'
    paperrag query "what is speech chain?"

Retrieval only (no LLM):

    paperrag query "what is speech chain?" --no-llm

**Note:** Each query reloads the index. For multiple questions, use REPL mode (see below).

Example output:

(PaperRAG) bagus@xps-9320:paper_rag$ paperrag query "what is speech chain?" --llm-mode local --llm-model llama3.2:3b
2026-02-15 23:57:26 [INFO] faiss.loader: Loading faiss with AVX2 support.
2026-02-15 23:57:26 [INFO] faiss.loader: Successfully loaded faiss with AVX2 support.
2026-02-15 23:57:27 [INFO] paper_rag.vectorstore: Loaded pickle metadata format
2026-02-15 23:57:27 [INFO] paper_rag.vectorstore: Loaded index v1 (342 vectors, dim=384)
2026-02-15 23:57:27 [INFO] paper_rag.embedder: Loading embedding model sentence-transformers/all-MiniLM-L6-v2 on cpu
2026-02-15 23:57:27 [INFO] sentence_transformers.SentenceTransformer: Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
2026-02-15 23:57:31 [INFO] paper_rag.embedder: Embedding dimension: 384
2026-02-15 23:57:31 [INFO] paper_rag.retriever: Retrieved 5 results for query: what is speech chain?
                                                           Retrieval Results                                                           
┏━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃  Score ┃ Paper                                    ┃ Section      ┃ Excerpt                                                      ┃
┡━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ 0.6351 │ Andros Tjandra , Sakriani Sakti , Member │ Introduction │ multaneously listening and speaking, the speaker can monitor │
│   │        │                                          │              │ her volume, articulation, and the general                    │
│   │        │                                          │              │ comprehensibility...                                         │
│ 2 │ 0.6251 │ Andros Tjandra , Sakriani Sakti , Member │ Introduction │ and TTS and performs on single-speaker task. - 2)            │
│   │        │                                          │              │ Multi-speaker speech chain with a speaker-embedding network  │
│   │        │                                          │              │ for handli...                                                │
│ 3 │ 0.5740 │ Andros Tjandra , Sakriani Sakti , Member │ Experiment   │ /speech-chain-multispk-demo.netlify.com/....                 │
│ 4 │ 0.5191 │ Andros Tjandra , Sakriani Sakti , Member │ Introduction │ S PEECH chain, a concept introduced by Denes et al. [1],     │
│   │        │                                          │              │ describes the basic mechanism involved in speech             │
│   │        │                                          │              │ communication ...                                            │
│ 5 │ 0.5173 │ Andros Tjandra , Sakriani Sakti , Member │ Discussion   │ ig. 8 illustrates the updated speech chain mechanism.        │
│   │        │                                          │              │ Similar to the earlier version, it consists of a             │
│   │        │                                          │              │ sequence-to-seque...                                         │
└───┴────────┴──────────────────────────────────────────┴──────────────┴──────────────────────────────────────────────────────────────┘
2026-02-15 23:57:31 [INFO] paper_rag.llm: Calling LLM (local, model=llama3.2:3b, temp=0.00)
2026-02-15 23:57:58 [INFO] httpx: HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
According to Fig. 1 in [4], a S PEECH chain describes the basic mechanism involved in speech communication when a spoken message travels from the 
speaker's mind to the listener's mind.

In more detail, as described in [5], a speech chain consists of a sequence-to-sequence ASR and a sequence-to-sequence TTS. The key idea is to 
jointly train these models by integrating a speaker recognition (SPKEMB) model inside the loop connection between ASR and TTS.

------------------------------------------------------------------------

### REPL Mode (Interactive)

**⭐ Best Practice - Start REPL with LLM configured:**

```bash
# With local Ollama (recommended for privacy & offline use)
paperrag --llm-mode local --llm-model llama3.2:3b

# With OpenAI (requires OPENAI_API_KEY env var)
paperrag --llm-model gpt-4

# With custom index directory
paperrag --index-dir /path/to/index --llm-mode local --llm-model llama3.2:3b

# Without LLM configured (will work but show tip to enable LLM)
paperrag
```

**CLI Options for REPL:**
- `--llm-mode <openai|local>` - Set LLM mode 
- `--llm-model <name>` - Set LLM model (e.g., llama3.2:3b, gpt-4)
- `--api-base <url>` - Custom API endpoint
- `--index-dir <path>` - Custom index directory

**Why REPL mode is recommended:**
- ✅ **Much faster** - Index loaded once, reused for all queries
- ✅ **Interactive** - Ask follow-up questions naturally  
- ✅ **Flexible** - Change settings (model, top-k) on the fly without restarting
- ✅ **Efficient** - Perfect for exploratory research sessions

**Example session:**

    $ paperrag --llm-mode local --llm-model llama3.2:3b
    
    PaperRAG version 0.1.0
    PDF directory: /home/bagus/Documents/Mendeley Desktop
    Found 15 PDF(s)
    LLM: local / llama3.2:3b
    Type help for commands.
    
    paperrag> what is speech chain?
    [retrieval results + answer displayed]
    
    paperrag> how does it work?
    [retrieval results + answer displayed]
    
    paperrag> topk 10
    top-k set to 10
    
    paperrag> exit
    Bye!

**REPL Commands:**

    <any text>           Query the indexed papers
    index                Re-index the PDF directory
    topk <n>             Set top-k for retrieval (default: 5)
    no-llm               Toggle LLM off/on (retrieval-only mode)
    llm-mode <mode>      Set LLM mode (openai or local)
    llm-model <name>     Set LLM model name
    config               Show current LLM configuration
    help                 Show help message
    exit / quit          Exit the REPL

**Example REPL session:**

    paperrag> config
    Current LLM Configuration:
      Mode: local
      Model: llama3.2:3b
      Base URL: default
      Temperature: 0.0
      Max tokens: 1024
      API key: ✓ set
    
    paperrag> llm-model qwen2.5-coder:7b
    LLM model set to qwen2.5-coder:7b

------------------------------------------------------------------------

## LLM Configuration

**Option 1: Local LLM with Ollama (Recommended)**

Install Ollama from https://ollama.com, then pull a model:

    ollama pull llama3.2:3b
    ollama pull qwen2.5-coder:7b

Use with PaperRAG:

    paperrag --llm-mode local --llm-model llama3.2:3b
    paperrag query "your question" --llm-mode local --llm-model llama3.2:3b

**Option 2: OpenAI API**

Set your API key:

    export OPENAI_API_KEY='sk-...'

Use with PaperRAG (openai mode is default):

    paperrag
    paperrag query "your question"

Or specify model:

    paperrag --llm-model gpt-4
    paperrag query "your question" --llm-model gpt-4

**Option 3: No LLM (Retrieval Only)**

LLM is optional. Use `--no-llm` for retrieval-only mode:

    paperrag query "your question" --no-llm

------------------------------------------------------------------------

## Project Structure

paper_rag/ ├── parser.py ├── chunker.py ├── embedder.py ├──
vectorstore.py ├── retriever.py ├── llm.py └── cli.py

------------------------------------------------------------------------

## Reproducibility

-   Embedding model version stored in index
-   File hash tracking
-   Deterministic chunk ordering
-   Temperature = 0.0 by default

------------------------------------------------------------------------

## License

MIT
