# AGENTS.md --- PaperRAG 

## 1. Project Overview

PaperRAG is a deterministic, scalable, local-first Retrieval-Augmented
Generation (RAG) system for querying academic PDF collections.

Default input directory:

/home/\$USER/Documents/Mendeley Desktop/

Must be configurable via CLI.

Design Goals:  
- Fully local-first  
- CPU first, GPU second  
- Balance of speed and quality  
- Deterministic indexing  
- Scalable to 20,000+ PDFs  
- Reproducible experiments  
- Modular architecture  
- Research-grade logging and evaluation hooks

------------------------------------------------------------------------

## 2. Environment Setup

### Python Version

Python \>= 3.11

### Virtual Environment (REQUIRED)

Use uv for all environment management.

Create environment:

    uv venv
    source .venv/bin/activate

Install dependencies:

    uv pip install <package>

No pip or conda allowed.

------------------------------------------------------------------------

## 3. Required Dependencies

Core: 
- docling  
- sentence-transformers  
- faiss-cpu  
- torch  
- numpy 
- pydantic  
- typer  
- rich  
- tqdm  
- python-dotenv

Optional (LLM):  
- transformers   
- accelerate  
- openai

Testing:  
- pytest

------------------------------------------------------------------------

## 4. Architecture

Pipeline:

PDF Folder -\> DOCLING parsing -\> Structured section extraction -\>
Deterministic chunking -\> Embedding -\> FAISS index -\> Retriever -\>
LLM

All steps must be modular and independently testable.

------------------------------------------------------------------------

## 5. Deterministic Indexing Requirements

1.  Each PDF must generate a SHA256 hash.
2.  Store metadata JSON alongside embeddings.
3.  Re-index only changed or new files.
4.  Embedding model name and version must be stored in index metadata.
5.  Index versioning required.

Index directory structure:

.paperrag-index/  
├── faiss.index   
├── metadata.json  
├── config_snapshot.json  
└── file_hashes.json  

------------------------------------------------------------------------

## 6. PDF Parsing (DOCLING Required)

Must extract: 
- Title 
- Authors  
- Sections (Abstract, Introduction Methods, Results, Conclusion)  
- Full raw text fallback

Must handle: 
- Corrupted files 
- Missing sections 
- Duplicate documents

------------------------------------------------------------------------

## 7. Chunking Strategy

-   Section-aware chunking
-   Default chunk_size: 1000 characters
-   Default chunk_overlap: 200
-   Deterministic order guaranteed

Chunks must include metadata:  
- paper_title  
- section_name  
- file_path  
- chunk_id  
- hash_id

------------------------------------------------------------------------

## 8. Embedding Layer

Default model:

sentence-transformers/all-MiniLM-L6-v2

Requirements:  
- Batched encoding  
- GPU support if available  
- Embedding dimension validation  
- Deterministic seed control

------------------------------------------------------------------------

## 9. Retrieval

-   FAISS IndexFlatIP preferred
-   Top-k retrieval (default k=3)
-   Return similarity score
-   Return metadata

Must support retrieval-only mode.

------------------------------------------------------------------------

## 10. LLM Layer

Two modes required:

1)  Local transformers-based model
2)  OpenAI-compatible API mode

Prompt template:

"Answer strictly using the provided context. If the answer is not in
context, respond: 'I don't know.'"

Temperature default: 0.0

------------------------------------------------------------------------

## 11. CLI Interface (Typer Required)

Commands:

Index:

    paperrag index
    paperrag index --input-dir <path>

Query:

    paperrag query "Your question"
    paperrag query --top-k 10
    paperrag query --no-llm

Evaluate:

    paperrag evaluate --benchmark-file <file>

------------------------------------------------------------------------

## 12. Logging

Structured logging required. Levels:  
- INFO  
- WARNING  
- ERROR

Indexing progress must use tqdm.

------------------------------------------------------------------------

## 13. Testing Requirements

Minimum test coverage: - PDF parsing - Chunk boundaries - Embedding
dimension - FAISS save/load - Retrieval correctness - Hash-based reindex
logic

pytest required.

------------------------------------------------------------------------

## 14. Evaluation Hooks

System must support:

-   Retrieval precision@k
-   MRR
-   Context length tracking
-   Latency logging

All evaluation must be reproducible.

------------------------------------------------------------------------

## 15. Non-Functional Requirements

-   No cloud dependency required
-   Offline capable
-   Reproducible experiments
-   Modular design
-   Clear separation of concerns
-   Clean pyproject.toml with pinned versions

------------------------------------------------------------------------

## 16. Deliverables Required From Agent

-   Complete project structure
-   Working CLI
-   Deterministic indexing
-   Unit tests
-   README
-   Example usage
-   No placeholder implementations
