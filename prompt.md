You are building a production-grade local RAG system named "PaperRAG".

STRICT REQUIREMENTS:
- Follow AGENTS.md exactly.
- Do not simplify architecture.
- Do not omit deterministic indexing requirements.
- No placeholder implementations.
- Code must be fully runnable.
- Use Python 3.11 only.
- Use uv for dependency management.
- Use DOCLING for PDF parsing.
- Use FAISS for vector storage.
- Use sentence-transformers for embeddings.
- Use Typer for CLI.
- Use Pydantic for configuration.

PROJECT FILES ALREADY PREPARED:
- README.md
- pyproject.toml
- AGENTS.md
- EVALUATION_TEMPLATE.md
- EXPERIMENT_LOGGING_SCHEMA.md
- evaluate_paper_rag.py

DEFAULT INPUT DIRECTORY:
/home/$USER/Documents/Mendeley Desktop/

OBJECTIVE:
Generate the complete project implementation with the following:

1. Modular package structure:
   paper_rag/
      parser.py
      chunker.py
      embedder.py
      vectorstore.py
      retriever.py
      llm.py
      config.py
      cli.py

2. Deterministic indexing:
   - SHA256 hashing per file
   - Reindex only changed files
   - Store metadata and config snapshot
   - Index versioning

3. CLI Commands:
   - index
   - query
   - evaluate

4. Embedding:
   - sentence-transformers/all-MiniLM-L6-v2
   - Batched encoding
   - GPU support if available

5. Retrieval:
   - FAISS IndexFlatIP
   - Top-k search
   - Return metadata + score

6. LLM:
   - Local transformers support (via Ollama)
   - OpenAI-compatible API support
   - Temperature = 0.0 default

7. Logging:
   - Structured logging
   - tqdm progress bar

8. Testing:
   - Include pytest-compatible test files
   - Ensure embedding dimension validation
   - Ensure FAISS persistence works

OUTPUT FORMAT:
- Provide complete file contents.
- No explanation.
- No commentary.
- Only code and file structure.

IMPORTANT:
Architecture must strictly match AGENTS.md.
All components must integrate correctly.
No shortcuts.
No mock implementations.
Try on 5 PDFs to ensure functionality.

---
We want to extend the CLI to support interactive REPL mode (I already implemented this in `repl.py`).

Constraints:
- Do not break existing commands.
- Keep Typer structure intact.
- Add REPL as default behavior when no subcommand is passed.
- Separate REPL logic into repl.py.
- Keep business logic outside CLI layer.
- Before starting REPL, print version, PDF directory info, and number of found.

I also tried running the CLI and it seems the REPL is not starting as expected. Please ensure that when I run `paperrag` (not `paper_rag`) without any subcommands, it should start the REPL interface.
bagus@xps-9320:PaperRAG$ source .venv/bin/activate
(PaperRAG) bagus@xps-9320:PaperRAG$ paper
paperconf    paperconfig  paper_rag/   
(PaperRAG) bagus@xps-9320:PaperRAG$ ls
AGENTS.md   evaluate_paper.py       EXPERIMENT_LOGGING_SCHEMA.md  paper_rag           prompt.md       README.md
Dockerfile  EVALUATION_TEMPLATE.md  opencode.json                 paper_rag.egg-info  pyproject.toml  tests
(PaperRAG) bagus@xps-9320:PaperRAG$ cd paper_rag
(PaperRAG) bagus@xps-9320:paper_rag$ paper
paperconf    paperconfig  paper_rag    
(PaperRAG) bagus@xps-9320:paper_rag$ paper_rag
PaperRAG version 0.1.1
PDF directory: /home/$USER/Documents/Mendeley Desktop/
Type 'help' for commands.

paperrag>: exit()
Unknown command.
paperrag>: what is CNN?
Unknown command.
paperrag>: 
