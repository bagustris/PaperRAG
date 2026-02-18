"""Interactive REPL for PaperRAG.
REPL: Read, Evaluate, Print, Loop!
This mode is first class in PaperRAG
"""

from __future__ import annotations

import logging
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from paperrag import __version__
from paperrag.config import PaperRAGConfig, load_rc
from paperrag.parser import discover_pdfs

console = Console()

HELP_TEXT = """\
[bold]Available commands:[/bold]
  [cyan]<any text>[/cyan]       Query the indexed papers (uses top-k retrieval + LLM)
  [cyan]index[/cyan]            Re-index the PDF directory
  [cyan]topk <n>[/cyan]         Set top-k for retrieval (default: 3)
  [cyan]threshold <n>[/cyan]    Set similarity threshold 0.0-1.0 (default: 0.15)
  [cyan]temperature <n>[/cyan]  Set LLM temperature 0.0-2.0 (default: 0.0)
  [cyan]max-tokens <n>[/cyan]   Set LLM max output tokens (default: 256)
  [cyan]model <name>[/cyan]     Set LLM model name
  [cyan]config[/cyan]           Show current configuration
  [cyan]rc[/cyan]               Show loaded .paperragrc files and values
  [cyan]help[/cyan]             Show this help message
  [cyan]exit[/cyan] / [cyan]quit[/cyan]      Exit the REPL
"""


def start_repl(cfg: PaperRAGConfig | None = None) -> None:
    """Launch the interactive REPL session."""
    cfg = cfg or PaperRAGConfig()
    pdf_dir = Path(cfg.input_dir)
    
    # Discover PDFs without logging
    import logging
    parser_logger = logging.getLogger('paperrag.parser')
    original_level = parser_logger.level
    parser_logger.setLevel(logging.WARNING)  # Suppress INFO logs temporarily
    pdfs = discover_pdfs(pdf_dir)
    parser_logger.setLevel(original_level)  # Restore original level

    console.print(f"\n[bold]PaperRAG[/bold] version [cyan]{__version__}[/cyan]")

    # Validate and display PDF directory
    # if not pdf_dir.exists():
    #     console.print(f"[yellow]Warning: PDF directory does not exist: {pdf_dir}[/yellow]")
    #     console.print("[dim]You can specify a different directory with --input-dir <path>[/dim]\n")
    # else:
    #     console.print(f"PDF directory: {pdf_dir}")

    from paperrag.vectorstore import VectorStore

    idx_dir = Path(cfg.index_dir)

    # Check if index exists and count indexed PDFs
    if VectorStore.exists(idx_dir):
        try:
            store = VectorStore.load(idx_dir)
            indexed_count = len(store.file_hashes)
            unindexed_count = len(pdfs) - indexed_count

            if unindexed_count > 0:
                console.print(
                    f"Found [green]{len(pdfs)}[/green] PDFs - "
                    f"[yellow]{unindexed_count} unindexed[/yellow]"
                )
            else:
                console.print(f"Found [green]{len(pdfs)}[/green] PDFs - [green]all indexed[/green]")
        except Exception as e:
            console.print(f"Found [green]{len(pdfs)}[/green] PDFs")
            console.print(f"[yellow]Warning: Could not load index: {e}[/yellow]")
    else:
        console.print(f"[red]Error: No index found at {idx_dir}[/red]")
        console.print("Run [bold]paperrag index[/bold] to create an index before using the REPL.")
        import sys
        sys.exit(1)

    console.print(f"LLM: [cyan]{cfg.llm.model_name}[/cyan]")

    # Display key accuracy parameters
    console.print(f"Top-k: [cyan]{cfg.retriever.top_k}[/cyan] (retrieve {cfg.retriever.top_k} chunks)")
    console.print(f"Threshold: [cyan]{cfg.retriever.score_threshold}[/cyan] (minimum similarity score)")
    console.print(f"Temperature: [cyan]{cfg.llm.temperature}[/cyan] (0.0=deterministic, higher=creative)")
    console.print(f"Max tokens: [cyan]{cfg.llm.max_tokens}[/cyan] (max output length)")

    console.print("Type [cyan]help[/cyan] for commands.\n")

    top_k = cfg.retriever.top_k

    # Eagerly load the retriever (including embedding model) at startup
    # so the first query doesn't pay the ~6s model-loading penalty.
    console.print("[dim]Loading embedding model...\n[/dim]", end="")
    retriever = _ensure_retriever(None, cfg)
    if retriever is not None:
        console.print("[green]Ready[/green]")
    else:
        console.print("[red]Failed[/red]")

    # Suppress INFO logs during interactive session to keep output clean.
    logging.getLogger().setLevel(logging.WARNING)

    # Create prompt session with history support for arrow keys
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            command = session.prompt("paperrag> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not command:
            continue

        if command in ("exit", "quit"):
            console.print("Bye!")
            break

        if command == "help":
            console.print(HELP_TEXT)
            continue

        if command == "index":
            _handle_index(cfg)
            retriever = None  # force reload after re-index
            continue

        if command.startswith("topk"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                console.print(f"top-k set to [cyan]{top_k}[/cyan]")
            else:
                console.print("[yellow]Usage: topk <number>[/yellow]")
            continue

        if command.startswith("threshold"):
            parts = command.split()
            if len(parts) == 2:
                try:
                    threshold_val = float(parts[1])
                    if 0.0 <= threshold_val <= 1.0:
                        cfg.retriever.score_threshold = threshold_val
                        console.print(f"Threshold set to [cyan]{threshold_val}[/cyan]")
                    else:
                        console.print("[yellow]Threshold must be between 0.0 and 1.0[/yellow]")
                except ValueError:
                    console.print("[yellow]Usage: threshold <number>[/yellow]")
            else:
                console.print("[yellow]Usage: threshold <number>[/yellow]")
            continue

        if command.startswith("temperature"):
            parts = command.split()
            if len(parts) == 2:
                try:
                    temp_val = float(parts[1])
                    if 0.0 <= temp_val <= 2.0:
                        cfg.llm.temperature = temp_val
                        console.print(f"Temperature set to [cyan]{temp_val}[/cyan]")
                    else:
                        console.print("[yellow]Temperature must be between 0.0 and 2.0[/yellow]")
                except ValueError:
                    console.print("[yellow]Usage: temperature <number>[/yellow]")
            else:
                console.print("[yellow]Usage: temperature <number>[/yellow]")
            continue

        if command.startswith("max-tokens"):
            parts = command.split()
            if len(parts) == 2 and parts[1].isdigit():
                cfg.llm.max_tokens = int(parts[1])
                console.print(f"Max tokens set to [cyan]{cfg.llm.max_tokens}[/cyan]")
            else:
                console.print("[yellow]Usage: max-tokens <number>[/yellow]")
            continue

        if command.startswith("model"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                cfg.llm.model_name = parts[1]
                console.print(f"LLM model set to {parts[1]}")
            else:
                console.print("[yellow]Usage: model <model-name>[/yellow]")
            continue

        if command == "config":
            console.print("\n[bold]Current Configuration:[/bold]")
            console.print("[bold]LLM:[/bold]")
            console.print(f"  Model: [cyan]{cfg.llm.model_name}[/cyan]")
            console.print(f"  Temperature: [cyan]{cfg.llm.temperature}[/cyan]")
            console.print(f"  Max tokens: [cyan]{cfg.llm.max_tokens}[/cyan]")
            console.print("[bold]Retrieval:[/bold]")
            console.print(f"  Top-k: [cyan]{top_k}[/cyan]")
            console.print(f"  Threshold: [cyan]{cfg.retriever.score_threshold}[/cyan]\n")
            continue

        if command == "rc":
            global_path = Path.home() / ".paperragrc"
            local_path = Path.cwd() / ".paperragrc"
            console.print("\n[bold].paperragrc files:[/bold]")
            for label, rc_path in [("Global", global_path), ("Local", local_path)]:
                if rc_path.is_file():
                    rc_data = load_rc(rc_path)
                    console.print(f"  [green]{label}[/green]: {rc_path}")
                    for k, v in rc_data.items():
                        console.print(f"    {k} = [cyan]{v}[/cyan]")
                else:
                    console.print(f"  [dim]{label}[/dim]: {rc_path} [dim](not found)[/dim]")
            console.print()
            continue

        # Anything else is treated as a query
        retriever = _ensure_retriever(retriever, cfg)
        if retriever is None:
            continue
        _handle_query(command, retriever, cfg, top_k=top_k)


def _ensure_retriever(retriever, cfg: PaperRAGConfig):
    """Lazy-load the retriever, returning None on failure."""
    if retriever is not None:
        return retriever
    try:
        from paperrag.retriever import Retriever

        return Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return None


def _handle_query(
    question: str,
    retriever,
    cfg: PaperRAGConfig,
    *,
    top_k: int,
) -> None:
    """Run retrieval and LLM for a user question."""
    import time

    t0 = time.perf_counter()
    results = retriever.retrieve(question, top_k=top_k)
    t_retrieval = time.perf_counter() - t0

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Show retrieved sources immediately so the user sees useful info
    # while waiting for the LLM to generate.
    console.print(f"\n[bold]Sources[/bold] [dim]({t_retrieval:.2f}s)[/dim]")
    seen_files: set[str] = set()
    for i, r in enumerate(results):
        filename = Path(r.file_path).name
        if filename not in seen_files:
            console.print(f"  [cyan][{i+1}][/cyan] {filename} [dim]({r.score:.2f})[/dim]")
            seen_files.add(filename)

    try:
        import sys

        from paperrag.llm import stream_answer

        context_chunks = [r.text for r in results]
        full_answer = ""
        header_printed = False
        t1 = time.perf_counter()
        for chunk in stream_answer(question, context_chunks, cfg.llm):
            if not header_printed:
                console.print("\n[bold green]Answer:[/bold green]")
                header_printed = True
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_answer += chunk
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        t_llm = time.perf_counter() - t1
        t_total = time.perf_counter() - t0
        console.print(f"[dim]Retrieval: {t_retrieval:.2f}s | LLM: {t_llm:.2f}s | Total: {t_total:.2f}s[/dim]\n")

    except ImportError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    except ValueError as exc:
        # LLM not configured - this is fine, just skip it
        console.print(f"\n[dim]💡 {exc}[/dim]\n")
    except Exception as exc:
        from paperrag.llm import describe_llm_error
        error_msg, hint = describe_llm_error(exc, cfg.llm.model_name)
        console.print(f"[red]{error_msg}[/red]")
        if hint:
            console.print(f"[yellow]Fix: {hint}[/yellow]")


def _handle_index(cfg: PaperRAGConfig) -> None:
    """Run the indexing pipeline from inside the REPL."""
    from paperrag.chunker import chunk_paper
    from paperrag.embedder import Embedder
    from paperrag.parser import compute_file_hashes_parallel, discover_pdfs, parse_pdf
    from paperrag.parallel import parallel_process_pdfs
    from paperrag.vectorstore import VectorStore

    pdf_dir = Path(cfg.input_dir)
    idx_dir = Path(cfg.index_dir)

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        return

    is_single_file = pdf_dir.is_file()

    embedder = Embedder(cfg.embedder)

    if VectorStore.exists(idx_dir):
        store = VectorStore.load(idx_dir)
        if store.dimension != embedder.dimension:
            store = VectorStore(idx_dir, embedder.dimension)
    else:
        store = VectorStore(idx_dir, embedder.dimension)

    # Remove deleted files from the index (skip for single-file mode)
    stale = []
    if not is_single_file:
        current_paths = {str(p) for p in pdfs}
        stale = [fp for fp in list(store.file_hashes) if fp not in current_paths]
        for fp in stale:
            store.remove_by_file(fp)
            del store.file_hashes[fp]
        if stale:
            console.print(f"Removed [red]{len(stale)}[/red] deleted file(s) from index.")

    # Determine which files need (re)indexing - use parallel hashing
    n_workers = cfg.indexing.get_n_workers()
    console.print(f"Computing hashes for {len(pdfs)} PDFs...")
    pdf_hashes = compute_file_hashes_parallel(pdfs, n_workers)

    to_index: list[Path] = []
    for pdf in pdfs:
        current_hash = pdf_hashes.get(str(pdf))
        stored_hash = store.get_file_hash(str(pdf))
        if stored_hash is None or stored_hash != current_hash:
            if stored_hash is not None:
                store.remove_by_file(str(pdf))
            to_index.append(pdf)

    if not to_index and not stale:
        console.print("[green]Index is up-to-date.[/green]")
        return

    if not to_index:
        store.version += 1
        store.save(config=cfg)
        console.print(f"[green]Done![/green] Index version: {store.version}")
        return

    total = len(to_index)
    console.print(f"Parsing [cyan]{total}[/cyan] PDF(s) with {n_workers} workers...")

    # Parallel parse + chunk phase
    parsed_results = parallel_process_pdfs(
        to_index,
        cfg.parser,
        cfg.chunker,
        n_workers,
        timeout=cfg.indexing.pdf_timeout
    )

    # Sequential embed + add phase
    console.print("Embedding and indexing chunks...")
    total_chunks = 0
    processed_count = 0
    checkpoint_interval = cfg.indexing.checkpoint_interval

    for i, (pdf_path, file_hash, chunks, error) in enumerate(parsed_results, 1):
        console.print(f"  [{i}/{total}] {pdf_path.name}", highlight=False)

        if error:
            console.print(f"    [red]Error: {error}[/red]")
            continue
        if not chunks:
            console.print(f"    [yellow]No chunks produced, skipping.[/yellow]")
            continue

        embeddings = embedder.embed([c.text for c in chunks])
        store.add(embeddings, chunks)
        store.set_file_hash(str(pdf_path), file_hash)
        total_chunks += len(chunks)
        processed_count += 1
        console.print(f"    [green]{len(chunks)} chunks[/green]")

        # Periodic checkpoint
        if checkpoint_interval > 0 and processed_count >= checkpoint_interval:
            try:
                store.save(config=cfg)
                console.print(f"    [dim]Checkpoint saved ({processed_count} PDFs, {total_chunks} chunks)[/dim]")
                processed_count = 0
            except Exception as e:
                console.print(f"    [yellow]Checkpoint save failed: {e}[/yellow]")

    store.version += 1
    store.save(config=cfg)
    console.print(
        f"[green]Done![/green] Indexed {total_chunks} chunks from "
        f"{len(to_index)} file(s). Index version: {store.version}"
    )
