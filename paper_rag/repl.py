"""Interactive REPL for PaperRAG."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from paper_rag import __version__
from paper_rag.config import PaperRAGConfig
from paper_rag.parser import discover_pdfs

console = Console()

HELP_TEXT = """\
[bold]Available commands:[/bold]
  [cyan]<any text>[/cyan]       Query the indexed papers (uses top-k retrieval + LLM)
  [cyan]index[/cyan]            Re-index the PDF directory
  [cyan]topk <n>[/cyan]         Set top-k for retrieval (default: 5)
  [cyan]no-llm[/cyan]           Toggle LLM off/on (retrieval-only mode)
  [cyan]llm-mode <mode>[/cyan]  Set LLM mode (openai or local)
  [cyan]llm-model <name>[/cyan] Set LLM model name
  [cyan]config[/cyan]           Show current LLM configuration
  [cyan]help[/cyan]             Show this help message
  [cyan]exit[/cyan] / [cyan]quit[/cyan]      Exit the REPL
"""


def start_repl(cfg: PaperRAGConfig | None = None) -> None:
    """Launch the interactive REPL session."""
    cfg = cfg or PaperRAGConfig()
    pdf_dir = Path(cfg.input_dir)
    pdfs = discover_pdfs(pdf_dir)

    console.print(f"\n[bold]PaperRAG[/bold] version [cyan]{__version__}[/cyan]")
    console.print(f"PDF directory: {pdf_dir}")

    from paper_rag.vectorstore import VectorStore

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
        console.print(
            f"Found [green]{len(pdfs)}[/green] PDFs - [yellow]{len(pdfs)} unindexed[/yellow]"
        )
        console.print(
            "[yellow]No index found. Run [bold]index[/bold] to build one before querying.[/yellow]"
        )

    console.print(f"LLM: [cyan]{cfg.llm.mode}[/cyan] / [cyan]{cfg.llm.model_name}[/cyan]")
    console.print("Type [cyan]help[/cyan] for commands.\n")

    top_k = cfg.retriever.top_k
    use_llm = True
    retriever = None  # lazy-loaded on first query

    while True:
        try:
            command = console.input("[bold]paperrag>[/bold] ").strip()
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

        if command == "no-llm":
            use_llm = not use_llm
            state = "[red]off[/red]" if not use_llm else "[green]on[/green]"
            console.print(f"LLM is now {state}")
            continue

        if command.startswith("llm-mode"):
            parts = command.split()
            if len(parts) == 2 and parts[1] in ("openai", "local"):
                cfg.llm.mode = parts[1]  # type: ignore[assignment]
                console.print(f"LLM mode set to [cyan]{parts[1]}[/cyan]")
            else:
                console.print("[yellow]Usage: llm-mode <openai|local>[/yellow]")
            continue

        if command.startswith("llm-model"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                cfg.llm.model_name = parts[1]
                console.print(f"LLM model set to [cyan]{parts[1]}[/cyan]")
            else:
                console.print("[yellow]Usage: llm-model <model-name>[/yellow]")
            continue

        if command == "config":
            console.print("\n[bold]Current LLM Configuration:[/bold]")
            console.print(f"  Mode: [cyan]{cfg.llm.mode}[/cyan]")
            console.print(f"  Model: [cyan]{cfg.llm.model_name}[/cyan]")
            console.print(f"  Base URL: [cyan]{cfg.llm.api_base or 'default'}[/cyan]")
            console.print(f"  Temperature: [cyan]{cfg.llm.temperature}[/cyan]")
            console.print(f"  Max tokens: [cyan]{cfg.llm.max_tokens}[/cyan]")
            api_key_status = "✓ set" if cfg.llm.resolve_api_key() else "✗ not set"
            console.print(f"  API key: [cyan]{api_key_status}[/cyan]\n")
            continue

        # Anything else is treated as a query
        retriever = _ensure_retriever(retriever, cfg)
        if retriever is None:
            continue
        _handle_query(command, retriever, cfg, top_k=top_k, use_llm=use_llm)


def _ensure_retriever(retriever, cfg: PaperRAGConfig):
    """Lazy-load the retriever, returning None on failure."""
    if retriever is not None:
        return retriever
    try:
        from paper_rag.retriever import Retriever

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
    use_llm: bool,
) -> None:
    """Run retrieval (and optionally LLM) for a user question."""
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Retrieval Results")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Paper", width=50, no_wrap=False)
    table.add_column("Section", width=15)
    table.add_column("Excerpt", max_width=60)
    for i, r in enumerate(results, 1):
        # Clean paper title - remove markdown artifacts
        paper_title = r.paper_title.replace("<!--- image -->", "").replace("<!-- image -->", "").strip()
        if not paper_title or paper_title == "Unknown":
            # Use filename as fallback
            paper_title = Path(r.file_path).stem

        table.add_row(
            str(i),
            f"{r.score:.4f}",
            paper_title[:80],
            r.section_name[:15],
            r.text[:120].replace("\n", " ") + "…",
        )
    console.print(table)

    if not use_llm:
        return

    try:
        from paper_rag.llm import generate_answer

        context_chunks = [r.text for r in results]
        answer = generate_answer(question, context_chunks, cfg.llm)

        # Display answer
        console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")

        # Display references mapping
        console.print("[bold]References:[/bold]")
        for i, r in enumerate(results, 1):
            # Clean paper title
            paper_title = r.paper_title.replace("<!--- image -->", "").strip()
            if not paper_title or paper_title == "Unknown":
                paper_title = Path(r.file_path).stem

            # Show reference with paper, section, and file path
            console.print(f"  [cyan][{i}][/cyan] {paper_title} - {r.section_name}")
            console.print(f"      [dim]{r.file_path}[/dim]")
        console.print()

    except ImportError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    except ValueError as exc:
        # LLM not configured - this is fine, just skip it
        console.print(f"\n[dim]💡 {exc}[/dim]\n")
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")


def _handle_index(cfg: PaperRAGConfig) -> None:
    """Run the indexing pipeline from inside the REPL."""
    from paper_rag.chunker import chunk_paper
    from paper_rag.embedder import Embedder
    from paper_rag.parser import compute_file_hashes_parallel, discover_pdfs, parse_pdf
    from paper_rag.parallel import parallel_process_pdfs
    from paper_rag.vectorstore import VectorStore

    pdf_dir = Path(cfg.input_dir)
    idx_dir = Path(cfg.index_dir)

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        return

    embedder = Embedder(cfg.embedder)

    if VectorStore.exists(idx_dir):
        store = VectorStore.load(idx_dir)
        if store.dimension != embedder.dimension:
            store = VectorStore(idx_dir, embedder.dimension)
    else:
        store = VectorStore(idx_dir, embedder.dimension)

    # Remove deleted files from the index
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
        n_workers
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
