"""Interactive REPL for PaperRAG."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from paperrag import __version__
from paperrag.config import PaperRAGConfig
from paperrag.parser import discover_pdfs

console = Console()

HELP_TEXT = """\
[bold]Available commands:[/bold]
  [cyan]<any text>[/cyan]       Query the indexed papers (uses top-k retrieval + LLM)
  [cyan]index[/cyan]            Re-index the PDF directory
  [cyan]topk <n>[/cyan]         Set top-k for retrieval (default: 5)
  [cyan]threshold <n>[/cyan]    Set similarity threshold 0.0-1.0 (default: 0.15)
  [cyan]temperature <n>[/cyan]  Set LLM temperature 0.0-2.0 (default: 0.0)
  [cyan]no-llm[/cyan]           Toggle LLM off/on (retrieval-only mode)
  [cyan]model <name>[/cyan]     Set LLM model name
  [cyan]config[/cyan]           Show current configuration
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

    console.print(f"LLM: [cyan]{cfg.llm.mode}[/cyan] / [cyan]{cfg.llm.model_name}[/cyan]")
    
    # Display threshold filtering info
    console.print(f"Threshold: [cyan]{cfg.retriever.score_threshold}[/cyan] (minimum similarity score)")
    
    console.print("Type [cyan]help[/cyan] for commands.\n")

    top_k = cfg.retriever.top_k
    use_llm = True
    retriever = None  # lazy-loaded on first query

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

        if command == "no-llm":
            use_llm = not use_llm
            state = "[red]off[/red]" if not use_llm else "[green]on[/green]"
            console.print(f"LLM is now {state}")
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
            console.print(f"  Mode: [cyan]{cfg.llm.mode}[/cyan]")
            console.print(f"  Model: [cyan]{cfg.llm.model_name}[/cyan]")
            console.print(f"  Base URL: [cyan]{cfg.llm.api_base or 'default'}[/cyan]")
            console.print(f"  Temperature: [cyan]{cfg.llm.temperature}[/cyan]")
            console.print(f"  Max tokens: [cyan]{cfg.llm.max_tokens}[/cyan]")
            api_key_status = "✓ set" if cfg.llm.resolve_api_key() else "✗ not set"
            console.print(f"  API key: [cyan]{api_key_status}[/cyan]")
            console.print("[bold]Retrieval:[/bold]")
            console.print(f"  Top-k: [cyan]{top_k}[/cyan]")
            console.print(f"  Threshold: [cyan]{cfg.retriever.score_threshold}[/cyan]\n")
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
    use_llm: bool,
) -> None:
    """Run retrieval (and optionally LLM) for a user question."""
    import time

    t0 = time.perf_counter()
    results = retriever.retrieve(question, top_k=top_k)
    t_retrieval = time.perf_counter() - t0

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if not use_llm:
        return

    try:
        from paperrag.llm import generate_answer

        context_chunks = [r.text for r in results]
        t1 = time.perf_counter()
        answer = generate_answer(question, context_chunks, cfg.llm)
        t_llm = time.perf_counter() - t1

        # Extract cited reference numbers from answer
        import re
        cited_nums = sorted(set(
            int(m) for m in re.findall(r'\[(\d+)\]', answer)
            if int(m) <= len(results)
        ))

        if cited_nums:
            # Map each cited chunk number to its unique filename,
            # then assign new numbers per unique file
            old_to_new = {}
            unique_files = []  # (new_num, filename)
            file_to_new = {}
            for old_num in cited_nums:
                filename = Path(results[old_num - 1].file_path).name
                if filename not in file_to_new:
                    new_num = len(file_to_new) + 1
                    file_to_new[filename] = new_num
                    unique_files.append((new_num, filename))
                old_to_new[old_num] = file_to_new[filename]

            def replace_citation(match):
                old_num = int(match.group(1))
                if old_num in old_to_new:
                    return f"[{old_to_new[old_num]}]"
                return match.group(0)

            answer = re.sub(r'\[(\d+)\]', replace_citation, answer)

        # Display answer
        console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")

        # Display references
        console.print("[bold]References:[/bold]")
        if cited_nums:
            for new_num, filename in unique_files:
                console.print(f"  [cyan][{new_num}][/cyan] {filename}")
        else:
            # Fallback: show all references deduplicated
            seen = set()
            ref_num = 1
            for r in results:
                filename = Path(r.file_path).name
                if filename not in seen:
                    console.print(f"  [cyan][{ref_num}][/cyan] {filename}")
                    seen.add(filename)
                    ref_num += 1
        t_total = time.perf_counter() - t0
        console.print(f"[dim]Retrieval: {t_retrieval:.2f}s | LLM: {t_llm:.2f}s | Total: {t_total:.2f}s[/dim]\n")

    except ImportError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    except ValueError as exc:
        # LLM not configured - this is fine, just skip it
        console.print(f"\n[dim]💡 {exc}[/dim]\n")
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")


def _handle_index(cfg: PaperRAGConfig) -> None:
    """Run the indexing pipeline from inside the REPL."""
    from paperrag.chunker import chunk_paper
    from paperrag.embedder import Embedder
    from paperrag.parser import compute_file_hashes_parallel, discover_pdfs, parse_pdf
    from paperrag.parallel import parallel_process_pdfs
    from paperrag.vectorstore import VectorStore

    # pdf_dir = Path(cfg.input_dir)
    idx_dir = Path(cfg.index_dir)

    # pdfs = discover_pdfs(pdf_dir)
    # if not pdfs:
    #     console.print("[red]No PDFs found.[/red]")
    #     return

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
