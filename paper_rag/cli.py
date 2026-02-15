"""Typer CLI for PaperRAG."""

from __future__ import annotations

import datetime
import gc
import json
import logging
import sys
from pathlib import Path

import multiprocessing
import psutil
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Force 'spawn' method for multiprocessing to avoid deadlocks with PyTorch/Docling/other libraries that use OpenMP/CUDA
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from paper_rag.config import PaperRAGConfig

app = typer.Typer(
    name="paperrag",
    help="PaperRAG - local RAG for academic PDFs.",
    invoke_without_command=True,
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper_rag")


@app.callback(invoke_without_command=True)
def entrypoint(
    ctx: typer.Context,
    index_dir: str = typer.Option(None, "--index-dir", help="Index directory"),
    llm_mode: str = typer.Option(None, "--llm-mode", help="LLM mode: openai or local"),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM model name"),
    api_base: str = typer.Option(None, "--api-base", help="API base URL for LLM"),
) -> None:
    """PaperRAG - local RAG for academic PDFs."""
    if ctx.invoked_subcommand is None:
        from paper_rag.repl import start_repl

        cfg = PaperRAGConfig()
        if index_dir:
            cfg.index_dir = index_dir
        if llm_mode:
            cfg.llm.mode = llm_mode  # type: ignore[assignment]
        if llm_model:
            cfg.llm.model_name = llm_model
        if api_base:
            cfg.llm.api_base = api_base

        start_repl(cfg)


# -- index -----------------------------------------------------------------
@app.command()
def index(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="PDF directory"),
    index_dir: str = typer.Option(None, "--index-dir", help="Index output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    checkpoint_interval: int = typer.Option(
        None,
        "--checkpoint-interval",
        "-c",
        help="Save index every N PDFs (0 to disable checkpointing)"
    ),
    workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (0 = auto-detect)"
    ),
) -> None:
    """Index PDF files into the FAISS vector store."""
    from paper_rag.chunker import chunk_paper
    from paper_rag.embedder import Embedder
    from paper_rag.parser import compute_file_hashes_parallel, discover_pdfs, parse_pdf
    from paper_rag.parallel import parallel_process_pdfs
    from paper_rag.vectorstore import VectorStore

    cfg = PaperRAGConfig()
    if input_dir:
        cfg.input_dir = input_dir
    if index_dir:
        cfg.index_dir = index_dir
    if checkpoint_interval is not None:
        cfg.indexing.checkpoint_interval = checkpoint_interval
    if workers is not None:
        cfg.indexing.n_workers = workers

    pdf_dir = Path(cfg.input_dir)
    idx_dir = Path(cfg.index_dir)

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        raise typer.Exit(1)

    console.print(f"Found [green]{len(pdfs)}[/green] PDF(s) in {pdf_dir}")

    embedder = Embedder(cfg.embedder)

    # Load or create store
    if VectorStore.exists(idx_dir) and not force:
        store = VectorStore.load(idx_dir)
        if store.dimension != embedder.dimension:
            console.print(
                "[yellow]Dimension mismatch - rebuilding index.[/yellow]"
            )
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
    if workers is None:
        n_workers = cfg.indexing.get_n_workers()
    else:
        n_workers = workers if workers > 0 else 1
    
    cfg.indexing.n_workers = n_workers

    console.print(f"Computing hashes for {len(pdfs)} PDFs with {n_workers} workers...")
    pdf_hashes = compute_file_hashes_parallel(pdfs, n_workers)

    to_index: list[Path] = []
    for pdf in pdfs:
        current_hash = pdf_hashes.get(str(pdf))
        stored_hash = store.get_file_hash(str(pdf))
        if force or stored_hash is None or stored_hash != current_hash:
            if stored_hash is not None:
                store.remove_by_file(str(pdf))
            to_index.append(pdf)

    if not to_index and not stale:
        console.print("[green]Index is up-to-date. No files to re-index.[/green]")
        raise typer.Exit(0)

    if not to_index:
        store.version += 1
        store.save(config=cfg)
        console.print(f"[green]Done![/green] Index version: {store.version}")
        raise typer.Exit(0)

    # Process in batches of 5 (reduced from 10 to save RAM)
    BATCH_SIZE = 5
    total_files = len(to_index)
    total_chunks = 0
    total_successes = 0
    total_failures = 0
    failed_pdfs: list[tuple[str, str]] = []  # (pdf_path, error_message)
    
    console.print(f"Indexing [cyan]{total_files}[/cyan] PDF(s) in batches of {BATCH_SIZE}...")
    console.print(f"Using [green]{n_workers}[/green] worker(s) for parallel processing.")
    
    # Helper function to log memory usage
    def log_memory():
        if cfg.indexing.log_memory_usage:
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(
                "Memory usage: RSS=%.1f MB, VMS=%.1f MB",
                mem_info.rss / 1024 / 1024,
                mem_info.vms / 1024 / 1024
            )
    
    for i in range(0, total_files, BATCH_SIZE):
        batch = to_index[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
        
        console.print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} files)...")
        log_memory()
        
        # Parallel parse + chunk phase for this batch
        parsed_results = parallel_process_pdfs(
            batch,
            cfg.parser,
            cfg.chunker,
            n_workers
        )

        # Sequential embed + add phase with comprehensive error handling
        batch_chunks = 0
        batch_successes = 0
        batch_failures = 0
        
        for pdf_path, file_hash, chunks, error in tqdm(parsed_results, desc="Embedding", unit="file"):
            # Handle parsing errors
            if error:
                logger.error("Failed to parse %s: %s", pdf_path.name, error)
                failed_pdfs.append((str(pdf_path), f"Parse error: {error}"))
                batch_failures += 1
                total_failures += 1
                
                # Check if we've exceeded max failures
                if cfg.indexing.max_failures > 0 and total_failures >= cfg.indexing.max_failures:
                    console.print(f"[red]Reached maximum failures ({cfg.indexing.max_failures}). Stopping.[/red]")
                    break
                
                if not cfg.indexing.continue_on_error:
                    console.print("[red]Stopping due to error (continue_on_error=False)[/red]")
                    raise typer.Exit(1)
                continue
            
            if not chunks:
                logger.warning("No chunks produced for %s", pdf_path.name)
                failed_pdfs.append((str(pdf_path), "No chunks produced"))
                batch_failures += 1
                total_failures += 1
                continue

            # Embedding phase with error handling
            try:
                texts = [c.text for c in chunks]
                embeddings = embedder.embed(texts)
                store.add(embeddings, chunks)
                store.set_file_hash(str(pdf_path), file_hash)
                batch_chunks += len(chunks)
                batch_successes += 1
                total_successes += 1
            except Exception as e:
                logger.error("Failed to embed/store %s: %s", pdf_path.name, e, exc_info=True)
                failed_pdfs.append((str(pdf_path), f"Embedding error: {e}"))
                batch_failures += 1
                total_failures += 1
                
                # Check if we've exceeded max failures
                if cfg.indexing.max_failures > 0 and total_failures >= cfg.indexing.max_failures:
                    console.print(f"[red]Reached maximum failures ({cfg.indexing.max_failures}). Stopping.[/red]")
                    break
                
                if not cfg.indexing.continue_on_error:
                    console.print("[red]Stopping due to error (continue_on_error=False)[/red]")
                    raise typer.Exit(1)
                continue

        total_chunks += batch_chunks
        
        # Report batch statistics
        console.print(
            f"Batch {batch_num} complete: [green]{batch_successes} succeeded[/green], "
            f"[red]{batch_failures} failed[/red], {batch_chunks} chunks added"
        )
        
        # Save after every batch for resumption with retry logic
        max_retries = 3
        save_success = False
        for retry in range(max_retries):
            try:
                store.save(config=cfg)
                console.print(f"✓ Index saved. Total chunks: {total_chunks}")
                save_success = True
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning("Failed to save index (attempt %d/%d): %s", retry + 1, max_retries, e)
                else:
                    logger.error("Failed to save index after %d attempts: %s", max_retries, e)
        
        if not save_success:
            console.print("[red]Critical: Failed to save index checkpoint![/red]")
            if not cfg.indexing.continue_on_error:
                raise typer.Exit(1)
        
        # Memory cleanup between batches
        if cfg.indexing.enable_gc_per_batch:
            gc.collect()
            log_memory()
        
        # Check if we should stop due to max failures
        if cfg.indexing.max_failures > 0 and total_failures >= cfg.indexing.max_failures:
            break

    # Final index save
    store.version += 1
    store.save(config=cfg)

    # Write failed PDFs log
    if failed_pdfs:
        failed_log_path = idx_dir / "failed_pdfs.log"
        with open(failed_log_path, "w") as f:
            f.write(f"# Failed PDFs Log - {total_failures} failures\n")
            f.write(f"# Generated: {datetime.datetime.now()}\n\n")
            for pdf_path, error_msg in failed_pdfs:
                f.write(f"{pdf_path}\n  Error: {error_msg}\n\n")
        console.print(f"\n[yellow]Failed PDFs logged to: {failed_log_path}[/yellow]")

    # Final summary
    console.print("\n" + "="*60)
    console.print("[bold]Indexing Summary[/bold]")
    console.print("="*60)
    console.print(f"Total PDFs processed: {total_files}")
    console.print(f"  [green]✓ Successful: {total_successes}[/green]")
    console.print(f"  [red]✗ Failed: {total_failures}[/red]")
    console.print(f"Total chunks indexed: {total_chunks}\n")
    console.print(f"Index version: {store.version}")
    console.print("="*60)


# -- query -----------------------------------------------------------------
@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Retrieval only, skip LLM"),
    index_dir: str = typer.Option(None, "--index-dir"),
    llm_mode: str = typer.Option(None, "--llm-mode", help="openai or local"),
    llm_model: str = typer.Option(None, "--llm-model"),
    api_base: str = typer.Option(None, "--api-base"),
) -> None:
    """Query the indexed papers."""
    from paper_rag.retriever import Retriever

    cfg = PaperRAGConfig()
    if index_dir:
        cfg.index_dir = index_dir
    if llm_mode:
        cfg.llm.mode = llm_mode  # type: ignore[assignment]
    if llm_model:
        cfg.llm.model_name = llm_model
    if api_base:
        cfg.llm.api_base = api_base

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

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
            r.text[:120].replace("\n", " ") + "...",
        )
    console.print(table)

    if no_llm:
        return

    from paper_rag.llm import generate_answer

    context_chunks = [r.text for r in results]
    try:
        answer = generate_answer(question, context_chunks, cfg.llm)
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(answer)

        # Display references mapping
        console.print("\n[bold]References:[/bold]")
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
        console.print(f"\n[dim]💡 {exc}[/dim]")
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")


# -- evaluate --------------------------------------------------------------
@app.command()
def evaluate(
    benchmark_file: str = typer.Argument(..., help="JSONL benchmark file"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    index_dir: str = typer.Option(None, "--index-dir"),
) -> None:
    """Evaluate retrieval quality using a JSONL benchmark.

    Each line: {"question": "...", "relevant_documents": ["path1", ...]}
    """
    from paper_rag.retriever import Retriever

    cfg = PaperRAGConfig()
    if index_dir:
        cfg.index_dir = index_dir

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    from evaluate_paper import evaluate as run_eval

    results = run_eval(
        benchmark_file,
        retriever_fn=lambda q: retriever.retrieve_file_paths(q, top_k=top_k),
        k=top_k,
    )

    console.print("\n[bold]Evaluation Results[/bold]")
    for metric, value in results.items():
        console.print(f"  {metric}: {value:.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
