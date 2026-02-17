"""Typer CLI for PaperRAG."""

from __future__ import annotations

import datetime
import gc
import json
import logging
import re
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

from paperrag.config import PaperRAGConfig
from paperrag import __version__

app = typer.Typer(
    name="paperrag",
    help="PaperRAG - local RAG for academic PDFs.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paperrag")

MIT_LICENSE = """
MIT License

Copyright (c) 2024 PaperRAG Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def version_callback(value: bool) -> None:
    """Display version and license information."""
    if value:
        console.print(f"[bold]PaperRAG[/bold] version [cyan]{__version__}[/cyan]")
        console.print(MIT_LICENSE)
        raise typer.Exit()



@app.callback(invoke_without_command=True)
def entrypoint(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version and license"),
    input_dir: str = typer.Option(None, "--input-dir", "-d", help="PDF directory or single PDF file to index"),
    index_dir: str = typer.Option(None, "--index-dir", "-i", help="Index directory (will auto-discover .paperrag-index subdirectory if needed)"),
    topk: int = typer.Option(None, "--topk", "-k", help="Number of chunks to retrieve for context (default: 3)"),
    model: str = typer.Option(None, "--model", "-m", help="LLM model name (e.g., qwen3:1.7b)"),
    threshold: float = typer.Option(None, "--threshold", "-t", help="Minimum similarity score threshold (0.0-1.0, default: 0.15)"),
    temperature: float = typer.Option(None, "--temperature", help="LLM temperature (0.0-2.0, default: 0.0)"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="LLM max output tokens (default: 512)"),
) -> None:
    """PaperRAG - local RAG for academic PDFs."""
    if ctx.invoked_subcommand is None:
        from paperrag.repl import start_repl

        # REPL mode requires index_dir
        if not index_dir:
            console.print("[red]Error: --index-dir is required for REPL mode[/red]")
            console.print("Usage: paperrag --index-dir <path> [options]")
            raise typer.Exit(1)

        cfg = PaperRAGConfig()

        # Auto-discover index location
        from paperrag.vectorstore import VectorStore

        if index_dir:
            index_path = Path(index_dir).resolve()

            # Check if index_dir points directly to an index
            if not (index_path / "version.json").exists():
                # Try subdirectory convention
                subdir_path = index_path / ".paperrag-index"
                if (subdir_path / "version.json").exists():
                    console.print(f"[dim]Found index at {subdir_path}[/dim]")
                    index_path = subdir_path
                else:
                    console.print(f"[red]No index found at {index_path} or {subdir_path}[/red]")
                    raise typer.Exit(1)

            cfg.index_dir = str(index_path)

        # Load config snapshot from discovered index
        if index_dir and VectorStore.exists(Path(cfg.index_dir)):
            snapshot_file = Path(cfg.index_dir) / "config_snapshot.json"
            if snapshot_file.exists():
                try:
                    loaded_cfg = PaperRAGConfig.load_snapshot(snapshot_file)
                    # Apply input_dir from snapshot (but keep CLI overrides)
                    if not input_dir:  # Only use snapshot if user didn't specify input_dir
                        cfg.input_dir = loaded_cfg.input_dir
                        console.print(f"[dim]Loaded PDF directory from index: {cfg.input_dir}[/dim]")
                except Exception as e:
                    logger.warning("Could not load config snapshot: %s", e)

        # Apply input_dir CLI override if specified
        if input_dir:
            cfg.input_dir = input_dir

        if topk is not None:
            cfg.retriever.top_k = topk

        if model:
            cfg.llm.model_name = model

        if threshold is not None:
            cfg.retriever.score_threshold = threshold

        if temperature is not None:
            cfg.llm.temperature = temperature

        if max_tokens is not None:
            cfg.llm.max_tokens = max_tokens

        start_repl(cfg)


# -- index -----------------------------------------------------------------
@app.command()
def index(
    input_dir: str = typer.Option(None, "--input-dir", "-d", help="PDF directory or single PDF file"),
    index_dir: str = typer.Option(None, "--index-dir", "-i", help="Index directory (default: <input-dir>/.paperrag-index)"),
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
    ocr: str = typer.Option(
        "auto",
        "--ocr",
        help="OCR mode: 'auto' (detect per PDF, recommended), 'always' (force), 'never' (disable)"
    ),
    manifest: str = typer.Option(
        None,
        "--manifest",
        help="CSV manifest file with columns: filename,title,authors,abstract,doi (speeds up indexing)"
    ),
) -> None:
    """Index PDF files into the FAISS vector store."""
    from paperrag.chunker import chunk_paper
    from paperrag.embedder import Embedder
    from paperrag.parser import compute_file_hashes_parallel, discover_pdfs, parse_pdf, load_manifest
    from paperrag.parallel import parallel_process_pdfs
    from paperrag.vectorstore import VectorStore

    cfg = PaperRAGConfig()
    if input_dir:
        cfg.input_dir = input_dir
    if index_dir:
        cfg.index_dir = index_dir
    if checkpoint_interval is not None:
        cfg.indexing.checkpoint_interval = checkpoint_interval
    if workers is not None:
        cfg.indexing.n_workers = workers
    
    # Set OCR mode with validation
    ocr_lower = ocr.lower()
    if ocr_lower in ["auto", "always", "never"]:
        cfg.parser.ocr_mode = ocr_lower  # type: ignore[assignment]
        if ocr_lower == "auto":
            console.print("[cyan]🔍 Adaptive OCR enabled (auto-detect per PDF)[/cyan]")
        elif ocr_lower == "never":
            console.print("[yellow]⚡ OCR disabled for all PDFs[/yellow]")
        elif ocr_lower == "always":
            console.print("[yellow]📄 OCR enabled for all PDFs[/yellow]")
    else:
        console.print(f"[red]Invalid OCR mode: {ocr}. Use 'auto', 'always', or 'never'.[/red]")
        raise typer.Exit(1)

    pdf_dir = Path(cfg.input_dir)
    idx_dir = Path(cfg.index_dir)

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        raise typer.Exit(1)

    is_single_file = pdf_dir.is_file()
    if is_single_file:
        console.print(f"Indexing single PDF: [green]{pdf_dir.name}[/green]")
    else:
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

    # Remove deleted files from the index (skip for single-file mode to avoid purging other files)
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
        # Ensure config snapshot is saved even if index is up-to-date
        # This helps persist input_dir for future REPL sessions
        cfg.save_snapshot(idx_dir / "config_snapshot.json")
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
    
    # Show memory and worker info for transparency
    try:
        mem = psutil.virtual_memory()
        console.print(
            f"Using [green]{n_workers}[/green] worker(s) for parallel processing "
            f"([dim]{mem.available / (1024**3):.1f}GB RAM available[/dim])"
        )
        if mem.available < 4 * 1024**3:  # Less than 4GB available
            console.print(
                "[yellow]⚠ Low memory detected. If indexing fails, try: --workers 2[/yellow]"
            )
    except Exception:
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
        
        # Memory guard: warn if running low before processing batch
        try:
            mem = psutil.virtual_memory()
            if mem.available < 2 * 1024**3:  # Less than 2GB available
                console.print(
                    f"[yellow]⚠ Low memory warning: {mem.available / (1024**3):.1f}GB available. "
                    "Consider reducing --workers if indexing fails.[/yellow]"
                )
        except Exception:
            pass
        
        log_memory()
        
        # Parallel parse + chunk phase for this batch
        parsed_results = parallel_process_pdfs(
            batch,
            cfg.parser,
            cfg.chunker,
            n_workers,
            timeout=cfg.indexing.pdf_timeout
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
                console.print(
                    f"✓ Checkpoint saved: [cyan]{total_successes}/{total_files}[/cyan] PDFs indexed "
                    f"([dim]{total_chunks} total chunks[/dim])"
                )
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
    threshold: float = typer.Option(None, "--threshold", "-t", help="Minimum similarity score threshold (0.0-1.0)"),
    temperature: float = typer.Option(None, "--temperature", help="LLM temperature (0.0-2.0, default: 0.0)"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="LLM max output tokens (default: 512)"),
    input_dir: str = typer.Option(None, "--input-dir", "-d", help="PDF directory or single PDF file"),
    index_dir: str = typer.Option(None, "--index-dir", "-i", help="Index directory (required)"),
    model: str = typer.Option(None, "--model", "-m", help="LLM model name (e.g., qwen3:1.7b)"),
) -> None:
    """Query the indexed papers."""
    from paperrag.retriever import Retriever

    # Query mode requires index_dir
    if not index_dir:
        console.print("[red]Error: --index-dir is required for query command[/red]")
        console.print("Usage: paperrag query <question> --index-dir <path> [options]")
        raise typer.Exit(1)

    cfg = PaperRAGConfig()
    if input_dir:
        cfg.input_dir = input_dir
    elif index_dir:
        # Try to load from index snapshot if input_dir not provided
        snapshot_path = Path(index_dir) / "config_snapshot.json"
        if snapshot_path.exists():
            try:
                loaded_cfg = PaperRAGConfig.load_snapshot(snapshot_path)
                # Use loaded input_dir
                cfg.input_dir = loaded_cfg.input_dir
            except Exception:
                pass  # Fallback to default if load fails

    if index_dir:
        cfg.index_dir = index_dir

    if model:
        cfg.llm.model_name = model

    if threshold is not None:
        cfg.retriever.score_threshold = threshold

    if temperature is not None:
        cfg.llm.temperature = temperature

    if max_tokens is not None:
        cfg.llm.max_tokens = max_tokens

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    import time
    t0 = time.perf_counter()
    results = retriever.retrieve(question, top_k=top_k)
    t_retrieval = time.perf_counter() - t0

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    import re
    import sys
    from paperrag.llm import stream_answer
    from pathlib import Path as PathlibPath

    context_chunks = [r.text for r in results]
    try:
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
        sys.stdout.write("\n")
        sys.stdout.flush()
        t_llm = time.perf_counter() - t1
        answer = full_answer.strip()

        # Extract cited reference numbers from the streamed answer (keep original numbers
        # so they match what was already printed to the terminal).
        cited_nums = sorted(set(
            int(m) for m in re.findall(r'\[(\d+)\]', answer)
            if int(m) <= len(results)
        ))

        # Group citation numbers by unique filename (same file may be cited multiple times)
        file_to_nums: dict[str, list[int]] = {}
        for num in cited_nums:
            filename = PathlibPath(results[num - 1].file_path).name
            file_to_nums.setdefault(filename, []).append(num)

        # Display references
        console.print("\n[bold]References:[/bold]")
        if file_to_nums:
            for filename, nums in file_to_nums.items():
                nums_str = "".join(f"[{n}]" for n in nums)
                console.print(f"  [cyan]{nums_str}[/cyan] {filename}")
        else:
            # Fallback: LLM produced no inline citations — list all retrieved files
            seen: set[str] = set()
            ref_num = 1
            for r in results:
                filename = PathlibPath(r.file_path).name
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
        console.print(f"\n[dim]💡 {exc}[/dim]")
    except Exception as exc:
        from paperrag.llm import describe_llm_error
        error_msg, hint = describe_llm_error(exc, cfg.llm.model_name)
        console.print(f"[red]{error_msg}[/red]")
        if hint:
            console.print(f"[yellow]Fix: {hint}[/yellow]")


# -- evaluate --------------------------------------------------------------
@app.command()
def evaluate(
    benchmark_file: str = typer.Argument(..., help="JSONL benchmark file"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    input_dir: str = typer.Option(None, "--input-dir", "-d", help="PDF directory or single PDF file"),
    index_dir: str = typer.Option(None, "--index-dir", "-i", help="Index directory (default: <input-dir>/.paperrag-index)"),
) -> None:
    """Evaluate retrieval quality using a JSONL benchmark.

    Each line: {"question": "...", "relevant_documents": ["path1", ...]}
    """
    from paperrag.retriever import Retriever

    cfg = PaperRAGConfig()
    if input_dir:
        cfg.input_dir = input_dir
    if index_dir:
        cfg.index_dir = index_dir

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    from paperrag.benchmark import evaluate as run_eval

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
