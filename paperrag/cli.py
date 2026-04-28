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

from paperrag.config import PaperRAGConfig, load_rc, apply_rc, PROMPT_PRESETS, PRESET_MAX_TOKENS
from paperrag import __version__

EXAMPLES_EPILOG = (
    "Examples:\n\n"
    "  paperrag                              # auto-discover index from CWD\n\n"
    "  paperrag --index-dir /path/to/index   # REPL with a specific index\n\n"
    "  paperrag index --input-dir ./papers   # index PDFs first\n\n"
    "  paperrag query \"What is attention?\"   # one-shot query\n\n"
    "  paperrag query \"What is attention?\" --no-llm   # raw retrieval results\n\n"
    "  paperrag review paper.pdf             # index + review a single PDF\n"
)

app = typer.Typer(
    name="paperrag",
    help="PaperRAG - local RAG for academic PDFs.",
    epilog=EXAMPLES_EPILOG,
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


def _print_gpu_info() -> None:
    """Detect and display GPU availability with an Ollama inference hint."""
    import subprocess
    import platform

    # --- NVIDIA: use nvidia-smi (same detection path as Ollama) ---
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().splitlines()[0]
            console.print(
                f"[green]GPU detected:[/green] {gpu_name} — Ollama will use it automatically for faster inference"
            )
            return
    except Exception:
        pass

    # --- Apple Silicon MPS ---
    if platform.system() == "Darwin":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                console.print(
                    "[green]GPU detected:[/green] Apple Silicon MPS — Ollama will use it automatically"
                )
                return
        except Exception:
            pass

    console.print("[dim]Running on CPU, No GPU detected[/dim]")


@app.callback(invoke_without_command=True)
def entrypoint(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and license",
    ),
    input_dir: str = typer.Option(
        None, "--input-dir", "-d", help="PDF directory or single PDF file to index"
    ),
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (will auto-discover .paperrag-index subdirectory if needed)",
    ),
    topk: int = typer.Option(
        None,
        "--top-k",
        "--topk",
        "-k",
        help="Number of chunks to retrieve for context (default: 3)",
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="LLM model name (e.g., qwen3:1.7b)"
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Minimum similarity score threshold (0.0-1.0, default: 0.15)",
    ),
    temperature: float = typer.Option(
        None, "--temperature", "--temp", help="LLM temperature (0.0-2.0, default: 0.0)"
    ),
    max_tokens: int = typer.Option(
        None, "--max-tokens", help="LLM max output tokens (default: 256)"
    ),
    ctx_size: int = typer.Option(
        None, "--ctx-size", min=512, help="LLM context window size (default: 2048)"
    ),
    system_prompt: str = typer.Option(
        None, "--system-prompt", "--prompt", help="LLM system prompt"
    ),
    think: bool = typer.Option(
        False, "--think/--no-think", help="Enable thinking/reasoning mode for supported models (e.g. Qwen3)"
    ),
) -> None:
    """PaperRAG - local RAG for academic PDFs.

    Starts an interactive REPL session using an existing index.
    """
    if ctx.invoked_subcommand is None:
        from paperrag.repl import start_repl

        if input_dir:
            console.print(
                "[yellow]⚠ Warning: --input-dir / -d does not auto-index PDFs when "
                "starting the REPL directly.\n"
                "  It may still be used by REPL features (for example, PDF discovery "
                "or as the default target for /index).\n"
                "  To index PDFs immediately run: paperrag index --input-dir <path>\n"
                "  To start the REPL with a specific index use: paperrag --index-dir "
                "<path>[/yellow]"
            )

        cfg = PaperRAGConfig()

        # Load .paperragrc: global first, then local overrides
        global_rc = load_rc(Path.home() / ".paperragrc")
        local_rc = load_rc(Path.cwd() / ".paperragrc")
        apply_rc(cfg, global_rc)
        apply_rc(cfg, local_rc)

        # Resolve effective index_dir: CLI arg takes priority over RC
        effective_index_dir = index_dir or cfg._index_dir

        # Auto-discover index from CWD if no explicit index_dir given
        from paperrag.vectorstore import VectorStore

        if not effective_index_dir:
            cwd = Path.cwd()
            for candidate in [cwd / ".paperrag-index", cwd]:
                if (candidate / "version.json").exists():
                    effective_index_dir = str(candidate)
                    console.print(f"[dim]Using index at {candidate}[/dim]")
                    break
            else:
                console.print(
                    "[red]Error: no index found. Pass --index-dir or run paperrag from an indexed folder.[/red]"
                )
                console.print(
                    "[dim]Tip: set index-dir in ~/.paperragrc to skip this flag[/dim]"
                )
                raise typer.Exit(1)

        index_path = Path(effective_index_dir).resolve()

        # Check if index_dir points directly to an index
        if not (index_path / "version.json").exists():
            # Try subdirectory convention
            subdir_path = index_path / ".paperrag-index"
            if (subdir_path / "version.json").exists():
                console.print(f"[dim]Found index at {subdir_path}[/dim]")
                index_path = subdir_path
            else:
                console.print(
                    f"[red]No index found at {index_path} or {subdir_path}[/red]"
                )
                raise typer.Exit(1)

        cfg.index_dir = str(index_path)

        # Load config snapshot from discovered index
        if VectorStore.exists(Path(cfg.index_dir)):
            snapshot_file = Path(cfg.index_dir) / "config_snapshot.json"
            if snapshot_file.exists():
                try:
                    loaded_cfg = PaperRAGConfig.load_snapshot(snapshot_file)
                    # Apply input_dir from snapshot (but keep CLI overrides)
                    if (
                        not input_dir
                    ):  # Only use snapshot if user didn't specify input_dir
                        cfg.input_dir = loaded_cfg.input_dir
                        console.print(
                            f"[dim]Loaded PDF directory from index: {cfg.input_dir}[/dim]"
                        )
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

        if ctx_size is not None:
            cfg.llm.ctx_size = ctx_size

        if system_prompt:
            cfg.llm.system_prompt = system_prompt

        if think:
            cfg.llm.think = think

        _print_gpu_info()
        start_repl(cfg)


# -- index -----------------------------------------------------------------
@app.command()
def index(
    input_dir: str = typer.Option(
        None, "--input-dir", "-d", help="PDF directory or single PDF file"
    ),
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (default: <input-dir>/.paperrag-index)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    checkpoint_interval: int = typer.Option(
        None,
        "--checkpoint-interval",
        "-c",
        help="Save index every N PDFs (0 to disable checkpointing)",
    ),
    workers: int = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers (0 = auto-detect)"
    ),
    ocr: str = typer.Option(
        "auto",
        "--ocr",
        help="OCR mode: 'auto' (detect per PDF, recommended), 'always' (force), 'never' (disable)",
    ),
    manifest: str = typer.Option(
        None,
        "--manifest",
        help="CSV manifest file with columns: filename,title,authors,abstract,doi (speeds up indexing)",
    ),
    embed_model: str = typer.Option(
        None,
        "--embed-model",
        help="Sentence-Transformers model name or local path for embedding (default: sentence-transformers/all-MiniLM-L6-v2)",
    ),
) -> None:
    """Index PDF files into the FAISS vector store."""
    from paperrag.chunker import chunk_paper
    from paperrag.embedder import Embedder
    from paperrag.parser import (
        compute_file_hashes_parallel,
        discover_pdfs,
        parse_pdf,
        load_manifest,
    )
    from paperrag.parallel import parallel_process_pdfs
    from paperrag.vectorstore import VectorStore

    cfg = PaperRAGConfig()

    # Load .paperragrc: global first, then local overrides
    global_rc = load_rc(Path.home() / ".paperragrc")
    local_rc = load_rc(Path.cwd() / ".paperragrc")
    apply_rc(cfg, global_rc)
    apply_rc(cfg, local_rc)

    if input_dir:
        cfg.input_dir = input_dir
    elif not global_rc.get("input-dir") and not local_rc.get("input-dir"):
        console.print("[red]Error: --input-dir (-d) is required[/red]")
        console.print("Usage: paperrag index --input-dir <path> [--index-dir <path>]")
        console.print(
            "[dim]Tip: set input-dir in ~/.paperragrc to skip this flag[/dim]"
        )
        raise typer.Exit(1)

    if index_dir:
        cfg.index_dir = index_dir
    if checkpoint_interval is not None:
        cfg.indexing.checkpoint_interval = checkpoint_interval
    if workers is not None:
        cfg.indexing.n_workers = workers
    if embed_model:
        cfg.embedder.model_name = embed_model
        console.print(f"[cyan]Embed model: {cfg.embedder.model_name}[/cyan]")

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
        console.print(
            f"[red]Invalid OCR mode: {ocr}. Use 'auto', 'always', or 'never'.[/red]"
        )
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

    _print_gpu_info()
    embedder = Embedder(cfg.embedder)

    # Load or create store
    if VectorStore.exists(idx_dir) and not force:
        store = VectorStore.load(idx_dir)
        if store.dimension != embedder.dimension:
            console.print("[yellow]Dimension mismatch - rebuilding index.[/yellow]")
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
            console.print(
                f"Removed [red]{len(stale)}[/red] deleted file(s) from index."
            )

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

    console.print(
        f"Indexing [cyan]{total_files}[/cyan] PDF(s) in batches of {BATCH_SIZE}..."
    )

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
        console.print(
            f"Using [green]{n_workers}[/green] worker(s) for parallel processing."
        )

    # Helper function to log memory usage
    def log_memory():
        if cfg.indexing.log_memory_usage:
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(
                "Memory usage: RSS=%.1f MB, VMS=%.1f MB",
                mem_info.rss / 1024 / 1024,
                mem_info.vms / 1024 / 1024,
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
            batch, cfg.parser, cfg.chunker, n_workers, timeout=cfg.indexing.pdf_timeout
        )

        # Sequential embed + add phase with comprehensive error handling
        batch_chunks = 0
        batch_successes = 0
        batch_failures = 0

        for pdf_path, file_hash, chunks, error in tqdm(
            parsed_results, desc="Embedding", unit="file"
        ):
            # Handle parsing errors
            if error:
                logger.error("Failed to parse %s: %s", pdf_path.name, error)
                failed_pdfs.append((str(pdf_path), f"Parse error: {error}"))
                batch_failures += 1
                total_failures += 1

                # Check if we've exceeded max failures
                if (
                    cfg.indexing.max_failures > 0
                    and total_failures >= cfg.indexing.max_failures
                ):
                    console.print(
                        f"[red]Reached maximum failures ({cfg.indexing.max_failures}). Stopping.[/red]"
                    )
                    break

                if not cfg.indexing.continue_on_error:
                    console.print(
                        "[red]Stopping due to error (continue_on_error=False)[/red]"
                    )
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
                logger.error(
                    "Failed to embed/store %s: %s", pdf_path.name, e, exc_info=True
                )
                failed_pdfs.append((str(pdf_path), f"Embedding error: {e}"))
                batch_failures += 1
                total_failures += 1

                # Check if we've exceeded max failures
                if (
                    cfg.indexing.max_failures > 0
                    and total_failures >= cfg.indexing.max_failures
                ):
                    console.print(
                        f"[red]Reached maximum failures ({cfg.indexing.max_failures}). Stopping.[/red]"
                    )
                    break

                if not cfg.indexing.continue_on_error:
                    console.print(
                        "[red]Stopping due to error (continue_on_error=False)[/red]"
                    )
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
                    logger.warning(
                        "Failed to save index (attempt %d/%d): %s",
                        retry + 1,
                        max_retries,
                        e,
                    )
                else:
                    logger.error(
                        "Failed to save index after %d attempts: %s", max_retries, e
                    )

        if not save_success:
            console.print("[red]Critical: Failed to save index checkpoint![/red]")
            if not cfg.indexing.continue_on_error:
                raise typer.Exit(1)

        # Memory cleanup between batches
        if cfg.indexing.enable_gc_per_batch:
            gc.collect()
            log_memory()

        # Check if we should stop due to max failures
        if (
            cfg.indexing.max_failures > 0
            and total_failures >= cfg.indexing.max_failures
        ):
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
    console.print("\n" + "=" * 60)
    console.print("[bold]Indexing Summary[/bold]")
    console.print("=" * 60)
    console.print(f"Total PDFs processed: {total_files}")
    console.print(f"  [green]✓ Successful: {total_successes}[/green]")
    console.print(f"  [red]✗ Failed: {total_failures}[/red]")
    console.print(f"Total chunks indexed: {total_chunks}\n")
    console.print(f"Index version: {store.version}")
    console.print("=" * 60)


# -- review ----------------------------------------------------------------
@app.command()
def review(
    input_path: str = typer.Argument(..., help="PDF file or directory to review"),
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (default: auto-derived from input path)",
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="LLM model name (e.g., qwen3:1.7b)"
    ),
    topk: int = typer.Option(
        None,
        "--top-k",
        "--topk",
        "-k",
        help="Number of chunks to retrieve for context (default: 3)",
    ),
    threshold: float = typer.Option(
        None, "--threshold", "-t", help="Minimum similarity score threshold (0.0-1.0)"
    ),
    temperature: float = typer.Option(
        None, "--temperature", "--temp", help="LLM temperature (0.0-2.0, default: 0.0)"
    ),
    max_tokens: int = typer.Option(
        None, "--max-tokens", help="LLM max output tokens (default: 256)"
    ),
    ctx_size: int = typer.Option(
        None, "--ctx-size", min=512, help="LLM context window size (default: 2048)"
    ),
    system_prompt: str = typer.Option(
        None, "--system-prompt", "--prompt", help="LLM system prompt"
    ),
    preset: str = typer.Option(
        None,
        "--preset",
        "-p",
        help=f"Named prompt preset: {', '.join(PROMPT_PRESETS.keys())}",
    ),
    n_gpu_layers: int = typer.Option(
        None, "--n-gpu-layers", "--ngl", help="GPU layers for llama.cpp (default: 0 = CPU)"
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Save Q&A session to this markdown file on exit",
    ),
    think: bool = typer.Option(
        False, "--think/--no-think", help="Enable thinking/reasoning mode for supported models (e.g. Qwen3)"
    ),
) -> None:
    """Index a PDF file (or directory) and start an interactive review session.

    Convenience command for focused paper review — equivalent to running:

        paperrag index --input-dir <path> && paperrag --index-dir <auto>

    Examples:

        paperrag review paper.pdf

        paperrag review paper.pdf --preset reviewer

        paperrag review paper.pdf --preset reviewer --output review.md

        paperrag review ./papers/ --topk 5

        paperrag review paper.pdf --index-dir /tmp/my-index
    """
    from paperrag.repl import _handle_index, start_repl

    path_obj = Path(input_path)
    if not path_obj.exists():
        console.print(f"[red]Error: Path does not exist: {input_path}[/red]")
        raise typer.Exit(1)

    cfg = PaperRAGConfig()

    # Load .paperragrc: global first, then local overrides
    global_rc = load_rc(Path.home() / ".paperragrc")
    local_rc = load_rc(Path.cwd() / ".paperragrc")
    apply_rc(cfg, global_rc)
    apply_rc(cfg, local_rc)

    cfg.input_dir = str(path_obj)
    if index_dir:
        cfg.index_dir = index_dir
    else:
        # Clear any RC-set index_dir so it auto-derives from input_path
        cfg._index_dir = None

    if model:
        cfg.llm.model_name = model
    if topk is not None:
        cfg.retriever.top_k = topk
    if threshold is not None:
        cfg.retriever.score_threshold = threshold
    if temperature is not None:
        cfg.llm.temperature = temperature
    if max_tokens is not None:
        cfg.llm.max_tokens = max_tokens
    if ctx_size is not None:
        cfg.llm.ctx_size = ctx_size
    if system_prompt:
        cfg.llm.system_prompt = system_prompt
    if n_gpu_layers is not None:
        cfg.llm.n_gpu_layers = n_gpu_layers
    if think:
        cfg.llm.think = think

    # Apply named preset (--system-prompt takes priority over --preset)
    if preset is not None:
        preset_lower = preset.lower()
        if preset_lower not in PROMPT_PRESETS:
            console.print(f"[red]Unknown preset '{preset}'. Valid: {', '.join(PROMPT_PRESETS)}[/red]")
            raise typer.Exit(1)
        if not system_prompt:
            cfg.llm.system_prompt = PROMPT_PRESETS[preset_lower]
        if max_tokens is None:
            cfg.llm.max_tokens = PRESET_MAX_TOKENS.get(preset_lower, cfg.llm.max_tokens)
    else:
        # review mode: bump max_tokens to at least 512 for richer responses
        if max_tokens is None:
            cfg.llm.max_tokens = max(cfg.llm.max_tokens, 512)

    # Validate that PDFs can be found before indexing
    from paperrag.parser import discover_pdfs

    pdfs = discover_pdfs(path_obj)
    if not pdfs:
        console.print(f"[red]Error: No PDFs found at {input_path}[/red]")
        raise typer.Exit(1)

    _print_gpu_info()

    # Step 1: Index the content
    _handle_index(cfg)

    # Step 2: Start interactive review session (auto-focus when reviewing a single PDF)
    auto_focus = pdfs[0] if len(pdfs) == 1 else None
    output_path = Path(output) if output else None
    start_repl(cfg, auto_focus=auto_focus, review_mode=True, output_path=output_path)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(3, "--top-k", "-k"),
    threshold: float = typer.Option(
        None, "--threshold", "-t", help="Minimum similarity score threshold (0.0-1.0)"
    ),
    temperature: float = typer.Option(
        None, "--temperature", help="LLM temperature (0.0-2.0, default: 0.0)"
    ),
    max_tokens: int = typer.Option(
        None, "--max-tokens", help="LLM max output tokens (default: 256)"
    ),
    ctx_size: int = typer.Option(
        None, "--ctx-size", min=512, help="LLM context window size (default: 2048)"
    ),
    system_prompt: str = typer.Option(
        None, "--system-prompt", "--prompt", help="LLM system prompt"
    ),
    input_dir: str = typer.Option(
        None, "--input-dir", "-d", help="PDF directory or single PDF file"
    ),
    index_dir: str = typer.Option(
        None, "--index-dir", "-i", help="Index directory (required)"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="LLM model name (e.g., qwen3:1.7b)"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Return raw retrieval results without calling the LLM"
    ),
    think: bool = typer.Option(
        False, "--think/--no-think", help="Enable thinking/reasoning mode for supported models (e.g. Qwen3)"
    ),
) -> None:
    """Query the indexed papers."""
    from paperrag.retriever import Retriever

    cfg = PaperRAGConfig()

    # Load .paperragrc: global first, then local overrides
    global_rc = load_rc(Path.home() / ".paperragrc")
    local_rc = load_rc(Path.cwd() / ".paperragrc")
    apply_rc(cfg, global_rc)
    apply_rc(cfg, local_rc)

    # Resolve effective index_dir: CLI arg takes priority over RC
    effective_index_dir = index_dir or cfg._index_dir

    # Query mode requires index_dir (from CLI or .paperragrc)
    if not effective_index_dir:
        console.print("[red]Error: --index-dir is required for query command[/red]")
        console.print("Usage: paperrag query <question> --index-dir <path> [options]")
        console.print(
            "[dim]Tip: set index-dir in ~/.paperragrc to skip this flag[/dim]"
        )
        raise typer.Exit(1)

    if input_dir:
        cfg.input_dir = input_dir
    elif effective_index_dir:
        # Try to load from index snapshot if input_dir not provided
        snapshot_path = Path(effective_index_dir) / "config_snapshot.json"
        if snapshot_path.exists():
            try:
                loaded_cfg = PaperRAGConfig.load_snapshot(snapshot_path)
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

    if ctx_size is not None:
        cfg.llm.ctx_size = ctx_size

    if system_prompt:
        cfg.llm.system_prompt = system_prompt

    if think:
        cfg.llm.think = think

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

    from pathlib import Path as PathlibPath

    if no_llm:
        console.print(f"\n[bold]Retrieved Chunks[/bold] [dim]({t_retrieval:.2f}s)[/dim]")
        for i, result in enumerate(results, start=1):
            filename = PathlibPath(result.file_path).name
            snippet = re.sub(r"\s+", " ", result.text).strip()
            if len(snippet) > 200:
                snippet = snippet[:197].rstrip() + "..."
            console.print(
                f"  [cyan][{i}][/cyan] {filename} | {result.section_name} | "
                f"chunk {result.chunk_id} [dim]({result.score:.2f})[/dim]"
            )
            console.print(f"      {snippet}")
        console.print(f"\n[dim]Retrieval only: {t_retrieval:.2f}s[/dim]\n")
        return

    import sys
    from paperrag.llm import stream_answer

    # Show retrieved sources immediately so the user sees useful info
    # while waiting for the LLM to generate.
    console.print(f"\n[bold]Sources[/bold] [dim]({t_retrieval:.2f}s)[/dim]")
    seen_files: dict[str, int] = {}
    for r in results:
        if r.file_path not in seen_files:
            seen_files[r.file_path] = len(seen_files) + 1
    for file_path, label in seen_files.items():
        filename = PathlibPath(file_path).name
        best_score = max(r.score for r in results if r.file_path == file_path)
        console.print(
            f"  [cyan][{label}][/cyan] {filename} [dim]({best_score:.2f})[/dim]"
        )

    context_chunks = [r.text for r in results]
    source_files = [r.file_path for r in results]
    try:
        full_answer = ""
        header_printed = False
        t1 = time.perf_counter()
        for chunk in stream_answer(
            question, context_chunks, cfg.llm, source_files=source_files
        ):
            if not header_printed:
                console.print("\n[bold green]Answer:[/bold green]")
                header_printed = True
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_answer += chunk
        sys.stdout.write("\n")
        sys.stdout.flush()
        t_llm = time.perf_counter() - t1
        t_total = time.perf_counter() - t0
        console.print(
            f"\n[dim]Retrieval: {t_retrieval:.2f}s | LLM: {t_llm:.2f}s | Total: {t_total:.2f}s[/dim]\n"
        )

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
    top_k: int = typer.Option(3, "--top-k", "-k"),
    input_dir: str = typer.Option(
        None, "--input-dir", "-d", help="PDF directory or single PDF file"
    ),
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (default: <input-dir>/.paperrag-index)",
    ),
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


@app.command()
def export(
    query: str = typer.Option(None, "--query", "-q", help="Question to query"),
    output_path: str = typer.Option(..., "--output", help="Output file path"),
    format: str = typer.Option(
        "markdown", "--format", help="Export format (markdown, csv, json)"
    ),
    top_k: int = typer.Option(3, "--top-k", "-k"),
    threshold: float = typer.Option(
        None, "--threshold", "-t", help="Minimum similarity score threshold (0.0-1.0)"
    ),
    input_dir: str = typer.Option(
        None, "--input-dir", "-d", help="PDF directory or single PDF file"
    ),
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (default: <input-dir>/.paperrag-index)",
    ),
) -> None:
    """Export query results to a file.

    Retrieves and saves results in the specified format.
    """
    from paperrag.retriever import Retriever

    # Prompt for query if not provided
    if not query:
        query = typer.prompt("Question", type=str)

    if query.strip() == "":
        console.print("[red]Error: Question cannot be empty[/red]")
        raise typer.Exit(1)

    cfg = PaperRAGConfig()
    if input_dir:
        cfg.input_dir = input_dir
    if index_dir:
        cfg.index_dir = index_dir

    # Resolve effective index_dir
    if not cfg._index_dir:
        console.print("[red]Error: --index-dir is required for export command[/red]")
        raise typer.Exit(1)

    try:
        retriever = Retriever(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    # Get retrieval results
    results = retriever.retrieve(
        query,
        top_k=top_k,
    )

    # Convert results for export
    from paperrag.export import export_results

    output_path = Path(output_path)
    export_results(results, output_path, format)

    console.print(f"[green]✓ Exported {len(results)} results to {output_path}[/green]")


# -- status ----------------------------------------------------------
@app.command()
def status(
    index_dir: str = typer.Option(
        None,
        "--index-dir",
        "-i",
        help="Index directory (auto-discovered if not provided)",
    ),
) -> None:
    """Show index health information."""
    from paperrag.parser import compute_file_hash, discover_pdfs
    from paperrag.vectorstore import VectorStore

    cfg = PaperRAGConfig()

    if index_dir:
        cfg.index_dir = index_dir

    idx_path = Path(cfg.index_dir).resolve()

    if not VectorStore.exists(idx_path):
        console.print("[red]No index found at specified location.[/red]")
        console.print("[dim]Use 'paperrag index' to create an index first.[/dim]")
        raise typer.Exit(1)

    try:
        store = VectorStore.load(idx_path)
    except Exception as e:
        console.print(f"[red]Error loading index: {e}[/red]")
        raise typer.Exit(1)

    # Create status table
    table = Table(title="✓ [bold]Index Status[/bold]")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green")

    # Vector count
    table.add_row("Vectors in index", str(store.index.ntotal))
    table.add_row("Indexed PDFs", str(len(store.file_hashes)))
    table.add_row("Index version", str(store.version))
    table.add_row("Embedding dimension", str(store.dimension))

    # Disk size
    try:
        index_size = idx_path.stat().st_size
        if index_size >= 10 * 1024 * 1024:
            size_str = f"{index_size / (1024 * 1024):.1f} MB"
        elif index_size >= 1024 * 1024:
            size_str = f"{index_size / 1024 / 1024:.2f} MB"
        else:
            size_str = f"{index_size / 1024:.2f} KB"

        table.add_row("Estimated size", size_str)
    except Exception:
        table.add_row("Estimated size", "Unknown")

    # Timestamp
    try:
        timestamp = datetime.datetime.fromtimestamp(idx_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        table.add_row("Last modified", timestamp)
    except Exception:
        pass

    console.print("\n" + "=" * 60)
    console.print(table)
    console.print("=" * 60 + "\n")

    # Check for files out of sync
    if cfg.input_dir:
        pdf_dir = Path(cfg.input_dir)
        try:
            pdfs = discover_pdfs(pdf_dir)

            if pdfs:
                if not VectorStore.exists(idx_path / ".paperrag-index"):
                    console.print(
                        "[yellow]⚠ Index directory not found. Auto-discovery enabled.[/yellow]"
                    )
                    idx_path = idx_path / ".paperrag-index"
                    if not VectorStore.exists(idx_path):
                        console.print(
                            "[yellow]ℹ No index found in either location.[/yellow]"
                        )
                        raise typer.Exit(0)

                # Detect modified/deleted files
                current_paths = {str(p) for p in pdfs}
                stored_hashes = set(store.file_hashes.keys())

                modified = []
                deleted = []

                for pdf_path in pdfs:
                    path_str = str(pdf_path)
                    stored_hash = store.get_file_hash(path_str)

                    if stored_hash is None:
                        deleted.append(pdf_path.name)
                    else:
                        # Check if hash has changed by comparing against disk
                        disk_hash = compute_file_hash(pdf_path)
                        if stored_hash != disk_hash:
                            modified.append(pdf_path.name)

                if not modified and not deleted:
                    console.print("[green]✓ Index is fully up-to-date[/green]")
                else:
                    if modified:
                        console.print(
                            f"[yellow]⚠ {len(modified)} file(s) have been modified[/yellow]"
                        )
                        for name in modified[:5]:
                            console.print(f"    - {name}")
                        if len(modified) > 5:
                            console.print(f"    ... and {len(modified) - 5} more")

                    if deleted:
                        console.print(
                            f"[red]✗ {len(deleted)} file(s) have been deleted[/red]"
                        )
                        for name in deleted[:5]:
                            console.print(f"    - {name}")
                        if len(deleted) > 5:
                            console.print(f"    ... and {len(deleted) - 5} more")
            else:
                console.print("[yellow]ℹ No PDFs found in input directory.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error checking file sync: {e}[/red]")
    else:
        console.print(
            "[yellow]ℹ No input directory configured. Run 'paperrag index --help' to set one.[/yellow]"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
