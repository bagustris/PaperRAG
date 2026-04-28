# Changelog 

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Major.Minor.Patch] - YYYY-MM-DD  

## [0.7.0] - 2026-04-28  
- Added finetuning script  

## [0.6.2] - 2026-04-20  
- Added retrieval-only query mode via `paperrag query --no-llm` to inspect raw retrieved chunks without calling an LLM.
- Added REPL retrieval-only toggle via `/no-llm`, with explicit `/no-llm on|off` controls.
- Updated README and CLI/usage docs to document retrieval-only mode in both one-shot query and REPL workflows.

## [0.6.1] - 2026-04-07
- Fixed a bug creating copy of index directory
- Expanded history to previous sessions in REPL 
- Added path completion for REPL commands and file paths
- Fixed wrong information about GPU use 

## [0.6.0] - 2026-04-06
- Added review-session export with `/export` and `--output`.
- Added prompt presets plus custom prompt overrides via `/preset`, `/prompt`, `--preset`, and `--system-prompt`.
- Added auto-discovery for `.paperrag-index` and a default input directory for local paper collections.
- Added query-result export to Markdown, CSV, or JSON.

## [0.5.0] - 2026-04-04
- Added autofocus mode for single PDF input  
- Fixed double reference numbers in a single sources (like [1], [2]) 

## [0.4.3] - 2026-04-03
- Added dynamic system prompt support via `--prompt` / `/prompt`.
- Added `/ctx-size` command to REPL and `--ctx-size` to CLI for manual context window control.
- Added `/focus` and `/focus list` to REPL for targeting specific papers.
- Updated default `max-tokens` to 256.
- Fixed legacy `n_ctx` migration from older config snapshots.
- Added pytest filters for third-party SWIG/Faiss warnings.
- Fixed a bug in index with -i (now it require -d)

## [0.4.2] - 2026-04-03
- Added `--temp` argument alias for `--temperature`.
- Updated docs to reflect new `review` command and llama.cpp backend support.

## [0.4.1] - 2026-04-02
- Added /ctx-size command to REPL to show current context size of the index (keep backward compatibility with older versions of the index  via n-ctx)

## [0.4.0] - 2026-04-01
- Added review mode (REPL) for interactive querying and testing of the index 

## [0.3.1] - 2026-02-18  
- Added Sources, removed References in REPL 
- Auto-detect GPU 

## [0.3.0] - 2026-02-17  
- Added documentation (`docs`)
- Added RC support (.paperragrc) 

## [0.2.0] - 2026-02-16  
- Added support for files, in addition to folders  

## [0.1.0] - 2026-2-15  
- Added plan (prompt) 
- Added --index-dir argument (for query/REPL)  
- Added --data-dir argument (only for index)  
