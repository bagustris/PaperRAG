# Changelog 

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [XX.YY.ZZ: Major.Minor.Patch] - YYYY-MM-DD  

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
