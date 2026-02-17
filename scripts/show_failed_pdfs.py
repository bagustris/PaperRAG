#!/usr/bin/env python3
"""Show failed PDFs and missing PDFs from the index."""

import pickle
import sys
from pathlib import Path


def show_failed_pdfs(index_dir: str = "paperrag/my_index", input_dir: str = None):
    """Display failed and missing PDFs."""
    index_path = Path(index_dir)
    
    if not index_path.exists():
        print(f"Error: Index directory {index_dir} does not exist")
        sys.exit(1)
    
    print("=" * 70)
    print("FAILED & MISSING PDF REPORT")
    print("=" * 70)
    
    # 1. Check for explicit failure log
    failed_log = index_path / "failed_pdfs.log"
    if failed_log.exists():
        print("\n[EXPLICIT FAILURES - from failed_pdfs.log]")
        print("-" * 70)
        with open(failed_log, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("\n[EXPLICIT FAILURES]")
        print("-" * 70)
        print("✓ No failed_pdfs.log found - all attempted PDFs indexed successfully!")
    
    # 2. Check for missing PDFs (discovered but not indexed)
    if input_dir:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"\n⚠ Input directory {input_dir} does not exist - skipping missing PDF check")
        else:
            # Load indexed files
            hashes_file = index_path / "file_hashes.json"
            with open(hashes_file, 'rb') as f:
                indexed_files = set(pickle.load(f).keys())
            
            # Discover all PDFs in input directory
            all_pdfs = set(str(p) for p in input_path.rglob("*.pdf"))
            
            # Find missing PDFs
            missing = all_pdfs - indexed_files
            
            print("\n[MISSING PDFs - in input directory but not indexed]")
            print("-" * 70)
            if missing:
                print(f"Found {len(missing)} PDF(s) not in index:\n")
                for pdf in sorted(missing):
                    print(f"  ✗ {pdf}")
            else:
                print("✓ All PDFs from input directory are indexed!")
    
    # 3. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    hashes_file = index_path / "file_hashes.json"
    with open(hashes_file, 'rb') as f:
        indexed_count = len(pickle.load(f))
    
    print(f"Indexed files: {indexed_count}")
    
    if input_dir:
        input_path = Path(input_dir)
        if input_path.exists():
            total_pdfs = len(list(input_path.rglob("*.pdf")))
            print(f"Total PDFs in input dir: {total_pdfs}")
            print(f"Success rate: {indexed_count}/{total_pdfs} ({100*indexed_count/total_pdfs:.1f}%)")
    
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Try to read input_dir from config
        try:
            import json
            config = json.load(open("paperrag/my_index/config_snapshot.json"))
            input_dir = config.get("input_dir")
            show_failed_pdfs("paperrag/my_index", input_dir)
        except:
            print("Usage: python3 scripts/show_failed_pdfs.py [index_dir] [input_dir]")
            print("\nOr run without arguments to auto-detect from config_snapshot.json")
            sys.exit(1)
    elif len(sys.argv) == 2:
        show_failed_pdfs(sys.argv[1])
    elif len(sys.argv) == 3:
        show_failed_pdfs(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 scripts/show_failed_pdfs.py [index_dir] [input_dir]")
        sys.exit(1)
