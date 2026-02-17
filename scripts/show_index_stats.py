#!/usr/bin/env python
"""Show statistics about the current index."""

import pickle
import sys
from pathlib import Path

def show_stats(index_dir: str = "paperrag/my_index"):
    """Display index statistics."""
    index_path = Path(index_dir)
    
    if not index_path.exists():
        print(f"Error: Index directory {index_dir} does not exist")
        sys.exit(1)
    
    hashes_file = index_path / "file_hashes.json"
    metadata_file = index_path / "metadata.json"
    
    if not hashes_file.exists():
        print(f"Error: {hashes_file} not found")
        sys.exit(1)
    
    # Load file hashes (pickle format)
    with open(hashes_file, 'rb') as f:
        file_hashes = pickle.load(f)
    
    # Load chunks metadata (pickle format)
    with open(metadata_file, 'rb') as f:
        chunks = pickle.load(f)
    
    print("=" * 60)
    print("INDEX STATISTICS")
    print("=" * 60)
    print(f"Total indexed files: {len(file_hashes)}")
    print(f"Total chunks:        {len(chunks)}")
    print()
    print("Indexed files:")
    print("-" * 60)
    for fp in sorted(file_hashes.keys()):
        print(f"  {fp}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_stats(sys.argv[1])
    else:
        show_stats()
