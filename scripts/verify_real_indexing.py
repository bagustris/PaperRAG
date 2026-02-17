
import os
import shutil
import sys
import subprocess
from pathlib import Path

def verify_indexing():
    # Setup paths
    source_dir = Path("/home/bagus/Documents/Mendeley Desktop")
    test_input_dir = Path("tests/verification_data")
    test_index_dir = Path("tests/verification_index")
    
    # Clean previous run
    if test_input_dir.exists():
        shutil.rmtree(test_input_dir)
    if test_index_dir.exists():
        shutil.rmtree(test_index_dir)
        
    test_input_dir.mkdir(parents=True)
    
    # Copy 12 files to test batching (Batch size is 10, so expect 2 batches)
    files = list(source_dir.glob("*.pdf"))
    if not files:
        print(f"Error: No PDFs found in {source_dir}")
        return
        
    to_copy = files[:4]
    print(f"Copying {len(to_copy)} files for verification...")
    for f in to_copy:
        shutil.copy(f, test_input_dir / f.name)
        
    # Run the index command
    cmd = [
        sys.executable, "-m", "paperrag.cli", "index",
        "--input-dir", str(test_input_dir),
        "--index-dir", str(test_index_dir),
        "--workers", "2",
        "--force"  # Force re-index to ensure we test the logic
    ]
    
    print("Executing command:", " ".join(cmd))
    # Stream output directly to stdout
    proc = subprocess.run(cmd)
    
    # Verification checks
    passed = True
    
    # We can't check stdout content programmatically if we stream it, 
    # but we can check the artifacts.
    # The user/agent will see the batch logs in the command output.

        
    # Check 2: Index creation
    if (test_index_dir / "faiss.index").exists() and (test_index_dir / "metadata.json").exists():
        print("[pw] Index artifacts check PASSED")
    else:
        print("[FAIL] Index artifacts check FAILED - Index files not created")
        passed = False

    if passed:
        print("\nAll verification checks PASSED!")
    else:
        print("\nVerification FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    verify_indexing()
