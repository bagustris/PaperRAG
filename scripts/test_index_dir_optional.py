#!/usr/bin/env python3
"""Test script to verify --index-dir is optional and works correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paperrag.config import PaperRAGConfig


def test_default_index_dir():
    """Test that index_dir defaults to input_dir/.paperrag-index"""
    cfg = PaperRAGConfig()
    cfg.input_dir = "/test/pdfs"
    
    expected = "/test/pdfs/.paperrag-index"
    actual = cfg.index_dir
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Default index_dir works: {actual}")


def test_custom_index_dir():
    """Test that custom index_dir can be set"""
    cfg = PaperRAGConfig()
    cfg.input_dir = "/test/pdfs"
    cfg.index_dir = "/custom/index/path"
    
    expected = "/custom/index/path"
    actual = cfg.index_dir
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Custom index_dir works: {actual}")


def test_index_dir_persistence():
    """Test that custom index_dir persists after changing input_dir"""
    cfg = PaperRAGConfig()
    cfg.input_dir = "/test/pdfs"
    cfg.index_dir = "/custom/index/path"
    
    # Change input_dir - custom index_dir should remain
    cfg.input_dir = "/different/pdfs"
    
    expected = "/custom/index/path"
    actual = cfg.index_dir
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Custom index_dir persists after input_dir change: {actual}")


def test_reset_to_default():
    """Test that setting index_dir to None resets to default behavior"""
    cfg = PaperRAGConfig()
    cfg.input_dir = "/test/pdfs"
    cfg.index_dir = "/custom/index/path"
    
    # Reset to default
    cfg._index_dir = None
    
    expected = "/test/pdfs/.paperrag-index"
    actual = cfg.index_dir
    
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Reset to default works: {actual}")


if __name__ == "__main__":
    print("Testing --index-dir optional functionality...\n")
    
    try:
        test_default_index_dir()
        test_custom_index_dir()
        test_index_dir_persistence()
        test_reset_to_default()
        
        print("\n✅ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
