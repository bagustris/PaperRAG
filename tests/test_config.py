"""Tests for the config module."""

import json
from pathlib import Path

from paper_rag.config import PaperRAGConfig


def test_default_config():
    cfg = PaperRAGConfig()
    assert cfg.chunker.chunk_size == 1000
    assert cfg.chunker.chunk_overlap == 200
    assert cfg.embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.top_k == 5
    assert cfg.llm.temperature == 0.0


def test_snapshot_roundtrip(tmp_path):
    cfg = PaperRAGConfig()
    path = tmp_path / "snap.json"
    cfg.save_snapshot(path)

    loaded = PaperRAGConfig.load_snapshot(path)
    assert loaded.chunker.chunk_size == cfg.chunker.chunk_size
    assert loaded.embedder.model_name == cfg.embedder.model_name


def test_snapshot_is_json_serialisable():
    cfg = PaperRAGConfig()
    s = cfg.snapshot()
    text = json.dumps(s)
    assert isinstance(text, str)
