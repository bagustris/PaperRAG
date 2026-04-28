"""Tests for the config module."""

import json
from pathlib import Path

from paperrag.config import PaperRAGConfig, apply_rc


def test_default_config():
    cfg = PaperRAGConfig()
    assert cfg.chunker.chunk_size == 1000
    assert cfg.chunker.chunk_overlap == 200
    assert cfg.embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.top_k == 2
    assert cfg.llm.temperature == 0.0
    assert cfg.llm.ctx_size == 2048
    assert "research assistant" in cfg.llm.system_prompt


def test_snapshot_roundtrip(tmp_path):
    cfg = PaperRAGConfig()
    cfg.llm.ctx_size = 4096
    cfg.llm.system_prompt = "Custom prompt"
    path = tmp_path / "snap.json"
    cfg.save_snapshot(path)

    loaded = PaperRAGConfig.load_snapshot(path)
    assert loaded.chunker.chunk_size == cfg.chunker.chunk_size
    assert loaded.embedder.model_name == cfg.embedder.model_name
    assert loaded.llm.ctx_size == 4096
    assert loaded.llm.system_prompt == "Custom prompt"


def test_legacy_snapshot_migration(tmp_path):
    """Verify that old snapshots with 'n_ctx' migrate correctly to 'ctx_size'."""
    legacy_data = {
        "llm": {
            "model_name": "test-model",
            "n_ctx": 1024
        }
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_data))

    cfg = PaperRAGConfig.load_snapshot(path)
    assert cfg.llm.model_name == "test-model"
    assert cfg.llm.ctx_size == 1024


def test_snapshot_is_json_serialisable():
    cfg = PaperRAGConfig()
    s = cfg.snapshot()
    text = json.dumps(s)
    assert isinstance(text, str)


def test_index_dir_for_directory(tmp_path):
    cfg = PaperRAGConfig(input_dir=str(tmp_path))
    assert cfg.index_dir == str(tmp_path / ".paperrag-index")


def test_index_dir_for_single_pdf(tmp_path):
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"")
    cfg = PaperRAGConfig(input_dir=str(pdf_file))
    assert cfg.index_dir == str(tmp_path / ".paperrag-index")


def test_rc_embed_model():
    """Verify that embed-model RC key sets cfg.embedder.model_name."""
    cfg = PaperRAGConfig()
    apply_rc(cfg, {"embed-model": "sentence-transformers/all-mpnet-base-v2"})
    assert cfg.embedder.model_name == "sentence-transformers/all-mpnet-base-v2"
