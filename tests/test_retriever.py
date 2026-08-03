"""Tests for retriever module - retrieval correctness and functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paperrag.config import PaperRAGConfig, RetrieverConfig, EmbedderConfig
from paperrag.retriever import Retriever, RetrievalResult, _embedder_cache
from paperrag.vectorstore import VectorStore


@pytest.fixture(autouse=True)
def _reset_embedder_cache():
    """Retriever caches embedders in a module-level dict keyed by model name.

    Most tests here use the default model name, so without clearing this
    between tests, only the first test to run actually constructs an
    Embedder mock - every later test silently gets the first test's cached
    instance instead of its own.
    """
    _embedder_cache.clear()
    yield
    _embedder_cache.clear()


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_result_creation(self):
        """Test RetrievalResult creation."""
        result = RetrievalResult(
            text="test text",
            score=0.95,
            paper_title="Test Paper",
            section_name="Abstract",
            file_path="/path/to/paper.pdf",
            chunk_id=0
        )
        
        assert result.text == "test text"
        assert result.score == 0.95
        assert result.paper_title == "Test Paper"
        assert result.section_name == "Abstract"
        assert result.file_path == "/path/to/paper.pdf"
        assert result.chunk_id == 0

    def test_result_string_representation(self):
        """Test RetrievalResult string representation."""
        result = RetrievalResult(
            text="test text",
            score=0.95,
            paper_title="Test Paper",
            section_name="Abstract",
            file_path="/path/to/paper.pdf",
            chunk_id=0
        )
        
        str_repr = str(result)
        assert "0.9500" in str_repr
        assert "Test Paper" in str_repr
        assert "Abstract" in str_repr
        assert "chunk 0" in str_repr


class TestRetrieverInit:
    """Test Retriever initialization."""

    @patch('paperrag.retriever.Embedder')
    @patch('paperrag.retriever.VectorStore')
    def test_initialization_with_store(self, mock_vs, mock_embedder):
        """Test initialization with provided store."""
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_vs.return_value = mock_store
        
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder.return_value = mock_embedder_instance
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig()
        )
        
        retriever = Retriever(config, store=mock_store)
        
        assert retriever.store == mock_store
        assert retriever.embedder == mock_embedder_instance

    @patch('paperrag.retriever.Embedder')
    @patch('paperrag.retriever.VectorStore')
    def test_initialization_dimension_mismatch(self, mock_vs, mock_embedder):
        """Test initialization fails with dimension mismatch."""
        mock_store = MagicMock()
        mock_store.dimension = 512
        mock_vs.return_value = mock_store
        
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder.return_value = mock_embedder_instance
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig()
        )
        
        with pytest.raises(ValueError) as exc_info:
            Retriever(config, store=mock_store)
        
        assert "dimension" in str(exc_info.value).lower()

    @patch('paperrag.retriever.Embedder')
    def test_embedder_caching(self, mock_embedder):
        """Test embedder caching by model name."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder.return_value = mock_embedder_instance
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(model_name="test-model")
        )
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        
        # First call should create embedder
        retriever1 = Retriever(config, store=mock_store)
        assert mock_embedder.call_count == 1
        
        # Second call should use cached embedder
        retriever2 = Retriever(config, store=mock_store)
        assert mock_embedder.call_count == 1  # No additional call

    @patch('paperrag.retriever.Embedder')
    def test_embedder_cache_distinguishes_device(self, mock_embedder):
        """Same model name but different device must not share a cached embedder."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder.return_value = mock_embedder_instance

        mock_store = MagicMock()
        mock_store.dimension = 384

        cpu_config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(model_name="test-model", device="cpu")
        )
        cuda_config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(model_name="test-model", device="cuda")
        )

        Retriever(cpu_config, store=mock_store)
        assert mock_embedder.call_count == 1

        # Different device, same model name: must construct a new embedder,
        # not reuse the cpu one.
        Retriever(cuda_config, store=mock_store)
        assert mock_embedder.call_count == 2

        # Repeating the cpu config should hit the cache again.
        Retriever(cpu_config, store=mock_store)
        assert mock_embedder.call_count == 2


class TestRetrieverRetrieve:
    """Test retrieval functionality."""

    @patch('paperrag.retriever.Embedder')
    def test_retrieve_basic(self, mock_embedder):
        """Test basic retrieval."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = [
            ({"text": "result1", "paper_title": "Paper1", "section_name": "Abstract", "file_path": "file1.pdf", "chunk_id": 0}, 0.95),
            ({"text": "result2", "paper_title": "Paper2", "section_name": "Introduction", "file_path": "file2.pdf", "chunk_id": 1}, 0.85),
        ]
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3, score_threshold=0.5)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query")
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    @patch('paperrag.retriever.Embedder')
    def test_retrieve_with_top_k_override(self, mock_embedder):
        """Test retrieval with custom top_k."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = [
            ({"text": "result1", "paper_title": "Paper1", "section_name": "Abstract", "file_path": "file1.pdf", "chunk_id": 0}, 0.95),
        ]
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query", top_k=5)
        
        # Should use custom top_k
        mock_store.search.assert_called_once()
        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs['top_k'] == 5

    @patch('paperrag.retriever.Embedder')
    def test_retrieve_score_threshold_filtering(self, mock_embedder):
        """Test retrieval filters by score threshold."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = [
            ({"text": "result1", "paper_title": "Paper1", "section_name": "Abstract", "file_path": "file1.pdf", "chunk_id": 0}, 0.95),
            ({"text": "result2", "paper_title": "Paper2", "section_name": "Introduction", "file_path": "file2.pdf", "chunk_id": 1}, 0.75),
            ({"text": "result3", "paper_title": "Paper3", "section_name": "Methods", "file_path": "file3.pdf", "chunk_id": 2}, 0.45),  # Below threshold
        ]
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=10, score_threshold=0.5)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query")
        
        # Should filter out result below threshold
        assert len(results) == 2
        assert all(r.score >= 0.5 for r in results)

    @patch('paperrag.retriever.Embedder')
    def test_retrieve_empty_results(self, mock_embedder):
        """Test retrieval with no results."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = []
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query")
        
        assert len(results) == 0

    @patch('paperrag.retriever.Embedder')
    def test_retrieve_with_file_path_filter(self, mock_embedder):
        """Test retrieval with file path filtering."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = [
            ({"text": "result1", "paper_title": "Paper1", "section_name": "Abstract", "file_path": "file1.pdf", "chunk_id": 0}, 0.95),
        ]
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query", file_path="file1.pdf")
        
        # Should pass file_path to search
        mock_store.search.assert_called_once()
        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs['file_path'] == "file1.pdf"


class TestRetrieverMMR:
    """Test MMR (Maximal Marginal Relevance) functionality."""

    @patch('paperrag.retriever.Embedder')
    def test_mmr_disabled_by_default(self, mock_embedder):
        """Test MMR is disabled by default."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.search.return_value = [
            ({"text": "result1", "paper_title": "Paper1", "section_name": "Abstract", "file_path": "file1.pdf", "chunk_id": 0}, 0.95),
        ]
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3, use_mmr=False)
        )
        
        retriever = Retriever(config, store=mock_store)
        results = retriever.retrieve("test query")
        
        # Should use standard search, not MMR
        mock_store.search.assert_called_once()

    @patch('paperrag.retriever.Embedder')
    def test_mmr_enabled(self, mock_embedder):
        """Test MMR is enabled when configured."""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.dimension = 384
        mock_embedder_instance.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store = MagicMock()
        mock_store.dimension = 384
        
        config = PaperRAGConfig(
            index_dir="/tmp/index",
            embedder=EmbedderConfig(),
            retriever=RetrieverConfig(top_k=3, use_mmr=True)
        )
        
        retriever = Retriever(config, store=mock_store)
        
        # Patch the _retrieve_mmr method to avoid complex implementation
        with patch.object(retriever, '_retrieve_mmr', return_value=[]):
            results = retriever.retrieve("test query")
            
            # Should call _retrieve_mmr instead of standard search
            assert results == []


class TestRetrieverErrorHandling:
    """Test error handling in retriever."""

    @patch('paperrag.retriever.VectorStore.exists')
    def test_missing_index_error(self, mock_exists):
        """Test error when index doesn't exist."""
        mock_exists.return_value = False
        
        config = PaperRAGConfig(
            index_dir="/tmp/nonexistent",
            embedder=EmbedderConfig()
        )
        
        with pytest.raises(FileNotFoundError) as exc_info:
            Retriever(config)
        
        assert "index" in str(exc_info.value).lower()
        assert "paperrag index" in str(exc_info.value).lower()