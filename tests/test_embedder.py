"""Tests for embedder module - sentence-transformers embedding."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from paperrag.config import EmbedderConfig
from paperrag.embedder import Embedder


class TestEmbedder:
    """Test Embedder class initialization and functionality."""

    @patch('paperrag.embedder.SentenceTransformer')
    def test_initialization_with_default_config(self, mock_transformer):
        """Test embedder initializes with default config."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        
        assert embedder.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.config.batch_size == 64  # Default batch size
        assert embedder.config.normalize is True
        assert embedder.config.seed == 42
        assert embedder.dimension == 384

    @patch('paperrag.embedder.SentenceTransformer')
    def test_initialization_with_custom_config(self, mock_transformer):
        """Test embedder initializes with custom config."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_model
        
        config = EmbedderConfig(
            model_name="custom-model",
            batch_size=16,
            normalize=False,
            seed=123,
            device="cpu"
        )
        embedder = Embedder(config)
        
        assert embedder.config.model_name == "custom-model"
        assert embedder.config.batch_size == 16
        assert embedder.config.normalize is False
        assert embedder.config.seed == 123
        assert embedder.dimension == 768

    @patch('paperrag.embedder.SentenceTransformer')
    @patch('paperrag.embedder.torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda, mock_transformer):
        """Test device selection when CUDA is available."""
        mock_cuda.return_value = True
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        
        mock_transformer.assert_called_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda",
            local_files_only=True
        )

    @patch('paperrag.embedder.SentenceTransformer')
    @patch('paperrag.embedder.torch.cuda.is_available')
    def test_device_selection_cuda_unavailable(self, mock_cuda, mock_transformer):
        """Test device selection when CUDA is unavailable."""
        mock_cuda.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        
        mock_transformer.assert_called_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            local_files_only=True
        )

    @patch('paperrag.embedder.SentenceTransformer')
    def test_device_selection_explicit_cpu(self, mock_transformer):
        """Test explicit CPU device selection."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        config = EmbedderConfig(device="cpu")
        embedder = Embedder(config)
        
        mock_transformer.assert_called_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            local_files_only=True
        )

    @patch('paperrag.embedder.SentenceTransformer')
    def test_seed_setting(self, mock_transformer):
        """Test that seed is set for reproducibility."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        config = EmbedderConfig(seed=999)
        embedder = Embedder(config)
        
        # Check that torch and numpy seeds were set
        # This is hard to test directly, but we can verify the method was called
        assert embedder.config.seed == 999

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_empty_list(self, mock_transformer):
        """Test embedding empty list returns empty array."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embeddings = embedder.embed([])
        
        assert embeddings.shape == (0, 384)
        assert embeddings.dtype == np.float32

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_single_text(self, mock_transformer):
        """Test embedding single text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embeddings = embedder.embed(["test text"])
        
        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32
        mock_model.encode.assert_called_once()

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_multiple_texts(self, mock_transformer):
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embeddings = embedder.embed(["text1", "text2", "text3"])
        
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_batch_size(self, mock_transformer):
        """Test that batch size is passed to model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        config = EmbedderConfig(batch_size=16)
        embedder = Embedder(config)
        embedder.embed(["text1", "text2"])
        
        # Check that batch_size was passed
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs['batch_size'] == 16

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_normalize_flag(self, mock_transformer):
        """Test that normalize flag is passed to model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        config = EmbedderConfig(normalize=False)
        embedder = Embedder(config)
        embedder.embed(["text1", "text2"])
        
        # Check that normalize_embeddings was passed
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs['normalize_embeddings'] is False

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_dimension_validation(self, mock_transformer):
        """Test that dimension validation works."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Return wrong dimension
        mock_model.encode.return_value = np.random.rand(2, 512).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        
        with pytest.raises(AssertionError) as exc_info:
            embedder.embed(["text1", "text2"])
        
        assert "Dimension mismatch" in str(exc_info.value)

    @patch('paperrag.embedder.SentenceTransformer')
    def test_embed_1d_array_reshaping(self, mock_transformer):
        """Test that 1D embeddings are reshaped to 2D."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Return 1D array
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embeddings = embedder.embed(["single text"])
        
        assert embeddings.ndim == 2
        assert embeddings.shape == (1, 384)

    @patch('paperrag.embedder.SentenceTransformer')
    def test_local_files_only_fallback(self, mock_transformer):
        """Test fallback when local_files_only fails."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # First call raises exception, second succeeds
        mock_transformer.side_effect = [
            Exception("Local files not found"),
            mock_model
        ]
        
        embedder = Embedder()
        
        # Should have tried local_files_only=True first, then without
        assert mock_transformer.call_count == 2

    @patch('paperrag.embedder.SentenceTransformer')
    def test_show_progress_bar_disabled(self, mock_transformer):
        """Test that progress bar is disabled."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embedder.embed(["text1", "text2"])
        
        # Check that show_progress_bar was False
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs['show_progress_bar'] is False

    @patch('paperrag.embedder.SentenceTransformer')
    def test_convert_to_numpy(self, mock_transformer):
        """Test that conversion to numpy is enabled."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embedder.embed(["text1", "text2"])
        
        # Check that convert_to_numpy was True
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs['convert_to_numpy'] is True


class TestEmbedderConfig:
    """Test EmbedderConfig default values."""

    def test_default_config_values(self):
        """Test that EmbedderConfig has expected defaults."""
        config = EmbedderConfig()
        
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.batch_size == 64  # Default batch size
        assert config.normalize is True
        assert config.seed == 42
        assert config.device is None