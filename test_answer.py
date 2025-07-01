#!/usr/bin/env python3
"""
Unit tests for the RAG Project answer.py module.
Tests happy-path scenarios and no-hit cases.

TEST COVERAGE SUMMARY:
======================

🔍 TestVectorRetriever (FAISS Vector Search):
  ✅ test_init_success - Successful FAISS index and metadata loading
  ✅ test_search_happy_path - Normal search returning relevant results (HAPPY PATH)
  ✅ test_search_no_results - Search with no valid results (NO-HIT CASE)
  ✅ test_search_invalid_indices - Search returning invalid FAISS indices (NO-HIT CASE)

🤖 TestRAGAnswerer (OpenAI Integration):
  ✅ test_init_success - Successful OpenAI client initialization
  ✅ test_get_query_embedding_success - Successful embedding generation (HAPPY PATH)
  ✅ test_create_context_prompt - Context prompt creation with retrieved chunks
  ✅ test_generate_answer_success - Successful answer generation (HAPPY PATH)
  ✅ test_generate_answer_error - API error handling (ERROR CASE)
  ✅ test_extract_sources - Source file extraction from search results

📂 TestLoadChunkTexts (File Processing):
  ✅ test_load_chunk_texts_success - Successful chunk text loading (HAPPY PATH)
  ✅ test_load_chunk_texts_missing_file - Missing source file handling (NO-HIT CASE)

Total: 12 tests covering initialization, happy paths, no-hit scenarios, and error handling
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the modules we want to test
from answer import VectorRetriever, RAGAnswerer, load_chunk_texts
from config import Config


class TestVectorRetriever:
    """Test cases for VectorRetriever class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "test_index.faiss"
        self.metadata_path = Path(self.temp_dir) / "test_metadata.json"
        
        # Mock metadata
        self.test_metadata = [
            {
                "source_file": "greetings.txt",
                "chunk_index": 0,
                "line_count": 3,
                "start_line": 1,
                "end_line": 3
            },
            {
                "source_file": "travel.txt", 
                "chunk_index": 0,
                "line_count": 3,
                "start_line": 1,
                "end_line": 3
            }
        ]
        
        # Create test metadata file
        with open(self.metadata_path, 'w') as f:
            json.dump(self.test_metadata, f)

    @patch('faiss.read_index')
    def test_init_success(self, mock_read_index):
        """Test successful initialization of VectorRetriever."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_read_index.return_value = mock_index
        
        # Test initialization
        retriever = VectorRetriever(self.index_path, self.metadata_path)
        
        assert retriever.index == mock_index
        assert retriever.metadata == self.test_metadata
        mock_read_index.assert_called_once_with(str(self.index_path))

    @patch('faiss.read_index')
    @patch('faiss.normalize_L2')
    def test_search_happy_path(self, mock_normalize, mock_read_index):
        """Test successful search with results."""
        # Mock FAISS index and search results
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.8, 0.6]]),  # scores
            np.array([[0, 1]])       # indices
        )
        mock_read_index.return_value = mock_index
        
        # Create retriever and perform search
        retriever = VectorRetriever(self.index_path, self.metadata_path)
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        results = retriever.search(query_embedding, k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["score"] == 0.8
        assert results[0]["index"] == 0
        assert results[0]["metadata"]["source_file"] == "greetings.txt"
        assert results[1]["score"] == 0.6
        assert results[1]["index"] == 1
        assert results[1]["metadata"]["source_file"] == "travel.txt"
        
        # Verify FAISS calls
        mock_normalize.assert_called_once()
        mock_index.search.assert_called_once()

    @patch('faiss.read_index')
    @patch('faiss.normalize_L2')
    def test_search_no_results(self, mock_normalize, mock_read_index):
        """Test search with no valid results (no-hit case)."""
        # Mock FAISS index with no results
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[]]),  # empty scores
            np.array([[]])   # empty indices
        )
        mock_read_index.return_value = mock_index
        
        # Create retriever and perform search
        retriever = VectorRetriever(self.index_path, self.metadata_path)
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        results = retriever.search(query_embedding, k=2)
        
        # Verify no results returned
        assert len(results) == 0

    @patch('faiss.read_index')
    @patch('faiss.normalize_L2') 
    def test_search_invalid_indices(self, mock_normalize, mock_read_index):
        """Test search with invalid indices (-1 values)."""
        # Mock FAISS index with invalid indices
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.8, 0.6]]),  # scores
            np.array([[-1, -1]])     # invalid indices
        )
        mock_read_index.return_value = mock_index
        
        # Create retriever and perform search
        retriever = VectorRetriever(self.index_path, self.metadata_path)
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        results = retriever.search(query_embedding, k=2)
        
        # Verify no results due to invalid indices
        assert len(results) == 0


class TestRAGAnswerer:
    """Test cases for RAGAnswerer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_api_key = "test_api_key_123"
        self.test_model = "gpt-3.5-turbo"

    @patch('answer.OpenAI')
    def test_init_success(self, mock_openai):
        """Test successful initialization of RAGAnswerer."""
        answerer = RAGAnswerer(self.test_api_key, self.test_model)
        
        assert answerer.chat_model == self.test_model
        assert answerer.embedding_model == Config.EMBEDDING_MODEL
        mock_openai.assert_called_once_with(api_key=self.test_api_key)

    @patch('answer.OpenAI')
    def test_get_query_embedding_success(self, mock_openai):
        """Test successful query embedding generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test embedding generation
        answerer = RAGAnswerer(self.test_api_key)
        result = answerer.get_query_embedding("test query")
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        
        # Verify OpenAI call
        mock_client.embeddings.create.assert_called_once_with(
            input=["test query"],
            model=Config.EMBEDDING_MODEL
        )

    @patch('answer.OpenAI')
    def test_create_context_prompt(self, mock_openai):
        """Test context prompt creation with relevant chunks."""
        answerer = RAGAnswerer(self.test_api_key)
        
        # Test data
        query = "How do you say hello?"
        relevant_chunks = [
            {
                "score": 0.9,
                "index": 0,
                "metadata": {"source_file": "greetings.txt"}
            },
            {
                "score": 0.7,
                "index": 1,
                "metadata": {"source_file": "travel.txt"}
            }
        ]
        chunk_texts = {
            0: "hola – hello\nbuenos días – good morning",
            1: "buen viaje – have a good trip\nadiós – goodbye"
        }
        
        prompt = answerer.create_context_prompt(query, relevant_chunks, chunk_texts)
        
        # Verify prompt contains expected elements
        assert "How do you say hello?" in prompt
        assert "hola – hello" in prompt
        assert "greetings.txt" in prompt
        assert "buen viaje – have a good trip" in prompt
        assert "travel.txt" in prompt
        assert "Spanish language learning assistant" in prompt

    @patch('answer.OpenAI')
    def test_generate_answer_success(self, mock_openai):
        """Test successful answer generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="  Test answer  "))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test answer generation
        answerer = RAGAnswerer(self.test_api_key)
        result = answerer.generate_answer("test prompt")
        
        # Verify result
        assert result == "Test answer"  # Should be stripped
        
        # Verify OpenAI call
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1,
            max_tokens=500
        )

    @patch('answer.OpenAI')
    def test_generate_answer_error(self, mock_openai):
        """Test answer generation with API error."""
        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        # Test answer generation
        answerer = RAGAnswerer(self.test_api_key)
        result = answerer.generate_answer("test prompt")
        
        # Verify error handling
        assert "encountered an error" in result

    def test_extract_sources(self):
        """Test source extraction from relevant chunks."""
        answerer = RAGAnswerer(self.test_api_key)
        
        relevant_chunks = [
            {"metadata": {"source_file": "greetings.txt"}},
            {"metadata": {"source_file": "travel.txt"}},
            {"metadata": {"source_file": "greetings.txt"}},  # duplicate
            {"metadata": {"source_file": "restaurant.txt"}}
        ]
        
        sources = answerer.extract_sources(relevant_chunks)
        
        # Verify unique sources, sorted
        assert sources == ["greetings.txt", "restaurant.txt", "travel.txt"]


class TestLoadChunkTexts:
    """Test cases for load_chunk_texts function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir)
        
        # Create test files
        with open(self.data_path / "greetings.txt", 'w') as f:
            f.write("hola – hello\nbuenos días – good morning\nbuenas tardes – good afternoon")
        
        with open(self.data_path / "travel.txt", 'w') as f:
            f.write("aeropuerto – airport\nestación – station\nautobus – bus")

    def test_load_chunk_texts_success(self):
        """Test successful loading of chunk texts."""
        metadata = [
            {
                "source_file": "greetings.txt",
                "start_line": 1,
                "end_line": 2
            },
            {
                "source_file": "travel.txt", 
                "start_line": 2,
                "end_line": 3
            }
        ]
        
        result = load_chunk_texts(self.data_path, metadata)
        
        # Verify results
        assert len(result) == 2
        assert "hola – hello\nbuenos días – good morning" in result[0]
        assert "estación – station\nautobus – bus" in result[1]

    def test_load_chunk_texts_missing_file(self):
        """Test handling of missing source file."""
        metadata = [
            {
                "source_file": "nonexistent.txt",
                "start_line": 1,
                "end_line": 2
            }
        ]
        
        result = load_chunk_texts(self.data_path, metadata)
        
        # Should handle missing file gracefully
        assert len(result) == 0 or 0 not in result


if __name__ == "__main__":
    # Run tests with: python -m pytest test_answer.py -v
    # Or run specific test: python -m pytest test_answer.py::TestVectorRetriever::test_search_happy_path -v
    pytest.main([__file__, "-v"])
