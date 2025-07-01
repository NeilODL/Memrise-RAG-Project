#!/usr/bin/env python3
"""
Unit tests for RAG answer system.
Tests happy-path and no-hit scenarios.
"""

import pytest # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from answer import VectorRetriever, RAGAnswerer, load_chunk_texts

class TestRAGSystem:
    """Test core RAG functionality - happy path and no-hit cases."""
    
    @patch('answer.faiss.read_index')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"source_file": "test1.txt", "chunk_index": 0}, {"source_file": "test2.txt", "chunk_index": 1}]')
    def test_vector_retriever_happy_path(self, mock_file, mock_faiss):
        """Test successful vector retrieval."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 10
        mock_index.search.return_value = (
            np.array([[0.9, 0.7]]),  # scores
            np.array([[0, 1]])       # indices
        )
        mock_faiss.return_value = mock_index
        
        retriever = VectorRetriever(Path("fake_index"), Path("fake_metadata"))
        query_embedding = np.random.rand(1536).astype(np.float32)
        
        results = retriever.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[0]["index"] == 0
        assert "metadata" in results[0]
    
    @patch('answer.faiss.read_index')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"source_file": "test.txt"}]')
    def test_vector_retriever_no_hits(self, mock_file, mock_faiss):
        """Test when no similar vectors are found."""
        # Mock FAISS index with no results
        mock_index = Mock()
        mock_index.ntotal = 10
        mock_index.search.return_value = (
            np.array([[-1.0]]),  # low scores
            np.array([[-1]])     # invalid indices
        )
        mock_faiss.return_value = mock_index
        
        retriever = VectorRetriever(Path("fake_index"), Path("fake_metadata"))
        query_embedding = np.random.rand(1536).astype(np.float32)
        
        results = retriever.search(query_embedding, k=1)
        
        assert len(results) == 0  # No valid results
    
    @patch('answer.OpenAI')
    def test_rag_answerer_happy_path(self, mock_openai):
        """Test successful answer generation."""
        # Mock OpenAI responses
        mock_client = Mock()
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Bonjour means hello"))]
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        mock_openai.return_value = mock_client
        
        answerer = RAGAnswerer("fake-api-key")
        
        # Test embedding generation
        embedding = answerer.get_query_embedding("How do you say hello?")
        assert embedding.shape == (1536,)
        
        # Test answer generation
        answer = answerer.generate_answer("Test prompt")
        assert answer == "Bonjour means hello"
    
    @patch('answer.OpenAI')
    def test_rag_answerer_api_error(self, mock_openai):
        """Test handling of API errors."""
        # Mock OpenAI client that raises exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        answerer = RAGAnswerer("fake-api-key")
        
        answer = answerer.generate_answer("Test prompt")
        assert "error" in answer.lower()
    
    def test_context_prompt_creation(self):
        """Test that context prompts include current language."""
        with patch('answer.OpenAI'):
            answerer = RAGAnswerer("fake-api-key")
            
            query = "How do you say hello?"
            relevant_chunks = [
                {
                    "index": 0,
                    "metadata": {"source_file": "greetings.txt"}
                }
            ]
            chunk_texts = {0: "bonjour – hello"}
            
            prompt = answerer.create_context_prompt(query, relevant_chunks, chunk_texts)
            
            # Should include current language and content
            assert Config.LANGUAGE.title() in prompt
            assert "bonjour – hello" in prompt
            assert "greetings.txt" in prompt
            assert query in prompt

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
