#!/usr/bin/env python3
"""
Answer script for RAG Project.
Loads FAISS index and answers questions using semantic search and LLM.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import typer
import faiss
from openai import OpenAI

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

class VectorRetriever:
    """Handles vector similarity search using FAISS."""
    
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Loaded metadata for {len(self.metadata)} chunks")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Perform similarity search and return relevant chunks with metadata.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunk info and metadata
        """
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search for similar vectors
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                chunk_metadata = self.metadata[idx]
                results.append({
                    "score": float(score),
                    "metadata": chunk_metadata,
                    "index": int(idx)
                })
        
        return results

class RAGAnswerer:
    """Handles the complete RAG pipeline for answering questions."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = Config.EMBEDDING_MODEL
        self.chat_model = model
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def create_context_prompt(self, query: str, relevant_chunks: List[Dict], chunk_texts: Dict[int, str]) -> str:
        """
        Create a prompt with context from relevant chunks.
        
        Args:
            query: The user's question
            relevant_chunks: List of relevant chunk metadata
            chunk_texts: Dictionary mapping chunk indices to their text content
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_idx = chunk["index"]
            chunk_text = chunk_texts.get(chunk_idx, "")
            source_file = chunk["metadata"]["source_file"]
            
            context_parts.append(f"Context {i} (from {source_file}):\n{chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Get prompt template from Config
        prompt_template = Config.get_prompt_template()
        prompt = prompt_template.format(context=context, query=query)
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using OpenAI's chat model."""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent answers
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
    
    def extract_sources(self, relevant_chunks: List[Dict]) -> List[str]:
        """Extract unique source files from relevant chunks."""
        sources = set()
        for chunk in relevant_chunks:
            sources.add(chunk["metadata"]["source_file"])
        return sorted(list(sources))

def load_chunk_texts(data_path: Path, metadata: List[Dict]) -> Dict[int, str]:
    """
    Load the actual text content for chunks by reconstructing from source files.
    
    Args:
        data_path: Path to the directory containing source .txt files
        metadata: List of chunk metadata
        
    Returns:
        Dictionary mapping chunk index to chunk text
    """
    # First, load all source files
    file_contents = {}
    for file_path in data_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                file_contents[file_path.name] = content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    # Reconstruct chunk texts
    chunk_texts = {}
    for idx, chunk_meta in enumerate(metadata):
        source_file = chunk_meta["source_file"]
        start_line = chunk_meta["start_line"]
        end_line = chunk_meta["end_line"]
        
        if source_file in file_contents:
            lines = file_contents[source_file].split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            # Extract the relevant lines (convert to 0-based indexing)
            chunk_lines = non_empty_lines[start_line-1:end_line]
            chunk_text = '\n'.join(chunk_lines)
            chunk_texts[idx] = chunk_text
    
    return chunk_texts

@app.command()
def answer(
    questions_path: str = typer.Option(None, help="Path to questions.json file (overrides config)"),
    data_path: str = typer.Option(None, help="Path to original data directory for chunk text lookup (overrides config)"),
    output_path: str = typer.Option(None, help="Path to save answers JSON file"),
    top_k: int = typer.Option(3, help="Number of relevant chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model to use")
):
    """
    Answer questions using the RAG system with FAISS index.
    Uses the language configured in Config.LANGUAGE.
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Determine paths
        if questions_path:
            questions_path = Path(questions_path)
        else:
            questions_path = Config.QUESTIONS_PATH
        
        if data_path:
            data_path = Path(data_path)
        else:
            data_path = Config.DATA_DIR
        
        if output_path:
            output_path = Path(output_path)
        else:
            # Save to answers directory relative to project root
            answers_dir = Path(__file__).parent.parent / "answers"
            answers_dir.mkdir(exist_ok=True)
            output_path = answers_dir / f"answers_{Config.LANGUAGE}.json"
        
        # Validate inputs
        if not questions_path.exists():
            typer.echo(f"Error: Questions file {questions_path} does not exist")
            raise typer.Exit(1)
        
        if not Config.INDEX_PATH.exists():
            typer.echo(f"Error: Index file {Config.INDEX_PATH} does not exist.")
            typer.echo(f"Run: poetry run python scripts/ingest.py ingest")
            raise typer.Exit(1)
        
        if not Config.METADATA_PATH.exists():
            typer.echo(f"Error: Metadata file {Config.METADATA_PATH} does not exist.")
            typer.echo(f"Run: poetry run python scripts/ingest.py ingest")
            raise typer.Exit(1)
        
        logger.info(f"Loading {Config.LANGUAGE.title()} questions...")
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"Loaded {len(questions)} questions for {Config.LANGUAGE.title()}")
        
        # Initialize components
        retriever = VectorRetriever(Config.INDEX_PATH, Config.METADATA_PATH)
        answerer = RAGAnswerer(Config.OPENAI_API_KEY, model)
        
        # Load chunk texts for context
        logger.info("Loading chunk texts...")
        chunk_texts = load_chunk_texts(data_path, retriever.metadata)
        
        # Process each question
        results = []
        for i, question_data in enumerate(questions, 1):
            question_id = question_data["id"]
            question = question_data["question"]
            
            logger.info(f"Processing question {i}/{len(questions)}: {question}")
            
            # Get query embedding
            query_embedding = answerer.get_query_embedding(question)
            
            # Retrieve relevant chunks
            relevant_chunks = retriever.search(query_embedding, k=top_k)
            
            # Generate answer
            prompt = answerer.create_context_prompt(question, relevant_chunks, chunk_texts)
            answer_text = answerer.generate_answer(prompt)
            
            # Extract sources
            sources = answerer.extract_sources(relevant_chunks)
            
            # Store result
            result = {
                "id": question_id,
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "retrieval_scores": [chunk["score"] for chunk in relevant_chunks],
                "language": Config.LANGUAGE
            }
            
            results.append(result)
            logger.info(f"Answer generated. Sources: {', '.join(sources)}")
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        typer.echo(f"SUCCESS: Answered {len(questions)} {Config.LANGUAGE.title()} questions successfully!")
        typer.echo(f"Results saved to: {output_path}")
        typer.echo(f"Language: {Config.LANGUAGE.title()}")
        
        # Print summary
        for result in results:
            typer.echo(f"\nQ{result['id']}: {result['question']}")
            typer.echo(f"A: {result['answer']}")
            typer.echo(f"Sources: {', '.join(result['sources'])}")
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

@app.command()
def list_languages():
    """List all available languages that can be used for answering questions."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages with data found.")
            typer.echo(f"Expected data directory: {Config.PROJECT_ROOT / 'sample_data'}")
            return
        
        typer.echo("Available languages:")
        for language in available_languages:
            data_dir = Config.PROJECT_ROOT / "sample_data" / language
            txt_files = list(data_dir.glob("*.txt"))
            typer.echo(f"  {language.title()} ({len(txt_files)} files)")
        
        typer.echo(f"\nTo use: Set Config.LANGUAGE to one of the above and run 'poetry run python scripts/answer.py answer'")
        typer.echo(f"Current language: {Config.LANGUAGE}")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 