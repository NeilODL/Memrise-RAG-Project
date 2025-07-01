#!/usr/bin/env python3
"""
Answer script for RAG Project.
Semantic search + LLM answer generation using FAISS index.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import typer
import faiss
from openai import OpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

class VectorRetriever:
    """FAISS-based semantic search for finding relevant chunks."""
    
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded index: {self.index.ntotal} vectors, {len(self.metadata)} chunks")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Find top-k most similar chunks to query."""
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "score": float(score),
                    "metadata": self.metadata[idx],
                    "index": int(idx)
                })
        
        return results

class RAGAnswerer:
    """Complete RAG pipeline: embedding generation + answer generation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = Config.EMBEDDING_MODEL
        self.chat_model = model
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        response = self.client.embeddings.create(input=[query], model=self.embedding_model)
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def create_context_prompt(self, query: str, relevant_chunks: List[Dict], chunk_texts: Dict[int, str]) -> str:
        """Build prompt with retrieved context."""
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_idx = chunk["index"]
            chunk_text = chunk_texts.get(chunk_idx, "")
            source_file = chunk["metadata"]["source_file"]
            context_parts.append(f"Context {i} (from {source_file}):\n{chunk_text}")
        
        context = "\n\n".join(context_parts)
        prompt_template = Config.get_prompt_template()
        return prompt_template.format(context=context, query=query)
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using OpenAI chat model."""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "Sorry, I encountered an error generating the answer."
    
    def extract_sources(self, relevant_chunks: List[Dict]) -> List[str]:
        """Get unique source files from chunks."""
        sources = {chunk["metadata"]["source_file"] for chunk in relevant_chunks}
        return sorted(sources)

def load_chunk_texts(data_path: Path, metadata: List[Dict]) -> Dict[int, str]:
    """Reconstruct chunk texts from source files."""
    # Load all source files
    file_contents = {}
    for file_path in data_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_contents[file_path.name] = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    # Reconstruct chunks
    chunk_texts = {}
    for idx, chunk_meta in enumerate(metadata):
        source_file = chunk_meta["source_file"]
        start_line = chunk_meta["start_line"]
        end_line = chunk_meta["end_line"]
        
        if source_file in file_contents:
            lines = file_contents[source_file].split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            chunk_lines = non_empty_lines[start_line-1:end_line]
            chunk_texts[idx] = '\n'.join(chunk_lines)
    
    return chunk_texts

def validate_files():
    """Check that required files exist."""
    if not Config.INDEX_PATH.exists():
        typer.echo(f"ERROR: Index missing: {Config.INDEX_PATH}")
        typer.echo("Run: poetry run python scripts/ingest.py ingest")
        raise typer.Exit(1)
    
    if not Config.METADATA_PATH.exists():
        typer.echo(f"ERROR: Metadata missing: {Config.METADATA_PATH}")
        typer.echo("Run: poetry run python scripts/ingest.py ingest")
        raise typer.Exit(1)

@app.command()
def answer(
    questions_path: str = typer.Option(None, help="Questions JSON file path"),
    data_path: str = typer.Option(None, help="Data directory path"),
    output_path: str = typer.Option(None, help="Output answers file"),
    top_k: int = typer.Option(3, help="Number of chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model")
):
    """
    Answer questions using RAG: retrieve relevant chunks → generate LLM answers.
    """
    try:
        Config.validate()
        validate_files()
        
        # Set paths
        questions_path = Path(questions_path) if questions_path else Config.QUESTIONS_PATH
        data_path = Path(data_path) if data_path else Config.DATA_DIR
        
        if not output_path:
            answers_dir = Path(__file__).parent.parent / "answers"
            answers_dir.mkdir(exist_ok=True)
            output_path = answers_dir / f"answers_{Config.LANGUAGE}.json"
        else:
            output_path = Path(output_path)
        
        if not questions_path.exists():
            typer.echo(f"ERROR: Questions file not found: {questions_path}")
            raise typer.Exit(1)
        
        # Load questions
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"Processing {len(questions)} {Config.LANGUAGE.title()} questions")
        
        # Initialize RAG components
        retriever = VectorRetriever(Config.INDEX_PATH, Config.METADATA_PATH)
        answerer = RAGAnswerer(Config.OPENAI_API_KEY, model)
        chunk_texts = load_chunk_texts(data_path, retriever.metadata)
        
        # Process questions
        results = []
        for i, question_data in enumerate(questions, 1):
            question_id = question_data["id"]
            question = question_data["question"]
            
            logger.info(f"Question {i}/{len(questions)}: {question}")
            
            # RAG pipeline: embed → search → generate
            query_embedding = answerer.get_query_embedding(question)
            relevant_chunks = retriever.search(query_embedding, k=top_k)
            prompt = answerer.create_context_prompt(question, relevant_chunks, chunk_texts)
            answer_text = answerer.generate_answer(prompt)
            sources = answerer.extract_sources(relevant_chunks)
            
            results.append({
                "id": question_id,
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "retrieval_scores": [chunk["score"] for chunk in relevant_chunks],
                "language": Config.LANGUAGE
            })
            
            logger.info(f"Generated answer. Sources: {', '.join(sources)}")
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        typer.echo(f"✅ SUCCESS: Answered {len(questions)} {Config.LANGUAGE.title()} questions!")
        typer.echo(f"Results: {output_path}")
        
        # Print summary
        for result in results:
            typer.echo(f"\nQ{result['id']}: {result['question']}")
            typer.echo(f"A: {result['answer']}")
            typer.echo(f"Sources: {', '.join(result['sources'])}")
        
    except Exception as e:
        logger.error(f"Answer pipeline failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

@app.command()
def list_languages():
    """List available languages for answering."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages found.")
            return
        
        typer.echo("Available languages:")
        for language in available_languages:
            data_dir = Config.PROJECT_ROOT / "sample_data" / language
            txt_files = list(data_dir.glob("*.txt"))
            typer.echo(f"  {language.title()} ({len(txt_files)} files)")
        
        typer.echo(f"\nCurrent: {Config.LANGUAGE}")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 