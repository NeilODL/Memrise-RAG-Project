#!/usr/bin/env python3
"""
Ingestion script for RAG Project.
Processes text files, generates embeddings, and builds a FAISS index.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
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

class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self, chunk_size: int = 3):
        self.chunk_size = chunk_size
    
    def read_text_files(self, data_path: Path) -> List[Tuple[str, str]]:
        """
        Read all .txt files from the given directory.
        
        Args:
            data_path: Path to directory containing .txt files
            
        Returns:
            List of tuples (filename, content)
        """
        files_content = []
        
        for file_path in data_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                files_content.append((file_path.name, content))
                logger.info(f"Read file: {file_path.name}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        return files_content
    
    def chunk_text(self, content: str, filename: str) -> List[Dict]:
        """
        Split text into chunks of consecutive non-empty lines.
        
        Args:
            content: Text content to chunk
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        lines = content.strip().split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        chunks = []
        for i in range(0, len(non_empty_lines), self.chunk_size):
            chunk_lines = non_empty_lines[i:i + self.chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            chunk_metadata = {
                "source_file": filename,
                "chunk_index": i // self.chunk_size,
                "line_count": len(chunk_lines),
                "start_line": i + 1,
                "end_line": i + len(chunk_lines)
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks

class EmbeddingGenerator:
    """Handles embedding generation using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process in each batch
            
        Returns:
            NumPy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        return np.array(all_embeddings, dtype=np.float32)

class FAISSIndexBuilder:
    """Builds and manages FAISS index."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
            index_type: Type of FAISS index to create
            
        Returns:
            FAISS index object
        """
        # Normalize embeddings for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(self.dimension)
        else:
            logger.warning(f"Unknown index type {index_type}, defaulting to IndexFlatIP")
            index = faiss.IndexFlatIP(self.dimension)
        
        index.add(embeddings)
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        
        return index
    
    def save_index(self, index: faiss.Index, index_path: Path):
        """Save FAISS index to file."""
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved index to {index_path}")

def save_metadata(chunks: List[Dict], metadata_path: Path):
    """Save chunk metadata to JSON file."""
    metadata = [chunk["metadata"] for chunk in chunks]
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {metadata_path}")

@app.command()
def ingest(
    chunk_size: int = typer.Option(Config.CHUNK_SIZE, help="Number of consecutive lines per chunk"),
    batch_size: int = typer.Option(100, help="Batch size for embedding generation"),
    data_path: str = typer.Option(None, help="Custom path to directory containing .txt files (overrides config)"),
):
    """
    Ingest text files, generate embeddings, and build FAISS index.
    Uses the language configured in Config.LANGUAGE.
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Determine data path
        if data_path:
            data_path = Path(data_path)
        else:
            data_path = Config.DATA_DIR
        
        # Validate data path
        if not data_path.exists():
            typer.echo(f"Error: Data path {data_path} does not exist")
            raise typer.Exit(1)
        
        if not data_path.is_dir():
            typer.echo(f"Error: {data_path} is not a directory")
            raise typer.Exit(1)
        
        logger.info(f"Starting ingestion for {Config.LANGUAGE.title()} from {data_path}")
        
        # Initialize components
        processor = DocumentProcessor(chunk_size=chunk_size)
        embedder = EmbeddingGenerator(Config.OPENAI_API_KEY, Config.EMBEDDING_MODEL)
        index_builder = FAISSIndexBuilder(Config.EMBEDDING_DIMENSION)
        
        # Process documents
        logger.info("Reading text files...")
        files_content = processor.read_text_files(data_path)
        
        if not files_content:
            typer.echo("No .txt files found in the specified directory")
            raise typer.Exit(1)
        
        # Generate chunks
        logger.info("Generating chunks...")
        all_chunks = []
        for filename, content in files_content:
            chunks = processor.chunk_text(content, filename)
            all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(files_content)} files")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedder.generate_embeddings(chunk_texts, batch_size=batch_size)
        
        # Build and save index
        logger.info("Building FAISS index...")
        index = index_builder.build_index(embeddings, Config.INDEX_TYPE)
        
        # Save outputs
        index_builder.save_index(index, Config.INDEX_PATH)
        save_metadata(all_chunks, Config.METADATA_PATH)
        
        typer.echo(f"SUCCESS: Ingestion completed successfully for {Config.LANGUAGE.title()}!")
        typer.echo(f"Index saved to: {Config.INDEX_PATH}")
        typer.echo(f"Metadata saved to: {Config.METADATA_PATH}")
        typer.echo(f"Total chunks: {len(all_chunks)}")
        typer.echo(f"Files processed: {len(files_content)}")
        typer.echo(f"Language: {Config.LANGUAGE.title()}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

@app.command()
def list_languages():
    """List all available languages that can be ingested."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages with data found.")
            typer.echo(f"Expected data directory: {Config.BASE_DATA_DIR}")
            return
        
        typer.echo("Available languages for ingestion:")
        for language in available_languages:
            data_dir = Config.PROJECT_ROOT / "sample_data" / language
            txt_files = list(data_dir.glob("*.txt"))
            typer.echo(f"  {language.title()} ({len(txt_files)} files)")
        
        typer.echo(f"\nTo ingest: Set Config.LANGUAGE and run 'poetry run python scripts/ingest.py ingest'")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
