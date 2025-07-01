#!/usr/bin/env python3
"""
Ingestion script for RAG Project.
Converts vocabulary files → chunks → embeddings → FAISS index for semantic search.
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
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

class DocumentProcessor:
    """Processes vocabulary files into chunks for embedding."""
    
    def __init__(self, chunk_size: int = 3):
        self.chunk_size = chunk_size
    
    def read_text_files(self, data_path: Path) -> List[Tuple[str, str]]:
        """Read all .txt files from directory."""
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
        """Split text into 3-line chunks with metadata."""
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
    """Generates embeddings using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings in batches to handle API limits."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                raise
        
        return np.array(all_embeddings, dtype=np.float32)

class FAISSIndexBuilder:
    """Builds FAISS index for similarity search."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
        """Build FAISS index. Normalizes embeddings for cosine similarity."""
        # Normalize for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(self.dimension)
        else:
            logger.warning(f"Unknown index type {index_type}, using IndexFlatIP")
            index = faiss.IndexFlatIP(self.dimension)
        
        index.add(embeddings)
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, index_path: Path):
        """Save FAISS index to file."""
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved index to {index_path}")

def save_metadata(chunks: List[Dict], metadata_path: Path):
    """Save chunk metadata to JSON."""
    metadata = [chunk["metadata"] for chunk in chunks]
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {metadata_path}")

@app.command()
def ingest(
    chunk_size: int = typer.Option(Config.CHUNK_SIZE, help="Lines per chunk"),
    batch_size: int = typer.Option(100, help="Embedding batch size"),
    data_path: str = typer.Option(None, help="Custom data directory path"),
):
    """
    Process vocabulary files into FAISS index for semantic search.
    
    Pipeline: .txt files → chunks → embeddings → FAISS index + metadata
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Set paths
        data_dir = Path(data_path) if data_path else Config.DATA_DIR
        
        if not data_dir.exists() or not data_dir.is_dir():
            typer.echo(f"ERROR: Invalid data directory: {data_dir}")
            raise typer.Exit(1)
        
        logger.info(f"Starting ingestion for {Config.LANGUAGE.title()}")
        
        # Initialize components
        processor = DocumentProcessor(chunk_size=chunk_size)
        embedder = EmbeddingGenerator(Config.OPENAI_API_KEY, Config.EMBEDDING_MODEL)
        index_builder = FAISSIndexBuilder(Config.EMBEDDING_DIMENSION)
        
        # Process files
        logger.info("Reading vocabulary files...")
        files_content = processor.read_text_files(data_dir)
        
        if not files_content:
            typer.echo("ERROR: No .txt files found")
            raise typer.Exit(1)
        
        # Generate chunks
        logger.info("Creating chunks...")
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
        
        index_builder.save_index(index, Config.INDEX_PATH)
        save_metadata(all_chunks, Config.METADATA_PATH)
        
        typer.echo(f"✅ SUCCESS: Ingestion completed for {Config.LANGUAGE.title()}!")
        typer.echo(f"Index: {Config.INDEX_PATH}")
        typer.echo(f"Metadata: {Config.METADATA_PATH}")
        typer.echo(f"Chunks: {len(all_chunks)}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

@app.command()
def list_languages():
    """List available languages for ingestion."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages found.")
            typer.echo(f"Expected directory: {Config.PROJECT_ROOT / 'sample_data'}")
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
