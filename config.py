import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the RAG project."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-ada-002
    
    # Chunking Configuration
    CHUNK_SIZE = 3  # Number of consecutive non-empty lines per chunk
    
    # FAISS Configuration
    INDEX_TYPE = "IndexFlatIP"  # Inner Product (cosine similarity after normalization)
    INDEX_FILE = "vocabulary_index.faiss"
    METADATA_FILE = "vocabulary_metadata.json"
    
    # File Configuration
    SUPPORTED_EXTENSIONS = [".txt"]
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "indexes"
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create output directory if it doesn't exist
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        
        return True 