import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the RAG project."""
    
    # Language Configuration - Change this to switch languages
    LANGUAGE = "italian"  # Change to "german", "french", etc.
    
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
    
    # Paths (all dynamic based on LANGUAGE setting)
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "sample_data" / LANGUAGE
    OUTPUT_DIR = PROJECT_ROOT / "indexes" / LANGUAGE
    QUESTIONS_PATH = DATA_DIR / "questions.json"
    INDEX_PATH = OUTPUT_DIR / INDEX_FILE
    METADATA_PATH = OUTPUT_DIR / METADATA_FILE
    
    @classmethod
    def get_prompt_template(cls) -> str:
        """Get the language-specific prompt template for answering questions."""
        return f"""You are a helpful {cls.LANGUAGE.title()} language learning assistant. Use the provided context to answer the user's question about {cls.LANGUAGE.title()} phrases and vocabulary.

Context from vocabulary files:
{{context}}

Question: {{query}}

Instructions:
1. Answer the question directly and accurately based on the provided context
2. If the exact phrase is in the context, provide it
3. If you need to infer or construct an answer, explain your reasoning
4. Keep your answer focused and practical for language learning
5. If the context doesn't contain relevant information, say so clearly

Answer:"""
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Check if language data directory exists
        if not cls.DATA_DIR.exists():
            raise ValueError(f"Language data directory does not exist: {cls.DATA_DIR}")
        
        # Check if language has any .txt files
        txt_files = list(cls.DATA_DIR.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found for language: {cls.LANGUAGE}")
        
        # Create output directory if it doesn't exist
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def list_available_languages(cls) -> list[str]:
        """List all available languages based on existing data directories."""
        available_languages = []
        base_data_dir = cls.PROJECT_ROOT / "sample_data"
        
        if base_data_dir.exists():
            for item in base_data_dir.iterdir():
                if item.is_dir() and item.name != "generation":
                    # Check if directory has .txt files
                    txt_files = list(item.glob("*.txt"))
                    if txt_files:
                        available_languages.append(item.name)
        
        return sorted(available_languages) 