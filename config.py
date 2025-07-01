import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()

class Config:
    """
    RAG system configuration. 
    
    Key design: 3-line chunks balance context (multiple phrases) with precision.
    FAISS IndexFlatIP uses cosine similarity for semantic search.
    """
    
    # Language setting - switches entire pipeline
    LANGUAGE = "spanish"  # Change to "german", "french", etc.
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Multilingual, 1536 dimensions
    EMBEDDING_DIMENSION = 1536
    
    # Chunking: 3 lines captures phrase groups while avoiding dilution
    CHUNK_SIZE = 3
    
    # FAISS: IndexFlatIP = exact cosine similarity after L2 normalization
    INDEX_TYPE = "IndexFlatIP"
    INDEX_FILE = "vocabulary_index.faiss"
    METADATA_FILE = "vocabulary_metadata.json"
    
    # File types
    SUPPORTED_EXTENSIONS = [".txt"]
    
    # Dynamic paths based on LANGUAGE setting
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "sample_data" / LANGUAGE
    OUTPUT_DIR = PROJECT_ROOT / "indexes" / LANGUAGE
    QUESTIONS_PATH = DATA_DIR / "questions.json"
    INDEX_PATH = OUTPUT_DIR / INDEX_FILE
    METADATA_PATH = OUTPUT_DIR / METADATA_FILE
    
    @classmethod
    def get_prompt_template(cls) -> str:
        """Language-specific prompt for LLM answer generation."""
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
        """Validate API key, data directory, and create output directory."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not cls.DATA_DIR.exists():
            raise ValueError(f"Language data directory does not exist: {cls.DATA_DIR}")
        
        # Check for vocabulary files
        txt_files = list(cls.DATA_DIR.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found for language: {cls.LANGUAGE}")
        
        # Create output directory for indexes
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return True
    
    @classmethod
    def list_available_languages(cls) -> list[str]:
        """List languages with vocabulary data (.txt files)."""
        available_languages = []
        base_data_dir = cls.PROJECT_ROOT / "sample_data"
        
        if base_data_dir.exists():
            for item in base_data_dir.iterdir():
                if item.is_dir() and item.name != "generation":
                    txt_files = list(item.glob("*.txt"))
                    if txt_files:
                        available_languages.append(item.name)
        
        return sorted(available_languages) 