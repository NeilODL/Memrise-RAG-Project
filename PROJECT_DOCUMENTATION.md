# RAG Project Documentation
## Multilingual Language Learning Assistant

### 📋 Overview

This project is a **Retrieval-Augmented Generation (RAG) system** designed for language learning, specifically built for Memrise-style vocabulary training. It combines semantic search with large language model (LLM) capabilities to create an intelligent language learning assistant that can answer questions about phrases and vocabulary in multiple languages.

### 🎯 Purpose

The system enables interactive language learning by:
- **Semantic Search**: Finding relevant vocabulary based on meaning rather than exact text matches
- **Contextual Answers**: Generating helpful explanations using retrieved vocabulary context
- **Multi-language Support**: Supporting French, German, Italian, Spanish, and more
- **Interactive Learning**: Providing a chat-like interface for asking language questions

---

## 🏗️ Architecture

### Core Components

1. **Document Processing & Ingestion** (`scripts/ingest.py`)
   - Reads vocabulary files from `sample_data/{language}/`
   - Chunks text into 3-line segments for optimal context
   - Generates embeddings using OpenAI's text-embedding-ada-002
   - Builds FAISS index for fast semantic search

2. **Vector Retrieval** (`scripts/answer.py`)
   - Uses FAISS IndexFlatIP for exact cosine similarity search
   - Retrieves top-k most relevant chunks for any query
   - Maintains metadata linking chunks back to source files

3. **Answer Generation** (`scripts/answer.py`)
   - Creates context-aware prompts with retrieved vocabulary
   - Uses OpenAI GPT models to generate educational responses
   - Provides source attribution for transparency

4. **Interactive Interface** (`scripts/ask.py`)
   - CLI-based chat interface for real-time learning
   - Supports conversation flow with exit commands
   - Shows confidence scores and source files

5. **Data Generation** (`scripts/generate_language_data.py`)
   - Auto-generates vocabulary files using GPT-4
   - Creates 10 categories (greetings, restaurant, travel, etc.)
   - Maintains consistent format across languages

### Technical Stack

- **Python 3.11+** with Poetry for dependency management
- **OpenAI API** for embeddings (ada-002) and chat completion (GPT-3.5/4)
- **FAISS** for high-performance vector similarity search
- **Typer** for command-line interfaces
- **NumPy** for numerical operations

---

## 📁 Project Structure

```
RAG Project/
├── config.py                 # Central configuration and language settings
├── pyproject.toml            # Poetry dependencies and project metadata
├── .gitignore                # Git ignore patterns
├── .env                      # Environment variables (create this)
│
├── scripts/                  # All executable scripts
│   ├── ingest.py            # Data processing and index building
│   ├── ask.py               # Interactive chat interface
│   ├── answer.py            # RAG pipeline and answer generation
│   ├── generate_language_data.py # Vocabulary file generation
│   ├── main.py              # Unified CLI entry point
│   └── test_answer.py       # Testing utilities
│
├── sample_data/             # Vocabulary source files
│   ├── french/
│   │   ├── greetings.txt    # 15 greeting phrases
│   │   ├── restaurant.txt   # 15 restaurant phrases
│   │   ├── travel.txt       # 15 travel phrases
│   │   ├── shopping.txt     # And so on...
│   │   └── questions.json   # Sample questions for testing
│   ├── german/              # Same structure for German
│   ├── italian/             # Same structure for Italian
│   └── spanish/             # Same structure for Spanish
│
├── indexes/                 # Generated FAISS indexes and metadata
│   ├── french/
│   │   ├── vocabulary_index.faiss    # Vector index
│   │   └── vocabulary_metadata.json  # Chunk metadata
│   └── [other languages]/
│
└── answers/                 # Generated answer files
    ├── answers_french.json
    ├── answers_german.json
    └── [other languages].json
```

---

## 🚀 Installation & Setup

### 1. Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- OpenAI API key

### 2. Environment Setup

```bash
# Clone or navigate to project directory
cd "RAG Project"

# Install dependencies with Poetry
poetry install

# Create environment file
cp .env.example .env  # If it exists, or create manually
```

### 3. Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Language Configuration

Edit `config.py` to set your desired language:
```python
LANGUAGE = "french"  # Change to "german", "italian", "spanish", etc.
```

---

## 🎮 Usage

### Basic Workflow

1. **Generate Vocabulary Data** (optional - data already provided)
2. **Build Search Index** (required for first-time setup)
3. **Start Interactive Learning** (main usage)

### 1. Generate Vocabulary Data (Optional)

```bash
# Generate vocabulary for a new language
poetry run python scripts/generate_language_data.py generate "Portuguese"

# This creates sample_data/portuguese/ with 10 vocabulary files
```

### 2. Build Search Index (Required)

```bash
# Process vocabulary files and build FAISS index
poetry run python scripts/ingest.py ingest

# This creates:
# - indexes/{language}/vocabulary_index.faiss
# - indexes/{language}/vocabulary_metadata.json
```

### 3. Interactive Learning

```bash
# Start the language learning assistant
poetry run python scripts/ask.py

# Example conversation:
# Your question: How do you say hello in French?
# Answer: In French, you say "Bonjour" for hello, which literally means "good day"...
# Found in: greetings.txt
# Confidence: 0.847
```

### 4. Batch Answer Generation

```bash
# Generate answers for predefined questions
poetry run python scripts/answer.py answer

# This creates answers/answers_{language}.json
```

### 5. List Available Languages

```bash
# See what languages have vocabulary data
poetry run python scripts/ask.py list-languages

# Output:
# Available languages:
#   • French (10 files)
#   • German (10 files)
#   • Italian (10 files)
#   • Spanish (10 files)
```

---

## ⚙️ Configuration

### Key Settings in `config.py`

```python
# Language selection (changes entire pipeline)
LANGUAGE = "french"  

# Chunking strategy (3 lines = phrase groups)
CHUNK_SIZE = 3

# OpenAI models
EMBEDDING_MODEL = "text-embedding-ada-002"  # 1536 dimensions
EMBEDDING_DIMENSION = 1536

# FAISS index type (IndexFlatIP = cosine similarity)
INDEX_TYPE = "IndexFlatIP"
```

### Language-Specific Paths

The system automatically adjusts paths based on `LANGUAGE` setting:
- Data: `sample_data/{language}/`
- Index: `indexes/{language}/`
- Questions: `sample_data/{language}/questions.json`

---

## 📊 Data Format

### Vocabulary Files (.txt)

Each file contains 15 phrases in this format:
```
French phrase – English translation
Comment ça va? – How are you? (informal)
Enchanté de faire votre connaissance – Pleased to meet you
```

### Questions File (questions.json)

```json
[
  {
    "id": 1,
    "question": "How do you say 'hello' in French?"
  },
  {
    "id": 2,
    "question": "How do you ask 'How are you?' in French?"
  }
]
```

### Vocabulary Categories

Each language includes 10 categories:
1. **greetings** - Basic introductions and social interactions
2. **restaurant** - Ordering food and dining etiquette
3. **travel** - Transportation and general travel needs
4. **shopping** - Retail interactions and purchases
5. **health** - Medical needs and pharmacy visits
6. **accommodation** - Hotel and lodging requests
7. **navigation** - Directions and finding locations
8. **weather** - Weather conditions and forecasts
9. **time** - Time expressions and scheduling
10. **internet** - Technology and digital communication

---

## 🔧 Technical Specifications

### Chunking Strategy

- **3-line chunks** balance context with precision
- Captures multiple related phrases without dilution
- Each chunk includes metadata (source file, line numbers, chunk index)

### Embedding & Search

- **OpenAI text-embedding-ada-002**: Multilingual, 1536 dimensions
- **FAISS IndexFlatIP**: Exact cosine similarity after L2 normalization
- **Top-k retrieval**: Default k=3 for answer generation

### Answer Generation

- **Language-specific prompts** tailored to each language
- **Context integration** from retrieved vocabulary chunks
- **Source attribution** for educational transparency
- **Temperature 0.1** for consistent, educational responses

### Performance

- **Vector search**: Sub-millisecond retrieval with FAISS
- **Batch processing**: 100 texts per embedding API call
- **Memory efficient**: Indexes stored on disk, loaded as needed

---

## 🛠️ Advanced Usage

### Custom Language Addition

1. Create directory: `sample_data/your_language/`
2. Add vocabulary files following the format
3. Update `config.py`: `LANGUAGE = "your_language"`
4. Run ingestion: `poetry run python scripts/ingest.py ingest`

### Debugging & Testing

```bash
# Test the answer pipeline
poetry run python scripts/test_answer.py

# Verbose logging during ingestion
poetry run python scripts/ingest.py ingest --help

# Custom chunk size experimentation
poetry run python scripts/ingest.py ingest --chunk-size 5
```

### Custom Models

```bash
# Use GPT-4 for better answers
poetry run python scripts/ask.py --model gpt-4

# Different retrieval parameters
poetry run python scripts/ask.py --top-k 5
```

---

## 🔍 How It Works (Technical Deep Dive)

### 1. Data Ingestion Pipeline

```
.txt files → Line splitting → 3-line chunks → Embeddings → FAISS index
```

1. **File Reading**: All `.txt` files in `sample_data/{language}/`
2. **Chunking**: Split into non-empty lines, group by 3
3. **Embedding Generation**: OpenAI API calls with batching
4. **Index Building**: FAISS IndexFlatIP with L2 normalization
5. **Metadata Storage**: JSON file linking chunks to sources

### 2. Query Processing Pipeline

```
User question → Embedding → FAISS search → Context prompt → LLM answer
```

1. **Query Embedding**: Convert question to vector representation
2. **Similarity Search**: Find top-k most similar vocabulary chunks
3. **Context Assembly**: Build prompt with retrieved chunks
4. **Answer Generation**: GPT model generates educational response
5. **Source Attribution**: Include file sources and confidence scores

### 3. Chunking Rationale

**Why 3-line chunks?**
- Single lines lack context
- Large chunks dilute semantic focus
- 3 lines typically contain related phrase groups
- Optimal balance for language learning content

### 4. FAISS Index Choice

**Why IndexFlatIP?**
- Exact cosine similarity (after normalization)
- No approximation - perfect for educational accuracy
- Fast enough for vocabulary-sized datasets
- Consistent, reproducible results

---

## 🚨 Common Issues & Solutions

### "Index or metadata missing"
```bash
# Solution: Run ingestion first
poetry run python scripts/ingest.py ingest
```

### "OPENAI_API_KEY environment variable is required"
```bash
# Solution: Set up .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### "No .txt files found for language"
```bash
# Solution: Check language setting and data directory
# Verify: ls sample_data/{your_language}/*.txt
```

### Empty or poor answers
- Check if vocabulary files contain relevant content
- Try different retrieval parameters (--top-k)
- Verify FAISS index was built correctly

---

## 📈 Future Enhancements

### Potential Improvements

1. **Audio Integration**: Text-to-speech for pronunciation
2. **Progress Tracking**: User learning progress and analytics
3. **Adaptive Learning**: Personalized difficulty adjustment
4. **Multi-modal**: Image-based vocabulary learning
5. **Conversation Practice**: Full dialogue simulation
6. **Mobile Interface**: Web or mobile app frontend
7. **Offline Mode**: Local model support
8. **Community Features**: Shared vocabulary sets

### Technical Upgrades

- **Vector Database**: Switch to Pinecone/Weaviate for production
- **Model Fine-tuning**: Custom embeddings for language learning
- **Caching Layer**: Redis for frequently asked questions
- **API Service**: REST API for frontend integration
- **Monitoring**: Usage analytics and performance metrics

---

## 📝 Summary

This RAG project represents a sophisticated language learning system that combines:

- **Modern NLP**: State-of-the-art embeddings and language models
- **Educational Focus**: Designed specifically for vocabulary acquisition
- **Practical Implementation**: Ready-to-use CLI interface
- **Extensible Architecture**: Easy to add new languages and features
- **Production Ready**: Proper configuration, error handling, and logging

The system demonstrates how RAG can be applied to educational use cases, providing accurate, contextual answers while maintaining transparency about information sources. It's particularly well-suited for Memrise-style flashcard learning, where students need quick, accurate access to vocabulary and phrases in context.

---

## 🤝 Contributing

To contribute to this project:

1. Follow the existing code structure and naming conventions
2. Add comprehensive logging for new features
3. Update configuration documentation for new settings
4. Test with multiple languages before submitting
5. Maintain the educational focus of the system

---

*This documentation covers the complete RAG Project architecture, setup, and usage. For specific implementation details, refer to the individual script files and their docstrings.* 