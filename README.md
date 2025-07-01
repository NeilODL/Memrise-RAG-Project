# RAG Project - Spanish Language Learning Assistant

A Retrieval-Augmented Generation system that answers Spanish vocabulary questions using semantic search and LLM generation.

## Setup

1. **Install Poetry** (if not installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   cd "Memrise/RAG Project"
   poetry install
   ```

3. **Configure OpenAI API**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage

1. **Build the search index**:
   ```bash
   poetry run python ingest.py sample_data
   ```

2. **Answer questions**:
   ```bash
   poetry run python answer.py sample_data/questions.json
   ```

3. **Run tests**:
   ```bash
   poetry run python -m pytest test_answer.py -v
   ```

## Technology & Design Rationale

**Chunking Strategy**: Text is split into 3-line chunks to balance context preservation with retrieval precision. This size captures complete Spanish phrases with translations while avoiding overly long chunks that dilute semantic meaning.

**Technology Selection**:
- **FAISS**: Provides fast cosine similarity search over high-dimensional embeddings, essential for real-time vocabulary lookup
- **OpenAI Embeddings**: text-embedding-ada-002 offers superior semantic understanding for multilingual content compared to alternatives
- **GPT Models**: Generate natural, contextual answers rather than simple phrase lookups

The RAG approach combines precise retrieval of relevant Spanish phrases with intelligent generation, ensuring answers are both accurate and pedagogically useful for language learners.

## Files

- `ingest.py` - Processes vocabulary files and builds FAISS index
- `answer.py` - Handles question answering pipeline  
- `test_answer.py` - Unit tests for happy-path and no-hit cases
- `sample_data/` - Spanish vocabulary organized by topics 