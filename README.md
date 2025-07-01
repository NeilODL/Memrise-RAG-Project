# RAG Project - Multilingual Language Learning Assistant

A Retrieval-Augmented Generation (RAG) system for language learning that uses semantic search and AI to answer vocabulary questions across multiple languages.

## Setup

**Prerequisites:**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd "Memrise/RAG Project"
poetry install

# Configure OpenAI API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_key_here
```

## Usage

**Generate vocabulary for a new language:**
```bash
poetry run python scripts/generate_language_data.py generate French
```

**Run complete workflow (choose language, build index, answer questions):**
```bash
poetry run python scripts/main.py
```

**Interactive language learning:**
```bash
poetry run python scripts/ask.py
```

**Individual components:**
```bash
# Build search index only
poetry run python scripts/ingest.py ingest

# Answer predefined questions
poetry run python scripts/answer.py answer
```

**Run tests:**
```bash
poetry run python -m pytest scripts/test_answer.py -v
```

## Technology & Design

**Architecture:** RAG (Retrieval-Augmented Generation) combines semantic search with large language models for accurate, context-aware responses.

**Stack:**
- **FAISS**: Fast approximate nearest neighbor search for semantic retrieval
- **OpenAI Embeddings**: `text-embedding-ada-002` for multilingual understanding
- **OpenAI Chat**: `gpt-3.5-turbo` for natural answer generation

**Chunking Strategy:** Text is split into 3-line chunks of consecutive non-empty lines. This balance provides sufficient context (multiple related phrases) while maintaining precision (focused topic boundaries). Smaller chunks lose context; larger chunks dilute relevance.

**Storage:** ~1MB per language (FAISS index + metadata + text files) 