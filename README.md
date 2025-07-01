# RAG Project - Multilingual Language Learning Assistant

A Retrieval-Augmented Generation system that answers vocabulary questions using semantic search and LLM generation. Supports multiple languages with easy configuration switching.

## Quick Start

### Prerequisites

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
   # Edit .env and add your OpenAI API key: OPENAI_API_KEY=your_key_here
   ```

## Simple Workflows

### Option 1: Use Existing Language (Spanish/German)

**Just run the main script and choose a language:**
```bash
poetry run python scripts/main.py
```

This will:
1. Show available languages
2. Let you choose one (by number or name)
3. Build the FAISS index
4. Answer sample questions
5. Save results to `answers/` folder

### Option 2: Generate New Language Data First

1. **Generate vocabulary for a new language:**
   ```bash
   poetry run python scripts/generate_language_data.py generate French
   ```

2. **Run the main workflow:**
   ```bash
   poetry run python scripts/main.py
   # Choose your new language when prompted
   ```

### Option 3: Specify Language Directly

```bash
# Skip the selection menu
poetry run python scripts/main.py --language spanish
poetry run python scripts/main.py --language german
```

### Interactive Learning

After running the main script, start chatting:
```bash
poetry run python scripts/ask.py
```

## Individual Commands

If you need to run components separately:

### Generate Language Data
```bash
# Generate vocabulary for a new language
poetry run python scripts/generate_language_data.py generate French

# Use different model or overwrite existing
poetry run python scripts/generate_language_data.py generate French --model gpt-4 --force
```

### Build Search Index Only
```bash
# First set language in config.py: LANGUAGE = "spanish"
poetry run python scripts/ingest.py ingest
```

### Answer Questions Only
```bash
# Requires existing index
poetry run python scripts/answer.py answer
```

### Interactive Mode
```bash
# Chat with your language assistant
poetry run python scripts/ask.py
```

### Run Tests
```bash
poetry run python -m pytest scripts/test_answer.py -v
```

## Generated Content Structure

When you generate a new language, you'll get:

```
sample_data/
└── french/                     # Your new language
    ├── greetings.txt           # Basic greetings (15 phrases)
    ├── restaurant.txt          # Restaurant/food (15 phrases)
    ├── travel.txt              # Transportation (15 phrases)
    ├── shopping.txt            # Shopping/retail (15 phrases)
    ├── health.txt              # Medical/health (15 phrases)
    ├── accommodation.txt       # Hotels/lodging (15 phrases)
    ├── navigation.txt          # Directions/locations (15 phrases)
    ├── weather.txt             # Weather/climate (15 phrases)
    ├── time.txt                # Time/scheduling (15 phrases)
    └── internet.txt            # Technology/wifi (15 phrases)
```

Each file contains phrases in this format:
```
bonjour – hello
comment allez-vous? – how are you?
je voudrais – I would like
```

## Usage Examples

### Complete New Language Setup
```bash
# 1. Generate Italian vocabulary
poetry run python scripts/generate_language_data.py generate Italian

# 2. Run main workflow
poetry run python scripts/main.py
# Select "Italian" when prompted

# 3. Start learning!
poetry run python scripts/ask.py
```

### Daily Usage
```bash
# Quick workflow for existing language
poetry run python scripts/main.py --language spanish

# Then start chatting
poetry run python scripts/ask.py
```

### Example Interactive Session
```bash
poetry run python scripts/ask.py

# Example questions you can ask:
# "How do you say hello?"
# "What's the word for restaurant?"
# "How do you ask for directions?"
```

## Project Structure

```
RAG Project/
├── config.py                    # Configuration settings
├── scripts/                     # All executable scripts
│   ├── main.py                 # Simple workflow orchestration
│   ├── generate_language_data.py # Generate new language vocab
│   ├── ingest.py               # Build FAISS indexes
│   ├── answer.py               # Question answering pipeline
│   ├── ask.py                  # Interactive CLI mode
│   └── test_answer.py          # Unit tests
├── answers/                     # Generated answer files
├── indexes/                     # FAISS indexes and metadata
├── sample_data/                 # Vocabulary organized by language
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

**"No languages with data found"**
```bash
# Generate data first:
poetry run python scripts/generate_language_data.py generate YourLanguage
```

**"OpenAI API key not found"**
```bash
# Set up your .env file:
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

**Need to see what's available?**
```bash
# Check available languages:
poetry run python scripts/ingest.py list-languages
```

### Getting Help
```bash
# Get help for any command:
poetry run python scripts/main.py --help
poetry run python scripts/ask.py --help
poetry run python scripts/generate_language_data.py --help
```

## Technical Details

- **FAISS**: Fast similarity search for semantic retrieval
- **OpenAI Embeddings**: `text-embedding-ada-002` for multilingual understanding  
- **OpenAI Chat**: `gpt-3.5-turbo` or `gpt-4` for natural answer generation
- **Chunking**: 3-line chunks for optimal context/precision balance
- **Storage**: ~1MB per language (index + metadata + text files)

---

**Happy Language Learning!** 