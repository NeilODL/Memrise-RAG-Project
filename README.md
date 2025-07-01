# RAG Project

A simple RAG (Retrieval-Augmented Generation) project for vocabulary generation.

## Prerequisites

- Python 3.9 or higher
- Poetry (Python dependency management tool)

### Installing Poetry (if not already installed)

```bash
# On macOS/Linux/WSL
curl -sSL https://install.python-poetry.org | python3 -

# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Alternative: Install via pip
pip install poetry
```

## Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd path/to/Memrise/RAG\ Project
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   
   **Option A: Install Poetry shell plugin (recommended for regular use)**
   ```bash
   # Install the shell plugin
   poetry self add poetry-plugin-shell
   
   # Then activate the shell
   poetry shell
   ```
   
   **Option B: Use the new env activate command**
   ```bash
   poetry env activate
   ```
   
   **Option C: Run commands without activating (simplest)**
   ```bash
   # Just prefix your commands with 'poetry run'
   poetry run python your_script.py
   ```

   *Note: Poetry 2.0.0+ doesn't include the shell command by default. Choose the option that works best for your workflow.*

## Project Structure

```
RAG Project/
├── sample_data/           # Sample vocabulary data files
│   ├── greetings.txt     # Greeting phrases
│   ├── restaurant.txt    # Restaurant-related vocabulary
│   ├── travel.txt        # Travel-related vocabulary
│   └── generation/       # Data generation utilities
├── pyproject.toml        # Project dependencies and configuration
└── README.md            # This file
```

## Usage

After setup, you can run the project using:

```bash
# If you've activated the poetry shell
python your_script.py

# Or without activating the shell
poetry run python your_script.py
```

## Dependencies

- **tiktoken**: Token counting and text processing utilities 