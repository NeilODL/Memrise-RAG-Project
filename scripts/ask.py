#!/usr/bin/env python3
"""
Interactive CLI for RAG Project.
Chat interface for asking language learning questions.
"""

import logging
from pathlib import Path
import typer

# Add parent directory to path for config import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Import components from answer.py
from answer import VectorRetriever, RAGAnswerer, load_chunk_texts

# Set up logging (less verbose for interactive use)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_path: str = typer.Option(None, help="Data directory path"),
    top_k: int = typer.Option(3, help="Number of chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model")
):
    """
    Interactive language learning assistant.
    Ask questions and get answers using RAG system.
    """
    if ctx.invoked_subcommand is None:
        interactive(data_path, top_k, model)

@app.command()
def interactive(
    data_path: str = typer.Option(None, help="Data directory path"),
    top_k: int = typer.Option(3, help="Number of chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model")
):
    """
    Start interactive language learning chat.
    Ask questions about phrases and vocabulary.
    Type 'quit', 'exit', or Ctrl+C to stop.
    """
    try:
        Config.validate()
        
        # Set paths
        data_path = Path(data_path) if data_path else Config.DATA_DIR
        
        # Validate required files
        if not Config.INDEX_PATH.exists() or not Config.METADATA_PATH.exists():
            typer.echo("ERROR: Index or metadata missing.")
            typer.echo("Run: poetry run python scripts/ingest.py ingest")
            raise typer.Exit(1)
        
        # Initialize RAG system
        language_title = Config.LANGUAGE.title()
        typer.echo(f"Initializing {language_title} Learning Assistant...")
        
        retriever = VectorRetriever(Config.INDEX_PATH, Config.METADATA_PATH)
        answerer = RAGAnswerer(Config.OPENAI_API_KEY, model)
        chunk_texts = load_chunk_texts(data_path, retriever.metadata)
        
        # Welcome message
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Welcome to your {language_title} Learning Assistant!")
        typer.echo(f"{'='*60}")
        typer.echo(f"Ask me anything about {language_title} phrases and vocabulary!")
        typer.echo("Type 'quit', 'exit', or press Ctrl+C to stop.")
        typer.echo(f"{'='*60}\n")
        
        # Interactive loop
        while True:
            try:
                question = input("Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', 'bye']:
                    typer.echo(f"\nThanks for learning {Config.LANGUAGE.title()} with me!")
                    break
                
                if not question:
                    typer.echo("Please ask a question!")
                    continue
                
                # Process question
                typer.echo("\nSearching for relevant information...")
                
                # RAG pipeline
                query_embedding = answerer.get_query_embedding(question)
                relevant_chunks = retriever.search(query_embedding, k=top_k)
                
                if not relevant_chunks:
                    typer.echo("Sorry, I couldn't find relevant information.")
                    typer.echo(f"Try rephrasing or asking about basic {Config.LANGUAGE.title()} phrases.\n")
                    continue
                
                prompt = answerer.create_context_prompt(question, relevant_chunks, chunk_texts)
                answer_text = answerer.generate_answer(prompt)
                sources = answerer.extract_sources(relevant_chunks)
                
                # Display results
                typer.echo(f"\n{'-'*50}")
                typer.echo(f"Answer: {answer_text}")
                typer.echo(f"\nFound in: {', '.join(sources)}")
                typer.echo(f"Confidence: {relevant_chunks[0]['score']:.3f}")
                typer.echo(f"{'-'*50}\n")
                
            except KeyboardInterrupt:
                typer.echo(f"\n\nThanks for learning {Config.LANGUAGE.title()} with me!")
                break
            except Exception as e:
                typer.echo(f"\nERROR: Something went wrong: {e}")
                typer.echo("Please try asking your question again.\n")
        
    except Exception as e:
        typer.echo(f"ERROR: Error starting the assistant: {e}")
        raise typer.Exit(1)

@app.command()
def list_languages():
    """List available languages for interactive learning."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages found.")
            return
        
        typer.echo("Available languages:")
        for language in available_languages:
            data_dir = Config.PROJECT_ROOT / "sample_data" / language
            txt_files = list(data_dir.glob("*.txt"))
            typer.echo(f"  • {language.title()} ({len(txt_files)} files)")
        
        typer.echo(f"\nCurrent: {Config.LANGUAGE}")
        typer.echo("To use: Set LANGUAGE in config.py and run 'poetry run python scripts/ask.py'")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 