#!/usr/bin/env python3
"""
Interactive CLI question answering script for RAG Project.
Allows user to ask questions interactively in a chat-like interface.
"""

import logging
from pathlib import Path
import typer

# Add parent directory to path for config import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Import the components from answer.py
from answer import VectorRetriever, RAGAnswerer, load_chunk_texts

# Set up logging (but make it less verbose for interactive use)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_path: str = typer.Option(None, help="Path to original data directory for chunk text lookup (overrides config)"),
    top_k: int = typer.Option(3, help="Number of relevant chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model to use")
):
    """
    Interactive language learning assistant CLI.
    Start asking questions immediately or use subcommands.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, run interactive mode
        interactive(data_path, top_k, model)

@app.command()
def interactive(
    data_path: str = typer.Option(None, help="Path to original data directory for chunk text lookup (overrides config)"),
    top_k: int = typer.Option(3, help="Number of relevant chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model to use")
):
    """
    Start an interactive language learning assistant.
    Ask questions about phrases and vocabulary and get immediate answers!
    Uses the language configured in Config.LANGUAGE.
    
    Type 'quit', 'exit', or press Ctrl+C to stop.
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Determine paths
        if data_path:
            data_path = Path(data_path)
        else:
            data_path = Config.DATA_DIR
        
        # Validate inputs
        if not Config.INDEX_PATH.exists():
            typer.echo(f"ERROR: Index file {Config.INDEX_PATH} does not exist.")
            typer.echo(f"Run: poetry run python scripts/ingest.py ingest")
            raise typer.Exit(1)
        
        if not Config.METADATA_PATH.exists():
            typer.echo(f"ERROR: Metadata file {Config.METADATA_PATH} does not exist.")
            typer.echo(f"Run: poetry run python scripts/ingest.py ingest")
            raise typer.Exit(1)
        
        # Initialize components (do this once at startup)
        language_title = Config.LANGUAGE.title()
        typer.echo(f"Initializing {language_title} Learning Assistant...")
        retriever = VectorRetriever(Config.INDEX_PATH, Config.METADATA_PATH)
        answerer = RAGAnswerer(Config.OPENAI_API_KEY, model)
        
        # Load chunk texts for context
        typer.echo("Loading vocabulary data...")
        chunk_texts = load_chunk_texts(data_path, retriever.metadata)
        
        # Welcome message
        typer.echo("\n" + "="*60)
        typer.echo(f"Welcome to your {language_title} Learning Assistant!")
        typer.echo("="*60)
        typer.echo(f"Ask me anything about {language_title} phrases and vocabulary!")
        typer.echo("Type 'quit', 'exit', or press Ctrl+C to stop.")
        typer.echo("="*60 + "\n")
        
        # Interactive loop
        while True:
            try:
                # Get user input
                question = input("Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', 'bye']:
                    typer.echo(f"\nThanks for learning {Config.LANGUAGE.title()} with me!")
                    break
                
                # Skip empty questions
                if not question:
                    typer.echo("Please ask a question!")
                    continue
                
                # Process the question
                typer.echo("\nSearching for relevant information...")
                
                # Get query embedding
                query_embedding = answerer.get_query_embedding(question)
                
                # Retrieve relevant chunks
                relevant_chunks = retriever.search(query_embedding, k=top_k)
                
                if not relevant_chunks:
                    typer.echo("Sorry, I couldn't find relevant information for your question.")
                    typer.echo(f"Try rephrasing or asking about basic {Config.LANGUAGE.title()} phrases.\n")
                    continue
                
                # Generate answer
                prompt = answerer.create_context_prompt(question, relevant_chunks, chunk_texts)
                answer_text = answerer.generate_answer(prompt)
                
                # Extract sources
                sources = answerer.extract_sources(relevant_chunks)
                
                # Display results in a nice format
                typer.echo("\n" + "-"*50)
                typer.echo(f"Answer: {answer_text}")
                typer.echo(f"\nFound in: {', '.join(sources)}")
                typer.echo(f"Confidence: {relevant_chunks[0]['score']:.3f}")
                typer.echo("-"*50 + "\n")
                
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
    """List all available languages that can be used for interactive learning."""
    try:
        available_languages = Config.list_available_languages()
        
        if not available_languages:
            typer.echo("No languages with data found.")
            typer.echo(f"Expected data directory: {Config.PROJECT_ROOT / 'sample_data'}")
            return
        
        typer.echo("Available languages for interactive learning:")
        for language in available_languages:
            data_dir = Config.PROJECT_ROOT / "sample_data" / language
            txt_files = list(data_dir.glob("*.txt"))
            typer.echo(f"  • {language.title()} ({len(txt_files)} files)")
        
        typer.echo(f"\nTo use: Set Config.LANGUAGE to one of the above and run 'poetry run python scripts/ask.py'")
        typer.echo(f"Current language: {Config.LANGUAGE}")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 