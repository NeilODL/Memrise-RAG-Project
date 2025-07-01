#!/usr/bin/env python3
"""
Interactive CLI question answering script for RAG Project.
Allows user to ask questions interactively in a chat-like interface.
"""

import logging
from pathlib import Path
import typer

# Import the components from answer.py
from answer import VectorRetriever, RAGAnswerer, load_chunk_texts
from config import Config

# Set up logging (but make it less verbose for interactive use)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def main(
    data_path: str = typer.Option("sample_data", help="Path to original data directory for chunk text lookup"),
    top_k: int = typer.Option(3, help="Number of relevant chunks to retrieve"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model to use")
):
    """
    Start an interactive Spanish learning assistant.
    Ask questions about Spanish phrases and get immediate answers!
    
    Type 'quit', 'exit', or press Ctrl+C to stop.
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Set up paths
        data_path = Path(data_path)
        index_path = Config.OUTPUT_DIR / Config.INDEX_FILE
        metadata_path = Config.OUTPUT_DIR / Config.METADATA_FILE
        
        # Validate that required files exist
        if not index_path.exists():
            typer.echo(f"❌ Error: Index file {index_path} does not exist. Run ingest.py first.")
            raise typer.Exit(1)
        
        if not metadata_path.exists():
            typer.echo(f"❌ Error: Metadata file {metadata_path} does not exist. Run ingest.py first.")
            raise typer.Exit(1)
        
        # Initialize components (do this once at startup)
        typer.echo("🚀 Initializing Spanish Learning Assistant...")
        retriever = VectorRetriever(index_path, metadata_path)
        answerer = RAGAnswerer(Config.OPENAI_API_KEY, model)
        
        # Load chunk texts for context
        typer.echo("📚 Loading vocabulary data...")
        chunk_texts = load_chunk_texts(data_path, retriever.metadata)
        
        # Welcome message
        typer.echo("\n" + "="*60)
        typer.echo("🇪🇸 Welcome to your Spanish Learning Assistant! 🇪🇸")
        typer.echo("="*60)
        typer.echo("Ask me anything about Spanish phrases and vocabulary!")
        typer.echo("Type 'quit', 'exit', or press Ctrl+C to stop.")
        typer.echo("="*60 + "\n")
        
        # Interactive loop
        while True:
            try:
                # Get user input
                question = input("❓ Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', 'bye']:
                    typer.echo("\n👋 ¡Adiós! Thanks for learning Spanish with me!")
                    break
                
                # Skip empty questions
                if not question:
                    typer.echo("Please ask a question!")
                    continue
                
                # Process the question
                typer.echo("\n🔍 Searching for relevant information...")
                
                # Get query embedding
                query_embedding = answerer.get_query_embedding(question)
                
                # Retrieve relevant chunks
                relevant_chunks = retriever.search(query_embedding, k=top_k)
                
                if not relevant_chunks:
                    typer.echo("❌ Sorry, I couldn't find relevant information for your question.")
                    typer.echo("Try rephrasing or asking about basic Spanish phrases.\n")
                    continue
                
                # Generate answer
                prompt = answerer.create_context_prompt(question, relevant_chunks, chunk_texts)
                answer_text = answerer.generate_answer(prompt)
                
                # Extract sources
                sources = answerer.extract_sources(relevant_chunks)
                
                # Display results in a nice format
                typer.echo("\n" + "-"*50)
                typer.echo(f"💡 {answer_text}")
                typer.echo(f"\n📚 Found in: {', '.join(sources)}")
                typer.echo(f"🎯 Confidence: {relevant_chunks[0]['score']:.3f}")
                typer.echo("-"*50 + "\n")
                
            except KeyboardInterrupt:
                typer.echo("\n\n👋 ¡Adiós! Thanks for learning Spanish with me!")
                break
            except Exception as e:
                typer.echo(f"\n❌ Oops! Something went wrong: {e}")
                typer.echo("Please try asking your question again.\n")
        
    except Exception as e:
        typer.echo(f"❌ Error starting the assistant: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 