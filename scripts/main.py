#!/usr/bin/env python3
"""
Main workflow orchestrator for RAG Project.
Handles language selection, runs ingestion, then answer generation.
"""

import subprocess
import sys
import re
from pathlib import Path
import typer

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

app = typer.Typer()

def run_step(script_name: str, command: str, description: str) -> bool:
    """Run a pipeline step and return success status."""
    try:
        typer.echo(f"\n{description}...")
        result = subprocess.run(
            ["poetry", "run", "python", f"scripts/{script_name}", command],
            cwd=Path(__file__).parent.parent,
            text=True
        )
        
        success = result.returncode == 0
        status = "✓ SUCCESS" if success else "✗ ERROR"
        typer.echo(f"{status}: {description} {'completed' if success else 'failed'}!")
        return success
    except Exception as e:
        typer.echo(f"✗ ERROR: {description} failed: {e}")
        return False

def select_language(provided_language: str) -> str:
    """Handle language selection with interactive fallback."""
    available_languages = Config.list_available_languages()
    
    if not available_languages:
        typer.echo("ERROR: No languages with data found.")
        typer.echo("Generate data first: poetry run python scripts/generate_language_data.py generate [Language]")
        raise typer.Exit(1)
    
    # Use provided language if valid
    if provided_language and provided_language.lower() in available_languages:
        return provided_language.lower()
    
    # Interactive selection
    typer.echo("Available languages:")
    for i, lang in enumerate(available_languages, 1):
        typer.echo(f"  {i}. {lang.title()}")
    
    while True:
        try:
            choice = typer.prompt("Select a language (number or name)")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_languages):
                    return available_languages[idx]
            elif choice.lower() in available_languages:
                return choice.lower()
            else:
                typer.echo("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            typer.echo("\nCancelled")
            raise typer.Exit(1)

def update_config_language(language: str):
    """Update config.py to use the selected language."""
    config_path = Config.PROJECT_ROOT / "config.py"
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    updated_content = re.sub(r'LANGUAGE = "[^"]*"', f'LANGUAGE = "{language}"', content)
    
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    typer.echo(f"Updated config to use {language.title()}")

def validate_language_data(language: str):
    """Check that language data exists."""
    data_dir = Config.PROJECT_ROOT / "sample_data" / language
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        typer.echo(f"ERROR: No data found for {language.title()}")
        typer.echo(f"Generate data: poetry run python scripts/generate_language_data.py generate {language.title()}")
        raise typer.Exit(1)

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    language: str = typer.Option(None, help="Language to use (e.g., 'spanish', 'german')")
):
    """
    Complete RAG workflow: language selection → FAISS indexing → question answering.
    """
    if ctx.invoked_subcommand is not None:
        return
    
    try:
        # Step 1: Select and validate language
        selected_language = select_language(language)
        typer.echo(f"\n🌍 Using {selected_language.title()}")
        
        # Step 2: Update config and validate data
        update_config_language(selected_language)
        validate_language_data(selected_language)
        
        # Step 3: Build FAISS index
        if not run_step("ingest.py", "ingest", f"Building FAISS index for {selected_language.title()}"):
            raise typer.Exit(1)
        
        # Step 4: Answer questions
        if not run_step("answer.py", "answer", f"Processing {selected_language.title()} questions"):
            typer.echo("⚠️  WARNING: Question answering had issues, continuing...")
        
        # Success
        typer.echo(f"\n🎉 RAG workflow completed for {selected_language.title()}!")
        typer.echo("Next: poetry run python scripts/ask.py")
        
    except KeyboardInterrupt:
        typer.echo("\nCancelled")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 