#!/usr/bin/env python3
"""
Simple main script for RAG Project.
Choose a language, run ingestion, then answer questions.
"""

import subprocess
import sys
from pathlib import Path
import typer

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

app = typer.Typer()

def run_script(script_name: str, command: str, description: str) -> bool:
    """Run a script command and return success status."""
    try:
        typer.echo(f"\n{description}...")
        result = subprocess.run(
            ["poetry", "run", "python", f"scripts/{script_name}", command],
            cwd=Path(__file__).parent.parent,
            text=True
        )
        
        if result.returncode == 0:
            typer.echo(f"SUCCESS: {description} completed!")
            return True
        else:
            typer.echo(f"ERROR: {description} failed!")
            return False
    except Exception as e:
        typer.echo(f"ERROR: {description} failed: {e}")
        return False

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    language: str = typer.Option(None, help="Language to use (e.g., 'spanish', 'german')")
):
    """
    Run the complete RAG workflow: choose language, ingest data, answer questions.
    """
    if ctx.invoked_subcommand is not None:
        return
    
    try:
        # Show available languages if none specified
        if not language:
            available_languages = Config.list_available_languages()
            if not available_languages:
                typer.echo("ERROR: No languages with data found.")
                typer.echo("Generate data first with: poetry run python scripts/generate_language_data.py generate [Language]")
                raise typer.Exit(1)
            
            typer.echo("Available languages:")
            for i, lang in enumerate(available_languages, 1):
                typer.echo(f"  {i}. {lang.title()}")
            
            # Get user choice
            while True:
                try:
                    choice = typer.prompt("Select a language (number or name)")
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(available_languages):
                            language = available_languages[idx]
                            break
                    elif choice.lower() in available_languages:
                        language = choice.lower()
                        break
                    else:
                        typer.echo("Invalid choice. Try again.")
                except (ValueError, KeyboardInterrupt):
                    typer.echo("\nCancelled by user")
                    raise typer.Exit(1)
        
        language = language.lower()
        typer.echo(f"\nRunning RAG workflow for {language.title()}")
        
        # Check if language data exists
        data_dir = Config.PROJECT_ROOT / "sample_data" / language
        if not data_dir.exists() or not list(data_dir.glob("*.txt")):
            typer.echo(f"ERROR: No data found for {language.title()}")
            typer.echo(f"Generate data first with: poetry run python scripts/generate_language_data.py generate {language.title()}")
            raise typer.Exit(1)
        
        # Update config file
        config_path = Config.PROJECT_ROOT / "config.py"
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace the LANGUAGE line
        import re
        new_content = re.sub(
            r'LANGUAGE = "[^"]*"',
            f'LANGUAGE = "{language}"',
            config_content
        )
        
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        typer.echo(f"Updated config.py to use {language.title()}")
        
        # Step 1: Run ingestion
        if not run_script("ingest.py", "ingest", f"Building FAISS index for {language.title()}"):
            raise typer.Exit(1)
        
        # Step 2: Answer questions
        if not run_script("answer.py", "answer", f"Answering {language.title()} questions"):
            typer.echo("WARNING: Question answering had issues, but continuing...")
        
        typer.echo(f"\nRAG workflow completed for {language.title()}!")
        typer.echo("You can now use the interactive mode:")
        typer.echo("   poetry run python scripts/ask.py")
        
    except KeyboardInterrupt:
        typer.echo("\nCancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"ERROR: Workflow failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 