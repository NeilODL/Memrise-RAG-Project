#!/usr/bin/env python3
"""
Language Data Generation Script for RAG Project.
Generates vocabulary files for language learning using OpenAI.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import typer # type: ignore
from openai import OpenAI # type: ignore

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

class LanguageDataGenerator:
    """Generates language learning data using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Define the categories we want to generate
        self.categories = {
            "greetings": "Basic greetings and introductions for everyday social interactions",
            "restaurant": "Ordering food, asking about menu items, and restaurant etiquette",
            "travel": "Transportation, booking accommodations, and general travel needs",
            "shopping": "Buying items, asking about prices, and retail interactions",
            "health": "Medical needs, pharmacy visits, and health-related conversations",
            "accommodation": "Hotel check-in/out, room requests, and lodging needs",
            "navigation": "Asking for directions, finding locations, and getting around",
            "weather": "Discussing weather conditions and forecasts",
            "time": "Telling time, scheduling, and time-related expressions",
            "internet": "Technology, wifi, and digital communication needs"
        }
    
    def generate_category_content(self, language: str, category: str, description: str) -> str:
        """
        Generate content for a specific category in the target language.
        
        Args:
            language: Target language (e.g., "German", "French")
            category: Category name (e.g., "greetings", "restaurant")
            description: Description of what the category should contain
            
        Returns:
            Generated content as a string
        """
        prompt = f"""Generate exactly 15 useful {language} phrases for language learners in the category: {category}.

Category description: {description}

Requirements:
1. Each line should follow this exact format: "{language} phrase – English translation"
2. Include both formal and informal expressions where appropriate
3. Focus on practical, commonly used phrases that beginners would need
4. Make sure the phrases are culturally appropriate for {language}-speaking countries
5. Include a mix of questions, statements, and requests
6. Ensure the {language} is grammatically correct and natural
7. Provide clear, concise English translations
8. Each phrase should be on its own line
9. Do not include any numbering, bullet points, or additional formatting
10. Generate exactly 15 phrases, no more, no less

Example format (for reference only, generate new content):
guten Tag – good day
wie geht es Ihnen? – how are you? (formal)
es freut mich, Sie kennenzulernen – pleased to meet you

Now generate 15 {category} phrases in {language}:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Some creativity but still consistent
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"Generated {category} content for {language}")
            return content
            
        except Exception as e:
            logger.error(f"Error generating {category} content for {language}: {e}")
            raise
    
    def validate_content(self, content: str, expected_lines: int = 15) -> bool:
        """
        Validate that the generated content meets our requirements.
        
        Args:
            content: Generated content to validate
            expected_lines: Expected number of non-empty lines
            
        Returns:
            True if content is valid, False otherwise
        """
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) != expected_lines:
            logger.warning(f"Expected {expected_lines} lines, got {len(lines)}")
            return False
        
        # Check that each line has the expected format (contains " – " or " - ")
        for i, line in enumerate(lines, 1):
            if " – " not in line and " - " not in line:
                logger.warning(f"Line {i} doesn't follow expected format: {line}")
                return False
        
        return True
    
    def save_category_file(self, content: str, output_path: Path):
        """
        Save generated content to a file.
        
        Args:
            content: Content to save
            output_path: Path where to save the file
        """
        try:
            # Normalize content: convert regular hyphens to en dashes for consistency
            normalized_content = content.replace(" - ", " – ")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(normalized_content)
                if not normalized_content.endswith('\n'):
                    f.write('\n')
            
            logger.info(f"Saved content to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving content to {output_path}: {e}")
            raise

@app.command()
def generate(
    language: str = typer.Argument(..., help="Target language to generate (e.g., 'German', 'French')"),
    output_dir: str = typer.Option("sample_data", help="Base directory for generated files"),
    model: str = typer.Option("gpt-4", help="OpenAI model to use for generation"),
    force: bool = typer.Option(False, help="Overwrite existing files if they exist")
):
    """
    Generate language learning vocabulary files for the specified language.
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Set up paths
        output_dir = Path(output_dir)
        language_dir = output_dir / language.lower()
        
        # Create language directory
        language_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files already exist
        if not force:
            existing_files = list(language_dir.glob("*.txt"))
            if existing_files:
                typer.echo(f"Warning: Files already exist in {language_dir}")
                typer.echo("Use --force to overwrite existing files")
                raise typer.Exit(1)
        
        logger.info(f"Generating {language} vocabulary files in {language_dir}")
        
        # Initialize generator
        generator = LanguageDataGenerator(Config.OPENAI_API_KEY, model)
        
        # Generate content for each category
        generated_files = []
        for category, description in generator.categories.items():
            typer.echo(f"Generating {category}...")
            
            # Generate content
            content = generator.generate_category_content(language, category, description)
            
            # Validate content
            if not generator.validate_content(content):
                typer.echo(f"ERROR: Generated content for {category} failed validation")
                continue
            
            # Save to file
            output_path = language_dir / f"{category}.txt"
            generator.save_category_file(content, output_path)
            generated_files.append(output_path)
        
        # Generate summary
        typer.echo(f"\nSUCCESS: Generated {len(generated_files)} files for {language}!")
        typer.echo(f"Files saved to: {language_dir}")
        
        # List generated files
        typer.echo("\nGenerated files:")
        for file_path in sorted(generated_files):
            file_size = file_path.stat().st_size
            typer.echo(f"  - {file_path.name} ({file_size} bytes)")
        
        # Create a simple report
        total_phrases = len(generated_files) * 15
        typer.echo(f"\nTotal phrases generated: {total_phrases}")
        typer.echo(f"Language: {language}")
        typer.echo(f"Categories: {len(generated_files)}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

@app.command()
def list_categories():
    """List all available categories that can be generated."""
    generator = LanguageDataGenerator("dummy_key")  # Just for accessing categories
    
    typer.echo("Available categories:")
    for category, description in generator.categories.items():
        typer.echo(f"  {category}: {description}")

if __name__ == "__main__":
    app() 