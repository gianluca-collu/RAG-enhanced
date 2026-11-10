import click
import sys
from pathlib import Path

from rag.config import Config
from rag.client import RagClient
from rag.chat import ChatEngine
from rag.utils import Logger


logger = Logger("RAGSimple-CLI")


@click.group()
def cli():
    """RAGSimple - Conversational RAG Framework"""
    pass


@cli.command()
@click.option("--env", default=".env", help="Environment file path")
def chat(env):
    """Interactive chat with RAG system."""
    click.echo("🤖 RAGSimple Interactive Chat")
    click.echo("=" * 50)
    
    try:
        config = Config.from_env(env)
        rag_client = RagClient(config)
        chat_engine = ChatEngine(rag_client=rag_client, enable_memory=True)
        
        click.echo("✅ Connected. Type 'quit' to exit.\n")
        
        while True:
            query = click.prompt("You")
            
            if query.lower() in ["quit", "exit", "q"]:
                click.echo("👋 Goodbye!")
                break
            
            if not query.strip():
                continue
            
            try:
                click.echo("🔄 Processing...", err=True)
                response = chat_engine.send_message(query)
                
                click.echo(f"\n🤖 Assistant: {response['answer']}\n")
                
                if response['sources']:
                    click.echo("📚 Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        score = source.get('score', 0)
                        click.echo(f"  {i}. (Score: {score:.2f})")
                    click.echo()
            
            except Exception as e:
                click.echo(f"❌ Error: {str(e)}", err=True)
    
    except Exception as e:
        click.echo(f"❌ Failed to initialize: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--file", required=True, help="File with queries (one per line)")
@click.option("--output", default="output.jsonl", help="Output file")
@click.option("--env", default=".env", help="Environment file path")
def query(file, output, env):
    """Process batch queries from file."""
    try:
        query_file = Path(file)
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {file}")
        
        config = Config.from_env(env)
        rag_client = RagClient(config)
        
        click.echo(f"📖 Processing queries from: {file}")
        click.echo(f"💾 Writing output to: {output}")
        click.echo("=" * 50)
        
        with open(output, "w") as out_f:
            with open(query_file, "r") as in_f:
                for i, line in enumerate(in_f, 1):
                    query_text = line.strip()
                    if not query_text:
                        continue
                    
                    try:
                        click.echo(f"Query {i}: {query_text[:50]}...", err=True)
                        result = rag_client.query(query_text)
                        import json
                        out_f.write(json.dumps(result) + "\n")
                    except Exception as e:
                        click.echo(f"  ❌ Error: {str(e)}", err=True)
        
        click.echo(f"✅ Completed! Results in: {output}")
    
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--env", default=".env", help="Environment file path")
def test(env):
    """Test connection to Azure services."""
    click.echo("🧪 Testing RAGSimple Configuration")
    click.echo("=" * 50)
    
    try:
        config = Config.from_env(env)
        
        tests = [
            ("OpenAI URL", lambda: config.openai_url),
            ("OpenAI Key", lambda: "✓" if config.openai_key else "✗"),
            ("Search URL", lambda: config.search_url),
            ("Search Key", lambda: "✓" if config.search_key else "✗"),
            ("Index Name", lambda: config.index_name),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                click.echo(f"✅ {test_name}: {result}")
            except Exception as e:
                click.echo(f"❌ {test_name}: {str(e)}", err=True)
        
        # Test RAG client initialization
        click.echo("\n🔧 Initializing RAG Client...")
        try:
            rag_client = RagClient(config)
            click.echo("✅ RAG Client initialized successfully")
        except Exception as e:
            click.echo(f"❌ RAG Client initialization failed: {str(e)}", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Configuration error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--file", "file_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--env", default=".env", help="Environment file path")
def ingest(file_path, env):
    """Chunk and index a document."""
    try:
        config = Config.from_env(env)
        rag_client = RagClient(config)
        result = rag_client.ingest_file(file_path)
        click.echo(
            f"Indexed {result['indexed_count']} chunks from {result['source']} "
            f"(document id: {result['doc_id']})"
        )
    except Exception as e:
        click.echo(f"❌ Ingestion failed: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
