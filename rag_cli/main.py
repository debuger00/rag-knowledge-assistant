"""CLI - Typer commands: rag ask, rag index, rag server."""
import os
from typing import Optional

import typer
from rich.console import Console

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from config import get_config

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

app = typer.Typer(
    name="rag",
    help="RAG knowledge assistant for Obsidian vaults",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable chat history"),
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by folder"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
):
    """Ask your knowledge base a question (streaming output)."""
    config = get_config()

    if not config.obsidian_vault_path:
        console.print("[red][ERROR] OBSIDIAN_VAULT_PATH not set. Check your .env file.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Vault: {config.obsidian_vault_path}[/dim]")

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    pipeline = RAGPipeline(store=store)

    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )
    try:
        result = watcher.full_sync()
        console.print(f"[dim]Index: {result['total']} notes[/dim]")
    except FileNotFoundError:
        console.print(f"[red][ERROR] Vault path not found: {config.obsidian_vault_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Searching...[/dim]\n")

    history: list[dict] = [] if no_history else []

    try:
        full_answer = ""
        if folder or tag:
            stream = pipeline.ask_with_filter(question, history, folder=folder, tag=tag)
        else:
            stream = pipeline.ask(question, history)

        # 用 sys.stdout 直接输出，避免 Rich 在 GBK 终端上的编码问题
        import sys
        for chunk in stream:
            if chunk:
                full_answer += chunk
                try:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    # GBK 无法编码的字符用 ? 替代
                    sys.stdout.write(chunk.encode("gbk", errors="replace").decode("gbk"))
                    sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        if not no_history:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})

    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Full rebuild of index"),
    sync: bool = typer.Option(False, "--sync", help="Incremental sync"),
    status: bool = typer.Option(False, "--status", help="Show index status"),
):
    """Manage knowledge base index."""
    config = get_config()

    if status:
        store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
        stats = store.get_stats()
        console.print(f"[bold]Index Status:[/bold]")
        console.print(f"   Parents: {stats['parent_count']}")
        console.print(f"   Children: {stats['child_count']}")
        console.print(f"   Last sync: {stats['last_sync']}")
        return

    if not config.obsidian_vault_path:
        console.print("[red][ERROR] OBSIDIAN_VAULT_PATH not set.[/red]")
        raise typer.Exit(code=1)

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )

    if rebuild:
        console.print("Rebuilding index...")
        result = watcher.rebuild()
        console.print(
            f"[green]Done: {result['total_parents']} parents, "
            f"{result['total_children']} children[/green]"
        )
    elif sync:
        console.print("Syncing...")
        result = watcher.full_sync()
        console.print(
            f"[green]Done: {result['total']} notes "
            f"({result['new']} new, {result['updated']} updated)[/green]"
        )


@app.command()
def server(
    port: int = typer.Option(8501, "--port", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    no_watch: bool = typer.Option(False, "--no-watch", help="Disable file watching"),
):
    """Start Web server."""
    config = get_config()
    config.server_port = port
    config.server_host = host

    console.print(f"Starting RAG Assistant server...")
    console.print(f"   Web UI: http://{host}:{port}")
    console.print(f"   API:    http://{host}:{port}/api/chat")

    from rag_server.app import run_server
    run_server(host=host, port=port, no_watch=no_watch)


if __name__ == "__main__":
    app()
