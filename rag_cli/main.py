"""CLI - Typer commands: rag ask, rag index, rag server."""
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rag_core.graph.builder import rebuild_structure_graph
from rag_core.graph.communities import (
    LLMCommunityReporter,
    build_communities,
    generate_community_reports,
)
from rag_core.graph.extractor import LLMGraphExtractor
from rag_core.graph.semantic import SemanticGraphIndexer
from rag_core.graph.summarizer import LLMDescriptionSummarizer
from rag_core.graph.store import GraphStore
from rag_core.indexing.loader import ObsidianLoader
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


def _get_history_path(session: str) -> Path:
    """获取指定 session 的历史记录文件路径。"""
    config = get_config()
    history_dir = Path(config.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"{session}.json"


def _load_history(session: str) -> list[dict]:
    """从文件加载指定 session 的历史记录。"""
    path = _get_history_path(session)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(session: str, history: list[dict]) -> None:
    """将指定 session 的历史记录保存到文件。"""
    path = _get_history_path(session)
    # 只保留最近 20 条
    trimmed = history[-20:]
    path.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable chat history"),
    session: str = typer.Option("default", "--session", "-s", help="Session name for history"),
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by folder"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
    clear: bool = typer.Option(False, "--clear", help="Clear history for this session"),
    mode: str = typer.Option(
        "auto", "--mode", help="Retrieval mode: auto, basic, local, global"
    ),
):
    """Ask your knowledge base a question (streaming output)."""
    config = get_config()

    if not config.obsidian_vault_path:
        console.print("[red][ERROR] documents.path not set. Check config.yaml.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Vault: {config.obsidian_vault_path}[/dim]")
    if not no_history:
        console.print(f"[dim]Session: {session}[/dim]")

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    graph_store = GraphStore(config.graph_db_path) if config.graph_enabled else None
    pipeline = RAGPipeline(store=store, graph_store=graph_store)

    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
        graph_store=graph_store,
    )
    try:
        result = watcher.full_sync()
        console.print(f"[dim]Index: {result['total']} notes[/dim]")
    except FileNotFoundError:
        console.print(f"[red][ERROR] Vault path not found: {config.obsidian_vault_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Searching...[/dim]\n")

    # --clear 清除该 session 的历史记录
    if clear:
        _save_history(session, [])
        console.print(f"[dim]History cleared for session '{session}'[/dim]")

    # 加载历史：--no-history 时为空，否则从文件加载
    if no_history:
        history: list[dict] = []
    else:
        history = _load_history(session)

    try:
        if folder or tag:
            response = pipeline.ask_with_filter(
                question, history, folder=folder, tag=tag, mode=mode
            )
        else:
            response = pipeline.ask(question, history, mode=mode)

        console.print_json(data=response)
        full_answer = " ".join(
            item["text"] for item in response.get("answer", [])
        ) or response.get("message", "")

        if not no_history:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})
            _save_history(session, history)

    except Exception as e:
        console.print(f"\n[red][ERROR] {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Full rebuild of index"),
    sync: bool = typer.Option(False, "--sync", help="Incremental sync"),
    status: bool = typer.Option(False, "--status", help="Show index status"),
    graph_only: bool = typer.Option(
        False, "--graph-only", help="Rebuild only the structural graph"
    ),
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
        if config.graph_enabled:
            graph_store = GraphStore(config.graph_db_path)
            graph_stats = graph_store.get_stats()
            console.print(f"   Graph nodes: {graph_stats['node_count']}")
            console.print(f"   Graph edges: {graph_stats['edge_count']}")
            console.print(
                "   Semantic evidence: "
                f"{graph_stats['semantic_node_evidence_count']} entities / "
                f"{graph_stats['semantic_edge_evidence_count']} relationships"
            )
            graph_store.close()
        return

    if not config.obsidian_vault_path:
        console.print("[red][ERROR] documents.path not set in config.yaml.[/red]")
        raise typer.Exit(code=1)

    if graph_only:
        if not config.graph_enabled:
            console.print("[red][ERROR] graph.enabled is false.[/red]")
            raise typer.Exit(code=1)
        console.print("Rebuilding structural graph...")
        graph_store = GraphStore(config.graph_db_path)
        docs = ObsidianLoader(
            config.obsidian_vault_path,
            list(config.obsidian_ignore_dirs),
        ).load()
        result = rebuild_structure_graph(
            graph_store,
            docs,
            child_chunk_size=config.child_chunk_size,
            child_chunk_overlap=config.child_chunk_overlap,
            child_max_len=config.child_max_len_before_split,
        )
        graph_store.close()
        console.print(
            f"[green]Done: {result['node_count']} nodes, "
            f"{result['edge_count']} edges[/green]"
        )
        return

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    graph_store = GraphStore(config.graph_db_path) if config.graph_enabled else None
    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
        graph_store=graph_store,
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


@app.command("graph-build")
def graph_build(
    changed_only: bool = typer.Option(
        True,
        "--changed-only/--all",
        help="Only extract sources whose semantic fingerprint changed",
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Only extract one vault-relative Markdown source"
    ),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Skip rebuilding entity embeddings"
    ),
):
    """Build the offline LLM entity/relationship graph."""
    config = get_config()
    if not config.graph_enabled:
        console.print("[red][ERROR] graph.enabled is false.[/red]")
        raise typer.Exit(code=1)
    if not config.graph_entity_extraction:
        console.print(
            "[red][ERROR] graph.entity_extraction is false. "
            "Enable it explicitly before running costly LLM extraction.[/red]"
        )
        raise typer.Exit(code=1)

    documents = ObsidianLoader(
        config.obsidian_vault_path,
        list(config.obsidian_ignore_dirs),
    ).load()
    graph_store = GraphStore(config.graph_db_path)
    try:
        console.print("Refreshing structural graph...")
        rebuild_structure_graph(
            graph_store,
            documents,
            child_chunk_size=config.child_chunk_size,
            child_chunk_overlap=config.child_chunk_overlap,
            child_max_len=config.child_max_len_before_split,
        )
        vector_store = (
            None
            if no_embeddings
            else VectorStoreManager(persist_dir=config.chroma_persist_dir)
        )
        indexer = SemanticGraphIndexer(
            graph_store,
            LLMGraphExtractor(config),
            config,
            vector_store=vector_store,
            summarizer=LLMDescriptionSummarizer(config),
        )
        result = indexer.build(
            documents,
            changed_only=changed_only,
            source=source,
        )
        if not result["documents_failed"] and config.graph_community_detection:
            result["communities"] = build_communities(graph_store, config)
            if config.graph_community_reports:
                result["community_reports"] = generate_community_reports(
                    graph_store, LLMCommunityReporter(config)
                )
            if vector_store is not None:
                result["community_embeddings"] = (
                    vector_store.rebuild_community_reports(
                        graph_store.list_community_reports()
                    )
                )
        console.print_json(data=result)
        if result["documents_failed"]:
            raise typer.Exit(code=1)
    finally:
        graph_store.close()


@app.command("graph-communities")
def graph_communities(
    reports: bool = typer.Option(
        False, "--reports", help="Also generate cached LLM community reports"
    ),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Skip rebuilding community embeddings"
    ),
):
    """Run Leiden over the current semantic entity graph."""
    config = get_config()
    if not config.graph_enabled:
        console.print("[red][ERROR] graph.enabled is false.[/red]")
        raise typer.Exit(code=1)
    graph_store = GraphStore(config.graph_db_path)
    try:
        result: dict = {"communities": build_communities(graph_store, config)}
        if reports:
            result["community_reports"] = generate_community_reports(
                graph_store, LLMCommunityReporter(config)
            )
        if not no_embeddings:
            vector_store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
            result["community_embeddings"] = vector_store.rebuild_community_reports(
                graph_store.list_community_reports()
            )
        console.print_json(data=result)
    finally:
        graph_store.close()


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
