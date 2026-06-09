"""CLI 入口 — Typer 命令：rag ask, rag index, rag server。"""
from typing import Optional

import typer
from rich.console import Console

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from config import get_config

app = typer.Typer(
    name="rag",
    help="个人知识库 RAG 问答助手",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="你的问题"),
    no_history: bool = typer.Option(False, "--no-history", help="不使用对话历史"),
    folder: Optional[str] = typer.Option(None, "--folder", help="限定搜索文件夹"),
    tag: Optional[str] = typer.Option(None, "--tag", help="限定搜索标签"),
):
    """向知识库提问（流式输出）。"""
    config = get_config()

    if not config.obsidian_vault_path:
        console.print("[red]❌ 请先设置 OBSIDIAN_VAULT_PATH 环境变量或 .env 文件[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]📚 仓库: {config.obsidian_vault_path}[/dim]")

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    pipeline = RAGPipeline(store=store)

    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )
    try:
        result = watcher.full_sync()
        console.print(f"[dim]📊 索引: {result['total']} 篇笔记[/dim]")
    except FileNotFoundError:
        console.print(f"[red]❌ Obsidian 仓库路径不存在: {config.obsidian_vault_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]🔍 正在检索...[/dim]\n")

    history: list[dict] = [] if no_history else []

    try:
        full_answer = ""
        if folder or tag:
            stream = pipeline.ask_with_filter(question, history, folder=folder, tag=tag)
        else:
            stream = pipeline.ask(question, history)

        for chunk in stream:
            if chunk:
                full_answer += chunk
                console.print(chunk, end="")

        console.print("\n")

        if not no_history:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})

    except Exception as e:
        console.print(f"\n[red]❌ 错误: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="全量重建索引"),
    sync: bool = typer.Option(False, "--sync", help="增量同步一次"),
    status: bool = typer.Option(False, "--status", help="查看索引状态"),
):
    """管理知识库索引。"""
    config = get_config()

    if status:
        store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
        stats = store.get_stats()
        console.print(f"[bold]📊 索引状态:[/bold]")
        console.print(f"   父文档数: {stats['parent_count']}")
        console.print(f"   子块数:   {stats['child_count']}")
        console.print(f"   最近同步: {stats['last_sync']}")
        return

    if not config.obsidian_vault_path:
        console.print("[red]❌ 请先设置 OBSIDIAN_VAULT_PATH[/red]")
        raise typer.Exit(code=1)

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )

    if rebuild:
        console.print("🔄 正在全量重建索引...")
        result = watcher.rebuild()
        console.print(
            f"[green]✅ 索引重建完成: {result['total_parents']} 篇父文档, "
            f"{result['total_children']} 个子块[/green]"
        )
    elif sync:
        console.print("🔄 正在增量同步...")
        result = watcher.full_sync()
        console.print(
            f"[green]✅ 同步完成: {result['total']} 篇笔记 "
            f"({result['new']} 新增, {result['updated']} 更新)[/green]"
        )


@app.command()
def server(
    port: int = typer.Option(8501, "--port", help="服务端口"),
    host: str = typer.Option("127.0.0.1", "--host", help="绑定地址"),
    no_watch: bool = typer.Option(False, "--no-watch", help="不启动文件监听"),
):
    """启动 Web 服务。"""
    config = get_config()
    config.server_port = port
    config.server_host = host

    console.print(f"🚀 启动 RAG 知识库助手服务...")
    console.print(f"   Web 界面: http://{host}:{port}")
    console.print(f"   API:      http://{host}:{port}/api/chat")

    from rag_server.app import run_server
    run_server(host=host, port=port, no_watch=no_watch)


if __name__ == "__main__":
    app()
