"""FastAPI 应用 — RAG 知识库助手后端服务。"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_core.graph.models import document_node_id
from rag_core.graph.store import GraphStore
from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from rag_server.chat import chat_answer
from config import get_config, Config

_pipeline: RAGPipeline | None = None
_store: VectorStoreManager | None = None
_watcher: VaultWatcher | None = None
_graph_store: GraphStore | None = None


def init_app(config: Config | None = None):
    global _pipeline, _store, _watcher, _graph_store
    cfg = config or get_config()

    _store = VectorStoreManager(persist_dir=cfg.chroma_persist_dir)
    _graph_store = GraphStore(cfg.graph_db_path) if cfg.graph_enabled else None
    _pipeline = RAGPipeline(store=_store, graph_store=_graph_store)

    if cfg.obsidian_vault_path:
        _watcher = VaultWatcher(
            store=_store,
            vault_path=cfg.obsidian_vault_path,
            ignore_dirs=list(cfg.obsidian_ignore_dirs),
            graph_store=_graph_store,
        )
        try:
            _watcher.full_sync()
        except FileNotFoundError:
            pass

        try:
            _watcher.start_watching()
        except Exception:
            pass


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_app(config)
        yield
        if _watcher:
            _watcher.stop_watching()
        if _graph_store:
            _graph_store.close()

    app = FastAPI(title="RAG 知识库助手", version="0.1.0", lifespan=lifespan)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"message": "RAG 知识库助手 API 服务运行中"}, status_code=200)

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空")

        session_id = body.get("session_id", "default")
        folder = body.get("folder")
        tag = body.get("tag")
        mode = body.get("mode", "auto")
        debug_retrieval = bool(body.get("debug_retrieval", False))
        if mode not in {"auto", "basic", "local", "global"}:
            raise HTTPException(
                status_code=400, detail="mode 必须是 auto、basic、local 或 global"
            )

        if _pipeline is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

        result = await chat_answer(
            _pipeline,
            question,
            session_id=session_id,
            folder=folder,
            tag=tag,
            mode=mode,
            debug_retrieval=debug_retrieval,
        )
        return JSONResponse(result)

    @app.get("/api/status")
    async def status():
        if _store is None:
            return JSONResponse({"status": "not_initialized"}, status_code=200)

        stats = _store.get_stats()
        payload = {
            "status": "ok",
            "index": stats,
            "config": {
                "vault_path": cfg.obsidian_vault_path,
                "embedding_model": cfg.embedding_model,
                "embedding_device": cfg.embedding_device,
                "llm_model": cfg.llm_model,
                "llm_base_url": cfg.llm_base_url,
                "retrieval_top_k": cfg.retrieval_top_k,
                "retrieval_score_threshold": cfg.retrieval_score_threshold,
                "rag_max_citations": cfg.rag_max_citations,
                "rag_max_retry": cfg.rag_max_retry,
                "rag_require_citations": cfg.rag_require_citations,
                "server_port": cfg.server_port,
                "graph_enabled": cfg.graph_enabled,
            },
        }
        if _graph_store is not None:
            payload["graph"] = _graph_store.get_stats()
        return JSONResponse(payload)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/reindex")
    async def reindex(scope: str = Query(default="all")):
        if _watcher is None:
            raise HTTPException(status_code=400, detail="未配置 Obsidian 仓库路径")

        try:
            if scope == "graph":
                if _graph_store is None:
                    raise HTTPException(status_code=400, detail="图索引未启用")
                result = _watcher.rebuild_graph()
            elif scope == "vector":
                result = _watcher.rebuild(include_graph=False)
            elif scope == "all":
                result = _watcher.rebuild()
            else:
                raise HTTPException(
                    status_code=400, detail="scope 必须是 all、vector 或 graph"
                )
            return JSONResponse({"status": "ok", **result})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/graph/status")
    async def graph_status():
        if _graph_store is None:
            return JSONResponse({"status": "disabled"})
        return JSONResponse({"status": "ok", **_graph_store.get_stats()})

    @app.get("/api/graph/neighbors/{node_id:path}")
    async def graph_neighbors(node_id: str, limit: int = Query(default=50, ge=1, le=200)):
        if _graph_store is None:
            raise HTTPException(status_code=404, detail="图索引未启用")
        result = _graph_store.neighbors(node_id, limit=limit)
        if result is None:
            raise HTTPException(status_code=404, detail="图节点不存在")
        return JSONResponse(result)

    @app.get("/api/graph/subgraph")
    async def graph_subgraph(
        source: str = Query(...),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        if _graph_store is None:
            raise HTTPException(status_code=404, detail="图索引未启用")
        result = _graph_store.neighbors(document_node_id(source), limit=limit)
        if result is None:
            raise HTTPException(status_code=404, detail=f"图中未找到笔记: {source}")
        return JSONResponse(result)

    @app.get("/api/sources/{source:path}")
    async def get_source(
        source: str,
        anchor: str | None = Query(default=None),
    ):
        if _store is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

        if anchor:
            cited_doc = _store.search_child_by_citation(source, anchor)
            if cited_doc is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到引用: {source}#{anchor}",
                )
            return JSONResponse({
                "source": source,
                "anchor": anchor,
                "content": cited_doc.page_content,
                "metadata": cited_doc.metadata,
            })

        docs = _store.search_parents_by_source(source)
        if not docs:
            raise HTTPException(status_code=404, detail=f"未找到笔记: {source}")

        doc = docs[0]
        return JSONResponse({
            "source": doc.metadata.get("source"),
            "content": doc.page_content,
            "metadata": doc.metadata,
        })

    return app


def run_server(host: str = "127.0.0.1", port: int = 8501, no_watch: bool = False):
    import uvicorn

    if no_watch and _watcher:
        _watcher.stop_watching()

    uvicorn.run(
        "rag_server.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="info",
    )
