"""FastAPI 应用 — RAG 知识库助手后端服务。"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from rag_server.chat import chat_stream
from config import get_config, Config

_pipeline: RAGPipeline | None = None
_store: VectorStoreManager | None = None
_watcher: VaultWatcher | None = None


def init_app(config: Config | None = None):
    global _pipeline, _store, _watcher
    cfg = config or get_config()

    _store = VectorStoreManager(persist_dir=cfg.chroma_persist_dir)
    _pipeline = RAGPipeline(store=_store)

    if cfg.obsidian_vault_path:
        _watcher = VaultWatcher(
            store=_store,
            vault_path=cfg.obsidian_vault_path,
            ignore_dirs=list(cfg.obsidian_ignore_dirs),
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

        if _pipeline is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

        async def sse_generator():
            async for event_data in chat_stream(
                _pipeline, question, session_id=session_id, folder=folder, tag=tag
            ):
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/status")
    async def status():
        if _store is None:
            return JSONResponse({"status": "not_initialized"}, status_code=200)

        stats = _store.get_stats()
        return JSONResponse({
            "status": "ok",
            "index": stats,
            "vault_path": cfg.obsidian_vault_path,
        })

    @app.post("/api/reindex")
    async def reindex():
        if _watcher is None:
            raise HTTPException(status_code=400, detail="未配置 Obsidian 仓库路径")

        try:
            result = _watcher.rebuild()
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sources/{source:path}")
    async def get_source(source: str):
        if _store is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

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
