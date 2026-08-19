"""复现 local→global 顺序触发 "Error finding id"，抓 global 完整堆栈。"""
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["RAG_CONFIG_PATH"] = "config.docker.yaml"

from config import get_config  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.store import VectorStoreManager  # noqa: E402
from rag_core.retrieval.pipeline import RAGPipeline  # noqa: E402

config = get_config()
store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
graph_store = GraphStore(config.graph_db_path)
pipeline = RAGPipeline(store=store, graph_store=graph_store)

Q = "Docker 构建镜像通常包含哪些主要步骤？"

print("=== 1) local 模式（此前成功） ===")
resp = pipeline.ask(Q, mode="local")
print("local status:", resp.get("status"))

print("\n=== 2) global 模式（同进程，此前失败） ===")
try:
    resp = pipeline.ask(Q, mode="global")
    print("global status:", resp.get("status"))
    print("global trace:", pipeline.get_retrieval_trace())
except Exception:
    traceback.print_exc()

graph_store.close()
