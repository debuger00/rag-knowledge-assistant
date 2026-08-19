"""单次单模式问答（模拟 CLI 每进程单查询）。用法: python test_single.py <mode> <问题>"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["RAG_CONFIG_PATH"] = "config.docker.yaml"

from config import get_config  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.store import VectorStoreManager  # noqa: E402
from rag_core.retrieval.pipeline import RAGPipeline  # noqa: E402

mode = sys.argv[1]
question = sys.argv[2]

config = get_config()
store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
graph_store = GraphStore(config.graph_db_path) if config.graph_enabled else None
pipeline = RAGPipeline(store=store, graph_store=graph_store)

resp = pipeline.ask(question, mode=mode)
status = resp.get("status")
answer = " ".join(item["text"] for item in resp.get("answer", []))
citations = len(resp.get("citations", []))
trace = pipeline.get_retrieval_trace()
print(f"[{mode}] status={status} 引用数={citations}")
print(f"  答案: {answer[:200]}")
if trace:
    print(f"  trace: {json.dumps(trace, ensure_ascii=False)[:300]}")
else:
    print("  trace: (空)")

if graph_store is not None:
    graph_store.close()
