"""Docker 文档集三模式问答对比测试（basic / local / global）。"""
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


def main() -> int:
    config = get_config()
    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    graph_store = GraphStore(config.graph_db_path) if config.graph_enabled else None
    pipeline = RAGPipeline(store=store, graph_store=graph_store)

    questions = [
        "Docker 构建镜像通常包含哪些主要步骤？",
        "docker compose 和 docker run 有什么区别？",
        "如何查看正在运行的容器？",
    ]
    modes = ["basic", "local", "global"]

    for q in questions:
        print(f"\n{'=' * 70}\n问题: {q}\n{'=' * 70}")
        for mode in modes:
            try:
                resp = pipeline.ask(q, mode=mode)
                trace = pipeline.get_retrieval_trace()
                status = resp.get("status")
                answer = " ".join(item["text"] for item in resp.get("answer", []))
                citations = len(resp.get("citations", []))
                print(f"\n[模式 {mode}] status={status} 引用数={citations}")
                print(f"  答案: {answer[:200]}")
                if trace:
                    print(f"  trace: {json.dumps(trace, ensure_ascii=False)[:300]}")
                else:
                    print("  trace: (空)")
            except Exception as exc:  # noqa: BLE001
                print(f"\n[模式 {mode}] 错误: {type(exc).__name__}: {exc}")

    if graph_store is not None:
        graph_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
