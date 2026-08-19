"""二分定位：local 中哪一步污染了 Chroma，导致后续 global 失败。"""
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


def run_global(label: str) -> bool:
    try:
        resp = pipeline.ask(Q, mode="global")
        print(f"[{label}] global status={resp.get('status')}")
        return True
    except Exception:
        print(f"[{label}] global FAILED")
        return False


# 基线：全新进程直接 global（预期成功）
print("=== 基线：直接 global ===")
ok0 = run_global("基线")

# 测试 1：先做一次 children filter 查询，再 global
print("\n=== 测试1：先 children filter 查询 ===")
store.similarity_search_with_scores(Q, k=1, filter_dict={"source": "04_image/4.5_build.md"})
ok1 = run_global("测试1")

# 测试 2：先做 entity 查询，再 global
print("\n=== 测试2：先 entity 查询 ===")
store.similarity_search_entities(Q, k=5)
ok2 = run_global("测试2")

# 测试 3：先做 children 无 filter 查询，再 global
print("\n=== 测试3：先 children 无 filter 查询 ===")
store.similarity_search_with_scores(Q, k=5)
ok3 = run_global("测试3")

# 测试 4：先做 community 查询，再 global
print("\n=== 测试4：先 community 查询 ===")
store.similarity_search_communities(Q, k=5)
ok4 = run_global("测试4")

print(f"\n结果: 基线={ok0} 测试1={ok1} 测试2={ok2} 测试3={ok3} 测试4={ok4}")
graph_store.close()
