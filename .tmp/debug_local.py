"""定位 local 模式 "Error finding id" 的完整堆栈。"""
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
from rag_core.retrieval.hybrid import HybridGraphRetriever  # noqa: E402

config = get_config()
store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
graph_store = GraphStore(config.graph_db_path)
hybrid = HybridGraphRetriever(store, graph_store)

print("=== 步骤 1: basic 检索 ===")
base = hybrid.basic.retrieve_with_scores("docker compose 和 docker run 有什么区别")
print(f"basic 命中 {len(base)} 条")
for doc, score in base[:3]:
    print(f"  score={score:.4f} source={doc.metadata.get('source')} anchor={doc.metadata.get('anchor')}")

seeds = [
    (str(doc.metadata.get("source", "")), str(doc.metadata.get("anchor", "")))
    for doc, _ in base[: config.graph_max_seed_nodes]
]
print(f"\n=== 步骤 2: 图扩展 seeds={len(seeds)} ===")
try:
    graph_hits = graph_store.expand_sources(
        seeds,
        max_hops=config.graph_max_hops,
        max_neighbors=config.graph_max_neighbors,
        max_results=config.graph_max_neighbors,
    )
    print(f"图扩展命中 {len(graph_hits)} 个 source")
    for hit in graph_hits[:5]:
        print(f"  score={hit.score:.4f} source={hit.source!r}")
except Exception:
    traceback.print_exc()

print(f"\n=== 步骤 3: 对每个扩展 source 做 filtered 检索 ===")
try:
    expanded = store.similarity_search_by_sources(
        "docker compose 和 docker run 有什么区别",
        [hit.source for hit in graph_hits],
        k_per_source=1,
    )
    print(f"filtered 检索命中 {len(expanded)} 条")
except Exception:
    traceback.print_exc()

print(f"\n=== 步骤 4: 单独对每个 source 检索，定位坏 source ===")
for hit in graph_hits:
    try:
        res = store.similarity_search_with_scores(
            "docker compose 和 docker run 有什么区别",
            k=1,
            filter_dict={"source": hit.source},
        )
    except Exception as exc:
        print(f"  BAD source={hit.source!r} -> {type(exc).__name__}: {str(exc)[:120]}")
        continue
    if not res:
        print(f"  (空) source={hit.source!r}")

graph_store.close()
