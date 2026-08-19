"""定位 global 模式里哪个 source 的 filter 查询触发 "Error finding id"。"""
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

config = get_config()
store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
graph_store = GraphStore(config.graph_db_path)

Q = "Docker 构建镜像通常包含哪些主要步骤？"

print("=== 1) 社区报告相似度检索 ===")
matches = store.similarity_search_communities(Q, k=config.graph_max_seed_nodes)
print(f"命中 {len(matches)} 个社区报告")
community_ids = [str(doc.metadata.get("community_id", "")) for doc, _ in matches if doc.metadata.get("community_id")]
print(f"社区 id 前几个: {community_ids[:3]}")

print("\n=== 2) sources_for_communities ===")
sources = graph_store.sources_for_communities(community_ids, limit=config.graph_max_neighbors)
print(f"返回 {len(sources)} 个 source:")
for s in sources:
    print(f"  {s!r}")

print("\n=== 3) 逐个对 source 做 filter 查询，找坏 source ===")
bad = []
for s in sources:
    try:
        res = store.similarity_search_with_scores(Q, k=1, filter_dict={"source": s})
    except Exception as exc:
        bad.append(s)
        print(f"  BAD  source={s!r} -> {type(exc).__name__}: {str(exc)[:80]}")
        continue
    if not res:
        print(f"  (空) source={s!r}")

print(f"\n坏 source 数量: {len(bad)} / {len(sources)}")
graph_store.close()
