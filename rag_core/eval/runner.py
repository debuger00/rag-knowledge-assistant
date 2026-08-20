"""RGB 中文 Retriever 检索性能离线评测主流程。

流程：
    RGB zh_refine.json (JSONL)
      -> 全局 Corpus（positive + negative 汇总去重，分配原始文档 source ID）
      -> parent_child_split 父子分块 + 结构图索引
      -> 直接调用项目现有 Retriever（basic / local）的 retrieve_with_scores
      -> 按原始文档 source 去重（同一文档的多个 chunk 只算一次）
      -> 与 positive Ground Truth 比对 -> Recall@K / HitRate@K / MRR@10 / nDCG@10
      -> 延迟统计（mean / P50 / P95）
      -> 输出 JSON + Markdown + CSV 报告

全程不调用 LLM。只评测项目已有检索能力，不新增 BM25/Reranker 等不存在功能。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config, set_config  # noqa: E402
from rag_core.eval.dataset import build_corpus, load_rgb_jsonl  # noqa: E402
from rag_core.eval.metrics import (  # noqa: E402
    aggregate_metrics,
    evaluate_query,
    latency_stats,
)
from rag_core.graph.builder import rebuild_structure_graph  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.splitter import parent_child_split  # noqa: E402
from rag_core.indexing.store import VectorStoreManager  # noqa: E402
from rag_core.retrieval.hybrid import HybridGraphRetriever  # noqa: E402
from rag_core.retrieval.retriever import ParentChildRetriever  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "data" / "02RGB" / "data" / "zh_refine.json"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "data" / "eval" / "rgb_retrieval"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "tests" / "eval" / "results"

RETRIEVER_NAMES = ("basic", "local")
EVAL_CHUNKS = 100  # 检索的 chunk 数：保证按原始文档去重后仍有 >= max(K) 篇
EVAL_MAX_K = 20    # 评测的最大 K


def _dedup_sources(
    scored: list[tuple[Any, float]],
) -> list[tuple[str, float]]:
    """按原始文档 source 去重，保留首个（得分最高的 chunk）的出现顺序。"""
    seen: set[str] = set()
    result: list[tuple[str, float]] = []
    for doc, score in scored:
        source = str(doc.metadata.get("source", ""))
        if source and source not in seen:
            seen.add(source)
            result.append((source, float(score)))
    return result


def _build_retrievers(
    vector_store: VectorStoreManager,
    graph_store: GraphStore,
) -> dict[str, Any]:
    basic = ParentChildRetriever(
        store=vector_store,
        top_k=EVAL_CHUNKS,
        enable_link_expansion=False,
    )
    local = HybridGraphRetriever(vector_store, graph_store)
    return {"basic": basic, "local": local}


def run(
    dataset_path: str | Path = DEFAULT_DATASET,
    work_root: str | Path = DEFAULT_WORK_ROOT,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    max_k: int = EVAL_MAX_K,
    rebuild: bool = True,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """执行完整评测并落盘报告，返回 report dict。

    - 评测期临时把 retrieval_top_k 调高、score_threshold 置 0（纯排序不过滤），
      结束后恢复原配置。
    - 返回前会写 three 份产物：JSON 详情 / Markdown 对比 / CSV 失败分析。
    """
    dataset_path = Path(dataset_path)
    work_root = Path(work_root)
    results_root = Path(results_root)
    work_root.mkdir(parents=True, exist_ok=True)

    samples = load_rgb_jsonl(dataset_path)
    if not samples:
        raise ValueError(f"RGB 数据集为空: {dataset_path}")
    corpus = build_corpus(samples)
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries 必须大于 0")
        corpus.samples = corpus.samples[:max_queries]

    original_config = get_config()
    eval_config = replace(
        original_config,
        retrieval_top_k=EVAL_CHUNKS,
        retrieval_score_threshold=0.0,
        enable_link_expansion=False,
    )
    set_config(eval_config)
    graph_store = GraphStore(str(work_root / "graph.sqlite3"))
    try:
        split_docs = parent_child_split(
            corpus.documents,
            child_chunk_size=eval_config.child_chunk_size,
            child_chunk_overlap=eval_config.child_chunk_overlap,
            child_max_len=eval_config.child_max_len_before_split,
        )
        parents = [
            doc for doc in split_docs
            if doc.metadata.get("doc_type") == "parent"
        ]
        children = [
            doc for doc in split_docs
            if doc.metadata.get("doc_type") == "child"
        ]
        vector_store = VectorStoreManager(str(work_root / "chroma"))
        if rebuild or vector_store.get_stats()["parent_count"] != len(parents):
            vector_store.rebuild(parents, children)
        graph_stats = rebuild_structure_graph(
            graph_store,
            corpus.documents,
            child_chunk_size=eval_config.child_chunk_size,
            child_chunk_overlap=eval_config.child_chunk_overlap,
            child_max_len=eval_config.child_max_len_before_split,
        )

        retrievers = _build_retrievers(vector_store, graph_store)
        per_query: dict[str, list[dict[str, Any]]] = {
            name: [] for name in retrievers
        }
        latencies: dict[str, list[float]] = {name: [] for name in retrievers}

        started = perf_counter()
        for index, sample in enumerate(corpus.samples, 1):
            gold = set(corpus.gold_sources(sample.sample_id))
            for name, retriever in retrievers.items():
                t0 = perf_counter()
                scored = retriever.retrieve_with_scores(sample.query)
                elapsed_ms = (perf_counter() - t0) * 1000.0
                latencies[name].append(elapsed_ms)
                deduped = _dedup_sources(scored)
                sources = [src for src, _ in deduped[:max_k]]
                scores = [round(score, 4) for _, score in deduped[:max_k]]
                per_query[name].append({
                    "sample_id": sample.sample_id,
                    "query": sample.query,
                    "gold_sources": sorted(gold),
                    "retrieved_sources": sources,
                    "retrieved_scores": scores,
                    "latency_ms": round(elapsed_ms, 4),
                    **evaluate_query(sources, gold),
                })
            if index % 25 == 0:
                print(f"[queries] {index}/{len(corpus.samples)}", flush=True)

        gold_documents = {
            src for golds in corpus.gold.values() for src in golds
        }
        report: dict[str, Any] = {
            "dataset": {
                "path": str(dataset_path),
                "queries": len(corpus.samples),
                "corpus_documents": len(corpus.documents),
                "gold_documents": len(gold_documents),
                "total_samples": len(samples),
            },
            "config": {
                "embedding_model": eval_config.embedding_model,
                "retrieval_chunks": EVAL_CHUNKS,
                "max_k": max_k,
                "score_threshold": eval_config.retrieval_score_threshold,
                "graph_enabled": eval_config.graph_enabled,
            },
            "index": {
                "parents": len(parents),
                "children": len(children),
                "graph": graph_stats,
            },
            "retrievers": {
                name: {
                    "metrics": aggregate_metrics(per_query[name]),
                    "latency": latency_stats(latencies[name]),
                }
                for name in retrievers
            },
            "per_query": per_query,
            "evaluation_seconds": round(perf_counter() - started, 3),
        }
    finally:
        graph_store.close()
        set_config(original_config)

    _write_report(report, results_root)
    return report


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------

def render_markdown(report: dict[str, Any]) -> str:
    retr = report["retrievers"]
    names = list(retr)
    first = retr[names[0]]
    lines = [
        "# RGB 中文 Retriever 检索性能评测报告", "",
        "## 数据与索引", "",
        f"- 数据集：`{report['dataset']['path']}`",
        f"- Query：{report['dataset']['queries']} 条"
        f"（样本共 {report['dataset']['total_samples']} 条）",
        f"- 全局 Corpus（positive + negative 去重）："
        f"{report['dataset']['corpus_documents']} 篇",
        f"- Gold 文档（至少被一个 query 作为 positive）："
        f"{report['dataset']['gold_documents']} 篇",
        f"- 索引：parent {report['index']['parents']} / "
        f"child {report['index']['children']}",
        f"- 图：node {report['index']['graph'].get('node_count', '?')} / "
        f"edge {report['index']['graph'].get('edge_count', '?')}",
        f"- 嵌入模型：`{report['config']['embedding_model']}`",
        f"- 评测窗口 max_k：{report['config']['max_k']}",
        "", "## 检索模式", "",
        "- `basic`：ParentChildRetriever（纯向量语义检索）",
        "- `local`：HybridGraphRetriever（向量 + 图扩展；"
        "RGB 语料无 wikilink，图扩展为空时退化为向量路径）",
        "", "## 指标对比", "",
        "| 指标 | " + " | ".join(names) + " |",
        "|" + "---:|" * (len(names) + 1),
    ]
    for key in first["metrics"]:
        cells = " | ".join(f"{retr[n]['metrics'][key]:.4f}" for n in names)
        lines.append(f"| {key} | {cells} |")
    lines += ["", "## 检索延迟（毫秒）", "",
              "| 统计 | " + " | ".join(names) + " |",
              "|" + "---:|" * (len(names) + 1)]
    for key in first["latency"]:
        cells = " | ".join(str(retr[n]['latency'][key]) for n in names)
        lines.append(f"| {key} | {cells} |")
    lines += ["", "## 失败案例分析", "",
              "- 每条 query 的 Top-K、gold、排名、是否命中见 "
              "`rgb-retrieval-eval-report.json` 的 `per_query` 字段；",
              "- 也见 `rgb-retrieval-eval-cases.csv`（每行一个 retriever × query）。", ""]
    return "\n".join(lines) + "\n"


def _render_csv(report: dict[str, Any]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "retriever", "sample_id", "query",
        "gold_sources", "top_k_sources",
        "first_hit_rank", "hit", "recall@20", "latency_ms",
    ])
    for name in report["retrievers"]:
        for case in report["per_query"][name]:
            writer.writerow([
                name,
                case["sample_id"],
                case["query"],
                "|".join(case["gold_sources"]),
                "|".join(case["retrieved_sources"]),
                case["first_hit_rank"] if case["first_hit_rank"] is not None else "",
                int(case["hit"]),
                case["recall@20"],
                case["latency_ms"],
            ])
    return out.getvalue()


def _write_report(report: dict[str, Any], results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "rgb-retrieval-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_root / "rgb-retrieval-eval-report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    # csv.writer 已输出 \r\n；用 newline="" 打开避免 Windows 文本模式二次翻译成 \r\r\n
    with (results_root / "rgb-retrieval-eval-cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        handle.write(_render_csv(report))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RGB 中文 Retriever 检索性能离线评测（仅检索，不调用 LLM）。"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--max-k", type=int, default=EVAL_MAX_K)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="只评测前 N 条 query（索引仍用全部语料）")
    parser.add_argument("--reuse-index", action="store_true",
                        help="复用已有向量索引，不重建")
    args = parser.parse_args()
    report = run(
        dataset_path=args.dataset.resolve(),
        work_root=args.work_root.resolve(),
        results_root=args.results_root.resolve(),
        max_k=args.max_k,
        rebuild=not args.reuse_index,
        max_queries=args.max_queries,
    )
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
