"""Run deterministic structure and retrieval evaluation for Local GraphRAG."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from rag_core.graph.builder import rebuild_structure_graph  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.loader import ObsidianLoader  # noqa: E402
from rag_core.indexing.splitter import parent_child_split  # noqa: E402
from rag_core.retrieval.hybrid import HybridGraphRetriever  # noqa: E402


EVAL_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = EVAL_ROOT / "fixtures" / "graph_vault"
DATASET_PATH = EVAL_ROOT / "datasets" / "graph_retrieval.jsonl"
EXPECTED_GRAPH_PATH = EVAL_ROOT / "datasets" / "expected_graph.json"
RESULTS_DIR = EVAL_ROOT / "results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _terms(text: str) -> set[str]:
    normalized = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.casefold()))
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index:index + 2] for index in range(len(normalized) - 1)}


def lexical_score(query: str, content: str) -> float:
    query_terms = _terms(query)
    content_terms = _terms(content)
    overlap = len(query_terms & content_terms)
    union = len(query_terms | content_terms) or 1
    return round(0.45 + 0.5 * (overlap / union), 6)


def _tag_values(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {item for item in str(value or "").split("|") if item}


class ControlledBasicRetriever:
    def __init__(self, scored):
        self.scored = scored

    def retrieve_with_scores(self, query: str, filter_dict=None):
        return list(self.scored)


class ControlledVectorStore:
    """A deterministic source-constrained reranker for graph-only evaluation."""

    def __init__(self, children_by_source):
        self.children_by_source = children_by_source

    def similarity_search_by_sources(
        self, query: str, sources: list[str], *, k_per_source=1, filter_dict=None
    ):
        results = []
        filters = dict(filter_dict or {})
        required_tag = filters.pop("__tag__", None)
        for source in dict.fromkeys(sources):
            candidates = list(self.children_by_source.get(source, ()))
            if "folder" in filters:
                candidates = [
                    doc for doc in candidates
                    if doc.metadata.get("folder") == filters["folder"]
                ]
            if required_tag:
                candidates = [
                    doc for doc in candidates
                    if required_tag in _tag_values(doc.metadata.get("tags"))
                ]
            ranked = sorted(
                (
                    (doc, lexical_score(query, doc.page_content))
                    for doc in candidates
                ),
                key=lambda item: (-item[1], item[0].metadata.get("anchor", "")),
            )
            results.extend(ranked[:k_per_source])
        return results


def build_fixture_index(db_path: Path):
    documents = ObsidianLoader(str(FIXTURE_ROOT)).load()
    children_by_source = defaultdict(list)
    for document in documents:
        for chunk in parent_child_split([document]):
            if chunk.metadata.get("doc_type") == "child":
                children_by_source[str(chunk.metadata["source"])].append(chunk)
    graph_store = GraphStore(str(db_path))
    started = perf_counter()
    stats = rebuild_structure_graph(
        graph_store,
        documents,
        child_chunk_size=800,
        child_chunk_overlap=100,
        child_max_len=2000,
    )
    stats["build_ms"] = round((perf_counter() - started) * 1000, 3)
    stats["markdown_files"] = len(documents)
    return graph_store, dict(children_by_source), stats


def evaluate_structure(graph_store: GraphStore, stats: dict) -> dict[str, Any]:
    expected = json.loads(EXPECTED_GRAPH_PATH.read_text(encoding="utf-8"))
    snapshot = graph_store.snapshot()
    node_by_id = {node["id"]: node for node in snapshot["nodes"]}
    resolved = set()
    unresolved = set()
    link_targets_by_source = defaultdict(set)
    for edge in snapshot["edges"]:
        if edge["type"] != "LINKS_TO":
            continue
        target = node_by_id[edge["target_id"]]
        if target["type"] == "document":
            resolved.add((edge["evidence_source"], target["source"]))
            link_targets_by_source[edge["evidence_source"]].add(target["source"])
        else:
            unresolved.add((
                edge["evidence_source"],
                str(edge["metadata"].get("resolution", "")),
                target["name"],
            ))

    expected_stats = {
        key: expected[key]
        for key in (
            "markdown_files", "node_count", "edge_count",
            "document_count", "unresolved_count", "nodes_by_type",
        )
    }
    actual_stats = {key: stats[key] for key in expected_stats}
    expected_resolved = {tuple(value) for value in expected["resolved_links"]}
    expected_unresolved = {
        (item["source"], item["status"], item["target"])
        for item in expected["unresolved_links"]
    }
    errors = []
    if actual_stats != expected_stats:
        errors.append("graph stats differ from Gold")
    if resolved != expected_resolved:
        errors.append("resolved LINKS_TO edges differ from Gold")
    if unresolved != expected_unresolved:
        errors.append("unresolved LINKS_TO edges differ from Gold")
    for source in expected["sources_without_links"]:
        if link_targets_by_source.get(source):
            errors.append(f"unexpected attachment link from {source}")
    if link_targets_by_source.get("code/source.md", set()) != set(
        expected["code_source_resolved_targets"]
    ):
        errors.append("code block or inline code produced a false edge")
    return {
        "passed": not errors,
        "errors": errors,
        "expected_stats": expected_stats,
        "actual_stats": actual_stats,
        "resolved_link_precision": round(
            len(resolved & expected_resolved) / len(resolved or {1}), 4
        ),
        "resolved_link_recall": round(
            len(resolved & expected_resolved) / len(expected_resolved or {1}), 4
        ),
        "unresolved_exact": unresolved == expected_unresolved,
    }


def _best_seed_chunk(query, source, children_by_source):
    return max(
        children_by_source[source],
        key=lambda doc: lexical_score(query, doc.page_content),
    )


def _first_rank(sources: list[str], gold: set[str]) -> int | None:
    for index, source in enumerate(sources, 1):
        if source in gold:
            return index
    return None


def evaluate_retrieval(graph_store, children_by_source) -> dict[str, Any]:
    cases = load_jsonl(DATASET_PATH)
    query_results = []
    graph_weight = 0.25
    for case in cases:
        seed_scored = [
            (
                _best_seed_chunk(case["question"], source, children_by_source),
                0.8,
            )
            for source in case["seed_sources"]
        ]
        basic_sources = [doc.metadata["source"] for doc, _ in seed_scored]
        retriever = object.__new__(HybridGraphRetriever)
        retriever.store = ControlledVectorStore(children_by_source)
        retriever.graph_store = graph_store
        retriever.basic = ControlledBasicRetriever(seed_scored)
        retriever.config = Config(
            retrieval_score_threshold=0.0,
            retrieval_top_k=20,
            rag_max_citations=20,
            graph_weight=graph_weight,
            graph_max_hops=2,
            graph_max_seed_nodes=10,
            graph_max_neighbors=30,
        )
        retriever.last_trace = {}
        local = retriever.retrieve_with_scores(
            case["question"], filter_dict=case.get("filters") or None
        )
        local_sources = list(dict.fromkeys(
            str(doc.metadata["source"]) for doc, _ in local
        ))
        gold = set(case["gold_sources"])
        forbidden = set(case.get("forbidden_sources", ()))
        basic_rank = _first_rank(basic_sources, gold)
        local_rank = _first_rank(local_sources, gold)
        graph_hits = graph_store.expand_sources(
            [
                (str(doc.metadata["source"]), str(doc.metadata.get("anchor", "")))
                for doc, _ in seed_scored
            ],
            max_hops=2,
            max_neighbors=30,
            max_results=30,
        )
        query_results.append({
            "id": case["id"],
            "category": case["category"],
            "basic_sources": basic_sources,
            "local_sources": local_sources,
            "expanded_sources": [asdict(hit) for hit in graph_hits],
            "gold_sources": sorted(gold),
            "basic_hit_at_5": bool(basic_rank and basic_rank <= 5),
            "local_hit_at_5": bool(local_rank and local_rank <= 5),
            "basic_rank": basic_rank,
            "local_rank": local_rank,
            "forbidden_in_local": sorted(forbidden & set(local_sources)),
            "safe_no_expansion": not gold and set(local_sources) <= set(case["seed_sources"]),
            "trace": dict(retriever.last_trace),
        })

    answerable = [item for item in query_results if item["gold_sources"]]
    safe = [item for item in query_results if not item["gold_sources"]]
    basic_hits = sum(item["basic_hit_at_5"] for item in answerable)
    local_hits = sum(item["local_hit_at_5"] for item in answerable)
    graph_wins = sum(
        item["local_hit_at_5"] and not item["basic_hit_at_5"]
        for item in answerable
    )
    graph_harms = sum(
        item["basic_hit_at_5"] and not item["local_hit_at_5"]
        for item in answerable
    )
    reciprocal_ranks = [
        1 / item["local_rank"] if item["local_rank"] else 0
        for item in answerable
    ]
    return {
        "query_count": len(query_results),
        "answerable_count": len(answerable),
        "safe_negative_count": len(safe),
        "basic_source_recall_at_5": round(basic_hits / len(answerable), 4),
        "local_source_recall_at_5": round(local_hits / len(answerable), 4),
        "recall_delta": round((local_hits - basic_hits) / len(answerable), 4),
        "local_mrr": round(sum(reciprocal_ranks) / len(answerable), 4),
        "graph_win_count": graph_wins,
        "graph_harm_count": graph_harms,
        "graph_harm_rate": round(graph_harms / max(basic_hits, 1), 4),
        "safe_negative_pass_rate": round(
            sum(item["safe_no_expansion"] for item in safe) / len(safe), 4
        ),
        "forbidden_leak_count": sum(bool(item["forbidden_in_local"]) for item in query_results),
        "cases": query_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    structure = report["structure"]
    retrieval = report["retrieval"]
    lines = [
        "# GraphRAG 可控评测报告",
        "",
        "## 结论",
        "",
        f"- 图结构 Gold 校验：{'通过' if structure['passed'] else '失败'}",
        f"- Basic Source Recall@5：{retrieval['basic_source_recall_at_5']:.2%}",
        f"- Local Source Recall@5：{retrieval['local_source_recall_at_5']:.2%}",
        f"- 图增益：{retrieval['recall_delta']:+.2%}",
        f"- Graph Win：{retrieval['graph_win_count']} 条",
        f"- Graph Harm：{retrieval['graph_harm_count']} 条",
        f"- Local MRR：{retrieval['local_mrr']:.4f}",
        f"- 安全负例通过率：{retrieval['safe_negative_pass_rate']:.2%}",
        f"- 禁止来源泄漏：{retrieval['forbidden_leak_count']} 条",
        "",
        "## 图结构",
        "",
        f"- 节点：{structure['actual_stats']['node_count']}",
        f"- 边：{structure['actual_stats']['edge_count']}",
        f"- resolved link precision：{structure['resolved_link_precision']:.2%}",
        f"- resolved link recall：{structure['resolved_link_recall']:.2%}",
        f"- 构建耗时：{report['index']['build_ms']:.3f} ms",
        "",
        "## 评测口径",
        "",
        "- 使用固定向量种子，隔离 embedding 模型波动，测量知识图带来的纯召回增量。",
        "- Local 候选仍需从目标 source 的原始 section 中重新选择，不把图节点直接当证据。",
        "- 本报告不调用 LLM，因此不代表最终答案正确率；端到端答案评测需另行使用人工答案要点和引用标注。",
        "- 评测实施时发现并修复：HAS_SECTION 错误消耗语义 hop、Windows folder 使用反斜杠导致过滤漏召回。",
        "",
        "## 查询明细",
        "",
        "| ID | 类别 | Basic@5 | Local@5 | Local Rank | 泄漏 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in retrieval["cases"]:
        lines.append(
            f"| {item['id']} | {item['category']} | "
            f"{'Y' if item['basic_hit_at_5'] else 'N'} | "
            f"{'Y' if item['local_hit_at_5'] else 'N'} | "
            f"{item['local_rank'] or '-'} | "
            f"{', '.join(item['forbidden_in_local']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def run(output_dir: Path | None = RESULTS_DIR) -> dict[str, Any]:
    with TemporaryDirectory(prefix="graph-eval-") as temp_dir:
        graph_store, children, stats = build_fixture_index(
            Path(temp_dir) / "graph.sqlite3"
        )
        try:
            report = {
                "dataset": {
                    "fixture_root": str(FIXTURE_ROOT.relative_to(PROJECT_ROOT)),
                    "query_file": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
                },
                "index": stats,
                "structure": evaluate_structure(graph_store, stats),
                "retrieval": evaluate_retrieval(graph_store, children),
            }
        finally:
            graph_store.close()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "graph-eval-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "graph-eval-report.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    result = run()
    print(render_markdown(result))
