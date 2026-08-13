"""Prepare and evaluate a realistic Chinese retrieval subset from CRUD-RAG."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config, set_config  # noqa: E402
from rag_core.graph.builder import rebuild_structure_graph  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.loader import ObsidianLoader  # noqa: E402
from rag_core.indexing.splitter import parent_child_split  # noqa: E402
from rag_core.indexing.store import VectorStoreManager  # noqa: E402
from rag_core.retrieval.hybrid import HybridGraphRetriever  # noqa: E402
from rag_core.retrieval.retriever import ParentChildRetriever  # noqa: E402


OFFICIAL_DATASET = (
    PROJECT_ROOT / "data" / "external" / "CRUD_RAG" /
    "data" / "crud_split" / "split_merged.json"
)
BACKGROUND_CORPUS = OFFICIAL_DATASET.parent.parent / "80000_docs"
WORK_ROOT = PROJECT_ROOT / "data" / "eval" / "crud_rag"
RESULTS_ROOT = PROJECT_ROOT / "tests" / "eval" / "results"
QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")


def _safe_scalar(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _case_id(task: str, item: dict[str, Any], index: int) -> str:
    raw_id = str(item.get("ID") or index)
    digest = hashlib.sha256(f"{task}\0{raw_id}\0{index}".encode("utf-8")).hexdigest()[:12]
    return f"{task}-{digest}"


def load_official_dataset(path: Path = OFFICIAL_DATASET) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CRUD-RAG dataset not found: {path}\n"
            "Run: python tests/eval/download_crud_rag.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = set(QA_TASKS) - set(data)
    if missing:
        raise ValueError(f"CRUD-RAG dataset is missing tasks: {sorted(missing)}")
    return data


def prepare_subset(
    dataset_path: Path,
    output_root: Path,
    per_task: int = 20,
    distractors: int = 1000,
    background_dir: Path | None = None,
) -> dict[str, Any]:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    if distractors < 0:
        raise ValueError("distractors must be non-negative")
    dataset = load_official_dataset(dataset_path)
    vault = output_root / "vault"
    cases_path = output_root / "cases.jsonl"
    if vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    gold_document_count = 0
    gold_hashes: set[str] = set()
    for task in QA_TASKS:
        items = dataset[task]
        if len(items) < per_task:
            raise ValueError(f"{task} has only {len(items)} rows, requested {per_task}")
        evidence_count = int(task.removeprefix("questanswer_").removesuffix("docs").removesuffix("doc"))
        for index, item in enumerate(items[:per_task]):
            case_id = _case_id(task, item, index)
            sources: list[str] = []
            for slot in range(1, evidence_count + 1):
                source = f"{task}/{case_id}-news{slot}.md"
                sources.append(source)
                path = vault / source
                path.parent.mkdir(parents=True, exist_ok=True)
                markdown = (
                    "---\n"
                    f"crud_id: {_safe_scalar(item.get('ID', ''))}\n"
                    f"crud_task: {_safe_scalar(task)}\n"
                    f"evidence_slot: {slot}\n"
                    "---\n\n"
                    f"# 新闻证据 {slot}\n\n"
                    f"{str(item.get(f'news{slot}', '')).strip()}\n"
                )
                path.write_text(markdown, encoding="utf-8")
                gold_document_count += 1
                gold_hashes.add(hashlib.sha256(
                    str(item.get(f"news{slot}", "")).strip().encode("utf-8")
                ).hexdigest())
            cases.append({
                "id": case_id,
                "official_id": str(item.get("ID", "")),
                "task": task,
                "question": str(item.get("questions", "")).strip(),
                "answer": str(item.get("answers", "")).strip(),
                "gold_sources": sources,
            })

    selected_distractors: list[tuple[str, str]] = []
    distractor_source = "qa_split"
    corpus_files = sorted(
        (background_dir or Path()).glob("documents_dup_part_*")
    ) if background_dir and background_dir.is_dir() else []
    if corpus_files and distractors:
        distractor_source = "80000_docs"
        rng = random.Random(20260813)
        seen_hashes: set[str] = set(gold_hashes)
        eligible_count = 0
        for corpus_file in corpus_files:
            with corpus_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    content = line.strip()
                    if not content:
                        continue
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)
                    eligible_count += 1
                    candidate = (content_hash, content)
                    if len(selected_distractors) < distractors:
                        selected_distractors.append(candidate)
                    else:
                        replacement = rng.randrange(eligible_count)
                        if replacement < distractors:
                            selected_distractors[replacement] = candidate
        selected_distractors.sort()
    else:
        distractor_candidates: dict[str, str] = {}
        for task in QA_TASKS:
            evidence_count = int(task.removeprefix("questanswer_").removesuffix("docs").removesuffix("doc"))
            for item in dataset[task]:
                for slot in range(1, evidence_count + 1):
                    content = str(item.get(f"news{slot}", "")).strip()
                    if not content:
                        continue
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if content_hash not in gold_hashes:
                        distractor_candidates.setdefault(content_hash, content)
        selected_distractors = sorted(distractor_candidates.items())[:distractors]
    for content_hash, content in selected_distractors:
        path = vault / "distractors" / f"{content_hash[:16]}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\ncrud_task: \"distractor\"\n---\n\n# 背景新闻\n\n"
            f"{content}\n",
            encoding="utf-8",
        )
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest = {
        "dataset": str(dataset_path),
        "per_task": per_task,
        "query_count": len(cases),
        "gold_document_count": gold_document_count,
        "distractor_count": len(selected_distractors),
        "distractor_source": distractor_source,
        "document_count": gold_document_count + len(selected_distractors),
        "tasks": {task: per_task for task in QA_TASKS},
        "vault": str(vault),
        "cases": str(cases_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_metrics(retrieved: list[str], gold: list[str], k: int) -> dict[str, Any]:
    top = retrieved[:k]
    gold_set = set(gold)
    hits = gold_set & set(top)
    first_rank = next((rank for rank, source in enumerate(top, 1) if source in gold_set), None)
    return {
        "evidence_recall": len(hits) / len(gold_set),
        "all_evidence_recalled": len(hits) == len(gold_set),
        "any_evidence_recalled": bool(hits),
        "first_relevant_rank": first_rank,
    }


def _aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
    count = len(items) or 1
    return {
        "evidence_recall_at_k": round(sum(x["evidence_recall"] for x in items) / count, 4),
        "all_evidence_recall_at_k": round(sum(x["all_evidence_recalled"] for x in items) / count, 4),
        "any_evidence_recall_at_k": round(sum(x["any_evidence_recalled"] for x in items) / count, 4),
        "mrr_at_k": round(sum(1 / x["first_relevant_rank"] if x["first_relevant_rank"] else 0 for x in items) / count, 4),
    }


def render_markdown(report: dict[str, Any]) -> str:
    basic, local = report["basic"], report["local"]
    lines = [
        "# CRUD-RAG 中文真实语料检索报告", "", "## 结论", "",
        f"- 查询：{report['dataset']['query_count']} 条",
        f"- 新闻文档：{report['dataset']['document_count']} 篇",
        f"- 其中 Gold：{report['dataset']['gold_document_count']} 篇",
        f"- Distractor：{report['dataset']['distractor_count']} 篇",
        f"- Distractor 来源：{report['dataset']['distractor_source']}",
        f"- Basic Evidence Recall@{report['top_k']}：{basic['evidence_recall_at_k']:.2%}",
        f"- Local Evidence Recall@{report['top_k']}：{local['evidence_recall_at_k']:.2%}",
        f"- Basic 全证据召回率：{basic['all_evidence_recall_at_k']:.2%}",
        f"- Local 全证据召回率：{local['all_evidence_recall_at_k']:.2%}",
        f"- 图增益：{report['graph_delta']:+.2%}", "",
        "## 图结构", "",
        f"- 节点：{report['graph']['node_count']}",
        f"- 边：{report['graph']['edge_count']}",
        f"- unresolved：{report['graph']['unresolved_count']}", "",
        "## 口径", "",
        "- 使用官方 CRUD-RAG `crud_split` 中的真实中文新闻、问题和参考答案。",
        "- 不使用固定向量种子，查询从真实 embedding 检索开始。",
        "- Markdown 中不注入答案、event、Gold 链接或 case 标签，避免数据泄漏。",
        "- CRUD-RAG 新闻没有天然 wikilink；当前图只有文档/章节结构，因此本报告主要验证中文向量检索。",
        "- 答案生成和参考答案评分未调用 LLM，本报告不代表最终答案准确率。", "",
        "## 分任务结果", "",
        "| 任务 | 查询数 | Basic Recall | Local Recall | Basic 全证据 | Local 全证据 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, values in report["by_task"].items():
        lines.append(
            f"| {task} | {values['count']} | "
            f"{values['basic']['evidence_recall_at_k']:.2%} | "
            f"{values['local']['evidence_recall_at_k']:.2%} | "
            f"{values['basic']['all_evidence_recall_at_k']:.2%} | "
            f"{values['local']['all_evidence_recall_at_k']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def run(
    dataset_path: Path = OFFICIAL_DATASET,
    work_root: Path = WORK_ROOT,
    results_root: Path = RESULTS_ROOT,
    per_task: int = 20,
    distractors: int = 1000,
    background_dir: Path | None = BACKGROUND_CORPUS,
    top_k: int = 10,
    rebuild: bool = True,
) -> dict[str, Any]:
    manifest = prepare_subset(
        dataset_path, work_root, per_task=per_task, distractors=distractors,
        background_dir=background_dir,
    )
    cases = load_cases(Path(manifest["cases"]))
    original_config = get_config()
    eval_config = replace(
        original_config,
        retrieval_top_k=top_k,
        retrieval_score_threshold=0.0,
        enable_link_expansion=False,
    )
    set_config(eval_config)
    graph_store = GraphStore(str(work_root / "graph.sqlite3"))
    try:
        documents = ObsidianLoader(manifest["vault"]).load()
        split_docs = []
        for document in documents:
            split_docs.extend(parent_child_split(
                [document],
                child_chunk_size=eval_config.child_chunk_size,
                child_chunk_overlap=eval_config.child_chunk_overlap,
                child_max_len=eval_config.child_max_len_before_split,
            ))
        parents = [doc for doc in split_docs if doc.metadata.get("doc_type") == "parent"]
        children = [doc for doc in split_docs if doc.metadata.get("doc_type") == "child"]
        vector_store = VectorStoreManager(str(work_root / "chroma"))
        if rebuild or vector_store.get_stats()["parent_count"] != len(parents):
            vector_store.rebuild(parents, children)
        graph_stats = rebuild_structure_graph(
            graph_store,
            documents,
            child_chunk_size=eval_config.child_chunk_size,
            child_chunk_overlap=eval_config.child_chunk_overlap,
            child_max_len=eval_config.child_max_len_before_split,
        )
        basic = ParentChildRetriever(store=vector_store, top_k=top_k, enable_link_expansion=False)
        local = HybridGraphRetriever(vector_store, graph_store)
        started = perf_counter()
        details = []
        for index, case in enumerate(cases, 1):
            basic_docs = basic.retrieve_with_scores(case["question"])
            local_docs = local.retrieve_with_scores(case["question"])
            basic_sources = list(dict.fromkeys(str(doc.metadata["source"]) for doc, _ in basic_docs))
            local_sources = list(dict.fromkeys(str(doc.metadata["source"]) for doc, _ in local_docs))
            details.append({
                "id": case["id"], "task": case["task"],
                "gold_sources": case["gold_sources"],
                "basic_sources": basic_sources, "local_sources": local_sources,
                "basic": _source_metrics(basic_sources, case["gold_sources"], top_k),
                "local": _source_metrics(local_sources, case["gold_sources"], top_k),
            })
            print(f"[queries] {index}/{len(cases)}", flush=True)
        basic_values = [item["basic"] for item in details]
        local_values = [item["local"] for item in details]
        by_task = {}
        for task in QA_TASKS:
            task_items = [item for item in details if item["task"] == task]
            by_task[task] = {
                "count": len(task_items),
                "basic": _aggregate([item["basic"] for item in task_items]),
                "local": _aggregate([item["local"] for item in task_items]),
            }
        report = {
            "dataset": manifest,
            "embedding_model": eval_config.embedding_model,
            "top_k": top_k,
            "index": {"parents": len(parents), "children": len(children)},
            "graph": graph_stats,
            "basic": _aggregate(basic_values),
            "local": _aggregate(local_values),
            "graph_delta": round(
                _aggregate(local_values)["evidence_recall_at_k"]
                - _aggregate(basic_values)["evidence_recall_at_k"], 4
            ),
            "evaluation_seconds": round(perf_counter() - started, 3),
            "by_task": by_task,
            "cases": details,
        }
    finally:
        graph_store.close()
        set_config(original_config)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "crud-rag-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_root / "crud-rag-eval-report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Chinese retrieval using CRUD-RAG.")
    parser.add_argument("--dataset", type=Path, default=OFFICIAL_DATASET)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=1000)
    parser.add_argument("--background-dir", type=Path, default=BACKGROUND_CORPUS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()
    report = run(
        dataset_path=args.dataset.resolve(), work_root=args.work_root.resolve(),
        results_root=args.results_root.resolve(), per_task=args.per_task,
        distractors=args.distractors, top_k=args.top_k,
        background_dir=args.background_dir.resolve(),
        rebuild=not args.reuse_index,
    )
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
