"""Prepare and run selectable Microsoft GraphRAG benchmark profiles."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from typing import Any
import zipfile


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


EVAL_ROOT = Path(__file__).resolve().parent
PROFILE_PATH = EVAL_ROOT / "datasets" / "microsoft_graphrag_profiles.json"
DATA_ROOT = PROJECT_ROOT / "data" / "external" / "microsoft-graphrag-benchmarking"
WORK_ROOT = PROJECT_ROOT / "data" / "eval" / "microsoft_graphrag"
RESULTS_ROOT = EVAL_ROOT / "results"
PART_RE = re.compile(r"(?i)-part(\d+)\.txt$")


def load_profiles() -> dict[str, dict[str, Any]]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_archive(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    entries.append((item.filename, archive.read(item).decode("utf-8-sig", errors="replace")))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for item in archive.getmembers():
                if item.isfile():
                    handle = archive.extractfile(item)
                    if handle:
                        entries.append((item.name, handle.read().decode("utf-8-sig", errors="replace")))
    else:
        raise ValueError(f"Unsupported archive format: {path}")
    return entries


def _safe_name(value: str, fallback: str) -> str:
    stem = PurePosixPath(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:90]
    return cleaned or fallback


def _write_markdown(path: Path, title: str, body: str, profile: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"benchmark_profile: {json.dumps(profile)}\n"
        "---\n\n"
        f"# {title}\n\n{body.strip()}\n",
        encoding="utf-8",
    )


def prepare_profile(
    profile_name: str,
    data_root: Path = DATA_ROOT,
    work_root: Path = WORK_ROOT,
    max_documents: int | None = None,
    max_questions: int | None = None,
) -> dict[str, Any]:
    catalog = load_profiles()
    if profile_name not in catalog:
        raise ValueError(f"Unknown dataset: {profile_name}; choose from {sorted(catalog)}")
    profile = catalog[profile_name]
    archive_path = data_root / profile["archive"]
    question_path = data_root / profile["questions"]
    if not archive_path.is_file() or not question_path.is_file():
        raise FileNotFoundError(
            "Microsoft benchmark files are missing. Run: "
            "python tests/eval/download_microsoft_graphrag.py"
        )
    entries = read_archive(archive_path)
    questions = read_questions(question_path)
    profile_root = work_root / profile_name
    vault = profile_root / "vault"
    if vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True)
    sources: list[str] = []
    cases: list[dict[str, Any]] = []

    if profile["corpus_kind"] == "hotpot_per_question":
        indexed = {}
        for archive_name, content in entries:
            match = re.search(r"test_(\d+)\.txt$", archive_name)
            if match:
                indexed[int(match.group(1))] = content
        # The Microsoft filter preserves the original test indices, so gaps are
        # expected (for example test_3 is absent).  CSV row order follows the
        # numerically sorted retained indices rather than a contiguous range.
        selected_indices = sorted(indexed)
        if max_documents is not None:
            selected_indices = selected_indices[:max_documents]
        if len(questions) != len(indexed):
            raise ValueError(
                f"HotpotQA question/context count mismatch: {len(questions)} != {len(indexed)}"
            )
        for index in selected_indices:
            source = f"contexts/test_{index}.md"
            _write_markdown(vault / source, f"HotpotQA Context {index}", indexed[index], profile_name)
            sources.append(source)
        paired_rows = list(zip(selected_indices, questions[:len(selected_indices)]))
        if max_questions is not None:
            paired_rows = paired_rows[:max_questions]
        for index, row in paired_rows:
            cases.append({
                "id": row["question_id"], "question": row["question_text"],
                "expected_sources": [f"contexts/test_{index}.md"],
            })
    elif profile["corpus_kind"] == "podcast_parts":
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for archive_name, content in entries:
            filename = PurePosixPath(archive_name).name
            match = PART_RE.search(filename)
            base = PART_RE.sub("", filename) if match else PurePosixPath(filename).stem
            groups[base].append((int(match.group(1)) if match else 0, content))
        items = sorted(groups.items())
        if max_documents is not None:
            items = items[:max_documents]
        for index, (base, parts) in enumerate(items):
            digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
            source = f"episodes/{index:03d}-{_safe_name(base, 'episode')}-{digest}.md"
            body = "\n\n".join(
                f"## Part {part}\n\n{text.strip()}" for part, text in sorted(parts)
            )
            _write_markdown(vault / source, base, body, profile_name)
            sources.append(source)
        selected_questions = questions if max_questions is None else questions[:max_questions]
        cases = [{
            "id": row["question_id"], "question": row["question_text"], "expected_sources": []
        } for row in selected_questions]
    else:
        items = sorted(entries)
        if max_documents is not None:
            items = items[:max_documents]
        for index, (archive_name, content) in enumerate(items):
            title = PurePosixPath(archive_name).stem
            source = f"transcripts/{index:03d}-{_safe_name(title, 'transcript')}.md"
            _write_markdown(vault / source, title, content, profile_name)
            sources.append(source)
        selected_questions = questions if max_questions is None else questions[:max_questions]
        cases = [{
            "id": row["question_id"], "question": row["question_text"], "expected_sources": []
        } for row in selected_questions]

    cases_path = profile_root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest = {
        "profile": profile_name, **profile, "document_count": len(sources),
        "question_count": len(cases), "vault": str(vault), "cases": str(cases_path),
        "global_pipeline_status": (
            "required_not_implemented" if profile["recommended_mode"] == "global" else "not_required"
        ),
    }
    (profile_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _aggregate_paired(details: list[dict[str, Any]], key: str, top_k: int) -> dict[str, float]:
    count = len(details) or 1
    ranks = []
    for item in details:
        expected = set(item["expected_sources"])
        rank = next((i for i, source in enumerate(item[key][:top_k], 1) if source in expected), None)
        ranks.append(rank)
    return {
        f"paired_context_recall_at_{top_k}": round(sum(rank is not None for rank in ranks) / count, 4),
        f"mrr_at_{top_k}": round(sum(1 / rank if rank else 0 for rank in ranks) / count, 4),
    }


def _diagnostic(details: list[dict[str, Any]], key: str, corpus_size: int) -> dict[str, float]:
    retrieved = [source for item in details for source in item[key]]
    count = len(details) or 1
    return {
        "nonempty_query_rate": round(sum(bool(item[key]) for item in details) / count, 4),
        "average_sources_per_query": round(sum(len(item[key]) for item in details) / count, 4),
        "corpus_source_coverage": round(len(set(retrieved)) / max(corpus_size, 1), 4),
    }


def render_markdown(report: dict[str, Any]) -> str:
    manifest = report["dataset"]
    lines = [
        f"# Microsoft GraphRAG 数据集报告：{manifest['profile']}", "",
        "## 数据集", "",
        f"- 名称：{manifest['display_name']}",
        f"- 文档：{manifest['document_count']}",
        f"- 问题：{manifest['question_count']}",
        f"- 推荐检索：{manifest['recommended_mode']}",
        f"- 评测模式：{manifest['evaluation_mode']}",
        f"- Global pipeline：{manifest['global_pipeline_status']}", "",
        "## 当前结果", "",
    ]
    for mode in ("basic", "local"):
        lines.append(f"### {mode.title()}")
        lines.append("")
        lines.extend(f"- {key}：{value:.2%}" if "rate" in key or "recall" in key or "coverage" in key else f"- {key}：{value:.4f}" for key, value in report[mode].items())
        lines.append("")
    lines.extend([
        "## 解释", "",
        "- HotpotQA 的过滤结果保留原始 `test_N` 编号（编号有缺口）；按编号排序后与 CSV 行一一对应。",
        "- HotpotQA 只能计算 paired-context recall；该上下文是问题配套输入，不等同于 Gold supporting facts。",
        "- 播客和财报 CSV 没有答案、supporting source 或 reference response，只报告检索覆盖诊断。",
        "- `kevin_scott` 和 `msft_multi` 需要社区发现、社区摘要和 Map-Reduce 才能进行真正的 Global GraphRAG 评测。",
        "- 当前报告不调用 LLM，不评价开放式答案质量。", "",
    ])
    return "\n".join(lines)


def run(
    dataset: str,
    data_root: Path = DATA_ROOT,
    work_root: Path = WORK_ROOT,
    results_root: Path = RESULTS_ROOT,
    max_documents: int | None = None,
    max_questions: int | None = None,
    top_k: int = 10,
    rebuild: bool = True,
    prepare_only: bool = False,
) -> dict[str, Any]:
    manifest = prepare_profile(
        dataset, data_root, work_root, max_documents=max_documents,
        max_questions=max_questions,
    )
    if prepare_only:
        return {"dataset": manifest, "status": "prepared"}
    original_config = get_config()
    eval_config = replace(
        original_config, retrieval_top_k=top_k, retrieval_score_threshold=0.0,
        enable_link_expansion=False,
    )
    set_config(eval_config)
    profile_root = work_root / dataset
    graph_store = GraphStore(str(profile_root / "graph.sqlite3"))
    try:
        documents = ObsidianLoader(manifest["vault"]).load()
        split_docs = []
        for document in documents:
            split_docs.extend(parent_child_split(
                [document], child_chunk_size=eval_config.child_chunk_size,
                child_chunk_overlap=eval_config.child_chunk_overlap,
                child_max_len=eval_config.child_max_len_before_split,
            ))
        parents = [item for item in split_docs if item.metadata.get("doc_type") == "parent"]
        children = [item for item in split_docs if item.metadata.get("doc_type") == "child"]
        store = VectorStoreManager(str(profile_root / "chroma"))
        if rebuild or store.get_stats()["parent_count"] != len(parents):
            store.rebuild(parents, children)
        graph_stats = rebuild_structure_graph(
            graph_store, documents, child_chunk_size=eval_config.child_chunk_size,
            child_chunk_overlap=eval_config.child_chunk_overlap,
            child_max_len=eval_config.child_max_len_before_split,
        )
        basic = ParentChildRetriever(store=store, top_k=top_k, enable_link_expansion=False)
        local = HybridGraphRetriever(store, graph_store)
        cases = [
            json.loads(line) for line in Path(manifest["cases"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        details = []
        for index, case in enumerate(cases, 1):
            basic_hits = basic.retrieve_with_scores(case["question"])
            local_hits = local.retrieve_with_scores(case["question"])
            details.append({
                **case,
                "basic_sources": list(dict.fromkeys(str(doc.metadata["source"]) for doc, _ in basic_hits)),
                "local_sources": list(dict.fromkeys(str(doc.metadata["source"]) for doc, _ in local_hits)),
            })
            print(f"[queries] {index}/{len(cases)}", flush=True)
        if manifest["has_paired_source"]:
            basic_metrics = _aggregate_paired(details, "basic_sources", top_k)
            local_metrics = _aggregate_paired(details, "local_sources", top_k)
        else:
            basic_metrics = _diagnostic(details, "basic_sources", len(documents))
            local_metrics = _diagnostic(details, "local_sources", len(documents))
        report = {
            "dataset": manifest, "embedding_model": eval_config.embedding_model,
            "top_k": top_k, "index": {"parents": len(parents), "children": len(children)},
            "graph": graph_stats, "basic": basic_metrics, "local": local_metrics,
            "cases": details,
        }
    finally:
        graph_store.close()
        set_config(original_config)
    results_root.mkdir(parents=True, exist_ok=True)
    stem = f"microsoft-{dataset}-report"
    (results_root / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_root / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    catalog = load_profiles()
    parser = argparse.ArgumentParser(description="Run a Microsoft GraphRAG benchmark profile.")
    parser.add_argument("--dataset", choices=sorted(catalog), required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    report = run(
        args.dataset, args.data_root.resolve(), args.work_root.resolve(),
        args.results_root.resolve(), max_documents=args.max_documents,
        max_questions=args.max_questions, top_k=args.top_k,
        rebuild=not args.reuse_index, prepare_only=args.prepare_only,
    )
    print(render_markdown(report) if "basic" in report else json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
