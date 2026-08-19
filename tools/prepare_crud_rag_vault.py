"""Prepare an isolated CRUD-RAG news vault without running evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "CRUD_RAG"
    / "data"
    / "crud_split"
    / "split_merged.json"
)
DEFAULT_BACKGROUND = DEFAULT_DATASET.parent.parent / "80000_docs"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "crud_rag_demo"
QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")


def _case_id(task: str, item: dict[str, Any], index: int) -> str:
    raw_id = str(item.get("ID") or index)
    value = f"{task}\0{raw_id}\0{index}".encode("utf-8")
    return f"{task}-{hashlib.sha256(value).hexdigest()[:12]}"


def _evidence_count(task: str) -> int:
    return int(
        task.removeprefix("questanswer_")
        .removesuffix("docs")
        .removesuffix("doc")
    )


def _write_news(path: Path, metadata: dict[str, object], content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = "\n".join(
        f"{key}: {json.dumps(str(value), ensure_ascii=False)}"
        for key, value in metadata.items()
    )
    path.write_text(
        f"---\n{frontmatter}\n---\n\n# 新闻正文\n\n{content.strip()}\n",
        encoding="utf-8",
    )


def prepare_vault(
    dataset_path: Path,
    background_dir: Path,
    output_root: Path,
    *,
    per_task: int = 2,
    distractors: int = 20,
    seed: int = 20260813,
) -> dict[str, Any]:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    if distractors < 0:
        raise ValueError("distractors must be non-negative")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"CRUD-RAG dataset not found: {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    missing = set(QA_TASKS) - set(dataset)
    if missing:
        raise ValueError(f"CRUD-RAG dataset is missing tasks: {sorted(missing)}")

    output_root.mkdir(parents=True, exist_ok=True)
    vault = output_root / "vault"
    if vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True)

    questions: list[dict[str, str]] = []
    content_sources: dict[str, str] = {}
    task_counts: dict[str, int] = {}

    for task in QA_TASKS:
        items = dataset[task]
        if len(items) < per_task:
            raise ValueError(
                f"{task} has only {len(items)} rows, requested {per_task}"
            )
        task_counts[task] = per_task
        for index, item in enumerate(items[:per_task]):
            case_id = _case_id(task, item, index)
            questions.append(
                {
                    "id": case_id,
                    "task": task,
                    "question": str(item.get("questions", "")).strip(),
                }
            )
            for slot in range(1, _evidence_count(task) + 1):
                content = str(item.get(f"news{slot}", "")).strip()
                if not content:
                    continue
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in content_sources:
                    continue
                source = f"{task}/{digest[:16]}-news{slot}.md"
                _write_news(
                    vault / source,
                    {
                        "crud_id": item.get("ID", ""),
                        "crud_task": task,
                        "evidence_slot": slot,
                        "content_hash": digest,
                    },
                    content,
                )
                content_sources[digest] = source

    corpus_files = (
        sorted(background_dir.glob("documents_dup_part_*"))
        if background_dir.is_dir()
        else []
    )
    if distractors and not corpus_files:
        raise FileNotFoundError(
            f"CRUD-RAG background corpus not found under: {background_dir}"
        )

    selected: list[tuple[str, str]] = []
    seen_hashes = set(content_sources)
    eligible_count = 0
    rng = random.Random(seed)
    for corpus_file in corpus_files:
        with corpus_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                content = line.strip()
                if not content:
                    continue
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                eligible_count += 1
                candidate = (digest, content)
                if len(selected) < distractors:
                    selected.append(candidate)
                else:
                    replacement = rng.randrange(eligible_count)
                    if replacement < distractors:
                        selected[replacement] = candidate

    for digest, content in sorted(selected):
        source = f"distractors/{digest[:16]}.md"
        _write_news(
            vault / source,
            {
                "crud_task": "distractor",
                "content_hash": digest,
            },
            content,
        )

    questions_path = output_root / "questions.jsonl"
    questions_path.write_text(
        "".join(
            json.dumps(question, ensure_ascii=False) + "\n"
            for question in questions
        ),
        encoding="utf-8",
    )

    manifest = {
        "dataset": str(dataset_path.resolve()),
        "background_dir": str(background_dir.resolve()),
        "vault": str(vault.resolve()),
        "questions": str(questions_path.resolve()),
        "per_task": per_task,
        "tasks": task_counts,
        "question_count": len(questions),
        "news_document_count": len(content_sources),
        "distractor_count": len(selected),
        "document_count": len(content_sources) + len(selected),
        "seed": seed,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a CRUD-RAG vault without evaluating it."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    manifest = prepare_vault(
        args.dataset.resolve(),
        args.background_dir.resolve(),
        args.output_root.resolve(),
        per_task=args.per_task,
        distractors=args.distractors,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
