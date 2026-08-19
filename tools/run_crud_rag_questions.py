"""Run CRUD-RAG questions through the real RAG pipeline without scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config  # noqa: E402
from rag_core.graph.store import GraphStore  # noqa: E402
from rag_core.indexing.store import VectorStoreManager  # noqa: E402
from rag_core.retrieval.pipeline import RAGPipeline  # noqa: E402


DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "crud_rag_demo" / "questions.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "crud_rag_demo"


def load_questions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"questions file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_questions(
    questions_path: Path,
    output_path: Path,
    *,
    mode: str,
    max_questions: int | None = None,
) -> dict[str, int | str]:
    questions = load_questions(questions_path)
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be positive")
        questions = questions[:max_questions]

    config = get_config()
    if not config.llm_api_key:
        raise ValueError("LLM_API_KEY or DEEPSEEK_API_KEY is not configured")

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    graph_store = GraphStore(config.graph_db_path) if config.graph_enabled else None
    pipeline = RAGPipeline(store=store, graph_store=graph_store)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answered = 0
    insufficient = 0
    errors = 0
    try:
        with output_path.open("w", encoding="utf-8") as output:
            for index, case in enumerate(questions, 1):
                record: dict[str, Any] = {
                    "id": case["id"],
                    "task": case["task"],
                    "question": case["question"],
                    "mode": mode,
                }
                try:
                    response = pipeline.ask(case["question"], mode=mode)
                    record["response"] = response
                    record["retrieval_trace"] = pipeline.get_retrieval_trace()
                    if response.get("status") == "answered":
                        answered += 1
                    else:
                        insufficient += 1
                except Exception as exc:  # keep later questions observable
                    errors += 1
                    record["error"] = f"{type(exc).__name__}: {exc}"
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[questions:{mode}] {index}/{len(questions)} "
                    f"answered={answered} insufficient={insufficient} errors={errors}",
                    flush=True,
                )
    finally:
        if graph_store is not None:
            graph_store.close()

    return {
        "mode": mode,
        "questions": len(questions),
        "answered": answered,
        "insufficient_evidence": insufficient,
        "errors": errors,
        "output": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CRUD-RAG questions without evaluating answers."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("basic", "local", "global"), required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    output_path = args.output_root.resolve() / f"answers-{args.mode}.jsonl"
    result = run_questions(
        args.questions.resolve(),
        output_path,
        mode=args.mode,
        max_questions=args.max_questions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
