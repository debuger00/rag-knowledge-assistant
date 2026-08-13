import json

from tests.eval.run_crud_rag_eval import prepare_subset, _source_metrics


def _item(prefix: str):
    return {
        "ID": prefix,
        "questions": f"{prefix}问题",
        "answers": f"{prefix}答案",
        "news1": f"{prefix}新闻一",
        "news2": f"{prefix}新闻二",
        "news3": f"{prefix}新闻三",
    }


def test_prepare_crud_rag_subset_without_answer_leakage(tmp_path):
    source = tmp_path / "split.json"
    source.write_text(json.dumps({
        "questanswer_1doc": [_item("one")],
        "questanswer_2docs": [_item("two")],
        "questanswer_3docs": [_item("three")],
    }, ensure_ascii=False), encoding="utf-8")

    manifest = prepare_subset(
        source, tmp_path / "derived", per_task=1, distractors=0,
        background_dir=None,
    )

    assert manifest["query_count"] == 3
    assert manifest["document_count"] == 6
    assert manifest["distractor_count"] == 0
    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "derived" / "vault").rglob("*.md")
    )
    assert "one新闻一" in markdown
    assert "one答案" not in markdown
    assert "one问题" not in markdown


def test_crud_rag_source_metrics_require_all_evidence():
    metrics = _source_metrics(["b.md", "x.md"], ["a.md", "b.md"], 2)
    assert metrics["evidence_recall"] == 0.5
    assert metrics["any_evidence_recalled"]
    assert not metrics["all_evidence_recalled"]
    assert metrics["first_relevant_rank"] == 1
