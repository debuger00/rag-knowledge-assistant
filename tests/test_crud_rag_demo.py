import json

import config as config_module
from tools.prepare_crud_rag_vault import prepare_vault


def _qa_item(prefix: str) -> dict:
    return {
        "ID": prefix,
        "questions": f"{prefix}问题",
        "answers": f"{prefix}答案不能进入知识库",
        "event": f"{prefix}事件不能进入知识库",
        "thoughts": f"{prefix}思考不能进入知识库",
        "news1": f"{prefix}新闻一",
        "news2": f"{prefix}新闻二",
        "news3": f"{prefix}新闻三",
    }


def test_get_config_uses_rag_config_path(tmp_path, monkeypatch):
    custom = tmp_path / "crud.yaml"
    custom.write_text(
        "documents:\n  path: ./isolated-vault\n"
        "storage:\n  chroma_dir: ./isolated-chroma\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_CONFIG_PATH", str(custom))
    monkeypatch.setattr(config_module, "_config", None)

    config = config_module.get_config()

    assert config.obsidian_vault_path == "./isolated-vault"
    assert config.chroma_persist_dir == "./isolated-chroma"


def test_prepare_demo_vault_excludes_answers_and_gold_metadata(tmp_path):
    dataset = {
        "questanswer_1doc": [_qa_item("one")],
        "questanswer_2docs": [_qa_item("two")],
        "questanswer_3docs": [_qa_item("three")],
    }
    dataset_path = tmp_path / "split.json"
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False), encoding="utf-8"
    )
    background = tmp_path / "background"
    background.mkdir()
    (background / "documents_dup_part_1_part_1").write_text(
        "背景新闻一\n背景新闻二\n", encoding="utf-8"
    )

    manifest = prepare_vault(
        dataset_path,
        background,
        tmp_path / "output",
        per_task=1,
        distractors=1,
    )

    vault_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "output" / "vault").rglob("*.md")
    )
    questions = [
        json.loads(line)
        for line in (tmp_path / "output" / "questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["question_count"] == 3
    assert manifest["news_document_count"] == 6
    assert manifest["distractor_count"] == 1
    assert len(questions) == 3
    assert set(questions[0]) == {"id", "task", "question"}
    assert "答案不能进入知识库" not in vault_text
    assert "事件不能进入知识库" not in vault_text
    assert "思考不能进入知识库" not in vault_text
