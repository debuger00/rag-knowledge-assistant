from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import Config, set_config
from rag_core.retrieval.pipeline import (
    DEFAULT_REFUSAL_REASON,
    RAGPipeline,
    SYSTEM_PROMPT,
    _build_response,
    _citation_quote,
    _format_docs,
    _format_history,
    insufficient_evidence,
    validate_answer_with_citations,
)


def _configure():
    set_config(Config(retrieval_score_threshold=0.35, rag_max_citations=5))


def _evidence():
    return [(
        Document(
            page_content="## 启动\n\n使用 Docker Compose 启动服务。",
            metadata={
                "source": "guide/deploy.md",
                "anchor": "启动",
                "heading": "启动",
                "section_title": "启动",
            },
        ),
        0.87,
    )]


def test_format_docs_uses_readonly_evidence_ids():
    result = _format_docs([_evidence()[0][0]])
    assert '"evidence_id": "ev_1"' in result
    assert "guide/deploy.md" in result
    assert "Docker Compose" in result


def test_format_docs_empty():
    assert "未找到相关笔记" in _format_docs([])


def test_format_history_and_truncation():
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"问题 {i}"})
    result = _format_history(history)
    assert "问题 0" not in result
    assert "问题 19" in result


def test_pipeline_refuses_without_calling_llm():
    _configure()

    class EmptyRetriever:
        def retrieve_with_scores(self, question):
            return []

    pipeline = object.__new__(RAGPipeline)
    pipeline.config = Config()
    pipeline.retriever = EmptyRetriever()
    response = pipeline.ask("文档外的问题")
    assert response == insufficient_evidence(DEFAULT_REFUSAL_REASON)


def test_build_response_only_maps_used_evidence():
    _configure()
    unused = Document(
        page_content="无关内容",
        metadata={"source": "other.md", "anchor": "无关", "heading": "无关"},
    )
    scored = _evidence() + [(unused, 0.8)]
    response = _build_response({
        "status": "answered",
        "answer": [
            {"text": "可使用 Docker Compose 启动服务。", "evidence_ids": ["ev_1"]},
            {"text": "该结论来自启动章节。", "evidence_ids": ["ev_1"]},
        ],
    }, scored)

    assert len(response["citations"]) == 1
    assert response["citations"][0]["document_path"] == "guide/deploy.md"
    assert response["answer"][0]["citation_ids"] == ["cite_1"]
    assert response["answer"][1]["citation_ids"] == ["cite_1"]
    assert response["citations"][0]["quote"] in scored[0][0].page_content


def test_validate_answer_rejects_unverified_or_unused_citations():
    _configure()
    response = _build_response({
        "status": "answered",
        "answer": [{"text": "使用 Compose。", "evidence_ids": ["ev_1"]}],
    }, _evidence())
    valid, errors = validate_answer_with_citations(response, _evidence())
    assert valid, errors

    response["citations"][0]["quote"] = "模型编写的伪造原文"
    valid, errors = validate_answer_with_citations(response, _evidence())
    assert not valid
    assert any("无法在检索原文定位" in error for error in errors)


def test_quote_is_exact_original_substring():
    content = "第一行。\n第二行原始证据。" * 50
    quote = _citation_quote(content)
    assert quote in content
    assert not quote.endswith("...")


def test_pipeline_retries_invalid_json_then_returns_validated_answer():
    _configure()

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="not-json")
            return AIMessage(content=(
                '{"status":"answered","answer":['
                '{"text":"使用 Docker Compose 启动服务。",'
                '"evidence_ids":["ev_1"]}]}'
            ))

    pipeline = object.__new__(RAGPipeline)
    pipeline.config = Config(rag_max_retry=1)
    pipeline.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    pipeline._llm = FakeLLM()
    response = pipeline.answer("如何启动？", scored_evidence=_evidence())

    assert pipeline._llm.calls == 2
    assert response["status"] == "answered"
    assert response["citations"][0]["id"] == "cite_1"
