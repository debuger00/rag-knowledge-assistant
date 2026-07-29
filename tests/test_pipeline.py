import pytest
from langchain_core.documents import Document

from rag_core.retrieval.pipeline import (
    REFUSAL_MESSAGE,
    RAGPipeline,
    _citation_footer,
    _format_docs,
    _format_history,
)


def test_format_docs():
    docs = [
        Document(
            page_content="bridge 是默认网络模式。",
            metadata={"source": "Docker/Docker 网络.md"},
        ),
    ]
    result = _format_docs(docs)
    assert "Docker/Docker 网络.md" in result
    assert "bridge 是默认网络模式" in result


def test_format_docs_empty():
    assert "未找到相关笔记" in _format_docs([])


def test_format_history():
    history = [
        {"role": "user", "content": "什么是 Docker？"},
        {"role": "assistant", "content": "Docker 是一个容器化平台。"},
    ]
    result = _format_history(history)
    assert "什么是 Docker？" in result
    assert "容器化平台" in result


def test_format_history_empty():
    assert "无历史对话" in _format_history([])


def test_format_history_truncates_to_6_turns():
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"问题 {i}"})
        history.append({"role": "assistant", "content": f"回答 {i}"})

    result = _format_history(history)
    assert "问题 0" not in result
    assert "问题 19" in result


def test_citation_footer_uses_path_and_anchor():
    footer = _citation_footer([
        {
            "path": "guide/deploy.md",
            "anchor": "docker-compose",
            "heading": "Docker Compose",
            "quote": "运行 docker compose up。",
            "score": 0.91,
        }
    ])
    assert "[guide/deploy.md#docker-compose]" in footer


def test_pipeline_refuses_without_calling_llm():
    class EmptyRetriever:
        def retrieve_with_scores(self, question):
            return []

    pipeline = object.__new__(RAGPipeline)
    pipeline.retriever = EmptyRetriever()

    assert "".join(pipeline.ask("文档外的问题")) == REFUSAL_MESSAGE


def test_retrieve_evidence_builds_verifiable_citation():
    class Retriever:
        def retrieve_with_scores(self, question):
            return [(
                Document(
                    page_content="使用 Docker Compose 启动服务。",
                    metadata={
                        "source": "guide/deploy.md",
                        "anchor": "启动",
                        "heading": "启动",
                    },
                ),
                0.87,
            )]

    pipeline = object.__new__(RAGPipeline)
    pipeline.retriever = Retriever()
    docs, citations = pipeline.retrieve_evidence("如何启动？")

    assert len(docs) == 1
    assert citations[0]["path"] == "guide/deploy.md"
    assert citations[0]["anchor"] == "启动"
    assert citations[0]["score"] == 0.87
