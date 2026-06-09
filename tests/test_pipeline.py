import pytest
from langchain_core.documents import Document

from rag_core.retrieval.pipeline import _format_docs, _format_history


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
