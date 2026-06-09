import pytest
from langchain_core.documents import Document

from rag_core.indexing.store import VectorStoreManager


@pytest.fixture
def store_manager(tmp_path):
    """使用临时目录的 Chroma 存储管理器。"""
    persist_dir = str(tmp_path / "chroma_test")
    return VectorStoreManager(persist_dir=persist_dir)


def test_add_and_search_parents(store_manager):
    """添加父文档后能通过 source 找到。"""
    docs = [
        Document(
            page_content="Docker 网络模式包括 bridge、host、none 三种。",
            metadata={"source": "Docker/Docker 网络.md", "doc_type": "parent"},
        ),
    ]
    store_manager.add_parents(docs)

    results = store_manager.search_parents_by_source("Docker/Docker 网络.md")
    assert len(results) == 1
    assert results[0].page_content == docs[0].page_content


def test_add_and_semantic_search_children(store_manager):
    """子文档可以通过语义搜索找到相关内容。"""
    docs = [
        Document(
            page_content="bridge 是 Docker 默认网络模式，容器之间通过 docker0 网桥通信。",
            metadata={"source": "Docker/Docker 网络.md", "doc_type": "child", "parent_id": "Docker/Docker 网络.md"},
        ),
        Document(
            page_content="Python asyncio 事件循环是异步编程的核心概念。",
            metadata={"source": "Python/asyncio 笔记.md", "doc_type": "child", "parent_id": "Python/asyncio 笔记.md"},
        ),
    ]
    store_manager.add_children(docs)

    results = store_manager.similarity_search("Docker 网络", k=2)
    assert len(results) >= 1
    # Docker 相关的应该排在前面
    assert "Docker" in results[0].page_content


def test_delete_by_source(store_manager):
    """按 source 删除文档。"""
    parent = Document(
        page_content="测试内容",
        metadata={"source": "test/doc.md", "doc_type": "parent"},
    )
    child = Document(
        page_content="测试内容子块",
        metadata={"source": "test/doc.md", "doc_type": "child", "parent_id": "test/doc.md"},
    )
    store_manager.add_parents([parent])
    store_manager.add_children([child])

    store_manager.delete_by_source("test/doc.md")

    results_p = store_manager.search_parents_by_source("test/doc.md")
    assert len(results_p) == 0


def test_get_index_stats(store_manager):
    """获取索引统计信息。"""
    docs = [
        Document(page_content="doc A content", metadata={"source": "a.md", "doc_type": "parent"}),
        Document(page_content="doc B content", metadata={"source": "b.md", "doc_type": "parent"}),
    ]
    store_manager.add_parents(docs)

    stats = store_manager.get_stats()
    assert stats["parent_count"] == 2
    assert "last_sync" in stats


def test_rebuild_clears_and_readds(store_manager):
    """rebuild 清空旧数据并重新添加。"""
    docs1 = [Document(page_content="v1", metadata={"source": "a.md", "doc_type": "parent"})]
    store_manager.add_parents(docs1)

    docs2 = [Document(page_content="v2", metadata={"source": "b.md", "doc_type": "parent"})]
    store_manager.rebuild(docs2, [])

    stats = store_manager.get_stats()
    assert stats["parent_count"] == 1


def test_get_parents_by_sources_batch(store_manager):
    """批量获取父文档。"""
    parents = [
        Document(page_content="Doc A", metadata={"source": "a.md", "doc_type": "parent"}),
        Document(page_content="Doc B", metadata={"source": "b.md", "doc_type": "parent"}),
        Document(page_content="Doc C", metadata={"source": "c.md", "doc_type": "parent"}),
    ]
    store_manager.add_parents(parents)

    results = store_manager.get_parents_by_sources(["a.md", "c.md", "nonexistent.md"])
    assert len(results) == 2
    sources = {d.metadata["source"] for d in results}
    assert sources == {"a.md", "c.md"}
