import pytest
from langchain_core.documents import Document

from rag_core.indexing.splitter import parent_child_split


def test_produces_one_parent_per_document():
    """每个输入文档应该产生一个父文档。"""
    docs = [
        Document(
            page_content="# Title\n\n内容段落\n\n## 第一节\n\n第一节内容",
            metadata={"source": "test/file.md", "filename": "file", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    parents = [d for d in result if d.metadata["doc_type"] == "parent"]
    assert len(parents) == 1
    assert parents[0].page_content == "# Title\n\n内容段落\n\n## 第一节\n\n第一节内容"


def test_parent_has_child_id_list():
    """每个子块的 metadata 应该指向父文档。"""
    docs = [
        Document(
            page_content="# A\n\nintro\n\n## Section 1\n\ncontent one\n\n## Section 2\n\ncontent two",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]

    assert len(children) >= 2
    for child in children:
        assert child.metadata["parent_id"] == "test/doc.md"
        assert child.metadata["anchor"]


def test_splits_on_h2_headings():
    """按 ## 二级标题切分。"""
    docs = [
        Document(
            page_content="# Title\n\nintro\n\n## Section A\n\nstuff A\n\n## Section B\n\nstuff B",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]
    # intro 是第一段，Section A, Section B
    assert len(children) == 3


def test_long_section_is_further_split():
    """超过 child_max_len 的子块会被 RecursiveCharacterTextSplitter 继续切分。"""
    long_section_content = "word " * 6000  # 约 36000 字符的段落
    docs = [
        Document(
            page_content=f"# Title\n\nintro\n\n## Big Section\n\n{long_section_content}",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]
    # intro + Big Section 被切分成多个块
    assert len(children) >= 3  # intro + 至少 2 个长段落切块


def test_doc_without_h2_still_splits():
    """文档如果没有 ## 标题，切分后至少产生一个子块（整个文档内容）。"""
    docs = [
        Document(
            page_content="# Title\n\n段落一有很多内容在这里写了很多东西\n\n段落二还有更多内容今天天气不错",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=200, child_chunk_overlap=0, child_max_len=200)

    parents = [d for d in result if d.metadata["doc_type"] == "parent"]
    children = [d for d in result if d.metadata["doc_type"] == "child"]

    assert len(parents) == 1
    assert len(children) >= 1


def test_child_metadata_inherits_from_parent():
    """子块的 metadata 应该继承父块的元数据。"""
    docs = [
        Document(
            page_content="# Title\n\nintro\n\n## Section\n\ncontent",
            metadata={
                "source": "test/doc.md",
                "filename": "doc",
                "folder": "test",
                "tags": ["python"],
                "links": ["other"],
                "doc_type": "raw",
            },
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]
    for child in children:
        assert child.metadata["filename"] == "doc"
        assert child.metadata["folder"] == "test"
        assert "python" in child.metadata["tags"]
        assert "other" in child.metadata["links"]
        assert child.metadata["source"] == "test/doc.md"


def test_duplicate_headings_get_unique_stable_anchors():
    docs = [
        Document(
            page_content=(
                "# Title\n\n## 部署步骤\n\n第一次\n\n"
                "## 部署步骤\n\n第二次"
            ),
            metadata={"source": "guide.md", "doc_type": "raw"},
        )
    ]

    result = parent_child_split(docs)
    children = [
        doc for doc in result if doc.metadata["doc_type"] == "child"
    ]

    anchors = [doc.metadata["anchor"] for doc in children]
    assert anchors[0] == "title"
    assert anchors[1] == "部署步骤"
    assert anchors[2].startswith("部署步骤-")
    assert "part-" not in " ".join(anchors)

    rerun = parent_child_split(docs)
    assert [
        doc.metadata["anchor"]
        for doc in rerun
        if doc.metadata["doc_type"] == "child"
    ] == anchors


def test_document_without_heading_uses_readable_stable_anchor():
    docs = [Document(
        page_content="这是没有标题的第一段内容。\n\n后续内容。",
        metadata={"source": "notes/plain.md", "doc_type": "raw"},
    )]
    children = [
        doc for doc in parent_child_split(docs)
        if doc.metadata["doc_type"] == "child"
    ]
    assert children[0].metadata["anchor"].startswith("这是没有标题的第一段内容-")
    assert children[0].metadata["section_title"] == ""
    assert children[0].metadata["anchor"] != "document-start"
