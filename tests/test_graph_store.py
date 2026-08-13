from langchain_core.documents import Document

from rag_core.graph.builder import rebuild_structure_graph
from rag_core.graph.models import document_node_id
from rag_core.graph.store import GraphStore


def _document(source: str, content: str, tags=None):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "filename": source.rsplit("/", 1)[-1][:-3],
            "folder": source.rsplit("/", 1)[0] if "/" in source else "",
            "tags": tags or [],
            "doc_type": "raw",
        },
    )


def test_rebuild_and_expand_follow_grounded_wikilinks(tmp_path):
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    docs = [
        _document("a.md", "# A\n\nSee [[b]].", ["topic"]),
        _document("b.md", "# B\n\nGrounded target content.", ["topic"]),
    ]

    stats = rebuild_structure_graph(
        store,
        docs,
        child_chunk_size=100,
        child_chunk_overlap=10,
        child_max_len=200,
    )
    hits = store.expand_sources([("a.md", "a")], max_hops=2)

    assert stats["document_count"] == 2
    assert stats["edge_count"] >= 5
    assert any(hit.source == "b.md" for hit in hits)
    assert store.neighbors(document_node_id("a.md")) is not None


def test_rebuild_is_idempotent_and_delete_removes_document_evidence(tmp_path):
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    docs = [_document("a.md", "# A\n\n[[missing]]")]
    kwargs = dict(child_chunk_size=100, child_chunk_overlap=10, child_max_len=200)

    first = rebuild_structure_graph(store, docs, **kwargs)
    second = rebuild_structure_graph(store, docs, **kwargs)
    assert first == second

    store.delete_by_source("a.md")
    stats = store.get_stats()
    assert stats["document_count"] == 0
    assert stats["edge_count"] == 0


def test_structural_section_edges_do_not_consume_semantic_hops(tmp_path):
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    docs = [
        _document("a.md", "# A\n\n## Seed\n\n[[b]]"),
        _document("b.md", "# B\n\n## Route\n\n[[c]]"),
        _document("c.md", "# C\n\n## Answer\n\nfinal evidence"),
    ]
    rebuild_structure_graph(
        store, docs,
        child_chunk_size=100,
        child_chunk_overlap=10,
        child_max_len=200,
    )

    hits = store.expand_sources([("a.md", "seed")], max_hops=2)

    assert {hit.source for hit in hits} >= {"b.md", "c.md"}
