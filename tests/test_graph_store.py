from langchain_core.documents import Document
import sqlite3

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


def test_schema_v1_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "graph.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '', anchor TEXT NOT NULL DEFAULT '',
                owner_source TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE edges (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                target_id TEXT NOT NULL, type TEXT NOT NULL, weight REAL NOT NULL,
                evidence_source TEXT NOT NULL DEFAULT '',
                evidence_anchor TEXT NOT NULL DEFAULT '',
                evidence_chunk_id TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
        """)

    store = GraphStore(str(path))
    try:
        node_columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(nodes)")
        }
        edge_columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(edges)")
        }
        assert "description" in node_columns
        assert {"description", "predicate"} <= edge_columns
        assert store.get_stats()["schema_version"] == 2
    finally:
        store.close()
