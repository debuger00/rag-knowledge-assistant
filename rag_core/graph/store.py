"""SQLite-backed graph storage with bounded in-process traversal."""
from __future__ import annotations

import heapq
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from rag_core.graph.models import (
    GraphEdge,
    GraphHit,
    GraphNode,
    document_node_id,
    section_node_id,
)


class GraphStore:
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS graph_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    anchor TEXT NOT NULL DEFAULT '',
                    owner_source TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source);
                CREATE INDEX IF NOT EXISTS idx_nodes_owner ON nodes(owner_source);
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    evidence_source TEXT NOT NULL DEFAULT '',
                    evidence_anchor TEXT NOT NULL DEFAULT '',
                    evidence_chunk_id TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_edges_evidence ON edges(evidence_source);
                CREATE TABLE IF NOT EXISTS document_state (
                    source TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    graph_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self._connection.execute(
                "INSERT OR REPLACE INTO graph_meta(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM edges")
            self._connection.execute("DELETE FROM nodes")
            self._connection.execute("DELETE FROM document_state")

    def replace_document(
        self,
        source: str,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        content_hash: str,
        graph_version: int,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM edges WHERE evidence_source = ?", (source,))
            self._connection.execute(
                "DELETE FROM nodes WHERE owner_source = ? AND type = 'section'", (source,)
            )
            for node in nodes:
                self._upsert_node(node)
            for edge in edges:
                self._upsert_edge(edge)
            self._connection.execute("""
                INSERT INTO document_state(source, content_hash, graph_version)
                VALUES(?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    graph_version=excluded.graph_version,
                    updated_at=CURRENT_TIMESTAMP
            """, (source, content_hash, graph_version))
            self._cleanup_orphans()

    def rebuild_documents(
        self,
        payloads: list[
            tuple[str, list[GraphNode], list[GraphEdge], str, int]
        ],
    ) -> None:
        """Replace the complete graph in one transaction."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM edges")
            self._connection.execute("DELETE FROM nodes")
            self._connection.execute("DELETE FROM document_state")
            for source, nodes, edges, content_hash, graph_version in payloads:
                for node in nodes:
                    self._upsert_node(node)
                for edge in edges:
                    self._upsert_edge(edge)
                self._connection.execute("""
                    INSERT INTO document_state(source, content_hash, graph_version)
                    VALUES(?, ?, ?)
                """, (source, content_hash, graph_version))
            self._cleanup_orphans()

    def _upsert_node(self, node: GraphNode) -> None:
        existing = self._connection.execute(
            "SELECT metadata_json FROM nodes WHERE id = ?", (node.id,)
        ).fetchone()
        metadata = json.loads(existing["metadata_json"] or "{}") if existing else {}
        metadata.update(node.metadata)
        self._connection.execute("""
            INSERT INTO nodes(
                id, type, name, source, anchor, owner_source,
                content_hash, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type,
                name=excluded.name,
                source=CASE WHEN excluded.source != '' THEN excluded.source ELSE nodes.source END,
                anchor=CASE WHEN excluded.anchor != '' THEN excluded.anchor ELSE nodes.anchor END,
                owner_source=CASE WHEN excluded.owner_source != '' THEN excluded.owner_source ELSE nodes.owner_source END,
                content_hash=CASE WHEN excluded.content_hash != '' THEN excluded.content_hash ELSE nodes.content_hash END,
                metadata_json=excluded.metadata_json
        """, (
            node.id, node.type, node.name, node.source, node.anchor,
            node.owner_source, node.content_hash,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        ))

    def _upsert_edge(self, edge: GraphEdge) -> None:
        self._connection.execute("""
            INSERT INTO edges(
                id, source_id, target_id, type, weight,
                evidence_source, evidence_anchor, evidence_chunk_id, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id, target_id=excluded.target_id,
                type=excluded.type, weight=excluded.weight,
                evidence_source=excluded.evidence_source,
                evidence_anchor=excluded.evidence_anchor,
                evidence_chunk_id=excluded.evidence_chunk_id,
                metadata_json=excluded.metadata_json
        """, (
            edge.id, edge.source_id, edge.target_id, edge.type, edge.weight,
            edge.evidence_source, edge.evidence_anchor, edge.evidence_chunk_id,
            json.dumps(edge.metadata, ensure_ascii=False, sort_keys=True),
        ))

    def delete_by_source(self, source: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM edges WHERE evidence_source = ?", (source,))
            self._connection.execute(
                "DELETE FROM nodes WHERE owner_source = ? AND type = 'section'", (source,)
            )
            node_id = document_node_id(source)
            row = self._connection.execute(
                "SELECT metadata_json FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row:
                metadata = json.loads(row["metadata_json"] or "{}")
                metadata["exists"] = False
                self._connection.execute(
                    "UPDATE nodes SET metadata_json = ?, content_hash = '' WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False, sort_keys=True), node_id),
                )
            self._connection.execute("DELETE FROM document_state WHERE source = ?", (source,))
            self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        self._connection.execute("""
            DELETE FROM nodes
            WHERE type IN ('tag', 'unresolved')
              AND id NOT IN (SELECT source_id FROM edges)
              AND id NOT IN (SELECT target_id FROM edges)
        """)

    def _node(self, node_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()

    def _seed_node_ids(self, seeds: list[tuple[str, str]]) -> list[str]:
        result: list[str] = []
        for source, anchor in seeds:
            section_id = section_node_id(source, anchor) if anchor else ""
            if section_id and self._node(section_id):
                result.append(section_id)
            document_id = document_node_id(source)
            if document_id not in result and self._node(document_id):
                result.append(document_id)
        return result

    def expand_sources(
        self,
        seeds: list[tuple[str, str]],
        max_hops: int = 2,
        max_neighbors: int = 30,
        max_results: int = 20,
    ) -> list[GraphHit]:
        if max_hops < 1 or not seeds:
            return []
        seed_sources = {source for source, _ in seeds}
        with self._lock:
            seed_ids = self._seed_node_ids(seeds)
            queue = [(-1.0, 0, node_id, (node_id,)) for node_id in seed_ids]
            heapq.heapify(queue)
            best = {node_id: 1.0 for node_id in seed_ids}
            hits: dict[str, GraphHit] = {}
            while queue:
                negative_score, depth, node_id, path = heapq.heappop(queue)
                score = -negative_score
                if score < best.get(node_id, 0.0):
                    continue
                rows = self._connection.execute("""
                    SELECT * FROM edges
                    WHERE source_id = ? OR target_id = ?
                    ORDER BY weight DESC, id LIMIT ?
                """, (node_id, node_id, max_neighbors)).fetchall()
                for edge in rows:
                    neighbor_id = edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]
                    # HAS_SECTION only moves between two representations of the
                    # same document. It attenuates rank, but must not consume a
                    # semantic graph hop such as LINKS_TO or TAGGED_WITH.
                    next_depth = depth if edge["type"] == "HAS_SECTION" else depth + 1
                    if next_depth > max_hops:
                        continue
                    next_score = score * float(edge["weight"]) * 0.85
                    if next_score <= best.get(neighbor_id, 0.0):
                        continue
                    best[neighbor_id] = next_score
                    next_path = path + (neighbor_id,)
                    heapq.heappush(queue, (-next_score, next_depth, neighbor_id, next_path))
                    node = self._node(neighbor_id)
                    if not node or node["type"] not in {"document", "section"}:
                        continue
                    source = str(node["source"] or "")
                    if not source or source in seed_sources:
                        continue
                    current = hits.get(source)
                    if current is None or next_score > current.score:
                        hits[source] = GraphHit(source, min(1.0, next_score), next_path)
            return sorted(hits.values(), key=lambda item: (-item.score, item.source))[:max_results]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            by_type = {
                row["type"]: row["count"]
                for row in self._connection.execute(
                    "SELECT type, COUNT(*) AS count FROM nodes GROUP BY type ORDER BY type"
                ).fetchall()
            }
            return {
                "node_count": self._connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                "edge_count": self._connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
                "document_count": self._connection.execute("SELECT COUNT(*) FROM document_state").fetchone()[0],
                "unresolved_count": by_type.get("unresolved", 0),
                "nodes_by_type": by_type,
                "schema_version": self.SCHEMA_VERSION,
            }

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Export a deterministic, content-free graph snapshot for evaluation."""
        with self._lock:
            nodes = self._connection.execute(
                "SELECT * FROM nodes ORDER BY id"
            ).fetchall()
            edges = self._connection.execute(
                "SELECT * FROM edges ORDER BY id"
            ).fetchall()
            return {
                "nodes": [self._row_to_node(row) for row in nodes],
                "edges": [self._row_to_edge(row) for row in edges],
            }

    def neighbors(self, node_id: str, limit: int = 50) -> dict[str, Any] | None:
        with self._lock:
            node = self._node(node_id)
            if node is None:
                return None
            edges = self._connection.execute("""
                SELECT * FROM edges WHERE source_id = ? OR target_id = ?
                ORDER BY weight DESC, id LIMIT ?
            """, (node_id, node_id, limit)).fetchall()
            neighbor_ids = {
                edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]
                for edge in edges
            }
            nodes = [self._node(value) for value in neighbor_ids]
            return {
                "node": self._row_to_node(node),
                "nodes": [self._row_to_node(value) for value in nodes if value],
                "edges": [self._row_to_edge(value) for value in edges],
            }

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "type": row["type"], "name": row["name"],
            "source": row["source"], "anchor": row["anchor"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "source_id": row["source_id"],
            "target_id": row["target_id"], "type": row["type"],
            "weight": row["weight"], "evidence_source": row["evidence_source"],
            "evidence_anchor": row["evidence_anchor"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }
