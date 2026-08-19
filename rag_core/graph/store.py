"""SQLite-backed graph storage with bounded in-process traversal."""
from __future__ import annotations

import heapq
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from collections import Counter

from rag_core.graph.models import (
    GraphEdge,
    GraphHit,
    GraphNode,
    document_node_id,
    section_node_id,
    stable_id,
)
from rag_core.graph.semantic_models import (
    GraphExtraction,
    entity_node_id,
    normalize_entity_name,
    semantic_edge_id,
)


class GraphStore:
    SCHEMA_VERSION = 2

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
                CREATE TABLE IF NOT EXISTS semantic_node_evidence (
                    node_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    anchor TEXT NOT NULL DEFAULT '',
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    raw_description TEXT NOT NULL,
                    evidence_quote TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    extraction_id TEXT NOT NULL,
                    PRIMARY KEY(node_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sem_node_source
                    ON semantic_node_evidence(source);
                CREATE TABLE IF NOT EXISTS semantic_edge_evidence (
                    edge_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    anchor TEXT NOT NULL DEFAULT '',
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    raw_description TEXT NOT NULL,
                    evidence_quote TEXT NOT NULL DEFAULT '',
                    strength REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    predicate TEXT NOT NULL DEFAULT 'RELATED_TO',
                    extraction_id TEXT NOT NULL,
                    PRIMARY KEY(edge_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sem_edge_source
                    ON semantic_edge_evidence(source);
                CREATE TABLE IF NOT EXISTS extraction_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    extractor_version INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ok',
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS semantic_source_state (
                    source TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS description_summary_cache (
                    summary_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    descriptions_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS communities (
                    id TEXT PRIMARY KEY,
                    cluster INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    parent_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    full_content_json TEXT NOT NULL DEFAULT '{}',
                    rank REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS community_members (
                    community_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    PRIMARY KEY(community_id, node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_community_member_node
                    ON community_members(node_id);
                CREATE TABLE IF NOT EXISTS community_report_cache (
                    report_key TEXT PRIMARY KEY,
                    community_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self._ensure_column("nodes", "description", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("edges", "description", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("edges", "predicate", "TEXT NOT NULL DEFAULT ''")
            self._connection.execute(
                "INSERT OR REPLACE INTO graph_meta(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM edges")
            self._connection.execute("DELETE FROM nodes")
            self._connection.execute("DELETE FROM document_state")
            self._connection.execute("DELETE FROM semantic_node_evidence")
            self._connection.execute("DELETE FROM semantic_edge_evidence")
            self._connection.execute("DELETE FROM semantic_source_state")
            self._connection.execute("DELETE FROM extraction_cache")
            self._connection.execute("DELETE FROM description_summary_cache")
            self._connection.execute("DELETE FROM community_members")
            self._connection.execute("DELETE FROM communities")
            self._connection.execute("DELETE FROM community_report_cache")

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
        """Replace the structural graph while preserving semantic evidence."""
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM edges WHERE type IN ('HAS_SECTION', 'LINKS_TO', 'TAGGED_WITH')"
            )
            self._connection.execute(
                "DELETE FROM nodes WHERE type IN ('document', 'section', 'tag', 'unresolved')"
            )
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
                content_hash, metadata_json, description
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type,
                name=excluded.name,
                source=CASE WHEN excluded.source != '' THEN excluded.source ELSE nodes.source END,
                anchor=CASE WHEN excluded.anchor != '' THEN excluded.anchor ELSE nodes.anchor END,
                owner_source=CASE WHEN excluded.owner_source != '' THEN excluded.owner_source ELSE nodes.owner_source END,
                content_hash=CASE WHEN excluded.content_hash != '' THEN excluded.content_hash ELSE nodes.content_hash END,
                metadata_json=excluded.metadata_json,
                description=CASE WHEN excluded.description != '' THEN excluded.description ELSE nodes.description END
        """, (
            node.id, node.type, node.name, node.source, node.anchor,
            node.owner_source, node.content_hash,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            node.description,
        ))

    def _upsert_edge(self, edge: GraphEdge) -> None:
        self._connection.execute("""
            INSERT INTO edges(
                id, source_id, target_id, type, weight,
                evidence_source, evidence_anchor, evidence_chunk_id, metadata_json,
                description, predicate
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id, target_id=excluded.target_id,
                type=excluded.type, weight=excluded.weight,
                evidence_source=excluded.evidence_source,
                evidence_anchor=excluded.evidence_anchor,
                evidence_chunk_id=excluded.evidence_chunk_id,
                metadata_json=excluded.metadata_json,
                description=excluded.description,
                predicate=excluded.predicate
        """, (
            edge.id, edge.source_id, edge.target_id, edge.type, edge.weight,
            edge.evidence_source, edge.evidence_anchor, edge.evidence_chunk_id,
            json.dumps(edge.metadata, ensure_ascii=False, sort_keys=True),
            edge.description, edge.predicate,
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
            self._delete_semantic_source_rows(source)
            self._refresh_semantic_graph()
            self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        self._connection.execute("""
            DELETE FROM nodes
            WHERE type IN ('tag', 'unresolved', 'entity')
              AND id NOT IN (SELECT source_id FROM edges)
              AND id NOT IN (SELECT target_id FROM edges)
        """)

    @staticmethod
    def extraction_cache_key(
        chunk_id: str,
        content_hash: str,
        model_id: str,
        prompt_hash: str,
        extractor_version: int,
    ) -> str:
        return stable_id(
            "extract-cache",
            chunk_id,
            content_hash,
            model_id,
            prompt_hash,
            str(extractor_version),
        )

    def get_cached_extraction(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM extraction_cache "
                "WHERE cache_key = ? AND status = 'ok'",
                (cache_key,),
            ).fetchone()
            return json.loads(row["response_json"]) if row else None

    def put_cached_extraction(
        self,
        *,
        cache_key: str,
        chunk_id: str,
        content_hash: str,
        model_id: str,
        prompt_hash: str,
        extractor_version: int,
        response: dict[str, Any],
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute("""
                INSERT INTO extraction_cache(
                    cache_key, chunk_id, content_hash, model_id, prompt_hash,
                    extractor_version, response_json, status, error
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 'ok', '')
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json=excluded.response_json,
                    status='ok', error='', updated_at=CURRENT_TIMESTAMP
            """, (
                cache_key, chunk_id, content_hash, model_id, prompt_hash,
                extractor_version,
                json.dumps(response, ensure_ascii=False, sort_keys=True),
            ))

    def get_semantic_source_fingerprint(self, source: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT fingerprint FROM semantic_source_state WHERE source = ?",
                (source,),
            ).fetchone()
            return str(row["fingerprint"]) if row else None

    def replace_semantic_source(
        self,
        source: str,
        fingerprint: str,
        records: list[tuple[str, str, str, GraphExtraction]],
        *,
        refresh: bool = True,
    ) -> None:
        """Atomically replace one source's semantic evidence and merged graph."""
        with self._lock, self._connection:
            self._delete_semantic_source_rows(source)
            for chunk_id, anchor, extraction_id, extraction in records:
                by_mention = {}
                for entity in extraction.entities:
                    node_id = entity.id
                    by_mention[normalize_entity_name(entity.name)] = entity
                    for alias in entity.aliases:
                        by_mention.setdefault(normalize_entity_name(alias), entity)
                    self._connection.execute("""
                        INSERT OR REPLACE INTO semantic_node_evidence(
                            node_id, chunk_id, source, anchor, entity_name,
                            entity_type, raw_description, evidence_quote,
                            confidence, aliases_json, extraction_id
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node_id, chunk_id, source, anchor, entity.name,
                        entity.type, entity.description, entity.evidence_quote,
                        entity.confidence,
                        json.dumps(entity.aliases, ensure_ascii=False),
                        extraction_id,
                    ))
                for relationship in extraction.relationships:
                    source_entity = by_mention.get(
                        normalize_entity_name(relationship.source)
                    )
                    target_entity = by_mention.get(
                        normalize_entity_name(relationship.target)
                    )
                    if not source_entity or not target_entity:
                        continue
                    edge_id = semantic_edge_id(source_entity.id, target_entity.id)
                    self._connection.execute("""
                        INSERT OR REPLACE INTO semantic_edge_evidence(
                            edge_id, chunk_id, source, anchor, source_node_id,
                            target_node_id, raw_description, evidence_quote,
                            strength, confidence, predicate, extraction_id
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        edge_id, chunk_id, source, anchor,
                        source_entity.id, target_entity.id,
                        relationship.description, relationship.evidence_quote,
                        relationship.strength, relationship.confidence,
                        relationship.predicate, extraction_id,
                    ))
            self._connection.execute("""
                INSERT INTO semantic_source_state(source, fingerprint)
                VALUES(?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    updated_at=CURRENT_TIMESTAMP
            """, (source, fingerprint))
            if refresh:
                self._refresh_semantic_graph()

    def delete_semantic_source(self, source: str) -> None:
        with self._lock, self._connection:
            self._delete_semantic_source_rows(source)
            self._refresh_semantic_graph()

    def prune_semantic_sources(
        self, valid_sources: set[str], *, refresh: bool = True
    ) -> int:
        with self._lock, self._connection:
            existing = {
                str(row["source"])
                for row in self._connection.execute(
                    "SELECT source FROM semantic_source_state"
                ).fetchall()
            }
            stale = existing - valid_sources
            for source in stale:
                self._delete_semantic_source_rows(source)
            if stale and refresh:
                self._refresh_semantic_graph()
            return len(stale)

    def refresh_semantic_graph(self) -> None:
        with self._lock, self._connection:
            self._refresh_semantic_graph()

    def _delete_semantic_source_rows(self, source: str) -> None:
        self._connection.execute(
            "DELETE FROM semantic_node_evidence WHERE source = ?", (source,)
        )
        self._connection.execute(
            "DELETE FROM semantic_edge_evidence WHERE source = ?", (source,)
        )
        self._connection.execute(
            "DELETE FROM semantic_source_state WHERE source = ?", (source,)
        )

    def _refresh_semantic_graph(self) -> None:
        """Rebuild merged semantic nodes/edges from provenance tables."""
        self._connection.execute(
            "DELETE FROM edges WHERE type IN ("
            "'MENTIONS', 'RELATED_TO', 'IN_COMMUNITY', 'PARENT_COMMUNITY')"
        )
        self._connection.execute(
            "DELETE FROM nodes WHERE type IN ('entity', 'community')"
        )
        self._connection.execute("DELETE FROM community_members")
        self._connection.execute("DELETE FROM communities")

        node_rows = self._connection.execute("""
            SELECT * FROM semantic_node_evidence
            ORDER BY node_id, confidence DESC, chunk_id
        """).fetchall()
        grouped_nodes: dict[str, list[sqlite3.Row]] = {}
        for row in node_rows:
            grouped_nodes.setdefault(str(row["node_id"]), []).append(row)

        for node_id, rows in grouped_nodes.items():
            descriptions = _unique_values(rows, "raw_description")
            aliases = []
            for row in rows:
                aliases.extend(json.loads(row["aliases_json"] or "[]"))
            chunks = list(dict.fromkeys(str(row["chunk_id"]) for row in rows))
            node = GraphNode(
                id=node_id,
                type="entity",
                name=str(rows[0]["entity_name"]),
                metadata={
                    "entity_type": str(rows[0]["entity_type"]),
                    "aliases": list(dict.fromkeys(aliases)),
                    "frequency": len(chunks),
                    "text_unit_ids": chunks,
                },
                description=_merge_descriptions(descriptions),
            )
            self._upsert_node(node)
            for row in rows:
                section_id = section_node_id(str(row["source"]), str(row["anchor"]))
                mention_id = stable_id(
                    "edge", section_id, node_id, "MENTIONS", str(row["chunk_id"])
                )
                self._upsert_edge(GraphEdge(
                    id=mention_id,
                    source_id=section_id,
                    target_id=node_id,
                    type="MENTIONS",
                    weight=0.8,
                    evidence_source=str(row["source"]),
                    evidence_anchor=str(row["anchor"]),
                    evidence_chunk_id=str(row["chunk_id"]),
                    metadata={"confidence": float(row["confidence"])},
                    description=str(row["raw_description"]),
                    predicate="MENTIONS",
                ))

        edge_rows = self._connection.execute("""
            SELECT * FROM semantic_edge_evidence ORDER BY edge_id, chunk_id
        """).fetchall()
        grouped_edges: dict[str, list[sqlite3.Row]] = {}
        for row in edge_rows:
            grouped_edges.setdefault(str(row["edge_id"]), []).append(row)
        for edge_id, rows in grouped_edges.items():
            descriptions = _unique_values(rows, "raw_description")
            strengths = [float(row["strength"]) for row in rows]
            predicates = Counter(str(row["predicate"]) for row in rows)
            predicate = predicates.most_common(1)[0][0] if predicates else "RELATED_TO"
            chunks = list(dict.fromkeys(str(row["chunk_id"]) for row in rows))
            average_strength = sum(strengths) / max(len(strengths), 1)
            self._upsert_edge(GraphEdge(
                id=edge_id,
                source_id=str(rows[0]["source_node_id"]),
                target_id=str(rows[0]["target_node_id"]),
                type="RELATED_TO",
                weight=max(0.1, min(0.95, average_strength / 10 * 0.7)),
                metadata={
                    "strength": sum(strengths),
                    "frequency": len(chunks),
                    "text_unit_ids": chunks,
                    "predicates": dict(predicates),
                },
                description=_merge_descriptions(descriptions),
                predicate=predicate,
            ))

    def list_entities(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM nodes WHERE type = 'entity' ORDER BY id"
            ).fetchall()
            return [self._row_to_node(row) for row in rows]

    def semantic_network(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            nodes = self._connection.execute(
                "SELECT * FROM nodes WHERE type = 'entity' ORDER BY id"
            ).fetchall()
            edges = self._connection.execute(
                "SELECT * FROM edges WHERE type = 'RELATED_TO' ORDER BY id"
            ).fetchall()
            return {
                "nodes": [self._row_to_node(row) for row in nodes],
                "edges": [self._row_to_edge(row) for row in edges],
            }

    def replace_communities(self, communities: list[dict[str, Any]]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM edges WHERE type IN ('IN_COMMUNITY', 'PARENT_COMMUNITY')"
            )
            self._connection.execute("DELETE FROM nodes WHERE type = 'community'")
            self._connection.execute("DELETE FROM community_members")
            self._connection.execute("DELETE FROM communities")
            for community in communities:
                community_id = str(community["id"])
                metadata = dict(community.get("metadata") or {})
                self._connection.execute("""
                    INSERT INTO communities(
                        id, cluster, level, parent_id, title, metadata_json
                    ) VALUES(?, ?, ?, ?, ?, ?)
                """, (
                    community_id,
                    int(community["cluster"]),
                    int(community["level"]),
                    str(community.get("parent_id") or ""),
                    str(community.get("title") or ""),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ))
                self._upsert_node(GraphNode(
                    id=community_id,
                    type="community",
                    name=str(community.get("title") or community_id),
                    metadata={
                        "cluster": int(community["cluster"]),
                        "level": int(community["level"]),
                        **metadata,
                    },
                ))
                for node_id in community.get("entity_ids", []):
                    self._connection.execute(
                        "INSERT INTO community_members(community_id, node_id) "
                        "VALUES(?, ?)",
                        (community_id, str(node_id)),
                    )
                    edge_id = stable_id(
                        "edge", str(node_id), community_id, "IN_COMMUNITY"
                    )
                    self._upsert_edge(GraphEdge(
                        id=edge_id,
                        source_id=str(node_id),
                        target_id=community_id,
                        type="IN_COMMUNITY",
                        weight=0.25,
                        predicate="IN_COMMUNITY",
                    ))
                parent_id = str(community.get("parent_id") or "")
                if parent_id:
                    edge_id = stable_id(
                        "edge", community_id, parent_id, "PARENT_COMMUNITY"
                    )
                    self._upsert_edge(GraphEdge(
                        id=edge_id,
                        source_id=community_id,
                        target_id=parent_id,
                        type="PARENT_COMMUNITY",
                        weight=0.15,
                        predicate="PARENT_COMMUNITY",
                    ))

    def community_contexts(self) -> list[dict[str, Any]]:
        with self._lock:
            communities = self._connection.execute(
                "SELECT * FROM communities ORDER BY level, id"
            ).fetchall()
            result = []
            for community in communities:
                members = self._connection.execute("""
                    SELECT n.* FROM nodes n
                    JOIN community_members m ON m.node_id = n.id
                    WHERE m.community_id = ? ORDER BY n.id
                """, (community["id"],)).fetchall()
                member_ids = {str(row["id"]) for row in members}
                relationships = []
                if member_ids:
                    rows = self._connection.execute(
                        "SELECT * FROM edges WHERE type = 'RELATED_TO' ORDER BY id"
                    ).fetchall()
                    relationships = [
                        self._row_to_edge(row)
                        for row in rows
                        if row["source_id"] in member_ids and row["target_id"] in member_ids
                    ]
                metadata = json.loads(community["metadata_json"] or "{}")
                result.append({
                    "id": str(community["id"]),
                    "level": int(community["level"]),
                    "title": str(community["title"]),
                    "entities": [self._row_to_node(row) for row in members],
                    "relationships": relationships,
                    "text_unit_ids": metadata.get("text_unit_ids", []),
                })
            return result

    def get_cached_community_report(self, report_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT report_json FROM community_report_cache WHERE report_key = ?",
                (report_key,),
            ).fetchone()
            return json.loads(row["report_json"]) if row else None

    def put_cached_community_report(
        self,
        *,
        report_key: str,
        community_id: str,
        model_id: str,
        prompt_hash: str,
        context_hash: str,
        report: dict[str, Any],
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute("""
                INSERT INTO community_report_cache(
                    report_key, community_id, model_id, prompt_hash,
                    context_hash, report_json
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_key) DO UPDATE SET
                    report_json=excluded.report_json, updated_at=CURRENT_TIMESTAMP
            """, (
                report_key, community_id, model_id, prompt_hash, context_hash,
                json.dumps(report, ensure_ascii=False, sort_keys=True),
            ))

    def apply_community_report(
        self, community_id: str, report: dict[str, Any]
    ) -> None:
        title = str(report.get("title", "")).strip()
        summary = str(report.get("summary", "")).strip()
        try:
            rank = max(0.0, min(10.0, float(report.get("rank", 0.0))))
        except (TypeError, ValueError):
            rank = 0.0
        with self._lock, self._connection:
            self._connection.execute("""
                UPDATE communities SET title = ?, summary = ?,
                    full_content_json = ?, rank = ? WHERE id = ?
            """, (
                title, summary,
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                rank, community_id,
            ))
            self._connection.execute(
                "UPDATE nodes SET name = ?, description = ? WHERE id = ?",
                (title or community_id, summary, community_id),
            )

    def list_community_reports(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("""
                SELECT id, level, title, summary, rank, full_content_json
                FROM communities WHERE summary != '' ORDER BY level, id
            """).fetchall()
            return [{
                "id": str(row["id"]),
                "level": int(row["level"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "rank": float(row["rank"]),
                "report": json.loads(row["full_content_json"] or "{}"),
            } for row in rows]

    def sources_for_communities(
        self, community_ids: list[str], limit: int = 100
    ) -> list[str]:
        if not community_ids:
            return []
        placeholders = ",".join("?" for _ in community_ids)
        with self._lock:
            rows = self._connection.execute(f"""
                SELECT DISTINCT e.source
                FROM community_members m
                JOIN semantic_node_evidence e ON e.node_id = m.node_id
                WHERE m.community_id IN ({placeholders})
                ORDER BY e.source LIMIT ?
            """, (*community_ids, limit)).fetchall()
            return [str(row["source"]) for row in rows]

    def semantic_description_groups(self) -> list[dict[str, Any]]:
        """Return merged items with their distinct raw descriptions."""
        with self._lock:
            groups = []
            node_rows = self._connection.execute("""
                SELECT node_id AS item_id, raw_description
                FROM semantic_node_evidence ORDER BY node_id, chunk_id
            """).fetchall()
            edge_rows = self._connection.execute("""
                SELECT edge_id AS item_id, raw_description
                FROM semantic_edge_evidence ORDER BY edge_id, chunk_id
            """).fetchall()
            for kind, rows in (("entity", node_rows), ("relationship", edge_rows)):
                grouped: dict[str, list[str]] = {}
                for row in rows:
                    grouped.setdefault(str(row["item_id"]), []).append(
                        str(row["raw_description"])
                    )
                for item_id, descriptions in grouped.items():
                    groups.append({
                        "kind": kind,
                        "item_id": item_id,
                        "descriptions": list(dict.fromkeys(descriptions)),
                    })
            return groups

    def get_cached_summary(self, summary_key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT summary FROM description_summary_cache WHERE summary_key = ?",
                (summary_key,),
            ).fetchone()
            return str(row["summary"]) if row else None

    def put_cached_summary(
        self,
        *,
        summary_key: str,
        kind: str,
        item_id: str,
        model_id: str,
        prompt_hash: str,
        descriptions: list[str],
        summary: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute("""
                INSERT INTO description_summary_cache(
                    summary_key, kind, item_id, model_id, prompt_hash,
                    descriptions_json, summary
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(summary_key) DO UPDATE SET
                    summary=excluded.summary, updated_at=CURRENT_TIMESTAMP
            """, (
                summary_key, kind, item_id, model_id, prompt_hash,
                json.dumps(descriptions, ensure_ascii=False), summary,
            ))

    def apply_semantic_summaries(self, summaries: dict[str, str]) -> None:
        with self._lock, self._connection:
            for item_id, summary in summaries.items():
                if not summary.strip():
                    continue
                self._connection.execute(
                    "UPDATE nodes SET description = ? "
                    "WHERE id = ? AND type = 'entity'",
                    (summary.strip(), item_id),
                )
                self._connection.execute(
                    "UPDATE edges SET description = ? "
                    "WHERE id = ? AND type = 'RELATED_TO'",
                    (summary.strip(), item_id),
                )

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
            return self._expand_node_ids(
                seed_ids,
                excluded_sources=seed_sources,
                max_hops=max_hops,
                max_neighbors=max_neighbors,
                max_results=max_results,
            )

    def expand_entities(
        self,
        entity_ids: list[str],
        *,
        max_hops: int = 2,
        max_neighbors: int = 30,
        max_results: int = 20,
    ) -> list[GraphHit]:
        with self._lock:
            valid_ids = [
                node_id
                for node_id in dict.fromkeys(entity_ids)
                if (node := self._node(node_id)) is not None and node["type"] == "entity"
            ]
            return self._expand_node_ids(
                valid_ids,
                excluded_sources=set(),
                max_hops=max_hops,
                max_neighbors=max_neighbors,
                max_results=max_results,
            )

    def _expand_node_ids(
        self,
        seed_ids: list[str],
        *,
        excluded_sources: set[str],
        max_hops: int,
        max_neighbors: int,
        max_results: int,
    ) -> list[GraphHit]:
        if max_hops < 1 or not seed_ids:
            return []
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
                neighbor_id = (
                    edge["target_id"]
                    if edge["source_id"] == node_id
                    else edge["source_id"]
                )
                # Representation edges do not consume a semantic hop.
                next_depth = (
                    depth
                    if edge["type"] in {"HAS_SECTION", "MENTIONS"}
                    else depth + 1
                )
                if next_depth > max_hops:
                    continue
                next_score = score * float(edge["weight"]) * 0.85
                if next_score <= best.get(neighbor_id, 0.0):
                    continue
                best[neighbor_id] = next_score
                next_path = path + (neighbor_id,)
                heapq.heappush(
                    queue, (-next_score, next_depth, neighbor_id, next_path)
                )
                node = self._node(neighbor_id)
                if not node or node["type"] not in {"document", "section"}:
                    continue
                source = str(node["source"] or "")
                if not source or source in excluded_sources:
                    continue
                current = hits.get(source)
                if current is None or next_score > current.score:
                    hits[source] = GraphHit(
                        source, min(1.0, next_score), next_path
                    )
        return sorted(
            hits.values(), key=lambda item: (-item.score, item.source)
        )[:max_results]

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
                "semantic_node_evidence_count": self._connection.execute(
                    "SELECT COUNT(*) FROM semantic_node_evidence"
                ).fetchone()[0],
                "semantic_edge_evidence_count": self._connection.execute(
                    "SELECT COUNT(*) FROM semantic_edge_evidence"
                ).fetchone()[0],
                "extraction_cache_count": self._connection.execute(
                    "SELECT COUNT(*) FROM extraction_cache WHERE status = 'ok'"
                ).fetchone()[0],
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
            "description": row["description"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "source_id": row["source_id"],
            "target_id": row["target_id"], "type": row["type"],
            "weight": row["weight"], "evidence_source": row["evidence_source"],
            "evidence_anchor": row["evidence_anchor"],
            "description": row["description"], "predicate": row["predicate"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }


def _unique_values(rows: list[sqlite3.Row], key: str) -> list[str]:
    return list(dict.fromkeys(
        str(row[key]).strip() for row in rows if str(row[key]).strip()
    ))


def _merge_descriptions(descriptions: list[str]) -> str:
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]
    return "；".join(value.rstrip("。；; ") for value in descriptions) + "。"
