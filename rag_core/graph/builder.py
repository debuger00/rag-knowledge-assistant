"""Build a provenance-preserving graph from Markdown documents."""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from langchain_core.documents import Document

from rag_core.graph.models import (
    GraphEdge,
    GraphNode,
    document_node_id,
    section_node_id,
    stable_id,
)
from rag_core.graph.parser import parse_wikilinks
from rag_core.graph.resolver import ObsidianLinkResolver
from rag_core.graph.store import GraphStore
from rag_core.indexing.splitter import parent_child_split


GRAPH_INDEX_VERSION = 1


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _metadata_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def build_document_graph(
    document: Document,
    children: list[Document],
    resolver: ObsidianLinkResolver,
) -> tuple[list[GraphNode], list[GraphEdge], str]:
    source = str(document.metadata.get("source", "")).replace("\\", "/")
    content_hash = _content_hash(document.page_content)
    document_id = document_node_id(source)
    nodes: dict[str, GraphNode] = {
        document_id: GraphNode(
            id=document_id,
            type="document",
            name=str(document.metadata.get("filename") or PurePosixPath(source).stem),
            source=source,
            owner_source=source,
            content_hash=content_hash,
            metadata={
                "exists": True,
                "folder": str(document.metadata.get("folder", "")),
                "aliases": _metadata_values(document.metadata.get("aliases")),
            },
        )
    }
    edges: dict[str, GraphEdge] = {}

    for tag in sorted(set(_metadata_values(document.metadata.get("tags")))):
        normalized = tag.casefold()
        tag_id = stable_id("tag", normalized)
        nodes[tag_id] = GraphNode(tag_id, "tag", tag, metadata={"normalized": normalized})
        edge_id = stable_id("edge", document_id, tag_id, "TAGGED_WITH")
        edges[edge_id] = GraphEdge(
            edge_id, document_id, tag_id, "TAGGED_WITH", 0.35,
            evidence_source=source,
        )

    for child in children:
        anchor = str(child.metadata.get("anchor", ""))
        chunk_id = str(child.metadata.get("chunk_id", ""))
        section_id = section_node_id(source, anchor)
        nodes[section_id] = GraphNode(
            id=section_id,
            type="section",
            name=str(child.metadata.get("section_title") or anchor or "未命名段落"),
            source=source,
            anchor=anchor,
            owner_source=source,
            content_hash=_content_hash(child.page_content),
            metadata={"chunk_id": chunk_id},
        )
        has_section_id = stable_id("edge", document_id, section_id, "HAS_SECTION")
        edges[has_section_id] = GraphEdge(
            has_section_id, document_id, section_id, "HAS_SECTION", 0.2,
            evidence_source=source, evidence_anchor=anchor,
            evidence_chunk_id=chunk_id,
        )

        for link in parse_wikilinks(child.page_content):
            resolution = resolver.resolve(link, source)
            if resolution.status == "attachment":
                continue
            if resolution.status == "resolved" and resolution.source:
                target_id = document_node_id(resolution.source)
                nodes[target_id] = GraphNode(
                    id=target_id,
                    type="document",
                    name=PurePosixPath(resolution.source).stem,
                    source=resolution.source,
                    owner_source=resolution.source,
                    metadata={"exists": True},
                )
            else:
                unresolved_key = link.target or f"#{link.anchor}"
                target_id = stable_id("unresolved", unresolved_key.casefold())
                nodes[target_id] = GraphNode(
                    id=target_id,
                    type="unresolved",
                    name=unresolved_key,
                    metadata={
                        "status": resolution.status,
                        "candidates": list(resolution.candidates),
                    },
                )
            edge_id = stable_id(
                "edge", section_id, target_id, "LINKS_TO", chunk_id, link.raw
            )
            edges[edge_id] = GraphEdge(
                edge_id, section_id, target_id, "LINKS_TO", 1.0,
                evidence_source=source,
                evidence_anchor=anchor,
                evidence_chunk_id=chunk_id,
                metadata={
                    "target_anchor": link.anchor,
                    "alias": link.alias,
                    "resolution": resolution.status,
                },
            )
    return list(nodes.values()), list(edges.values()), content_hash


def rebuild_structure_graph(
    graph_store: GraphStore,
    documents: list[Document],
    *,
    child_chunk_size: int,
    child_chunk_overlap: int,
    child_max_len: int,
) -> dict:
    resolver = ObsidianLinkResolver(
        {str(document.metadata.get("source", "")) for document in documents},
        {
            str(document.metadata.get("source", "")): _metadata_values(
                document.metadata.get("aliases")
            )
            for document in documents
        },
    )
    payloads = []
    for document in documents:
        split_docs = parent_child_split(
            [document],
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            child_max_len=child_max_len,
        )
        children = [
            item for item in split_docs if item.metadata.get("doc_type") == "child"
        ]
        nodes, edges, content_hash = build_document_graph(document, children, resolver)
        payloads.append((
            str(document.metadata["source"]),
            nodes,
            edges,
            content_hash,
            GRAPH_INDEX_VERSION,
        ))

    graph_store.rebuild_documents(payloads)
    return graph_store.get_stats()
