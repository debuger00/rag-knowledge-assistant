"""Data structures shared by graph indexing and retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


def stable_id(prefix: str, *parts: str) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


def document_node_id(source: str) -> str:
    return stable_id("doc", source.replace("\\", "/").lower())


def section_node_id(source: str, anchor: str) -> str:
    return stable_id("section", source.replace("\\", "/").lower(), anchor)


@dataclass(frozen=True)
class WikiLink:
    target: str
    anchor: str = ""
    alias: str = ""
    embedded: bool = False
    raw: str = ""


@dataclass(frozen=True)
class LinkResolution:
    status: str
    source: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class GraphNode:
    id: str
    type: str
    name: str
    source: str = ""
    anchor: str = ""
    owner_source: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    evidence_source: str = ""
    evidence_anchor: str = ""
    evidence_chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphHit:
    source: str
    score: float
    path: tuple[str, ...]

