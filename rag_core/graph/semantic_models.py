"""Validated data structures for LLM-based semantic graph extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any

from rag_core.graph.models import stable_id


EXTRACTOR_VERSION = 1


def normalize_entity_name(value: str) -> str:
    """Conservative normalization used for deterministic entity identity."""
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n,，。.;；:：")
    return value.casefold()


def normalize_entity_type(value: str) -> str:
    return normalize_entity_name(value).replace(" ", "_")


def entity_node_id(entity_type: str, name: str) -> str:
    return stable_id(
        "entity", normalize_entity_type(entity_type), normalize_entity_name(name)
    )


def semantic_edge_id(source_id: str, target_id: str) -> str:
    return stable_id("semantic-edge", source_id, target_id)


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    description: str
    aliases: tuple[str, ...] = ()
    evidence_quote: str = ""
    confidence: float = 1.0

    @property
    def id(self) -> str:
        return entity_node_id(self.type, self.name)


@dataclass(frozen=True)
class ExtractedRelationship:
    source: str
    target: str
    description: str
    strength: float = 1.0
    predicate: str = "RELATED_TO"
    evidence_quote: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class GraphExtraction:
    entities: tuple[ExtractedEntity, ...] = field(default_factory=tuple)
    relationships: tuple[ExtractedRelationship, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [asdict(value) for value in self.entities],
            "relationships": [asdict(value) for value in self.relationships],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        text: str,
        entity_types: tuple[str, ...] | list[str],
        min_confidence: float = 0.0,
    ) -> "GraphExtraction":
        if not isinstance(payload, dict):
            raise ValueError("抽取结果必须是 JSON 对象")
        raw_entities = payload.get("entities", [])
        raw_relationships = payload.get("relationships", [])
        if not isinstance(raw_entities, list) or not isinstance(raw_relationships, list):
            raise ValueError("entities 和 relationships 必须是数组")

        allowed_types = {normalize_entity_type(value) for value in entity_types}
        entities: dict[tuple[str, str], ExtractedEntity] = {}
        by_mention: dict[str, ExtractedEntity] = {}
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            entity_type = normalize_entity_type(str(raw.get("type", "")))
            description = str(raw.get("description", "")).strip()
            quote = str(raw.get("evidence_quote", "")).strip()
            confidence = _bounded_float(raw.get("confidence", 1.0), 0.0, 1.0)
            if not name or not entity_type or not description:
                continue
            if allowed_types and entity_type not in allowed_types:
                continue
            if confidence < min_confidence:
                continue
            if not quote or quote not in text:
                continue
            aliases_value = raw.get("aliases", [])
            aliases = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in aliases_value
                    if str(value).strip()
                )
            ) if isinstance(aliases_value, list) else ()
            entity = ExtractedEntity(
                name=name,
                type=entity_type,
                description=description,
                aliases=aliases,
                evidence_quote=quote,
                confidence=confidence,
            )
            key = (entity_type, normalize_entity_name(name))
            entities[key] = entity
            by_mention[normalize_entity_name(name)] = entity
            for alias in aliases:
                by_mention.setdefault(normalize_entity_name(alias), entity)

        relationships: dict[tuple[str, str, str], ExtractedRelationship] = {}
        for raw in raw_relationships:
            if not isinstance(raw, dict):
                continue
            source_name = str(raw.get("source", "")).strip()
            target_name = str(raw.get("target", "")).strip()
            source_entity = by_mention.get(normalize_entity_name(source_name))
            target_entity = by_mention.get(normalize_entity_name(target_name))
            description = str(raw.get("description", "")).strip()
            quote = str(raw.get("evidence_quote", "")).strip()
            confidence = _bounded_float(raw.get("confidence", 1.0), 0.0, 1.0)
            if not source_entity or not target_entity or source_entity.id == target_entity.id:
                continue
            if not description or confidence < min_confidence:
                continue
            if not quote or quote not in text:
                continue
            predicate = str(raw.get("predicate", "RELATED_TO")).strip().upper()
            predicate = re.sub(r"[^A-Z0-9_]+", "_", predicate).strip("_")
            relationship = ExtractedRelationship(
                source=source_entity.name,
                target=target_entity.name,
                description=description,
                strength=_bounded_float(raw.get("strength", 1.0), 0.0, 10.0),
                predicate=predicate or "RELATED_TO",
                evidence_quote=quote,
                confidence=confidence,
            )
            key = (source_entity.id, target_entity.id, relationship.description)
            relationships[key] = relationship

        return cls(tuple(entities.values()), tuple(relationships.values()))

    def merged_with(self, other: "GraphExtraction") -> "GraphExtraction":
        entities = {
            (value.type, normalize_entity_name(value.name)): value
            for value in self.entities
        }
        entities.update({
            (value.type, normalize_entity_name(value.name)): value
            for value in other.entities
        })
        relationships = {
            (
                normalize_entity_name(value.source),
                normalize_entity_name(value.target),
                value.description,
            ): value
            for value in self.relationships
        }
        relationships.update({
            (
                normalize_entity_name(value.source),
                normalize_entity_name(value.target),
                value.description,
            ): value
            for value in other.relationships
        })
        return GraphExtraction(tuple(entities.values()), tuple(relationships.values()))


def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return min(maximum, max(minimum, number))
