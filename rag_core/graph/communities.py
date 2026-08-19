"""Hierarchical Leiden communities and grounded community reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

import graspologic_native

from config import Config, get_config
from rag_core.graph.models import stable_id
from rag_core.graph.store import GraphStore
from rag_core.llm.deepseek import create_llm


class CommunityReporter(Protocol):
    model_id: str
    prompt_hash: str

    def report(self, context: dict[str, Any]) -> dict[str, Any]: ...


def build_communities(
    graph_store: GraphStore,
    config: Config | None = None,
) -> dict[str, Any]:
    config = config or get_config()
    network = graph_store.semantic_network()
    nodes = {str(node["id"]): node for node in network["nodes"]}
    edge_list = [
        (
            str(edge["source_id"]),
            str(edge["target_id"]),
            max(0.0001, float((edge.get("metadata") or {}).get("strength", 1.0))),
        )
        for edge in network["edges"]
        if edge["source_id"] in nodes and edge["target_id"] in nodes
    ]

    memberships: dict[tuple[int, int], set[str]] = {}
    parent_clusters: dict[tuple[int, int], int | None] = {}
    if edge_list:
        clusters = graspologic_native.hierarchical_leiden(
            edge_list,
            max_cluster_size=config.graph_community_max_cluster_size,
            seed=config.graph_community_seed,
        )
        for value in clusters:
            key = (int(value.level), int(value.cluster))
            memberships.setdefault(key, set()).add(str(value.node))
            parent_clusters[key] = (
                int(value.parent_cluster)
                if value.parent_cluster is not None
                else None
            )

    clustered_nodes = set().union(*memberships.values()) if memberships else set()
    next_cluster = max((cluster for _, cluster in memberships), default=-1) + 1
    for node_id in sorted(set(nodes) - clustered_nodes):
        memberships[(0, next_cluster)] = {node_id}
        parent_clusters[(0, next_cluster)] = None
        next_cluster += 1

    ids_by_key = {
        key: stable_id(
            "community",
            str(key[0]),
            "\0".join(sorted(entity_ids)),
        )
        for key, entity_ids in memberships.items()
    }
    communities = []
    for (level, cluster), entity_ids in sorted(memberships.items()):
        parent_cluster = parent_clusters.get((level, cluster))
        parent_id = ids_by_key.get((level - 1, parent_cluster), "")
        names = [str(nodes[node_id]["name"]) for node_id in sorted(entity_ids)]
        text_unit_ids = []
        for node_id in sorted(entity_ids):
            text_unit_ids.extend(
                (nodes[node_id].get("metadata") or {}).get("text_unit_ids", [])
            )
        communities.append({
            "id": ids_by_key[(level, cluster)],
            "cluster": cluster,
            "level": level,
            "parent_id": parent_id,
            "title": " / ".join(names[:3]),
            "entity_ids": sorted(entity_ids),
            "metadata": {
                "size": len(entity_ids),
                "text_unit_ids": list(dict.fromkeys(text_unit_ids)),
            },
        })
    graph_store.replace_communities(communities)
    return {
        "community_count": len(communities),
        "levels": max((value["level"] for value in communities), default=-1) + 1,
        "entity_count": len(nodes),
        "relationship_count": len(edge_list),
    }


class LLMCommunityReporter:
    def __init__(self, config: Config | None = None, llm=None):
        self.config = config or get_config()
        self.model_id = self.config.llm_model
        path = Path(self.config.graph_community_report_prompt)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"社区报告 Prompt 不存在: {path}")
        self.prompt = path.read_text(encoding="utf-8")
        self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(streaming=False)
        return self._llm

    def report(self, context: dict[str, Any]) -> dict[str, Any]:
        response = self.llm.invoke(
            self.prompt
            + "\n\n<COMMUNITY>\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
            + "\n</COMMUNITY>"
        )
        content = response.content if hasattr(response, "content") else response
        report = _parse_json(str(content))
        if not str(report.get("title", "")).strip():
            raise ValueError("社区报告缺少 title")
        if not str(report.get("summary", "")).strip():
            raise ValueError("社区报告缺少 summary")
        findings = report.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError("社区报告 findings 必须是数组")
        return report


def generate_community_reports(
    graph_store: GraphStore,
    reporter: CommunityReporter,
) -> dict[str, Any]:
    calls = 0
    cache_hits = 0
    errors = []
    for context in graph_store.community_contexts():
        hash_context = {key: value for key, value in context.items() if key != "title"}
        context_json = json.dumps(hash_context, ensure_ascii=False, sort_keys=True)
        context_hash = hashlib.sha256(context_json.encode("utf-8")).hexdigest()
        report_key = stable_id(
            "community-report",
            str(context["id"]),
            reporter.model_id,
            reporter.prompt_hash,
            context_hash,
        )
        report = graph_store.get_cached_community_report(report_key)
        if report is None:
            try:
                report = reporter.report(context)
            except Exception as exc:
                errors.append({"community_id": context["id"], "error": str(exc)})
                continue
            graph_store.put_cached_community_report(
                report_key=report_key,
                community_id=str(context["id"]),
                model_id=reporter.model_id,
                prompt_hash=reporter.prompt_hash,
                context_hash=context_hash,
                report=report,
            )
            calls += 1
        else:
            cache_hits += 1
        graph_store.apply_community_report(str(context["id"]), report)
    return {
        "community_report_llm_calls": calls,
        "community_report_cache_hits": cache_hits,
        "community_report_errors": errors,
    }


def _parse_json(content: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("社区报告中没有 JSON 对象")
    value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("社区报告必须是 JSON 对象")
    return value
