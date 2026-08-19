"""Grounded global search seeded by GraphRAG community reports."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import get_config
from rag_core.graph.store import GraphStore
from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.retriever import ParentChildRetriever


class GlobalCommunityRetriever:
    def __init__(self, store: VectorStoreManager, graph_store: GraphStore):
        self.store = store
        self.graph_store = graph_store
        self.config = get_config()
        self.basic = ParentChildRetriever(
            store=store,
            top_k=self.config.retrieval_top_k,
            enable_link_expansion=False,
        )
        self.last_trace: dict[str, Any] = {}

    def retrieve_with_scores(
        self, query: str, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        matches = self.store.similarity_search_communities(
            query, k=self.config.graph_max_seed_nodes
        )
        source_scores: dict[str, tuple[float, list[str]]] = {}
        for report, report_score in matches:
            community_id = str(report.metadata.get("community_id", ""))
            if not community_id:
                continue
            sources = self.graph_store.sources_for_communities(
                [community_id], limit=self.config.graph_max_neighbors
            )
            for source in sources:
                score, community_ids = source_scores.get(source, (0.0, []))
                source_scores[source] = (
                    max(score, float(report_score)),
                    list(dict.fromkeys([*community_ids, community_id])),
                )

        if not source_scores:
            result = self.basic.retrieve_with_scores(query, filter_dict=filter_dict)
            self.last_trace = {
                "community_hits": 0,
                "expanded_sources": 0,
                "selected_evidence": len(result),
                "fallback": "basic",
            }
            return result

        ordered_sources = sorted(
            source_scores,
            key=lambda source: (-source_scores[source][0], source),
        )[: self.config.graph_max_neighbors]
        chunks = self.store.similarity_search_by_sources(
            query,
            ordered_sources,
            k_per_source=1,
            filter_dict=filter_dict,
        )
        ranked = []
        for document, text_score in chunks:
            source = str(document.metadata.get("source", ""))
            community_score, community_ids = source_scores.get(source, (0.0, []))
            final_score = 0.75 * float(text_score) + 0.25 * community_score
            if final_score < self.config.retrieval_score_threshold:
                continue
            document.metadata["community_ids"] = community_ids
            document.metadata["community_score"] = round(community_score, 4)
            ranked.append((document, min(1.0, final_score)))
        ranked.sort(
            key=lambda item: (
                -item[1],
                str(item[0].metadata.get("source", "")),
                str(item[0].metadata.get("anchor", "")),
            )
        )
        selected = ranked[: self.config.retrieval_top_k]
        self.last_trace = {
            "community_hits": len(matches),
            "expanded_sources": len(ordered_sources),
            "selected_evidence": len(selected),
            "fallback": None,
        }
        return selected
