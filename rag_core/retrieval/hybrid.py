"""Hybrid vector and graph retrieval while returning grounded text chunks."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import get_config
from rag_core.graph.store import GraphStore
from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.retriever import ParentChildRetriever


class HybridGraphRetriever:
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
        base = self.basic.retrieve_with_scores(query, filter_dict=filter_dict)
        seeds = [
            (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("anchor", "")),
            )
            for doc, _ in base[: self.config.graph_max_seed_nodes]
        ]
        graph_hits = self.graph_store.expand_sources(
            seeds,
            max_hops=self.config.graph_max_hops,
            max_neighbors=self.config.graph_max_neighbors,
            max_results=self.config.graph_max_neighbors,
        )
        entity_seeds = []
        entity_search = getattr(self.store, "similarity_search_entities", None)
        entity_expand = getattr(self.graph_store, "expand_entities", None)
        entity_hits = []
        if callable(entity_search) and callable(entity_expand):
            entity_matches = entity_search(
                query, k=self.config.graph_max_seed_nodes
            )
            entity_seeds = [
                str(doc.metadata.get("entity_id", ""))
                for doc, score in entity_matches
                if doc.metadata.get("entity_id")
                and float(score) >= self.config.retrieval_score_threshold
            ]
            if entity_seeds:
                entity_hits = entity_expand(
                    entity_seeds,
                    max_hops=self.config.graph_max_hops,
                    max_neighbors=self.config.graph_max_neighbors,
                    max_results=self.config.graph_max_neighbors,
                )
        hits_by_source = {}
        for hit in [*graph_hits, *entity_hits]:
            current = hits_by_source.get(hit.source)
            if current is None or hit.score > current.score:
                hits_by_source[hit.source] = hit
        graph_hits = sorted(
            hits_by_source.values(), key=lambda hit: (-hit.score, hit.source)
        )[: self.config.graph_max_neighbors]
        expanded = self.store.similarity_search_by_sources(
            query,
            [hit.source for hit in graph_hits],
            k_per_source=1,
            filter_dict=filter_dict,
        ) if graph_hits else []

        graph_by_source = {
            hit.source: (hit.score, rank, hit.path)
            for rank, hit in enumerate(graph_hits, 1)
        }
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for vector_rank, (doc, score) in enumerate(base, 1):
            key = self._key(doc)
            candidates[key] = {
                "doc": doc,
                "text_score": float(score),
                "vector_rank": vector_rank,
                "graph_score": 0.0,
                "graph_rank": None,
                "path": (),
            }
        for offset, (doc, score) in enumerate(expanded, 1):
            source = str(doc.metadata.get("source", ""))
            graph_score, graph_rank, path = graph_by_source.get(source, (0.0, None, ()))
            key = self._key(doc)
            candidate = candidates.setdefault(key, {
                "doc": doc,
                "text_score": float(score),
                "vector_rank": len(base) + offset,
                "graph_score": 0.0,
                "graph_rank": None,
                "path": (),
            })
            candidate["text_score"] = max(candidate["text_score"], float(score))
            candidate["graph_score"] = max(candidate["graph_score"], graph_score)
            candidate["graph_rank"] = graph_rank
            candidate["path"] = path

        graph_weight = self.config.graph_weight
        threshold = self.config.retrieval_score_threshold
        ranked: list[tuple[float, float, Document]] = []
        for candidate in candidates.values():
            text_score = candidate["text_score"]
            graph_score = candidate["graph_score"]
            final_score = (
                (1 - graph_weight) * text_score + graph_weight * graph_score
                if graph_score
                else text_score
            )
            if final_score < threshold:
                continue
            rank_score = 1 / (60 + candidate["vector_rank"])
            if candidate["graph_rank"] is not None:
                rank_score += graph_weight / (60 + candidate["graph_rank"])
                candidate["doc"].metadata["graph_path"] = list(candidate["path"])
                candidate["doc"].metadata["graph_score"] = round(graph_score, 4)
            ranked.append((rank_score, min(1.0, final_score), candidate["doc"]))
        ranked.sort(key=lambda item: (-item[0], -item[1], self._key(item[2])))

        selected = [(doc, score) for _, score, doc in ranked]
        self.last_trace = {
            "vector_hits": len(base),
            "seed_nodes": len(seeds),
            "entity_seed_nodes": len(entity_seeds),
            "expanded_sources": len(graph_hits),
            "graph_candidates": len(expanded),
            "selected_evidence": len(selected),
        }
        return selected

    @staticmethod
    def _key(doc: Document) -> tuple[str, str]:
        return (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("anchor", "")),
        )
