"""Offline, incremental semantic GraphRAG indexing pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from typing import Any, TYPE_CHECKING

from langchain_core.documents import Document

from config import Config, get_config
from rag_core.graph.extractor import SemanticExtractor
from rag_core.graph.semantic_models import GraphExtraction
from rag_core.graph.store import GraphStore
from rag_core.graph.summarizer import DescriptionSummarizer, summary_cache_key
from rag_core.indexing.splitter import parent_child_split

if TYPE_CHECKING:
    from rag_core.indexing.store import VectorStoreManager


SEMANTIC_INDEX_VERSION = 1


class SemanticGraphIndexer:
    def __init__(
        self,
        graph_store: GraphStore,
        extractor: SemanticExtractor,
        config: Config | None = None,
        vector_store: "VectorStoreManager | None" = None,
        summarizer: DescriptionSummarizer | None = None,
    ):
        self.graph_store = graph_store
        self.extractor = extractor
        self.config = config or get_config()
        self.vector_store = vector_store
        self.summarizer = summarizer

    def build(
        self,
        documents: list[Document],
        *,
        changed_only: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        selected = [
            document
            for document in documents
            if source is None
            or str(document.metadata.get("source", "")).replace("\\", "/")
            == source.replace("\\", "/")
        ]
        if source is not None and not selected:
            raise ValueError(f"未找到文档: {source}")

        stats: dict[str, Any] = {
            "documents_total": len(selected),
            "documents_indexed": 0,
            "documents_skipped": 0,
            "documents_failed": 0,
            "chunks_total": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "summary_llm_calls": 0,
            "summary_cache_hits": 0,
            "pruned_sources": 0,
            "errors": [],
        }

        for document in selected:
            document_source = str(document.metadata.get("source", "")).replace(
                "\\", "/"
            )
            children = self._children(document)
            fingerprint = self._source_fingerprint(children)
            if (
                changed_only
                and self.graph_store.get_semantic_source_fingerprint(document_source)
                == fingerprint
            ):
                stats["documents_skipped"] += 1
                continue

            stats["chunks_total"] += len(children)
            records, document_stats, errors = self._extract_children(children)
            stats["llm_calls"] += document_stats["llm_calls"]
            stats["cache_hits"] += document_stats["cache_hits"]
            if errors:
                stats["documents_failed"] += 1
                stats["errors"].append({"source": document_source, "errors": errors})
                continue
            self.graph_store.replace_semantic_source(
                document_source, fingerprint, records, refresh=False
            )
            stats["documents_indexed"] += 1

        if source is None:
            valid_sources = {
                str(document.metadata.get("source", "")).replace("\\", "/")
                for document in documents
            }
            stats["pruned_sources"] = self.graph_store.prune_semantic_sources(
                valid_sources, refresh=False
            )
        if stats["documents_indexed"] or stats["pruned_sources"]:
            self.graph_store.refresh_semantic_graph()
        if self.summarizer is not None and (
            stats["documents_indexed"] or stats["pruned_sources"]
        ):
            summary_stats = self._summarize_descriptions()
            stats.update(summary_stats)
        if self.vector_store is not None and (
            stats["documents_indexed"] or stats["pruned_sources"]
        ):
            stats["entity_embeddings"] = self.vector_store.rebuild_entities(
                self.graph_store.list_entities()
            )
        stats["graph"] = self.graph_store.get_stats()
        return stats

    def _summarize_descriptions(self) -> dict[str, int]:
        summaries: dict[str, str] = {}
        calls = 0
        cache_hits = 0
        for group in self.graph_store.semantic_description_groups():
            descriptions = group["descriptions"]
            if len(descriptions) < 2:
                continue
            key = summary_cache_key(
                group["kind"],
                group["item_id"],
                descriptions,
                self.summarizer.model_id,
                self.summarizer.prompt_hash,
            )
            summary = self.graph_store.get_cached_summary(key)
            if summary is None:
                try:
                    summary = self.summarizer.summarize(
                        group["kind"], descriptions
                    )
                except Exception:
                    continue
                self.graph_store.put_cached_summary(
                    summary_key=key,
                    kind=group["kind"],
                    item_id=group["item_id"],
                    model_id=self.summarizer.model_id,
                    prompt_hash=self.summarizer.prompt_hash,
                    descriptions=descriptions,
                    summary=summary,
                )
                calls += 1
            else:
                cache_hits += 1
            summaries[group["item_id"]] = summary
        self.graph_store.apply_semantic_summaries(summaries)
        return {"summary_llm_calls": calls, "summary_cache_hits": cache_hits}

    def _children(self, document: Document) -> list[Document]:
        split = parent_child_split(
            [document],
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
            child_max_len=self.config.child_max_len_before_split,
        )
        return [value for value in split if value.metadata.get("doc_type") == "child"]

    def _source_fingerprint(self, children: list[Document]) -> str:
        payload = {
            "version": SEMANTIC_INDEX_VERSION,
            "model": self.extractor.model_id,
            "prompt": self.extractor.prompt_hash,
            "extractor_version": self.extractor.extractor_version,
            "chunks": [
                {
                    "id": str(child.metadata.get("chunk_id", "")),
                    "hash": _content_hash(child.page_content),
                }
                for child in children
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _extract_children(
        self, children: list[Document]
    ) -> tuple[
        list[tuple[str, str, str, GraphExtraction]],
        dict[str, int],
        list[str],
    ]:
        records_by_id: dict[str, tuple[str, str, str, GraphExtraction]] = {}
        missing: list[tuple[Document, str, str]] = []
        cache_hits = 0
        errors: list[str] = []
        for child in children:
            chunk_id = str(child.metadata.get("chunk_id", ""))
            content_hash = _content_hash(child.page_content)
            cache_key = self.graph_store.extraction_cache_key(
                chunk_id,
                content_hash,
                self.extractor.model_id,
                self.extractor.prompt_hash,
                self.extractor.extractor_version,
            )
            cached = self.graph_store.get_cached_extraction(cache_key)
            if cached is None:
                missing.append((child, cache_key, content_hash))
                continue
            try:
                extraction = GraphExtraction.from_dict(
                    cached,
                    text=child.page_content,
                    entity_types=self.extractor.entity_types,
                    min_confidence=0.0,
                )
            except ValueError:
                missing.append((child, cache_key, content_hash))
                continue
            records_by_id[chunk_id] = (
                chunk_id,
                str(child.metadata.get("anchor", "")),
                cache_key,
                extraction,
            )
            cache_hits += 1

        if missing:
            workers = min(self.config.graph_extraction_concurrency, len(missing))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.extractor.extract, child.page_content): (
                        child, cache_key, content_hash
                    )
                    for child, cache_key, content_hash in missing
                }
                for future in as_completed(futures):
                    child, cache_key, content_hash = futures[future]
                    chunk_id = str(child.metadata.get("chunk_id", ""))
                    try:
                        extraction = future.result()
                    except Exception as exc:
                        errors.append(f"{chunk_id}: {exc}")
                        continue
                    self.graph_store.put_cached_extraction(
                        cache_key=cache_key,
                        chunk_id=chunk_id,
                        content_hash=content_hash,
                        model_id=self.extractor.model_id,
                        prompt_hash=self.extractor.prompt_hash,
                        extractor_version=self.extractor.extractor_version,
                        response=extraction.to_dict(),
                    )
                    records_by_id[chunk_id] = (
                        chunk_id,
                        str(child.metadata.get("anchor", "")),
                        cache_key,
                        extraction,
                    )

        ordered_records = [
            records_by_id[str(child.metadata.get("chunk_id", ""))]
            for child in children
            if str(child.metadata.get("chunk_id", "")) in records_by_id
        ]
        return (
            ordered_records,
            {"llm_calls": len(missing), "cache_hits": cache_hits},
            errors,
        )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
