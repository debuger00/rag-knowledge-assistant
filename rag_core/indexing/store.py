"""Chroma 向量存储管理 — 双 Collection（父文档 + 子块）。"""
import uuid
from datetime import datetime, timezone

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from rag_core.indexing.embedder import create_embedder


class VectorStoreManager:
    """管理 Chroma 中的两个 Collection：rag_parents 和 rag_children。"""

    def __init__(
        self,
        persist_dir: str,
        embedder: Embeddings | None = None,
    ):
        self.persist_dir = persist_dir
        self._embedder = embedder or create_embedder()

        self._client = chromadb.PersistentClient(path=persist_dir)

        self._parent_store = Chroma(
            collection_name="rag_parents",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._children_store = Chroma(
            collection_name="rag_children",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._entity_store = Chroma(
            collection_name="rag_entities",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._community_store = Chroma(
            collection_name="rag_community_reports",
            embedding_function=self._embedder,
            client=self._client,
        )

    _BATCH_SIZE = 20  # 较小批次，降低内存峰值

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        """清理 metadata，移除 ChromaDB 不支持的值（空列表、嵌套结构等）。"""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                if len(value) == 0:
                    continue  # ChromaDB rejects empty lists
                cleaned[key] = "|".join(str(v) for v in value)
            elif isinstance(value, dict):
                continue  # ChromaDB rejects nested dicts
            elif isinstance(value, (str, int, float, bool, type(None))):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def add_parents(self, documents: list[Document]) -> list[str]:
        """添加父文档到 rag_parents 集合（分批处理避免内存溢出）。"""
        if not documents:
            return []
        total = len(documents)
        all_ids = []
        for i in range(0, total, self._BATCH_SIZE):
            batch = documents[i:i + self._BATCH_SIZE]
            for doc in batch:
                doc.metadata = self._clean_metadata(doc.metadata)
            ids = [
                str(doc.metadata.get("document_id"))
                if doc.metadata.get("document_id")
                else f"parent_{doc.metadata.get('source', uuid.uuid4())}"
                for doc in batch
            ]
            self._parent_store.add_documents(batch, ids=ids)
            all_ids.extend(ids)
            done = min(i + self._BATCH_SIZE, total)
            print(f"  [parents] {done}/{total}", flush=True)
        return all_ids

    def add_children(self, documents: list[Document]) -> list[str]:
        """添加子文档到 rag_children 集合（分批处理避免内存溢出）。"""
        if not documents:
            return []
        total = len(documents)
        all_ids = []
        for i in range(0, total, self._BATCH_SIZE):
            batch = documents[i:i + self._BATCH_SIZE]
            for doc in batch:
                doc.metadata = self._clean_metadata(doc.metadata)
            ids = [
                str(doc.metadata.get("chunk_id"))
                if doc.metadata.get("chunk_id")
                else f"child_{uuid.uuid4().hex[:12]}_{doc.metadata.get('parent_id', 'unknown')}"
                for doc in batch
            ]
            self._children_store.add_documents(batch, ids=ids)
            all_ids.extend(ids)
            done = min(i + self._BATCH_SIZE, total)
            print(f"  [children] {done}/{total}", flush=True)
        return all_ids

    def similarity_search(
        self, query: str, k: int = 10, filter_dict: dict | None = None
    ) -> list[Document]:
        """在子块中进行语义搜索。"""
        return self._children_store.similarity_search(query, k=k, filter=filter_dict)

    def similarity_search_with_scores(
        self, query: str, k: int = 10, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Search child chunks and return normalized relevance scores."""
        return self._children_store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )

    @staticmethod
    def combine_filters(
        first: dict | None, second: dict | None
    ) -> dict | None:
        if not first:
            return second
        if not second:
            return first
        return {"$and": [first, second]}

    def similarity_search_by_sources(
        self,
        query: str,
        sources: list[str],
        *,
        k_per_source: int = 1,
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the best semantic chunks from graph-expanded sources."""
        effective_filter = dict(filter_dict or {})
        tag = effective_filter.pop("__tag__", None)
        results: list[tuple[Document, float]] = []
        for source in dict.fromkeys(sources):
            source_filter = self.combine_filters(
                effective_filter or None, {"source": source}
            )
            candidates = self.similarity_search_with_scores(
                query,
                k=max(k_per_source * 5, k_per_source) if tag else k_per_source,
                filter_dict=source_filter,
            )
            if tag:
                candidates = [
                    (doc, score) for doc, score in candidates
                    if tag in {
                        value.strip()
                        for value in str(doc.metadata.get("tags", "")).split("|")
                        if value.strip()
                    }
                ]
            results.extend(candidates[:k_per_source])
        return results

    def rebuild_entities(self, entities: list[dict]) -> int:
        """Replace the semantic-entity embedding collection."""
        try:
            self._client.delete_collection("rag_entities")
        except Exception:
            pass
        self._entity_store = Chroma(
            collection_name="rag_entities",
            embedding_function=self._embedder,
            client=self._client,
        )
        documents = [
            Document(
                page_content=(
                    str(entity.get("name", ""))
                    + "\n"
                    + str(entity.get("description", ""))
                ).strip(),
                metadata={
                    "entity_id": str(entity.get("id", "")),
                    "entity_type": str(
                        (entity.get("metadata") or {}).get("entity_type", "")
                    ),
                },
            )
            for entity in entities
            if entity.get("id") and entity.get("name")
        ]
        for index in range(0, len(documents), self._BATCH_SIZE):
            batch = documents[index:index + self._BATCH_SIZE]
            self._entity_store.add_documents(
                batch,
                ids=[str(value.metadata["entity_id"]) for value in batch],
            )
        return len(documents)

    def similarity_search_entities(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        if self._entity_store._collection.count() == 0:
            return []
        return self._entity_store.similarity_search_with_relevance_scores(query, k=k)

    def rebuild_community_reports(self, reports: list[dict]) -> int:
        try:
            self._client.delete_collection("rag_community_reports")
        except Exception:
            pass
        self._community_store = Chroma(
            collection_name="rag_community_reports",
            embedding_function=self._embedder,
            client=self._client,
        )
        documents = [
            Document(
                page_content=(
                    str(report.get("title", ""))
                    + "\n"
                    + str(report.get("summary", ""))
                ).strip(),
                metadata={
                    "community_id": str(report.get("id", "")),
                    "level": int(report.get("level", 0)),
                    "rank": float(report.get("rank", 0.0)),
                },
            )
            for report in reports
            if report.get("id") and report.get("summary")
        ]
        for index in range(0, len(documents), self._BATCH_SIZE):
            batch = documents[index:index + self._BATCH_SIZE]
            self._community_store.add_documents(
                batch,
                ids=[str(value.metadata["community_id"]) for value in batch],
            )
        return len(documents)

    def similarity_search_communities(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        if self._community_store._collection.count() == 0:
            return []
        return self._community_store.similarity_search_with_relevance_scores(
            query, k=k
        )

    def list_parent_sources(self) -> set[str]:
        """Return every source currently present in the parent collection."""
        result = self._parent_store.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return {
            str(metadata["source"])
            for metadata in metadatas
            if metadata and metadata.get("source")
        }

    def search_parents_by_source(self, source: str) -> list[Document]:
        """按 source 查找父文档。"""
        result = self._parent_store.get(where={"source": source})
        if not result or not result["documents"]:
            return []
        docs = []
        for i, content in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def search_child_by_citation(
        self, source: str, anchor: str
    ) -> Document | None:
        """Find the exact evidence chunk referenced by path and anchor."""
        result = self._children_store.get(
            where={"$and": [{"source": source}, {"anchor": anchor}]}
        )
        documents = result.get("documents") or []
        if not documents:
            return None
        metadatas = result.get("metadatas") or [{}]
        return Document(
            page_content=documents[0],
            metadata=metadatas[0] or {},
        )

    def get_parents_by_sources(self, sources: list[str]) -> list[Document]:
        """批量按 source 获取父文档。"""
        docs = []
        seen: set[str] = set()
        for source in sources:
            if source in seen:
                continue
            seen.add(source)
            docs.extend(self.search_parents_by_source(source))
        return docs

    def delete_by_source(self, source: str) -> None:
        """删除指定 source 的所有文档（父 + 子）。"""
        for store in [self._parent_store, self._children_store]:
            try:
                store._collection.delete(where={"source": source})
            except Exception:
                pass

    def rebuild(self, parents: list[Document], children: list[Document]) -> None:
        """清空所有数据并重建。"""
        for name in ["rag_parents", "rag_children"]:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

        self._parent_store = Chroma(
            collection_name="rag_parents",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._children_store = Chroma(
            collection_name="rag_children",
            embedding_function=self._embedder,
            client=self._client,
        )

        self.add_parents(parents)
        self.add_children(children)

    def get_stats(self) -> dict:
        """获取索引统计信息。"""
        try:
            parent_count = self._parent_store._collection.count()
        except Exception:
            parent_count = 0
        try:
            child_count = self._children_store._collection.count()
        except Exception:
            child_count = 0
        try:
            entity_count = self._entity_store._collection.count()
        except Exception:
            entity_count = 0
        try:
            community_report_count = self._community_store._collection.count()
        except Exception:
            community_report_count = 0
        return {
            "parent_count": parent_count,
            "child_count": child_count,
            "entity_count": entity_count,
            "community_report_count": community_report_count,
            "last_sync": datetime.now(timezone.utc).isoformat(),
        }
