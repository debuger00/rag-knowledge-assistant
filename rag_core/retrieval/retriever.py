"""Grounded child-chunk vector retriever."""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_core.indexing.store import VectorStoreManager
from config import get_config


class ParentChildRetriever(BaseRetriever):
    """LangChain Retriever：在 rag_children 中返回可直接引用的子块。

    检索流程：
    1. 在 rag_children 中语义搜索 top-k 个子块
    2. 执行 metadata 和相关度过滤
    3. 按 path + anchor 去重

    图扩展由 HybridGraphRetriever 负责，避免把不可直接引用的父文档
    混入 grounded evidence。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: VectorStoreManager
    top_k: int = 10
    enable_link_expansion: bool = True
    filter_dict: dict | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query)]

    def retrieve_with_scores(
        self, query: str, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Return grounded child chunks above the configured score threshold."""
        config = get_config()
        self.top_k = config.retrieval_top_k
        self.enable_link_expansion = config.enable_link_expansion

        # Step 1: 在子块中检索
        effective_filter = dict(
            (self.filter_dict if filter_dict is None else filter_dict) or {}
        )
        tag = effective_filter.pop("__tag__", None)
        scored_children = self.store.similarity_search_with_scores(
            query,
            k=self.top_k * 5 if tag else self.top_k,
            filter_dict=effective_filter or None,
        )
        if tag:
            scored_children = [
                (doc, score) for doc, score in scored_children
                if tag in {
                    value.strip()
                    for value in str(doc.metadata.get("tags", "")).split("|")
                    if value.strip()
                }
            ]
        threshold = config.retrieval_score_threshold
        grounded = [
            (doc, score)
            for doc, score in scored_children
            if score >= threshold
        ]

        # Keep only the strongest evidence for each path + anchor pair.
        result: list[tuple[Document, float]] = []
        seen: set[tuple[str, str]] = set()
        for doc, score in grounded:
            key = (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("anchor", "document-start")),
            )
            if key not in seen:
                seen.add(key)
                result.append((doc, score))
        return result[: self.top_k]
