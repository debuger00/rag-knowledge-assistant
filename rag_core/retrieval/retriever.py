"""父子检索器 — 子块语义检索 + 父文档补齐。"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_core.indexing.store import VectorStoreManager
from config import get_config


class ParentChildRetriever(BaseRetriever):
    """LangChain Retriever：在 rag_children 中搜索，返回完整父文档。

    检索流程：
    1. 在 rag_children 中语义搜索 top-k 个子块
    2. 按 parent_id 去重分组
    3. 从 rag_parents 取出完整父文档
    4. 可选：通过 [[链接]] 一阶扩展检索
    """

    store: VectorStoreManager
    top_k: int = 10
    enable_link_expansion: bool = True
    filter_dict: dict | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query)]

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Return grounded child chunks above the configured score threshold."""
        config = get_config()
        self.top_k = config.retrieval_top_k
        self.enable_link_expansion = config.enable_link_expansion

        # Step 1: 在子块中检索
        scored_children = self.store.similarity_search_with_scores(
            query, k=self.top_k, filter_dict=self.filter_dict
        )
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
        return result

    def _expand_by_links(self, parent_docs: list[Document]) -> list[Document]:
        """通过 [[链接]] 一阶扩展查找关联文档。"""
        all_links: set[str] = set()
        for doc in parent_docs:
            links = doc.metadata.get("links", [])
            all_links.update(links)

        if not all_links:
            return []

        # 链接可能不带 .md 后缀
        link_sources = []
        for link in all_links:
            link = link.replace("\\", "/")
            link_src = link if link.endswith(".md") else f"{link}.md"
            link_sources.append(link_src)

        return self.store.get_parents_by_sources(link_sources)
