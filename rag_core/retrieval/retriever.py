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
        config = get_config()
        self.top_k = config.retrieval_top_k
        self.enable_link_expansion = config.enable_link_expansion

        # Step 1: 在子块中检索
        children = self.store.similarity_search(
            query, k=self.top_k, filter_dict=self.filter_dict
        )

        if not children:
            return []

        # Step 2: 按 parent_id 去重分组
        seen_parents: set[str] = set()
        for child in children:
            pid = child.metadata.get("parent_id", "")
            if pid:
                seen_parents.add(pid)

        # Step 3: 取出完整父文档
        parent_docs = self.store.get_parents_by_sources(list(seen_parents))

        # Step 4: 可选 — 链接扩展检索
        if self.enable_link_expansion:
            linked_docs = self._expand_by_links(parent_docs)
            existing_sources = {d.metadata.get("source") for d in parent_docs}
            for ld in linked_docs:
                if ld.metadata.get("source") not in existing_sources:
                    parent_docs.append(ld)
                    existing_sources.add(ld.metadata["source"])

        return parent_docs

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
