"""Grounded RAG pipeline with deterministic refusal and citations."""
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_core.llm.deepseek import create_llm
from rag_core.retrieval.retriever import ParentChildRetriever
from rag_core.indexing.store import VectorStoreManager
from config import get_config


REFUSAL_MESSAGE = "根据当前文档集，无法找到足够可靠的依据回答该问题。"


class Citation(TypedDict):
    path: str
    anchor: str
    heading: str
    quote: str
    score: float


SYSTEM_PROMPT = """你是技术文档问答助手。只能根据提供的证据回答问题。
证据已经通过相关度阈值检查。不得使用训练知识补充证据中不存在的信息。

以下是从文档集中检索到的证据：

{context}

<对话历史>
{history}

用户问题：{question}

要求：
- 用中文回答
- 结论必须能从证据中直接推出
- 不得编造文件、标题、数字或事实
- 不要自行生成来源列表，系统会附加经过校验的引用"""


def _format_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为 Prompt 使用的上下文字符串。"""
    if not docs:
        return "（未找到相关笔记）"

    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        anchor = doc.metadata.get("anchor", "document-start")
        parts.append(
            f"--- 证据 {i + 1}: {source}#{anchor} ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(parts)


def _format_history(history: list[dict]) -> str:
    """将对话历史格式化为字符串。"""
    if not history:
        return "（无历史对话）"
    lines = []
    for turn in history[-6:]:  # 最近 6 轮
        role = "用户" if turn["role"] == "user" else "助手"
        lines.append(f"{role}：{turn['content']}")
    return "\n".join(lines)


class RAGPipeline:
    """RAG 问答管线。

    用法:
        pipeline = RAGPipeline(store)
        answer = pipeline.ask("Docker 网络模式有哪些？")
    """

    def __init__(self, store: VectorStoreManager):
        self.store = store
        self.config = get_config()

        self.retriever = ParentChildRetriever(
            store=store,
            top_k=self.config.retrieval_top_k,
            enable_link_expansion=self.config.enable_link_expansion,
        )
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self._llm = None

    @property
    def llm(self):
        """Create the gateway client only when an answer needs generation."""
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def _retrieve_and_format(self, query: str) -> str:
        docs, _ = self.retrieve_evidence(query)
        return _format_docs(docs)

    def retrieve_evidence(
        self, question: str
    ) -> tuple[list[Document], list[Citation]]:
        scored_docs = self.retriever.retrieve_with_scores(question)
        docs = [doc for doc, _ in scored_docs]
        citations = [
            Citation(
                path=str(doc.metadata.get("source", "未知来源")),
                anchor=str(doc.metadata.get("anchor", "document-start")),
                heading=str(doc.metadata.get("heading", "")),
                quote=_citation_quote(doc.page_content),
                score=round(float(score), 4),
            )
            for doc, score in scored_docs
        ]
        return docs, citations

    def retrieve_evidence_with_filter(
        self,
        question: str,
        folder: str | None = None,
        tag: str | None = None,
    ) -> tuple[list[Document], list[Citation]]:
        filter_dict = None
        if folder:
            filter_dict = {"folder": folder}
        if tag:
            filter_dict = filter_dict or {}
            filter_dict["tags"] = {"$contains": tag}

        original_filter = self.retriever.filter_dict
        self.retriever.filter_dict = filter_dict
        try:
            return self.retrieve_evidence(question)
        finally:
            self.retriever.filter_dict = original_filter

    def _stream_with_inputs(
        self,
        context: str,
        question: str,
        history_str: str,
        citations: list[Citation],
    ) -> Any:
        """同步流式 —— CLI 使用。"""
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "history": history_str,
        })
        parser = StrOutputParser()
        def generate():
            yield from parser.transform(self.llm.stream(prompt_value))
            yield _citation_footer(citations)

        return generate()

    async def _astream_with_inputs(
        self,
        context: str,
        question: str,
        history_str: str,
        citations: list[Citation],
    ) -> Any:
        """异步流式 —— SSE / Web 使用。"""
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "history": history_str,
        })
        async for chunk in self.llm.astream(prompt_value):
            yield chunk.content
        yield _citation_footer(citations)

    def ask(self, question: str, history: list[dict] | None = None) -> Any:
        """执行问答（同步），返回 LangChain stream 对象。"""
        history = history or []
        docs, citations = self.retrieve_evidence(question)
        if not docs:
            return iter([REFUSAL_MESSAGE])
        context = _format_docs(docs)
        history_str = _format_history(history)
        return self._stream_with_inputs(
            context, question, history_str, citations
        )

    async def aask(self, question: str, history: list[dict] | None = None) -> Any:
        """执行问答（异步），返回 async generator。"""
        history = history or []
        docs, citations = self.retrieve_evidence(question)
        if not docs:
            return _async_single(REFUSAL_MESSAGE)
        context = _format_docs(docs)
        history_str = _format_history(history)
        return self._astream_with_inputs(
            context, question, history_str, citations
        )

    async def aask_with_evidence(
        self,
        question: str,
        docs: list[Document],
        citations: list[Citation],
        history: list[dict] | None = None,
    ) -> Any:
        """Answer using evidence already retrieved by the API layer."""
        if not docs:
            return _async_single(REFUSAL_MESSAGE)
        return self._astream_with_inputs(
            _format_docs(docs),
            question,
            _format_history(history or []),
            citations,
        )

    def ask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """带过滤条件的问答（同步）。"""
        history = history or []
        docs, citations = self.retrieve_evidence_with_filter(
            question, folder, tag
        )
        if not docs:
            return iter([REFUSAL_MESSAGE])
        context = _format_docs(docs)
        history_str = _format_history(history)
        return self._stream_with_inputs(
            context, question, history_str, citations
        )

    async def aask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """带过滤条件的问答（异步）。"""
        history = history or []
        docs, citations = self.retrieve_evidence_with_filter(
            question, folder, tag
        )
        if not docs:
            return _async_single(REFUSAL_MESSAGE)
        context = _format_docs(docs)
        history_str = _format_history(history)
        return self._astream_with_inputs(
            context, question, history_str, citations
        )


def _citation_quote(content: str, max_length: int = 180) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= max_length else compact[:max_length] + "..."


def _citation_footer(citations: list[Citation]) -> str:
    if not citations:
        return ""
    unique = dict.fromkeys(
        f"[{citation['path']}#{citation['anchor']}]" for citation in citations
    )
    return "\n\n来源：" + "、".join(unique)


async def _async_single(value: str):
    yield value
