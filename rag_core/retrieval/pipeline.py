"""RAG Chain — LCEL 管线：检索 → 格式化 → Prompt → LLM → 解析。"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from rag_core.llm.deepseek import create_deepseek_llm
from rag_core.retrieval.retriever import ParentChildRetriever
from rag_core.indexing.store import VectorStoreManager
from config import get_config


SYSTEM_PROMPT = """你是个人知识库问答助手。根据用户笔记内容回答问题。
如果笔记中没有相关信息，请明确说明，不要编造。

以下是从用户 Obsidian 知识库中检索到的相关笔记：

{context}

<对话历史>
{history}

用户问题：{question}

要求：
- 用中文回答
- 引用具体笔记时，注明来源（笔记文件名）
- 如果涉及多个笔记的观点，请分别说明
- 可以综合多篇笔记进行分析"""


def _format_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为 Prompt 使用的上下文字符串。"""
    if not docs:
        return "（未找到相关笔记）"

    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"--- 笔记 {i + 1}: {source} ---\n{doc.page_content}\n")
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
        self.llm = create_deepseek_llm()

        self.chain = (
            {
                "context": RunnableLambda(func=self._retrieve_and_format),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(func=lambda _: ""),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_and_format(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return _format_docs(docs)

    def _stream_with_inputs(self, context: str, question: str, history_str: str) -> Any:
        """同步流式 —— CLI 使用。"""
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "history": history_str,
        })
        parser = StrOutputParser()
        return parser.transform(self.llm.stream(prompt_value))

    async def _astream_with_inputs(self, context: str, question: str, history_str: str) -> Any:
        """异步流式 —— SSE / Web 使用。"""
        prompt_value = self.prompt.invoke({
            "context": context,
            "question": question,
            "history": history_str,
        })
        async for chunk in self.llm.astream(prompt_value):
            yield chunk.content

    def ask(self, question: str, history: list[dict] | None = None) -> Any:
        """执行问答（同步），返回 LangChain stream 对象。"""
        history = history or []
        context = self._retrieve_and_format(question)
        history_str = _format_history(history)
        return self._stream_with_inputs(context, question, history_str)

    async def aask(self, question: str, history: list[dict] | None = None) -> Any:
        """执行问答（异步），返回 async generator。"""
        history = history or []
        context = self._retrieve_and_format(question)
        history_str = _format_history(history)
        return self._astream_with_inputs(context, question, history_str)

    def ask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """带过滤条件的问答（同步）。"""
        history = history or []

        filter_dict = None
        if folder:
            filter_dict = filter_dict or {}
            filter_dict["folder"] = folder
        if tag:
            filter_dict = filter_dict or {}
            filter_dict["tags"] = {"$contains": tag}

        original_filter = self.retriever.filter_dict
        self.retriever.filter_dict = filter_dict
        try:
            context = self._retrieve_and_format(question)
            history_str = _format_history(history)
            return self._stream_with_inputs(context, question, history_str)
        finally:
            self.retriever.filter_dict = original_filter

    async def aask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """带过滤条件的问答（异步）。"""
        history = history or []

        filter_dict = None
        if folder:
            filter_dict = filter_dict or {}
            filter_dict["folder"] = folder
        if tag:
            filter_dict = filter_dict or {}
            filter_dict["tags"] = {"$contains": tag}

        original_filter = self.retriever.filter_dict
        self.retriever.filter_dict = filter_dict
        try:
            context = self._retrieve_and_format(question)
            history_str = _format_history(history)
            return self._astream_with_inputs(context, question, history_str)
        finally:
            self.retriever.filter_dict = original_filter
