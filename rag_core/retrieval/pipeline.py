"""有证据约束的 RAG 生成、引用映射与确定性校验。"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import get_config
from rag_core.graph.store import GraphStore
from rag_core.retrieval.global_search import GlobalCommunityRetriever
from rag_core.indexing.store import VectorStoreManager
from rag_core.llm.deepseek import create_llm
from rag_core.retrieval.hybrid import HybridGraphRetriever
from rag_core.retrieval.retriever import ParentChildRetriever


REFUSAL_MESSAGE = (
    "根据当前文档集，未找到能够可靠回答该问题的依据，因此无法给出答案。"
)
DEFAULT_REFUSAL_REASON = "没有检索到能够直接支持答案的文档片段"


class Citation(TypedDict):
    id: str
    document_path: str
    anchor: str
    section_title: str
    quote: str
    score: float


class AnswerItem(TypedDict):
    text: str
    citation_ids: list[str]


class AnswerResponse(TypedDict, total=False):
    status: str
    answer: list[AnswerItem]
    citations: list[Citation]
    message: str
    reason: str
    mode: str
    retrieval_trace: dict[str, Any]


SYSTEM_PROMPT = """你只能根据提供的文档证据回答问题。

规则：
1. 不得使用文档之外的知识。
2. 每个结论都必须关联至少一个 evidence_id。
3. 不得创造文档路径、anchor、标题或引用原文。
4. 证据不足时输出 insufficient_evidence。
5. 不要为了回答而推测。
6. 不得引用与结论无关的证据。
7. 回答应简洁、直接；每个 answer 项只表达一个由证据直接支持的结论。
8. 引用文字由程序从原始检索片段中提取，你只选择 evidence_id。
9. 输出必须严格符合指定 JSON schema，不要输出额外说明或 Markdown 代码块。

JSON schema：
{{
  "status": "answered | insufficient_evidence",
  "answer": [
    {{"text": "结论句", "evidence_ids": ["ev_1"]}}
  ],
  "reason": "仅在证据不足时说明原因"
}}

只读文档证据：
{context}

对话历史仅用于理解指代，不能作为事实证据：
{history}

用户问题：{question}
"""


def _format_docs(docs: list[Document]) -> str:
    """兼容旧调用方；以只读 evidence JSON 格式化文档。"""
    if not docs:
        return "（未找到相关笔记）"
    evidence = []
    for index, doc in enumerate(docs, 1):
        evidence.append({
            "evidence_id": f"ev_{index}",
            "document_path": str(doc.metadata.get("source", "")),
            "anchor": str(doc.metadata.get("anchor", "")),
            "section_title": _section_title(doc),
            "content": doc.page_content,
        })
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "（无历史对话）"
    lines = []
    for turn in history[-6:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{turn.get('content', '')}")
    return "\n".join(lines)


def insufficient_evidence(reason: str = DEFAULT_REFUSAL_REASON) -> AnswerResponse:
    return {
        "status": "insufficient_evidence",
        "answer": [],
        "citations": [],
        "message": REFUSAL_MESSAGE,
        "reason": reason or DEFAULT_REFUSAL_REASON,
    }


def validate_answer_with_citations(
    response: dict[str, Any],
    retrieved_chunks: list[tuple[Document, float]] | list[Document],
    min_score: float | None = None,
    require_citations: bool | None = None,
) -> tuple[bool, list[str]]:
    """校验回答的引用闭环，绝不信任模型提供的引用元数据。"""
    config = get_config()
    threshold = (
        config.retrieval_score_threshold if min_score is None else min_score
    )
    citations_required = (
        config.rag_require_citations
        if require_citations is None
        else require_citations
    )
    errors: list[str] = []

    chunks: list[tuple[Document, float]] = []
    for item in retrieved_chunks:
        if isinstance(item, tuple):
            chunks.append((item[0], float(item[1])))
        else:
            chunks.append((item, float(item.metadata.get("score", 0.0))))

    evidence_by_key = {
        (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("anchor", "")),
        ): (doc, score)
        for doc, score in chunks
    }
    citations = response.get("citations")
    answers = response.get("answer")
    if response.get("status") != "answered":
        errors.append("status 不是 answered")
    if not isinstance(answers, list) or not answers:
        errors.append("回答为空")
        answers = []
    if not isinstance(citations, list):
        errors.append("citations 不是数组")
        citations = []

    citation_by_id: dict[str, dict[str, Any]] = {}
    for citation in citations:
        if not isinstance(citation, dict):
            errors.append("citation 不是对象")
            continue
        citation_id = citation.get("id")
        if not isinstance(citation_id, str) or not citation_id:
            errors.append("citation id 为空")
            continue
        if citation_id in citation_by_id:
            errors.append(f"引用 ID 重复: {citation_id}")
        citation_by_id[citation_id] = citation

        path = citation.get("document_path")
        anchor = citation.get("anchor")
        quote = citation.get("quote")
        if not all(isinstance(value, str) and value for value in (path, anchor, quote)):
            errors.append(f"引用元数据为空: {citation_id}")
            continue
        matched = evidence_by_key.get((path, anchor))
        if matched is None:
            errors.append(f"引用不属于检索证据: {citation_id}")
            continue
        doc, retrieved_score = matched
        if quote not in doc.page_content:
            errors.append(f"quote 无法在检索原文定位: {citation_id}")
        score = citation.get("score")
        if not isinstance(score, (int, float)):
            errors.append(f"引用分数无效: {citation_id}")
        elif float(score) < threshold or retrieved_score < threshold:
            errors.append(f"引用分数低于阈值: {citation_id}")
        if citation.get("section_title") != _section_title(doc):
            errors.append(f"section_title 与检索元数据不一致: {citation_id}")

    used_ids: set[str] = set()
    for index, answer in enumerate(answers, 1):
        if not isinstance(answer, dict) or not str(answer.get("text", "")).strip():
            errors.append(f"第 {index} 个回答句子为空")
            continue
        ids = answer.get("citation_ids")
        if citations_required and (not isinstance(ids, list) or not ids):
            errors.append(f"第 {index} 个回答句子没有引用")
            continue
        for citation_id in ids or []:
            if citation_id not in citation_by_id:
                errors.append(f"引用 ID 不存在: {citation_id}")
            else:
                used_ids.add(citation_id)

    unused_ids = set(citation_by_id) - used_ids
    if unused_ids:
        errors.append("存在未使用引用: " + ", ".join(sorted(unused_ids)))
    if citations_required and not citation_by_id:
        errors.append("回答没有任何引用")
    if len(citation_by_id) > config.rag_max_citations:
        errors.append("引用数量超过配置上限")
    return not errors, errors


class RAGPipeline:
    def __init__(
        self,
        store: VectorStoreManager,
        graph_store: GraphStore | None = None,
    ):
        self.store = store
        self.graph_store = graph_store
        self.config = get_config()
        self.basic_retriever = ParentChildRetriever(
            store=store,
            top_k=self.config.retrieval_top_k,
            enable_link_expansion=False,
        )
        self.hybrid_retriever = (
            HybridGraphRetriever(store, graph_store)
            if graph_store is not None and self.config.graph_enabled
            else None
        )
        self.global_retriever = (
            GlobalCommunityRetriever(store, graph_store)
            if graph_store is not None and self.config.graph_enabled
            else None
        )
        self.retriever = self.hybrid_retriever or self.basic_retriever
        self.last_retrieval_mode = "local" if self.hybrid_retriever else "basic"
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(streaming=False)
        return self._llm

    def retrieve_scored_evidence(
        self,
        question: str,
        mode: str = "auto",
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        retriever, resolved_mode = self._retriever_for_mode(mode)
        self.last_retrieval_mode = resolved_mode
        if filter_dict is None:
            scored = retriever.retrieve_with_scores(question)
        else:
            scored = retriever.retrieve_with_scores(question, filter_dict=filter_dict)
        return scored[
            : self.config.rag_max_citations
        ]

    def _retriever_for_mode(self, mode: str):
        normalized = (mode or "auto").lower()
        if normalized not in {"auto", "basic", "local", "global"}:
            raise ValueError("mode 必须是 auto、basic、local 或 global")
        basic = getattr(self, "basic_retriever", None)
        hybrid = getattr(self, "hybrid_retriever", None)
        fallback = getattr(self, "retriever")
        if normalized == "basic":
            return basic or fallback, "basic"
        if normalized == "global":
            global_retriever = getattr(self, "global_retriever", None)
            return global_retriever or basic or fallback, (
                "global" if global_retriever is not None else "basic"
            )
        if hybrid is not None:
            return hybrid, "local"
        return basic or fallback, "basic"

    def get_retrieval_trace(self) -> dict[str, Any]:
        if self.last_retrieval_mode == "global":
            global_retriever = getattr(self, "global_retriever", None)
            return dict(global_retriever.last_trace) if global_retriever else {}
        hybrid = getattr(self, "hybrid_retriever", None)
        if hybrid is None or self.last_retrieval_mode != "local":
            return {}
        return dict(hybrid.last_trace)

    def retrieve_evidence(
        self, question: str
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """兼容旧调用方：第二项改为只读 evidence 描述，不再是前端来源。"""
        scored = self.retrieve_scored_evidence(question)
        docs = []
        evidence = []
        for index, (doc, score) in enumerate(scored, 1):
            doc.metadata["score"] = float(score)
            docs.append(doc)
            evidence.append(_evidence_descriptor(index, doc, score))
        return docs, evidence

    def retrieve_scored_evidence_with_filter(
        self,
        question: str,
        folder: str | None = None,
        tag: str | None = None,
        mode: str = "auto",
    ) -> list[tuple[Document, float]]:
        filter_dict = {"folder": folder} if folder else None
        if tag:
            filter_dict = filter_dict or {}
            filter_dict["__tag__"] = tag
        return self.retrieve_scored_evidence(
            question, mode=mode, filter_dict=filter_dict
        )

    def retrieve_evidence_with_filter(
        self,
        question: str,
        folder: str | None = None,
        tag: str | None = None,
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        scored = self.retrieve_scored_evidence_with_filter(question, folder, tag)
        return [item[0] for item in scored], [
            _evidence_descriptor(index, doc, score)
            for index, (doc, score) in enumerate(scored, 1)
        ]

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        scored_evidence: list[tuple[Document, float]] | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        scored = (
            self.retrieve_scored_evidence(question, mode=mode)
            if scored_evidence is None
            else scored_evidence
        )
        refusal = _retrieval_refusal(scored, self.config.retrieval_score_threshold)
        if refusal:
            return insufficient_evidence(refusal)

        prompt_value = self.prompt.invoke({
            "context": _format_scored_evidence(scored),
            "question": question,
            "history": _format_history(history or []),
        })
        validation_errors: list[str] = []
        for _ in range(self.config.rag_max_retry + 1):
            try:
                raw = self.llm.invoke(prompt_value)
                model_output = _parse_model_output(raw.content)
                response = _build_response(model_output, scored)
                if response.get("status") == "insufficient_evidence":
                    return response
                valid, validation_errors = validate_answer_with_citations(
                    response, scored
                )
                if valid:
                    return response
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                validation_errors = [str(exc)]
        reason = "引用校验失败"
        if validation_errors:
            reason += "：" + "；".join(validation_errors)
        return insufficient_evidence(reason)

    async def aanswer(
        self,
        question: str,
        history: list[dict] | None = None,
        scored_evidence: list[tuple[Document, float]] | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        scored = (
            self.retrieve_scored_evidence(question, mode=mode)
            if scored_evidence is None
            else scored_evidence
        )
        refusal = _retrieval_refusal(scored, self.config.retrieval_score_threshold)
        if refusal:
            return insufficient_evidence(refusal)

        prompt_value = self.prompt.invoke({
            "context": _format_scored_evidence(scored),
            "question": question,
            "history": _format_history(history or []),
        })
        validation_errors: list[str] = []
        for _ in range(self.config.rag_max_retry + 1):
            try:
                raw = await self.llm.ainvoke(prompt_value)
                model_output = _parse_model_output(raw.content)
                response = _build_response(model_output, scored)
                if response.get("status") == "insufficient_evidence":
                    return response
                valid, validation_errors = validate_answer_with_citations(
                    response, scored
                )
                if valid:
                    return response
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                validation_errors = [str(exc)]
        reason = "引用校验失败"
        if validation_errors:
            reason += "：" + "；".join(validation_errors)
        return insufficient_evidence(reason)

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        return self.answer(question, history, mode=mode)

    async def aask(
        self,
        question: str,
        history: list[dict] | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        return await self.aanswer(question, history, mode=mode)

    def ask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        scored = self.retrieve_scored_evidence_with_filter(
            question, folder, tag, mode=mode
        )
        return self.answer(question, history, scored, mode=mode)

    async def aask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
        mode: str = "auto",
    ) -> AnswerResponse:
        scored = self.retrieve_scored_evidence_with_filter(
            question, folder, tag, mode=mode
        )
        return await self.aanswer(question, history, scored, mode=mode)


def _evidence_descriptor(
    index: int, doc: Document, score: float
) -> dict[str, Any]:
    return {
        "evidence_id": f"ev_{index}",
        "document_path": str(doc.metadata.get("source", "")),
        "anchor": str(doc.metadata.get("anchor", "")),
        "section_title": _section_title(doc),
        "content": doc.page_content,
        "score": round(float(score), 4),
    }


def _format_scored_evidence(scored: list[tuple[Document, float]]) -> str:
    return json.dumps(
        [
            _evidence_descriptor(index, doc, score)
            for index, (doc, score) in enumerate(scored, 1)
        ],
        ensure_ascii=False,
        indent=2,
    )


def _parse_model_output(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise ValueError("模型输出不是 JSON 字符串")
    parsed = json.loads(content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("模型输出 JSON 顶层必须是对象")
    return parsed


def _build_response(
    model_output: dict[str, Any],
    scored: list[tuple[Document, float]],
) -> AnswerResponse:
    if model_output.get("status") == "insufficient_evidence":
        return insufficient_evidence(
            str(model_output.get("reason") or "文档证据不能直接回答该问题")
        )
    if model_output.get("status") != "answered":
        raise ValueError("模型 status 无效")

    evidence_map = {
        f"ev_{index}": (doc, score)
        for index, (doc, score) in enumerate(scored, 1)
    }
    answer_items = model_output.get("answer")
    if not isinstance(answer_items, list) or not answer_items:
        raise ValueError("模型回答为空")

    normalized_answers: list[AnswerItem] = []
    evidence_order: list[str] = []
    for item in answer_items:
        if not isinstance(item, dict):
            raise ValueError("answer 项必须是对象")
        text = str(item.get("text", "")).strip()
        evidence_ids = item.get("evidence_ids")
        if not text or not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("每个结论必须包含文本和 evidence_ids")
        unique_evidence_ids = list(dict.fromkeys(str(value) for value in evidence_ids))
        if any(value not in evidence_map for value in unique_evidence_ids):
            raise ValueError("模型引用了不存在的 evidence_id")
        for evidence_id in unique_evidence_ids:
            if evidence_id not in evidence_order:
                evidence_order.append(evidence_id)
        normalized_answers.append({
            "text": text,
            "citation_ids": unique_evidence_ids,
        })

    citation_id_map = {
        evidence_id: f"cite_{index}"
        for index, evidence_id in enumerate(evidence_order, 1)
    }
    citations: list[Citation] = []
    for evidence_id in evidence_order:
        doc, score = evidence_map[evidence_id]
        citations.append({
            "id": citation_id_map[evidence_id],
            "document_path": str(doc.metadata.get("source", "")),
            "anchor": str(doc.metadata.get("anchor", "")),
            "section_title": _section_title(doc),
            "quote": _citation_quote(doc.page_content),
            "score": round(float(score), 4),
        })
    for item in normalized_answers:
        item["citation_ids"] = [
            citation_id_map[evidence_id] for evidence_id in item["citation_ids"]
        ]
    return {"status": "answered", "answer": normalized_answers, "citations": citations}


def _section_title(doc: Document) -> str:
    return str(
        doc.metadata.get("section_title")
        or doc.metadata.get("heading")
        or doc.metadata.get("filename")
        or "未命名段落"
    )


def _citation_quote(content: str, max_length: int = 500) -> str:
    """直接截取原始 chunk，保留原文字符以便精确定位。"""
    stripped = content.strip()
    return stripped[:max_length]


def _retrieval_refusal(
    scored: list[tuple[Document, float]], threshold: float
) -> str | None:
    if not scored:
        return DEFAULT_REFUSAL_REASON
    if max(float(score) for _, score in scored) < threshold:
        return "最高检索相关度低于可靠回答阈值"
    for doc, score in scored:
        if score >= threshold and all(
            str(doc.metadata.get(key, "")).strip()
            for key in ("source", "anchor")
        ) and doc.page_content.strip():
            return None
    return "检索片段缺少可追溯的路径、anchor 或原文"


def _citation_footer(citations: list[dict[str, Any]]) -> str:
    """旧调用兼容；新接口不再拼接 footer。"""
    if not citations:
        return ""
    refs = []
    for citation in citations:
        path = citation.get("document_path", citation.get("path", ""))
        anchor = citation.get("anchor", "")
        if path and anchor:
            refs.append(f"[{path}#{anchor}]")
    return "\n\n来源：" + "、".join(dict.fromkeys(refs)) if refs else ""
