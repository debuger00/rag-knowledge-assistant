"""聊天服务：返回经过引用校验的结构化回答。"""
from rag_core.retrieval.pipeline import (
    AnswerResponse,
    RAGPipeline,
    build_retrieval_path,
)

_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    if len(_sessions) > 100:
        del _sessions[next(iter(_sessions))]
    return _sessions[session_id]


async def chat_answer(
    pipeline: RAGPipeline,
    question: str,
    session_id: str = "default",
    folder: str | None = None,
    tag: str | None = None,
    mode: str = "auto",
    debug_retrieval: bool = False,
) -> AnswerResponse:
    history = get_history(session_id)
    filter_dict = {"folder": folder} if folder else None
    if tag:
        filter_dict = filter_dict or {}
        filter_dict["__tag__"] = tag
    scored, resolved_mode = pipeline.retrieve_scored_evidence_with_mode(
        question, mode=mode, filter_dict=filter_dict
    )
    response = await pipeline.aanswer(
        question,
        history,
        scored_evidence=scored,
        mode=resolved_mode,
    )

    response["mode"] = resolved_mode
    retrieval_path = build_retrieval_path(
        question, resolved_mode, response, scored
    )
    if retrieval_path is not None:
        response["retrieval_path"] = retrieval_path
    if debug_retrieval:
        response["retrieval_trace"] = pipeline.get_retrieval_trace()

    answer_text = " ".join(
        item["text"] for item in response.get("answer", [])
    ) or response.get("message", "")
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer_text})
    if len(history) > 20:
        _sessions[session_id] = history[-20:]
    return response
