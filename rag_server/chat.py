"""聊天服务：返回经过引用校验的结构化回答。"""
from rag_core.retrieval.pipeline import AnswerResponse, RAGPipeline

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
    if folder or tag:
        response = await pipeline.aask_with_filter(
            question, history, folder=folder, tag=tag, mode=mode
        )
    else:
        response = await pipeline.aask(question, history, mode=mode)

    response["mode"] = pipeline.last_retrieval_mode
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
