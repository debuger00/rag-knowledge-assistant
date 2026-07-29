"""聊天端点 — SSE 流式问答。"""
import json
from typing import AsyncIterator

from rag_core.retrieval.pipeline import RAGPipeline

_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    if len(_sessions) > 100:
        oldest = list(_sessions.keys())[0]
        del _sessions[oldest]
    return _sessions[session_id]


async def chat_stream(
    pipeline: RAGPipeline,
    question: str,
    session_id: str = "default",
    folder: str | None = None,
    tag: str | None = None,
) -> AsyncIterator[dict]:
    """SSE 流式返回答案。"""
    history = get_history(session_id)

    try:
        if folder or tag:
            docs, citations = pipeline.retrieve_evidence_with_filter(
                question, folder, tag
            )
        else:
            docs, citations = pipeline.retrieve_evidence(question)
        stream = await pipeline.aask_with_evidence(
            question, docs, citations, history
        )

        full_answer = ""
        yield {"event": "thinking", "data": "正在检索笔记..."}
        yield {"event": "sources", "data": citations}

        async for chunk in stream:
            if chunk:
                full_answer += chunk
                yield {"event": "token", "data": chunk}

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        if len(history) > 20:
            _sessions[session_id] = history[-20:]

        yield {"event": "done", "data": full_answer}

    except Exception as e:
        yield {"event": "error", "data": str(e)}
