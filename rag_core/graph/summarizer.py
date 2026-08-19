"""LLM description summarization for merged semantic graph items."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from config import Config, get_config
from rag_core.llm.deepseek import create_llm


class DescriptionSummarizer(Protocol):
    model_id: str
    prompt_hash: str

    def summarize(self, kind: str, descriptions: list[str]) -> str: ...


class LLMDescriptionSummarizer:
    def __init__(self, config: Config | None = None, llm=None):
        self.config = config or get_config()
        self.model_id = self.config.llm_model
        path = Path(self.config.graph_summary_prompt)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"描述汇总 Prompt 不存在: {path}")
        self.prompt = path.read_text(encoding="utf-8")
        self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(streaming=False)
        return self._llm

    def summarize(self, kind: str, descriptions: list[str]) -> str:
        prompt = (
            self.prompt
            + "\n\n对象类型："
            + kind
            + "\n输入描述：\n"
            + json.dumps(descriptions, ensure_ascii=False, indent=2)
        )
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        summary = str(content).strip()
        if not summary:
            raise ValueError("描述汇总模型返回空内容")
        return summary


def summary_cache_key(
    kind: str,
    item_id: str,
    descriptions: list[str],
    model_id: str,
    prompt_hash: str,
) -> str:
    payload = json.dumps(
        sorted(dict.fromkeys(descriptions)), ensure_ascii=False
    )
    return hashlib.sha256(
        f"{kind}\0{item_id}\0{model_id}\0{prompt_hash}\0{payload}".encode("utf-8")
    ).hexdigest()
