"""LLM graph extractor with strict JSON parsing and grounded validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

from config import Config, get_config
from rag_core.graph.semantic_models import (
    EXTRACTOR_VERSION,
    GraphExtraction,
)
from rag_core.llm.deepseek import create_llm


class SemanticExtractor(Protocol):
    model_id: str
    prompt_hash: str
    extractor_version: int
    entity_types: tuple[str, ...]

    def extract(self, text: str) -> GraphExtraction: ...


class LLMGraphExtractor:
    """Extract entities and relationships from one TextUnit at a time."""

    extractor_version = EXTRACTOR_VERSION

    def __init__(self, config: Config | None = None, llm=None):
        self.config = config or get_config()
        self.entity_types = tuple(self.config.graph_entity_types)
        self.model_id = self.config.llm_model
        self._prompt = _read_prompt(self.config.graph_extraction_prompt)
        signature = self._prompt + "\0" + json.dumps(
            self.entity_types, ensure_ascii=False
        ) + "\0" + str(self.config.graph_min_confidence)
        self.prompt_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(streaming=False)
        return self._llm

    def extract(self, text: str) -> GraphExtraction:
        result = self._invoke_and_validate(self._render_prompt(text))
        for _ in range(self.config.graph_max_gleanings):
            gleaning_prompt = (
                self._render_prompt(text)
                + "\n\n这是补抽轮次。只返回上一轮可能遗漏的实体和关系；"
                "仍须返回相同 JSON 结构，不要重复已有内容。"
            )
            result = result.merged_with(self._invoke_and_validate(gleaning_prompt))
        return result

    def _invoke_and_validate(self, prompt: str) -> GraphExtraction:
        last_error: Exception | None = None
        current_prompt = prompt
        for _ in range(self.config.graph_extraction_max_retries + 1):
            try:
                response = self.llm.invoke(current_prompt)
                content = response.content if hasattr(response, "content") else response
                payload = _parse_json_object(str(content))
                text = _extract_text_from_prompt(prompt)
                return GraphExtraction.from_dict(
                    payload,
                    text=text,
                    entity_types=self.entity_types,
                    min_confidence=self.config.graph_min_confidence,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                current_prompt = (
                    prompt
                    + "\n\n上一次输出无法通过 JSON Schema 校验。"
                    "请只输出一个合法 JSON 对象，不要添加 Markdown 代码围栏。"
                )
        raise ValueError(f"实体关系抽取失败: {last_error}")

    def _render_prompt(self, text: str) -> str:
        return (
            self._prompt.replace(
                "{{entity_types}}",
                json.dumps(self.entity_types, ensure_ascii=False),
            )
            .replace("{{text}}", text)
        )


def _read_prompt(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"图抽取 Prompt 不存在: {path}")
    return path.read_text(encoding="utf-8")


def _parse_json_object(content: str) -> dict:
    stripped = content.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型输出中没有 JSON 对象")
    payload = json.loads(stripped[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return payload


def _extract_text_from_prompt(prompt: str) -> str:
    marker = "<TEXT_UNIT>"
    end_marker = "</TEXT_UNIT>"
    start = prompt.find(marker)
    end = prompt.find(end_marker, start + len(marker))
    if start < 0 or end < 0:
        raise ValueError("抽取 Prompt 缺少 TEXT_UNIT 标记")
    return prompt[start + len(marker):end].strip("\r\n")
