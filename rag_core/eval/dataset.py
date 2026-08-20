"""RGB 数据集的加载与全局 Corpus 构建（无 LLM、无检索逻辑）。

数据流：RGB zh_refine.json (JSONL)
  -> 汇总全部样本的 positive + negative
  -> 按正文去重得到全局唯一文档集合
  -> 为每篇唯一文档分配稳定 source ID（= 原始文档 ID）
  -> 建立 query(sample_id) -> 该样本 positive 的 source ID 列表（Ground Truth）

说明：
- `source` 是评估口径下的"原始文档 ID"；父子分块后子 chunk 会保留该字段，
  评测时按 `source` 去重即可避免把同一文档的多个 chunk 当成多个结果。
- 同一正文若在某些样本是 positive、另一些是 negative（role=both），仍是一篇文档；
  对具体 query 是否属于 gold 只看该 query 自己的 positive 列表。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


@dataclass
class RGBSample:
    """RGB 单条样本：query + positive（Ground Truth）+ negative（干扰文档）。"""

    sample_id: str
    query: str
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)


@dataclass
class RGBCorpus:
    """全局 Corpus：去重后的文档集合 + 逐 query 的 gold 映射。"""

    samples: list[RGBSample]
    documents: list[Document]
    source_by_content: dict[str, str]
    gold: dict[str, list[str]]

    def gold_sources(self, sample_id: str) -> list[str]:
        """给定样本 ID，返回其 positive 文档的 source ID 列表。"""
        return list(self.gold.get(sample_id, []))


def _normalize(text: str) -> str:
    return text.strip()


def load_rgb_jsonl(path: str | Path) -> list[RGBSample]:
    """解析 RGB JSONL（每行一个样本：id / query / positive / negative）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RGB 数据集不存在: {path}")
    samples: list[RGBSample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append(
                RGBSample(
                    sample_id=str(raw.get("id", "")),
                    query=str(raw.get("query", "")).strip(),
                    positive=[_normalize(p) for p in raw.get("positive", [])],
                    negative=[_normalize(n) for n in raw.get("negative", [])],
                )
            )
    return samples


def build_corpus(
    samples: Iterable[RGBSample],
    *,
    source_prefix: str = "rgb",
) -> RGBCorpus:
    """把样本的 positive + negative 汇总去重，构建统一全局 Corpus。

    每篇唯一文档分配稳定、可读的 source（f"{prefix}/doc_序号.md"），
    并在 metadata 中记录 rgb_doc_id / rgb_role / rgb_sample_ids 便于失败分析。
    """
    entries: dict[str, dict] = {}
    for sample in samples:
        for passage in [*sample.positive, *sample.negative]:
            key = _normalize(passage)
            if not key:
                continue
            entry = entries.setdefault(
                key, {"content": key, "roles": set(), "sample_ids": []}
            )
            role = "positive" if _normalize(passage) in sample.positive else "negative"
            entry["roles"].add(role)
            entry["sample_ids"].append(sample.sample_id)

    documents: list[Document] = []
    source_by_content: dict[str, str] = {}
    for index, (content, entry) in enumerate(
        sorted(entries.items(), key=lambda item: item[0]), 1
    ):
        source = f"{source_prefix}/doc_{index:04d}.md"
        source_by_content[content] = source
        roles = entry["roles"]
        role = "both" if len(roles) == 2 else next(iter(roles))
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "doc_type": "raw",
                    "rgb_doc_id": index,
                    "rgb_role": role,
                    "rgb_sample_ids": list(dict.fromkeys(entry["sample_ids"])),
                },
            )
        )

    gold: dict[str, list[str]] = {}
    for sample in samples:
        gold[sample.sample_id] = [
            source_by_content[_normalize(p)]
            for p in sample.positive
            if _normalize(p) in source_by_content
        ]

    return RGBCorpus(
        samples=list(samples),
        documents=documents,
        source_by_content=source_by_content,
        gold=gold,
    )
