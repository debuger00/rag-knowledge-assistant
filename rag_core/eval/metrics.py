"""Retriever 检索性能的离线评测指标（纯函数，无检索/LLM 依赖）。

口径说明（按原始文档去重后的排序列表计算）：
- Recall@K    : top-K 命中的 gold 文档数 / gold 文档总数
- HitRate@K   : top-K 是否至少命中一个 gold（0/1，跨 query 求均值即"至少一次命中"比例）
- MRR@K       : 首个 gold 命中排名的倒数（未命中记 0）
- nDCG@K      : 二值相关（gold=1），DCG 用 top-K 的理想 DCG 归一化
- first_hit_rank / hit : 供逐 query 失败分析使用（hit 取最大 K 窗口）
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

DEFAULT_KS = (1, 3, 5, 10, 20)
MRR_K = 10
NDCG_K = 10


def metric_keys(
    ks: Sequence[int] = DEFAULT_KS,
    mrr_k: int = MRR_K,
    ndcg_k: int = NDCG_K,
) -> tuple[str, ...]:
    keys: list[str] = []
    for k in ks:
        keys.append(f"recall@{k}")
    for k in ks:
        keys.append(f"hit_rate@{k}")
    keys.append(f"mrr@{mrr_k}")
    keys.append(f"ndcg@{ndcg_k}")
    return tuple(keys)


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Recall@K：top-K 命中的 gold 占全部 gold 的比例。gold 为空返回 0。"""
    if not gold or k < 1:
        return 0.0
    hits = set(retrieved[:k]) & gold
    return len(hits) / len(gold)


def hit_rate_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """HitRate@K：top-K 是否至少命中一个 gold（0.0 或 1.0）。"""
    if not gold or k < 1:
        return 0.0
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def first_hit_rank(
    retrieved: Sequence[str], gold: set[str], k: int | None = None
) -> int | None:
    """返回首个命中 gold 的 1 起排名；未命中或超窗返回 None。"""
    limit = len(retrieved) if k is None else min(k, len(retrieved))
    for index in range(limit):
        if retrieved[index] in gold:
            return index + 1
    return None


def mrr_at_k(retrieved: Sequence[str], gold: set[str], k: int = MRR_K) -> float:
    """MRR@K：首个命中排名的倒数，未命中记 0。"""
    rank = first_hit_rank(retrieved, gold, k)
    return 1.0 / rank if rank is not None else 0.0


def _dcg(relevances: Iterable[float]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg_at_k(
    retrieved: Sequence[str], gold: set[str], k: int = NDCG_K
) -> float:
    """nDCG@K：二值相关（命中 gold 为 1）。理想 DCG 取 min(len(gold), k)。"""
    if not gold or k < 1:
        return 0.0
    top = retrieved[:k]
    dcg = _dcg(1.0 if item in gold else 0.0 for item in top)
    ideal = _dcg([1.0] * min(len(gold), k))
    return dcg / ideal if ideal > 0.0 else 0.0


def evaluate_query(
    retrieved: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    mrr_k: int = MRR_K,
    ndcg_k: int = NDCG_K,
) -> dict:
    """对单条 query 计算全部指标，含失败分析字段（first_hit_rank / hit）。"""
    gold_set = set(gold)
    result: dict = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, gold_set, k)
        result[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, gold_set, k)
    result[f"mrr@{mrr_k}"] = mrr_at_k(retrieved, gold_set, mrr_k)
    result[f"ndcg@{ndcg_k}"] = ndcg_at_k(retrieved, gold_set, ndcg_k)
    rank = first_hit_rank(retrieved, gold_set, max(ks))
    result["first_hit_rank"] = rank
    result["hit"] = rank is not None
    return result


def aggregate_metrics(
    per_query: Sequence[dict],
    ks: Sequence[int] = DEFAULT_KS,
    mrr_k: int = MRR_K,
    ndcg_k: int = NDCG_K,
) -> dict[str, float]:
    """对多条 query 的 evaluate_query 结果做均值聚合。"""
    if not per_query:
        return {}
    count = len(per_query)
    return {
        key: sum(float(item.get(key, 0.0)) for item in per_query) / count
        for key in metric_keys(ks, mrr_k, ndcg_k)
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """nearest-rank 百分位：ceil(p * n) 位置的排序值。"""
    index = math.ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def latency_stats(latencies_ms: Iterable[float]) -> dict:
    """检索延迟统计：count / mean / p50 / p95 / min / max（nearest-rank）。"""
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 4),
        "p50_ms": round(_percentile(values, 0.50), 4),
        "p95_ms": round(_percentile(values, 0.95), 4),
        "min_ms": round(values[0], 4),
        "max_ms": round(values[-1], 4),
    }
