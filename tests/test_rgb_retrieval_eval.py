"""单元测试：RGB Retriever 离线评测的指标计算与数据转换。

覆盖两个纯逻辑层：
- rag_core.eval.metrics —— Recall@K / HitRate@K / MRR@K / nDCG@K / 延迟统计
- rag_core.eval.dataset —— RGB JSONL 解析、全局 Corpus 去重构建、gold 映射

评测主流程（runner）通过真实检索集成运行，不在此 mock。
"""
import math
import json

import pytest

from rag_core.eval.dataset import RGBSample, build_corpus, load_rgb_jsonl
from rag_core.eval.metrics import (
    aggregate_metrics,
    evaluate_query,
    hit_rate_at_k,
    latency_stats,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def test_recall_at_k_counts_gold_in_top_k():
    retrieved = ["a", "b", "c", "d"]
    gold = {"a", "d"}
    assert recall_at_k(retrieved, gold, 1) == 0.5
    assert recall_at_k(retrieved, gold, 2) == 0.5   # 仅 "a" 进入 top2
    assert recall_at_k(retrieved, gold, 4) == 1.0


def test_recall_at_k_zero_when_no_hit():
    assert recall_at_k(["x", "y"], {"z"}, 2) == 0.0


def test_recall_at_k_empty_gold_is_zero():
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_hit_rate_at_k_is_binary_any_hit():
    assert hit_rate_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"c"}, 2) == 0.0
    assert hit_rate_at_k(["a", "b"], {"b"}, 1) == 0.0  # "b" 不在 top1


def test_mrr_uses_reciprocal_of_first_hit_rank():
    assert mrr_at_k(["x", "a", "y"], {"a"}, 10) == pytest.approx(1 / 2)
    assert mrr_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_mrr_respects_k_boundary():
    retrieved = [f"d{i}" for i in range(20)]
    assert mrr_at_k(retrieved, {"d9"}, 10) == pytest.approx(1 / 10)
    assert mrr_at_k(retrieved, {"d10"}, 10) == 0.0   # 首个命中在 rank 11


def test_ndcg_at_k_binary_relevance():
    # 两个 gold 都在 top2：DCG == IDCG
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a", "b"}, 10) == pytest.approx(1.0)
    # retrieved [x, a, b]：rank2/rank3 命中
    expected = (
        (1 / math.log2(3) + 1 / math.log2(4))
        / (1 + 1 / math.log2(3))
    )
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_caps_ideal_by_k():
    gold = {"a", "b", "c"}          # 3 个 gold，但 k=2
    assert ndcg_at_k(["a", "b"], gold, 2) == pytest.approx(1.0)
    expected = 1 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x"], gold, 2) == pytest.approx(expected)


def test_ndcg_single_gold_at_rank_two():
    # DCG = 1/log2(3)，IDCG = 1
    assert ndcg_at_k(["a", "b", "c"], {"b"}, 10) == pytest.approx(1 / math.log2(3))


def test_evaluate_query_returns_expected_metrics():
    result = evaluate_query(["a", "b", "c"], {"b"})
    assert result["recall@1"] == 0.0
    assert result["recall@3"] == 1.0
    assert result["hit_rate@1"] == 0.0
    assert result["hit_rate@3"] == 1.0
    assert result["mrr@10"] == pytest.approx(1 / 2)
    assert result["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert result["first_hit_rank"] == 2
    assert result["hit"] is True


def test_evaluate_query_no_hit():
    result = evaluate_query(["x", "y", "z"], {"a"})
    assert result["hit_rate@5"] == 0.0
    assert result["recall@5"] == 0.0
    assert result["mrr@10"] == 0.0
    assert result["first_hit_rank"] is None
    assert result["hit"] is False


def test_aggregate_metrics_averages_across_queries():
    per_query = [
        {"recall@5": 1.0, "hit_rate@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
        {"recall@5": 0.0, "hit_rate@5": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
    ]
    agg = aggregate_metrics(per_query)
    assert agg["recall@5"] == pytest.approx(0.5)
    assert agg["hit_rate@5"] == pytest.approx(0.5)
    assert agg["mrr@10"] == pytest.approx(0.5)
    assert agg["ndcg@10"] == pytest.approx(0.5)


def test_latency_stats_mean_p50_p95_nearest_rank():
    latencies = [10.0, 20.0, 30.0, 40.0, 100.0]
    stats = latency_stats(latencies)
    assert stats["count"] == 5
    assert stats["mean_ms"] == pytest.approx(40.0)
    assert stats["p50_ms"] == pytest.approx(30.0)
    assert stats["p95_ms"] == pytest.approx(100.0)


def test_latency_stats_empty():
    assert latency_stats([])["count"] == 0


# ---------------------------------------------------------------------------
# 数据转换：RGB JSONL -> 全局 Corpus
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, text):
    path = tmp_path / "zh_refine.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_rgb_jsonl_parses_samples(tmp_path):
    path = _write_jsonl(tmp_path, (
        '{"id": "q1", "query": "问题一", "answer": ["答1"],'
        ' "positive": ["文档A内容", "文档B内容"], "negative": ["噪声1", "噪声2"]}\n'
        '{"id": "q2", "query": "问题二", "answer": ["答2"],'
        ' "positive": ["文档A内容"], "negative": ["噪声1", "噪声3"]}\n'
    ))
    samples = load_rgb_jsonl(path)
    assert len(samples) == 2
    assert samples[0].sample_id == "q1"
    assert samples[0].query == "问题一"
    assert samples[0].positive == ["文档A内容", "文档B内容"]
    assert samples[1].negative == ["噪声1", "噪声3"]


def test_build_corpus_dedups_and_maps_gold(tmp_path):
    samples = [
        RGBSample("q1", "问题一", ["A", "B"], ["N1"]),
        RGBSample("q2", "问题二", ["A"], ["N1", "N2"]),
    ]
    corpus = build_corpus(samples)
    assert len(corpus.documents) == 4                      # A, B, N1, N2
    assert len(set(corpus.source_by_content.values())) == 4
    assert set(corpus.gold["q1"]) == {
        corpus.source_by_content["A"], corpus.source_by_content["B"],
    }
    assert set(corpus.gold["q2"]) == {corpus.source_by_content["A"]}
    for doc in corpus.documents:
        assert doc.metadata.get("source")
        assert doc.metadata.get("rgb_doc_id") is not None


def test_build_corpus_handles_posneg_conflict(tmp_path):
    # 同一文本在 q1 是 positive、在 q2 是 negative：仍是一篇文档，且仅对 q1 是 gold
    samples = [
        RGBSample("q1", "问题一", ["X"], []),
        RGBSample("q2", "问题二", [], ["X"]),
    ]
    corpus = build_corpus(samples)
    assert len(corpus.documents) == 1
    assert corpus.gold["q1"] == [corpus.source_by_content["X"]]
    assert corpus.gold["q2"] == []


def test_corpus_documents_split_preserve_document_identity(tmp_path):
    from rag_core.indexing.splitter import parent_child_split

    samples = [
        RGBSample("q1", "问题一", ["A" * 5, "B" * 3], ["N1"]),
        RGBSample("q2", "问题二", ["A" * 5], ["N2"]),
    ]
    corpus = build_corpus(samples)
    split = parent_child_split(corpus.documents)
    children = [d for d in split if d.metadata.get("doc_type") == "child"]
    parents = [d for d in split if d.metadata.get("doc_type") == "parent"]

    assert len(parents) == len(corpus.documents)
    # 每个子块都保留原始文档的 source 和 document_id
    for child in children:
        assert child.metadata["source"] in corpus.source_by_content.values()
        assert child.metadata["document_id"]
    # 按 source 去重后能恢复原始文档集合
    assert {c.metadata["source"] for c in children} == set(
        corpus.source_by_content.values()
    )


def test_load_rgb_jsonl_missing_file():
    with pytest.raises(FileNotFoundError):
        load_rgb_jsonl("no-such-file.jsonl")
