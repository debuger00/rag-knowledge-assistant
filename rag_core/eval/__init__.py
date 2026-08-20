"""仅评测 Retriever 检索性能的离线评测子包。

- metrics : Recall@K / HitRate@K / MRR@K / nDCG@K / 延迟统计（纯函数）
- dataset : RGB JSONL 加载与全局 Corpus 构建
- runner  : 索引构建 -> 调用现有 Retriever -> 指标 + 延迟 -> 报告
"""
