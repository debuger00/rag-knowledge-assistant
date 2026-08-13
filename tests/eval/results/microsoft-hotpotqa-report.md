# Microsoft GraphRAG 数据集报告：hotpotqa

## 数据集

- 名称：Microsoft Filtered HotpotQA
- 文档：500
- 问题：100
- 推荐检索：local
- 评测模式：paired_context_recall
- Global pipeline：not_required

## 当前结果

### Basic

- paired_context_recall_at_10：91.00%
- mrr_at_10：0.7883

### Local

- paired_context_recall_at_10：90.00%
- mrr_at_10：0.6396

## 解释

- HotpotQA 的过滤结果保留原始 `test_N` 编号（编号有缺口）；按编号排序后与 CSV 行一一对应。
- HotpotQA 只能计算 paired-context recall；该上下文是问题配套输入，不等同于 Gold supporting facts。
- 播客和财报 CSV 没有答案、supporting source 或 reference response，只报告检索覆盖诊断。
- `kevin_scott` 和 `msft_multi` 需要社区发现、社区摘要和 Map-Reduce 才能进行真正的 Global GraphRAG 评测。
- 当前报告不调用 LLM，不评价开放式答案质量。
