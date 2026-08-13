# Microsoft GraphRAG 数据集报告：kevin_scott

## 数据集

- 名称：Kevin Scott Behind the Tech Podcasts
- 文档：60
- 问题：20
- 推荐检索：global
- 评测模式：retrieval_diagnostic
- Global pipeline：required_not_implemented

## 当前结果

### Basic

- nonempty_query_rate：100.00%
- average_sources_per_query：8.6500
- corpus_source_coverage：90.00%

### Local

- nonempty_query_rate：100.00%
- average_sources_per_query：8.6500
- corpus_source_coverage：90.00%

## 解释

- HotpotQA 的过滤结果保留原始 `test_N` 编号（编号有缺口）；按编号排序后与 CSV 行一一对应。
- HotpotQA 只能计算 paired-context recall；该上下文是问题配套输入，不等同于 Gold supporting facts。
- 播客和财报 CSV 没有答案、supporting source 或 reference response，只报告检索覆盖诊断。
- `kevin_scott` 和 `msft_multi` 需要社区发现、社区摘要和 Map-Reduce 才能进行真正的 Global GraphRAG 评测。
- 当前报告不调用 LLM，不评价开放式答案质量。
