# CRUD-RAG 中文真实语料检索报告

## 结论

- 查询：60 条
- 新闻文档：1120 篇
- 其中 Gold：120 篇
- Distractor：1000 篇
- Distractor 来源：80000_docs
- Basic Evidence Recall@10：97.78%
- Local Evidence Recall@10：97.78%
- Basic 全证据召回率：96.67%
- Local 全证据召回率：96.67%
- 图增益：+0.00%

## 图结构

- 节点：2242
- 边：1122
- unresolved：0

## 口径

- 使用官方 CRUD-RAG `crud_split` 中的真实中文新闻、问题和参考答案。
- 不使用固定向量种子，查询从真实 embedding 检索开始。
- Markdown 中不注入答案、event、Gold 链接或 case 标签，避免数据泄漏。
- CRUD-RAG 新闻没有天然 wikilink；当前图只有文档/章节结构，因此本报告主要验证中文向量检索。
- 答案生成和参考答案评分未调用 LLM，本报告不代表最终答案准确率。

## 分任务结果

| 任务 | 查询数 | Basic Recall | Local Recall | Basic 全证据 | Local 全证据 |
|---|---:|---:|---:|---:|---:|
| questanswer_1doc | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| questanswer_2docs | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| questanswer_3docs | 20 | 93.33% | 93.33% | 90.00% | 90.00% |
