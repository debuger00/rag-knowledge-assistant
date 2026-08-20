# RGB 中文 Retriever 检索性能评测报告

## 数据与索引

- 数据集：`E:\program\agent\0000personal-projects\02bankSuperpowers\data\02RGB\data\zh_refine.json`
- Query：300 条（样本共 300 条）
- 全局 Corpus（positive + negative 去重）：7337 篇
- Gold 文档（至少被一个 query 作为 positive）：2472 篇
- 索引：parent 7337 / child 7337
- 图：node 14674 / edge 7337
- 嵌入模型：`BAAI/bge-small-zh-v1.5`
- 评测窗口 max_k：20

## 检索模式

- `basic`：ParentChildRetriever（纯向量语义检索）
- `local`：HybridGraphRetriever（向量 + 图扩展；RGB 语料无 wikilink，图扩展为空时退化为向量路径）

## 指标对比

| 指标 | basic | local |
|---:|---:|---:|
| recall@1 | 0.1067 | 0.1067 |
| recall@3 | 0.2974 | 0.2974 |
| recall@5 | 0.4602 | 0.4602 |
| recall@10 | 0.6982 | 0.6982 |
| recall@20 | 0.8712 | 0.8712 |
| hit_rate@1 | 0.7267 | 0.7267 |
| hit_rate@3 | 0.9100 | 0.9100 |
| hit_rate@5 | 0.9500 | 0.9500 |
| hit_rate@10 | 0.9767 | 0.9767 |
| hit_rate@20 | 0.9867 | 0.9867 |
| mrr@10 | 0.8262 | 0.8262 |
| ndcg@10 | 0.7244 | 0.7244 |

## 检索延迟（毫秒）

| 统计 | basic | local |
|---:|---:|---:|
| count | 300 | 300 |
| mean_ms | 47.1815 | 52.1214 |
| p50_ms | 45.7785 | 50.5867 |
| p95_ms | 53.7998 | 59.4372 |
| min_ms | 40.3254 | 47.3661 |
| max_ms | 207.4424 | 98.0675 |

## 失败案例分析

- 每条 query 的 Top-K、gold、排名、是否命中见 `rgb-retrieval-eval-report.json` 的 `per_query` 字段；
- 也见 `rgb-retrieval-eval-cases.csv`（每行一个 retriever × query）。

