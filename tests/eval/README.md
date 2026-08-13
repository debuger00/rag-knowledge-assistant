# GraphRAG 可控评测集

该评测集包含 27 篇人工 Markdown，覆盖直接 wikilink、backlink、共享标签、两跳关系、环路、同名文件、断链、附件、代码伪链接、folder/tag 过滤、alias 和相对路径。

## 评测层次

1. `expected_graph.json`：固定节点、边和解析结果；
2. `graph_retrieval.jsonl`：固定向量种子，隔离 embedding 波动，比较 Basic 与 Local；
3. `run_graph_eval.py`：生成结构与检索指标报告。

运行：

```powershell
python tests/eval/run_graph_eval.py
```

结果默认写入：

```text
tests/eval/results/graph-eval-report.json
tests/eval/results/graph-eval-report.md
```

主基准使用受控种子评估“图带来的纯增量”，不会调用 LLM。它不能代替真实 embedding 和端到端答案评测，但能够稳定发现建图、遍历、过滤和融合回归。

## 图结构可视化

生成无需联网的交互式 HTML，以及 Gephi/Cytoscape 可读取的 GraphML 和 CSV：

```powershell
python tests/eval/visualize_graph.py
```

输出目录：

```text
tests/eval/results/visualization/
├── graph.html
├── graph.graphml
├── nodes.csv
└── edges.csv
```

浏览器打开 `graph.html` 即可缩放、平移、按节点/边类型过滤并点击查看属性。页面默认隐藏 section 节点和 `HAS_SECTION` 边，并在显示层把 section 发出的链接折叠到所属文档；需要时可在左侧勾选 section 查看原始结构。GraphML/CSV 始终保存未经折叠的完整图。

## CRUD-RAG 中文真实语料评测

下载官方评测切分：

```powershell
python tests/eval/download_crud_rag.py
```

需要完整的 8 万篇背景新闻时运行：

```powershell
python tests/eval/download_crud_rag.py --include-corpus
```

使用真实中文 embedding 运行平衡子集评测。默认单/双/三文档问答各 20 条，并从官方 `80000_docs` 中确定性抽取 1,000 篇去重背景新闻作为 distractor：

```powershell
python -u tests/eval/run_crud_rag_eval.py
```

可以调整规模：

```powershell
python -u tests/eval/run_crud_rag_eval.py --per-task 50 --distractors 5000 --top-k 10
```

生成的第三方数据、Markdown 和独立 Chroma 索引位于 `data/`，不会污染正式知识库；报告写入：

```text
tests/eval/results/crud-rag-eval-report.md
tests/eval/results/crud-rag-eval-report.json
```

CRUD-RAG 新闻不包含天然 wikilink。转换器不会把答案、事件摘要或 Gold case 标签注入 Markdown，因此当前评测主要测量中文向量检索；只有在后续增加实体/事件边抽取后，才应期待 Local GraphRAG 相比 Basic 产生真实增益。

如果没有下载 `80000_docs`，脚本会退回到 `crud_split` 中未入选的问答证据作为 distractor，并在报告的 `Distractor 来源` 中明确标注。

## Microsoft GraphRAG 官方基准选择

下载微软公开的全部基准文件：

```powershell
python tests/eval/download_microsoft_graphrag.py
```

可选择四个 profile：

```powershell
python -u tests/eval/run_microsoft_graphrag_eval.py --dataset hotpotqa
python -u tests/eval/run_microsoft_graphrag_eval.py --dataset kevin_scott
python -u tests/eval/run_microsoft_graphrag_eval.py --dataset msft_single
python -u tests/eval/run_microsoft_graphrag_eval.py --dataset msft_multi
```

首次建议先准备或运行小样本：

```powershell
python tests/eval/run_microsoft_graphrag_eval.py `
  --dataset hotpotqa --max-documents 500 --max-questions 100 --prepare-only
```

Profile 定义在 `tests/eval/datasets/microsoft_graphrag_profiles.json`。其中：

- `hotpotqa`：按保留的原始 `test_N` 编号排序后与问题一一对应，可计算 Paired-context Recall@K；它不是 Gold supporting-fact 指标；
- `kevin_scott`：跨播客主题总结，推荐 Global；
- `msft_single`：单财报问题，推荐 Local；
- `msft_multi`：跨 41 份财报的聚合问题，推荐 Global。

微软的播客/财报问题 CSV 没有 Gold 答案和 supporting source，因此当前只生成检索覆盖诊断，不把它误报为准确率。`kevin_scott` 和 `msft_multi` 会明确显示 `global_pipeline_status=required_not_implemented`，待社区发现、社区摘要和 Map-Reduce 实现后再作为 Global GraphRAG 验收集。
