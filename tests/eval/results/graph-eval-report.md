# GraphRAG 可控评测报告

## 结论

- 图结构 Gold 校验：通过
- Basic Source Recall@5：9.09%
- Local Source Recall@5：100.00%
- 图增益：+90.91%
- Graph Win：10 条
- Graph Harm：0 条
- Local MRR：0.9091
- 安全负例通过率：100.00%
- 禁止来源泄漏：0 条

## 图结构

- 节点：90
- 边：77
- resolved link precision：100.00%
- resolved link recall：100.00%
- 构建耗时：20.331 ms

## 评测口径

- 使用固定向量种子，隔离 embedding 模型波动，测量知识图带来的纯召回增量。
- Local 候选仍需从目标 source 的原始 section 中重新选择，不把图节点直接当证据。
- 本报告不调用 LLM，因此不代表最终答案正确率；端到端答案评测需另行使用人工答案要点和引用标注。
- 评测实施时发现并修复：HAS_SECTION 错误消耗语义 hop、Windows folder 使用反斜杠导致过滤漏召回。

## 查询明细

| ID | 类别 | Basic@5 | Local@5 | Local Rank | 泄漏 |
|---|---|---:|---:|---:|---|
| direct_1hop | direct_link | N | Y | 1 | - |
| backlink_1hop | backlink | N | Y | 1 | - |
| shared_tag_2hop | shared_tag | N | Y | 1 | - |
| two_wikilinks | two_hop | N | Y | 2 | - |
| cycle_bounded | cycle | N | Y | 1 | - |
| ambiguous_safe | ambiguous | N | N | - | - |
| unresolved_safe | unresolved | N | N | - | - |
| attachment_safe | attachment | N | N | - | - |
| code_false_positive | code_block | N | Y | 1 | - |
| folder_filter | folder_filter | N | Y | 1 | - |
| tag_filter | tag_filter | N | Y | 1 | - |
| alias_resolution | alias | N | Y | 1 | - |
| relative_resolution | relative_path | N | Y | 1 | - |
| basic_safe | basic_safe | Y | Y | 2 | - |
