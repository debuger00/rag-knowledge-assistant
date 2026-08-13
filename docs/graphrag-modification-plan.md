# GraphRAG 改造方案

## 1. 改造目标

在保留现有 Grounded RAG、Chroma 父子分块和 `path + anchor` 引用校验的前提下，增加一个可增量维护的知识图谱层，使系统同时支持：

- Basic Search：原有的纯向量检索；
- Local Graph Search：从向量命中的笔记或章节出发，沿 Obsidian 链接、标签和章节关系扩展证据；
- Global Search：后续可基于实体社区和社区报告回答全库主题类问题；
- 所有回答最终仍必须引用可定位的原始 Markdown 子块，图关系和摘要只负责发现证据。

当前知识库约 840 篇 Markdown、9.5 MB 文本，包含约 1,626 次 `[[...]]`。这一规模适合使用 Chroma 保存向量、SQLite 保存图结构、NetworkX 执行内存图算法，不需要先引入独立图数据库。

## 实施状态（2026-08-11）

已完成：

- Phase 1 结构图 MVP：稳定 ID、代码感知 wikilink 解析、alias/anchor/附件处理、SQLite GraphStore、全量和 Watchdog 增量同步；
- Phase 2 Hybrid Local Search：向量种子、双向图扩展、RRF 排序、语义相关度融合、folder/tag 约束和原始证据回查；
- `basic/local/auto` API、CLI 和 Web 模式；
- 图状态、邻居、子图和分范围重建接口；
- `rag index --graph-only`；
- 解析器、Resolver、GraphStore、Hybrid Retriever、配置和 API 测试。

当前真实库图索引结果为 18,605 个节点、18,053 条边、7 个 unresolved 目标；批量事务优化后的纯结构图构建约 8.55 秒。运行时数据库保存在 `graph_data/graph.sqlite3`，不提交 Git。

尚未默认启用：LLM 实体/关系抽取、社区报告和 Global Search。这些属于 Phase 3 的高成本能力，配置字段与扩展边界已经保留，需在专门的评测集和成本预算确定后实施，避免未经校准就向 17,000 多个章节发起模型抽取。

## 2. 现有系统与问题

当前链路为：

```text
Markdown -> ObsidianLoader -> 父子分块 -> Chroma
         -> 子块向量检索 -> 阈值过滤 -> LLM -> 引用校验
```

已有优势：

- Loader 已提取标签和 wikilink；
- Splitter 已生成稳定可读 anchor；
- Chroma 区分父文档和检索子块；
- Watchdog 支持新增、修改、删除同步；
- Pipeline 已强制执行原文引用闭环和证据不足拒答。

主要缺口：

1. `_expand_by_links()` 未进入实际检索主路径；
2. `links` 写入 Chroma 后由列表变成字符串，与扩展代码的数据假设不一致；
3. 链接解析没有实现 Obsidian 的路径、basename、alias、anchor 和附件规则；
4. 代码块中的 `[[nodiscard]]`、Shell 表达式和图片嵌入可能被误识别；
5. 子块使用随机 UUID，无法稳定关联图节点和增量状态；
6. 向量更新与未来的图更新缺少统一协调；
7. 当前检索只有文本相似度，没有跨文档关系召回、社区和全库推理。

## 3. 目标架构

```text
                       +-- Chroma: parent/child vectors
Markdown -> IndexCoordinator
                       +-- SQLite: nodes/edges/aliases/state
                                      |
Question -> basic/local/auto router -> HybridRetriever
             |                        |
             |                 vector seeds + graph expansion
             |                        |
             +-----------------> grounded child chunks
                                      |
                              RAGPipeline + citation validation
```

第一阶段只构建零 LLM 成本的 Obsidian 结构图；实体抽取、社区检测和社区报告通过稳定接口逐步加入，并默认关闭高成本能力。

## 4. 图数据模型

### 4.1 节点

| 类型 | 稳定 ID | 说明 |
|---|---|---|
| `document` | `doc:{sha256(source)}` | 一篇笔记 |
| `section` | `section:{sha256(source, anchor)}` | 可引用章节 |
| `tag` | `tag:{normalized_tag}` | frontmatter 或正文标签 |
| `entity` | `entity:{type}:{canonical_name}` | 后续 LLM/NLP 抽取实体 |
| `community` | `community:{level}:{hash}` | 图社区和社区摘要 |
| `unresolved` | `unresolved:{hash(target)}` | 无法或暂未解析的链接目标 |

### 4.2 边

- `HAS_SECTION`：document -> section
- `LINKS_TO`：document/section -> document/unresolved
- `TAGGED_WITH`：document -> tag
- `MENTIONS`：section -> entity
- `RELATED_TO`：entity -> entity
- `IN_COMMUNITY`：document/entity -> community
- `PARENT_COMMUNITY`：子社区 -> 父社区

每条关系保存 `source`、`anchor`、`chunk_id`、`extractor`、`confidence` 和 `content_hash`，保证能回溯到原文。

## 5. 模块设计

```text
rag_core/
├── graph/
│   ├── models.py          # GraphNode、GraphEdge、GraphHit
│   ├── parser.py          # Obsidian 链接/标签语法解析
│   ├── resolver.py        # wikilink 目标解析
│   ├── store.py           # SQLite GraphStore
│   ├── builder.py         # 文档结构图构建
│   ├── communities.py     # 社区检测和失效边界
│   └── retrieval.py       # 图扩展和路径解释
├── indexing/
│   └── coordinator.py     # Chroma 与 GraphStore 统一更新
└── retrieval/
    ├── hybrid.py          # 多路召回、RRF 和去重
    └── models.py          # RetrievalBundle/Trace
```

## 6. 索引流程

### 6.1 结构图

1. Loader 解析文档正文、frontmatter、标题、标签和 wikilink；
2. 链接扫描跳过 fenced code 和 inline code；
3. 区分普通链接、嵌入附件、alias 和 heading anchor；
4. Resolver 按完整路径、当前目录相对路径、vault 内唯一 basename 依次解析；
5. 重名链接标记为 ambiguous，断链保留为 unresolved；
6. Splitter 生成稳定 `document_id`、`section_id` 和 `chunk_id`；
7. IndexCoordinator 同时更新 Chroma 和 GraphStore；
8. `doc_state` 记录 content hash、向量索引版本和图索引版本。

### 6.2 增量更新

修改文档时，先构建新文档的完整索引数据，再替换该 source 的 Chroma 和图数据。删除文档时删除其父/子块、拥有的章节节点和出边，并清理无引用的孤立节点。受影响社区标记为 dirty，由批处理或手动重建统一更新。

## 7. 检索流程

### 7.1 Basic

保留原有子块向量检索，作为回退、兼容和 A/B 基线。

### 7.2 Local

1. 向量检索 top-N 子块；
2. 将命中的 section/document 作为图种子；
3. 在限定的 hop 和邻居预算内沿图扩展；
4. 将扩展节点映射回原始子块；
5. 使用 Reciprocal Rank Fusion 融合向量排名和图排名；
6. 每个 source 设置证据上限并去重；
7. folder/tag 过滤同时约束种子和扩展结果；
8. 最终证据继续经过现有引用校验。

初始边权：

```yaml
LINKS_TO: 1.0
MENTIONS: 0.8
RELATED_TO: 0.7
TAGGED_WITH: 0.35
HAS_SECTION: 0.2
```

### 7.3 Global（后续阶段）

社区报告只用于主题发现和 map-reduce。任何最终结论必须回查社区成员对应的原始 chunk；无法回查的结论删除或拒答。社区报告不能直接绕过引用验证成为事实证据。

## 8. 配置和接口

新增配置：

```yaml
graph:
  enabled: true
  db_path: ./graph_data/graph.sqlite3
  max_hops: 2
  max_seed_nodes: 10
  max_neighbors: 30
  graph_weight: 0.25
  entity_extraction: false
  community_detection: false
  community_reports: false
```

`POST /api/chat` 增加可选字段 `mode: basic | local | auto` 和 `debug_retrieval`。响应保持现有 `answer/citations` 兼容，可增加 `mode` 与不包含正文的 `retrieval_trace`。

新增：

- `GET /api/graph/status`
- `GET /api/graph/neighbors/{node_id}`
- `GET /api/graph/subgraph?source=...&depth=2`
- `POST /api/reindex?scope=all|vector|graph`

## 9. 实施阶段

### Phase 1：结构图 MVP

- 稳定 ID；
- Obsidian 链接解析和解析器；
- SQLite GraphStore；
- Chroma/图统一增删改；
- 图状态 CLI/API。

### Phase 2：Hybrid Local Search

- 向量种子与图扩展；
- RRF、多源去重和证据预算；
- `basic/local/auto` 模式；
- 保持 AnswerResponse 与引用协议兼容。

### Phase 3：实体和社区

- 带缓存、版本和 JSON 校验的实体/关系抽取；
- 社区检测、社区失效和摘要；
- Global Search 与原始证据回查。

### Phase 4：UI、评测和运维

- 查询模式与图路径展示；
- 图重建进度、统计和诊断；
- Basic/Local/Global A/B 评测；
- 索引成本、查询延迟和引用质量监控。

## 10. 测试与验收

新增解析器、Resolver、GraphStore、GraphIndexer、HybridRetriever、API 和 grounding 测试。必须覆盖代码块误识别、alias/anchor、附件、重名文件、断链、重复索引、修改删除、环路、hop 限制、过滤范围以及图关系缺少原文证据时拒答。

评测集至少包含单文档事实、跨文档关系、全库主题和应拒答问题。核心指标为 Recall@K、引用有效率、拒答准确率、跨文档答案正确率、P95 延迟和索引成本。任何 GraphRAG 模式都不得降低引用可验证性。
