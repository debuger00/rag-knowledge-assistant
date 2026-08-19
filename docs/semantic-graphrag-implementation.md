# 语义 GraphRAG 建图实施方案

## 实施状态（2026-08-13）

已实现：

- GraphStore v2、语义证据表、抽取缓存与来源状态；
- 严格 JSON 实体/关系抽取、类型/置信度/原文 quote 校验；
- 跨 TextUnit 实体与关系合并、描述汇总及汇总缓存；
- `MENTIONS`、`RELATED_TO` 和实体描述向量；
- 显式 `graph-build`、changed-only 和单 source 索引；
- hierarchical Leiden、社区报告及报告缓存；
- Local 实体种子和基于社区报告的 grounded Global 检索；
- 删除来源、结构图重建、重复索引和引用回查兼容测试。

当前 Global 模式采用“社区报告发现范围 -> 原始 chunk 回查”的 grounded 检索，
没有直接采用官方无引用的 report map-reduce 结果作为答案。若后续增加完整 map-reduce，
仍必须为每个最终结论回查原始 TextUnit。

## 1. 目标与边界

在现有 Obsidian 结构图、Chroma 父子分块和引用校验能力之上，增加与 Standard GraphRAG 等价的语义建图链路：

```text
Document -> child chunk/TextUnit -> LLM Entity+Relationship
         -> 跨 TextUnit 合并 -> 描述汇总 -> 语义知识图谱
         -> Leiden 社区 -> Community Report
```

必须满足以下约束：

- 复用现有稳定 `chunk_id/source/anchor`，不重复切块；
- 图只负责发现证据，最终回答仍引用可定位的原始 child chunk；
- LLM 抽取不进入服务器启动时的同步全量建图；
- 支持按内容哈希增量抽取，未修改的 TextUnit 不重复调用模型；
- 删除文档时只删除该文档贡献的证据，不误删共享实体；
- 高成本的社区报告默认关闭。

## 2. 目标架构

```text
Markdown -> parent_child_split -> Chroma parent/child
                         |
                         +-> SemanticGraphIndexer
                               |
                               +-> extraction cache
                               +-> raw entity/relationship evidence
                               +-> deterministic merge
                               +-> optional LLM description summary
                               +-> GraphStore v2

Question -> child vector seeds + entity vector seeds
         -> section -MENTIONS-> entity -RELATED_TO-> entity
         -> related sections/chunks
         -> RRF/semantic rerank
         -> original chunks
         -> answer + citation validation
```

保留现有结构边：`HAS_SECTION`、`LINKS_TO`、`TAGGED_WITH`。新增：

- `MENTIONS`：section -> entity；
- `RELATED_TO`：entity -> entity，关系语义保存在 `description/predicate`；
- 后续 `IN_COMMUNITY`：entity -> community。

## 3. LLM 抽取协议

每个 child chunk 作为一个 TextUnit。模型必须输出严格 JSON：

```json
{
  "entities": [
    {
      "name": "ABC科技公司",
      "type": "organization",
      "description": "一家总部位于上海的科技公司",
      "aliases": ["ABC公司"],
      "evidence_quote": "ABC 科技公司总部位于上海"
    }
  ],
  "relationships": [
    {
      "source": "ABC科技公司",
      "target": "上海",
      "description": "ABC科技公司的总部位于上海",
      "strength": 8.0,
      "predicate": "LOCATED_IN",
      "evidence_quote": "ABC 科技公司总部位于上海"
    }
  ]
}
```

校验规则：

1. 实体类型必须在配置的 `entity_types` 中；
2. 关系端点必须能解析到同次抽取中的实体；
3. `strength` 限制在 0 到 10；
4. `evidence_quote` 必须是 TextUnit 原文子串；
5. 非法 JSON、截断输出或不满足 Schema 时按配置重试；
6. 描述汇总不能创建新实体、新关系或新事实。

## 4. 实体规范化与合并

默认进行高精度确定性合并：

```text
entity key = (normalized entity_type, normalized canonical_name)
entity id  = entity:{sha256(entity_type, canonical_name)}
edge id    = semantic-edge:{sha256(source_entity_id, target_entity_id)}
```

规范化只处理 Unicode、空白、英文大小写和首尾标点，不擅自删除公司后缀或合并同名人物。别名消歧作为后续可选能力，通过人工映射或高置信度 resolver 完成。

同一实体或关系的多份原始描述首先去重；只有存在多份不同描述时才调用汇总模型。关系权重为所有关系实例 `strength` 的和。

## 5. GraphStore v2

现有 `nodes/edges` 继续保存最终可遍历图，增加以下数据：

```text
nodes: description
edges: description, predicate

node_evidence:
  node_id, chunk_id, source, anchor, raw_description,
  evidence_quote, confidence, extraction_id

edge_evidence:
  edge_id, chunk_id, source, anchor, raw_description,
  evidence_quote, strength, confidence, extraction_id

extraction_cache:
  chunk_id, content_hash, model_id, prompt_hash,
  extractor_version, response_json, status, error
```

证据表使用 `(node_id|edge_id, chunk_id)` 唯一约束。重新抽取单篇文档时，先替换该 source 的语义证据，再重算受影响节点和关系；没有任何证据的语义节点和边才会被清理。

## 6. 索引与命令

结构图仍由当前同步路径维护。语义图使用显式离线命令：

```powershell
uv run rag graph extract --changed-only
uv run rag graph extract --source path/to/note.md
uv run rag graph summarize
uv run rag graph status
```

第一阶段可以合并为一个用户入口：

```powershell
uv run rag graph-build --changed-only
```

索引器步骤：

1. 加载 Markdown 并生成 child chunks；
2. 根据 `content_hash + model_id + prompt_hash + extractor_version` 查缓存；
3. 对缺失或失效 TextUnit 调用 LLM；
4. 校验并保存原始抽取结果；
5. 替换该 source 的语义证据；
6. 合并实体和关系并生成描述；
7. 更新实体描述向量；
8. 将受影响社区标记为 dirty。

## 7. 检索

Local Search 使用两类种子：

- 原有 child chunk 向量结果；
- 问题与实体 `name + description` 的向量结果。

典型扩展路径：

```text
section -> MENTIONS -> entity -> RELATED_TO -> entity -> MENTIONS -> section
```

遍历可以双向进行，但路径中保留原始关系方向。最终候选必须映射回原始 child chunk，经过现有分数阈值、数量预算和引用校验。社区报告与实体描述不能直接作为最终事实证据。

## 8. 配置

```yaml
graph:
  enabled: true
  entity_extraction: false
  community_detection: false
  community_reports: false
  extraction:
    prompt: prompts/extract_graph.txt
    summary_prompt: prompts/summarize_descriptions.txt
    entity_types:
      - concept
      - language
      - library
      - class
      - function
      - api
      - tool
      - error
    max_gleanings: 0
    concurrency: 4
    max_retries: 2
    min_confidence: 0.65
```

默认关闭实体抽取，避免用户启动服务时意外产生大额 LLM 调用。试点阶段先使用 `max_gleanings: 0`，评测实体召回后再决定是否增加补抽轮次。

## 9. 实施阶段

### Phase A：语义图 MVP

- GraphStore v2 迁移；
- JSON 抽取模型、Prompt、校验和缓存；
- TextUnit 级抽取；
- `MENTIONS/RELATED_TO` 写入与跨块合并；
- 手动 CLI 和 changed-only 模式。

### Phase B：Local Search

- 实体描述向量；
- 实体/关系路径扩展；
- 图路径解释和证据回查；
- Basic/Local A/B 评测。

### Phase C：社区和 Global Search

- Leiden 层级社区；
- Community Reports；
- Global map-reduce；
- 所有结论回查原始 TextUnit。

## 10. 验收标准

- 示例文本生成 4 个实体和 3 条关系；
- 跨 TextUnit 的同名同类型实体合并为一个节点；
- 每个实体和关系均可追溯到 `source + anchor + chunk_id + evidence_quote`；
- 相同内容第二次索引不调用 LLM；
- 修改一个 TextUnit 只重抽该 TextUnit；
- 删除一篇文档不会删除仍有其他证据支持的共享实体；
- 图检索输出最终均为原始 child chunk；
- 实体抽取关闭时，现有 Basic/Local 行为保持兼容；
- 单元测试不依赖真实 LLM，使用确定性 fake extractor。
