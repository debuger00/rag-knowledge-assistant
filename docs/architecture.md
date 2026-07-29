# 架构与数据流

## 组件关系

```text
比赛 Markdown 文档集
        |
        v
ObsidianLoader -> Anchor-aware Splitter -> BGE Embedding -> Chroma
                                                        |
Web UI -> FastAPI/SSE -> Score-threshold Retriever -----+
                            |
                            +-- 无可靠证据 -> 固定拒答
                            |
                            +-- 有证据 -> OpenAI-compatible LLM Gateway
                                            |
                                            v
                                 回答 + 结构化可验证引用
```

## 索引数据流

1. Loader 读取 UTF-8 Markdown，提取相对路径、标题、标签和链接。
2. Splitter 按二级标题建立分块，为每块生成稳定 anchor。
3. 父文档用于原文查看，子块用于语义检索。
4. Watchdog 在文件新增、修改、删除后约 2 秒更新索引。
5. 服务启动时执行全量差异检查，清理停机期间删除的文档。

## 问答数据流

1. Retriever 返回带相关度分数的候选子块。
2. 低于 `RETRIEVAL_SCORE_THRESHOLD` 的候选被丢弃。
3. 没有可靠证据时直接拒答，不调用 LLM。
4. 有可靠证据时，仅把证据块和有限对话历史发送到统一 LLM 网关。
5. 后端从检索元数据生成引用，不接受 LLM 自报来源。
6. SSE 分别发送引用和答案；引用 API 可按 `path + anchor` 读取原证据。

## 外部依赖

| 依赖 | 用途 | 数据边界 |
|---|---|---|
| 比赛统一 LLM 网关 | 答案生成 | 仅发送检索证据、问题和有限历史 |
| Hugging Face 模型缓存 | 首次获取公开嵌入模型 | 不发送比赛文档 |
| Chroma | 本地向量存储 | 数据保留在部署环境 |

## 安全与合规

- 最终环境的所有生成模型调用必须经过比赛统一网关。
- `.env`、文档集、Chroma 数据和聊天记录不进入 Git。
- 应在网关和应用日志层关闭正文日志或完成脱敏。
- Demo 只使用比赛允许的公开、指定或合成数据。

## 已知约束

- 当前输入格式为 UTF-8 Markdown。
- 阈值需要使用正式评测集校准。
- 多实例部署需要把内存会话和文件监听协调迁移到共享服务。
