# 技术文档智能问答与引用溯源系统

面向公开技术文档集的 Grounded RAG 系统。回答仅基于检索证据，提供可验证的
`文档路径 + 段落 anchor` 引用；证据不足时由程序直接拒答。

## 核心能力

- Markdown 文档加载、父子分块和本地 Chroma 向量索引
- 结构化引用：`path`、`anchor`、证据摘录和相关度分数
- 相关度阈值拒答，不依赖 LLM 自觉判断
- 新增、修改、删除实时同步，服务重启后清理已删除文档
- FastAPI + 结构化 JSON 问答 Web UI
- OpenAI-compatible LLM 网关，开发阶段使用 DeepSeek API 占位
- LLM 实体/关系抽取、跨块合并、Leiden 社区与可追溯 GraphRAG 检索
- Docker Compose 一键运行和 GitHub Actions 自动测试

## 一键运行

### 1. 准备文档和配置

将比赛文档集放入 `documents/`，仅使用允许参赛的公开或指定数据。

```bash
cp .env.example .env
```

`.env` 只保存密钥：

```ini
LLM_API_KEY=your-key
```

其他参数统一在 `config.yaml` 中维护。开发阶段默认使用 DeepSeek；比赛网关下发
后，修改 `llm.base_url` 和 `llm.model`，业务代码无需修改。禁止在最终比赛环境中
绕过统一网关直接访问外部模型。

### 2. 启动

```bash
docker compose up --build
```

首次启动需要下载嵌入模型。服务就绪后访问：

- Web UI：<http://localhost:8501>
- 健康检查：<http://localhost:8501/health>
- 索引状态：<http://localhost:8501/api/status>

也可以使用 `make run`。

## 本地开发

需要 Python 3.10 或更高版本：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
python -m rag_cli.main index --rebuild
python -m rag_cli.main server
```

Linux/macOS 使用 `source .venv/bin/activate` 激活环境。

常用命令：

```bash
python -m rag_cli.main ask "文档中的部署步骤是什么？"
python -m rag_cli.main index --sync
python -m rag_cli.main index --status
python -m rag_cli.main graph-build --changed-only
python -m rag_cli.main graph-communities --reports
python -m pytest tests -q
```

由于比赛版新增了 anchor 元数据，从旧面试项目升级后必须执行一次
`index --rebuild`。

### 语义 GraphRAG

语义建图是显式的离线操作，不会随服务启动自动调用 LLM。先在 `config.yaml`
中设置：

```yaml
graph:
  entity_extraction: true
```

然后运行：

```powershell
uv run rag graph-build --changed-only
```

该命令复用现有 child chunk 作为 TextUnit，抽取实体与关系、校验证据原文、
缓存模型结果、合并跨块描述并更新实体向量。未修改的 TextUnit 不会重复调用模型。

若要启用 Leiden 社区和社区报告，可设置：

```yaml
graph:
  community_detection: true
  community_reports: true
```

也可对已有语义图单独运行：

```powershell
uv run rag graph-communities --reports
```

问答支持 `--mode basic|local|global|auto`。`local` 使用实体关系路径扩展，
`global` 使用社区报告发现相关文档，但两者最终都只把可定位的原始 child chunk
作为回答证据。

## 回答与引用协议

`POST /api/chat`：

```json
{
  "question": "系统支持哪些部署方式？",
  "session_id": "demo"
}
```

成功回答返回（只包含答案实际使用的证据）：

```json
{
  "status": "answered",
  "answer": [
    {"text": "使用 Docker Compose 启动服务。", "citation_ids": ["cite_1"]}
  ],
  "citations": [
    {
      "id": "cite_1",
      "document_path": "guide.md",
      "anchor": "部署",
      "section_title": "部署",
      "quote": "从检索片段直接截取的原文",
      "score": 0.82
    }
  ]
}
```

引用可通过下面的 API 复核：

```text
GET /api/sources/guide.md?anchor=部署
```

若没有证据达到门槛、模型判断证据不能直接回答，或引用校验失败，返回：

```json
{
  "status": "insufficient_evidence",
  "answer": [],
  "citations": [],
  "message": "根据当前文档集，未找到能够可靠回答该问题的依据，因此无法给出答案。",
  "reason": "没有检索到能够直接支持答案的文档片段"
}
```

## 配置

密钥只通过 `.env` 或部署环境的 `LLM_API_KEY` 注入。普通参数位于
`config.yaml`：

| YAML 配置 | 默认值 | 说明 |
|---|---|---|
| `llm.base_url` | DeepSeek URL | OpenAI-compatible 网关地址 |
| `llm.model` | `deepseek-chat` | 网关模型名 |
| `documents.path` | `./documents` | Markdown 文档目录 |
| `storage.chroma_dir` | `./chroma_data` | 向量索引目录 |
| `embedding.model` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `embedding.device` | `cpu` | 嵌入计算设备 |
| `retrieval.top_k` | `10` | 候选证据数 |
| `retrieval.score_threshold` | `0.35` | 确定性拒答阈值 |
| `retrieval.max_citations` | `5` | 最多传给模型并允许返回的证据数 |
| `retrieval.max_retry` | `1` | 结构或引用校验失败后的重试次数 |
| `retrieval.require_citations` | `true` | 强制每个回答结论包含引用 |

上述证据约束也可分别通过 `RAG_MIN_RETRIEVAL_SCORE`、
`RAG_MAX_CITATIONS`、`RAG_MAX_RETRY`、`RAG_REQUIRE_CITATIONS` 环境变量覆盖。

`embedding.device` 常用选项：

- `cpu`：通用选项，无需 GPU，评委环境默认使用此项；
- `cuda`：使用默认 NVIDIA GPU，需要正确安装 CUDA 版 PyTorch；
- `cuda:0`、`cuda:1`：指定某一块 NVIDIA GPU；
- `mps`：Apple Silicon macOS 使用 Metal 加速。

阈值必须使用比赛公开评测集校准，不应只凭人工体验设置。

## 文档

- [架构与数据流](docs/architecture.md)
- [语义 GraphRAG 实施方案](docs/semantic-graphrag-implementation.md)
- [自测报告](docs/self-test-report.md)
- [2～3 分钟 Demo 录制脚本](docs/demo-script.md)
- [原始设计记录](docs/superpowers/specs/2026-06-09-rag-knowledge-assistant-design.md)

## 数据合规

仓库不提交 `.env`、向量索引、聊天记录或比赛文档集。Prompt、日志和 Demo
不得包含真实员工、客户、合作伙伴或内部业务数据。

## License

[MIT](LICENSE)
