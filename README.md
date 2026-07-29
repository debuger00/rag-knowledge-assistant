# 技术文档智能问答与引用溯源系统

面向公开技术文档集的 Grounded RAG 系统。回答仅基于检索证据，提供可验证的
`文档路径 + 段落 anchor` 引用；证据不足时由程序直接拒答。

## 核心能力

- Markdown 文档加载、父子分块和本地 Chroma 向量索引
- 结构化引用：`path`、`anchor`、证据摘录和相关度分数
- 相关度阈值拒答，不依赖 LLM 自觉判断
- 新增、修改、删除实时同步，服务重启后清理已删除文档
- FastAPI + SSE 流式问答 Web UI
- OpenAI-compatible LLM 网关，开发阶段使用 DeepSeek API 占位
- Docker Compose 一键运行和 GitHub Actions 自动测试

## 一键运行

### 1. 准备文档和配置

将比赛文档集放入 `documents/`，仅使用允许参赛的公开或指定数据。

```bash
cp .env.example .env
```

开发阶段配置 DeepSeek：

```ini
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

比赛网关下发后，只替换以上三个值，业务代码无需修改。禁止在最终比赛环境中绕过
统一网关直接访问外部模型。

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
python -m pytest tests -q
```

由于比赛版新增了 anchor 元数据，从旧面试项目升级后必须执行一次
`index --rebuild`。

## 回答与引用协议

`POST /api/chat`：

```json
{
  "question": "系统支持哪些部署方式？",
  "session_id": "demo"
}
```

SSE 依次返回：

```text
{"event":"thinking","data":"正在检索笔记..."}
{"event":"sources","data":[{"path":"guide.md","anchor":"部署","quote":"...","score":0.82}]}
{"event":"token","data":"..."}
{"event":"done","data":"..."}
```

引用可通过下面的 API 复核：

```text
GET /api/sources/guide.md?anchor=部署
```

若没有证据达到 `RETRIEVAL_SCORE_THRESHOLD`，系统不调用 LLM，固定返回：

```text
根据当前文档集，无法找到足够可靠的依据回答该问题。
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 无 | 统一网关密钥 |
| `LLM_BASE_URL` | DeepSeek URL | OpenAI-compatible 网关地址 |
| `LLM_MODEL` | `deepseek-chat` | 网关模型名 |
| `OBSIDIAN_VAULT_PATH` | `./documents` | Markdown 文档目录 |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | 向量索引目录 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` 或 `cuda` |
| `RETRIEVAL_TOP_K` | `10` | 候选证据数 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | 确定性拒答阈值 |

阈值必须使用比赛公开评测集校准，不应只凭人工体验设置。

## 文档

- [架构与数据流](docs/architecture.md)
- [自测报告](docs/self-test-report.md)
- [2～3 分钟 Demo 录制脚本](docs/demo-script.md)
- [原始设计记录](docs/superpowers/specs/2026-06-09-rag-knowledge-assistant-design.md)

## 数据合规

仓库不提交 `.env`、向量索引、聊天记录或比赛文档集。Prompt、日志和 Demo
不得包含真实员工、客户、合作伙伴或内部业务数据。

## License

[MIT](LICENSE)
