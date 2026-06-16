# RAG 知识库助手

基于 Obsidian 笔记仓库的个人 RAG（Retrieval-Augmented Generation）问答助手。提供 CLI 命令行和 Web 界面两种交互方式。

**技术栈**: Python · PyTorch (GPU/CUDA) · LangChain · ChromaDB · BGE 中文嵌入模型 · DeepSeek API · FastAPI · Alpine.js

## 目录结构

```
rag-assistant/
├── config.py                  # 全局配置（从 .env 读取）
├── pyproject.toml             # 项目依赖与构建配置
├── .env.example               # 环境变量模板
│
├── rag_core/                  # 核心 RAG 引擎
│   ├── indexing/              # 索引子系统
│   │   ├── loader.py          # Obsidian .md 文件加载器
│   │   ├── splitter.py        # 父子分块（按 ## 标题切分）
│   │   ├── embedder.py        # BGE 嵌入模型封装
│   │   └── store.py           # ChromaDB 双集合管理（父文档 + 子块）
│   ├── retrieval/             # 检索子系统
│   │   ├── retriever.py       # 父子检索器（语义搜索子块 → 返回父文档）
│   │   └── pipeline.py        # RAG 问答管线（检索 → 格式化 → LLM → 输出）
│   ├── llm/                   # LLM 子系统
│   │   └── deepseek.py        # DeepSeek API 封装
│   └── watcher.py             # 文件监听器（watchdog，自动同步 Obsidian 变更）
│
├── rag_server/                # Web 服务端
│   ├── app.py                 # FastAPI 应用（/api/chat, /api/status, /api/reindex）
│   ├── chat.py                # SSE 流式聊天 + 会话管理
│   └── static/                # 前端静态文件
│       ├── index.html         # Alpine.js 聊天界面
│       └── style.css          # 样式（含暗色模式）
│
├── rag_cli/                   # CLI 命令行
│   └── main.py                # Typer 入口（rag ask / index / server）
│
├── tests/                     # 测试
│   ├── conftest.py            # Pytest fixtures（临时 Obsidian 仓库）
│   ├── test_loader.py         # 加载器测试（7 个）
│   ├── test_splitter.py       # 分块器测试（6 个）
│   ├── test_store.py          # 向量存储测试（6 个）
│   └── test_pipeline.py       # 管线测试（5 个）
│
└── docs/superpowers/          # 设计文档
    ├── specs/                 # 设计规格说明
    └── plans/                 # 实施计划
```

## 快速开始

### 1. 环境准备

需要 Python >= 3.10，推荐使用 conda 环境以获得 GPU 加速推理。

```bash
# 克隆仓库后，激活 conda 环境（需提前安装 PyTorch CUDA 版本）
conda activate pytorch251

# 安装依赖
pip install -e .
```

> **GPU 推理**：本项目使用 PyTorch CUDA 版本进行嵌入模型推理。如无 GPU，可降级为 CPU 版 PyTorch（推理速度较慢）。


### 2. 配置

复制并编辑 `.env` 文件：

```bash
cp .env.example .env
```

必填项：

```ini
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
OBSIDIAN_VAULT_PATH=E:/你的Obsidian仓库路径
```

可选配置（有默认值，按需修改）：

```ini
# 嵌入模型（默认 bge-small-zh-v1.5，100MB 轻量中文模型）
# 也可换用 BAAI/bge-m3（2GB，更强但更慢）
DEEPSEEK_MODEL=deepseek-chat

# 检索参数
RETRIEVAL_TOP_K=10          # 检索返回的文档数

# 分块参数
CHILD_CHUNK_SIZE=800        # 子块最大字符数
CHILD_CHUNK_OVERLAP=100     # 块间重叠字符数
CHILD_MAX_LEN=2000          # 超过此长度的段落会二次切分
```

### 3. 建立索引

首次使用需要将 Obsidian 笔记转为向量索引：

```bash
rag index --rebuild
```

进度示例：

```
Rebuilding index...
  [1/3] 加载笔记...
  [1/3] 加载完成: 526 篇笔记
  [2/3] 文本切分...
  [2/3] 切分完成: 7050 个文档块
  [3/3] 向量嵌入: 526 父文档 + 6524 子块
  [parents] 526/526
  [children] 6524/6524
Done: 526 parents, 6524 children
```

后续增量同步：

```bash
rag index --sync       # 增量更新（仅处理变更的文件）
rag index --status     # 查看索引统计
```

## 使用方式

### CLI 问答

```bash
rag ask "Docker 有哪几种网络模式？"
```

流式输出，实时显示 DeepSeek 的回答。支持过滤：

```bash
rag ask "这段代码怎么优化？" --folder "000C++/c++刷题" --tag "DP"
```

### Web 界面

```bash
rag server
```

打开浏览器访问 `http://127.0.0.1:8501`。

功能：
- 🗨️ 流式聊天（SSE 实时推送）
- 📄 查看引用笔记来源
- ⚙️ 设置面板（本地存储，无需数据库）
- 🌙 自动适配系统暗色模式
- 🔄 网页端一键重建索引

### 命令总览

```
rag ask <问题>        # 提问
rag ask --folder <目录> --tag <标签>  # 带过滤的提问
rag index --rebuild    # 全量重建索引
rag index --sync       # 增量同步
rag index --status     # 查看索引状态
rag server             # 启动 Web 服务
rag server --port 8080 # 指定端口
```

## 工作原理

```text
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
│ Obsidian    │───▶│ 父子分块      │───▶│ ChromaDB    │───▶│ RAG 管线  │
│ .md 文件    │    │ 父文档+子块   │    │ 双集合存储   │    │ 检索+生成  │
└─────────────┘    └──────────────┘    └─────────────┘    └───────────┘
                          │                   │                  │
                    BGE 嵌入模型        语义向量索引        DeepSeek LLM
```

**父子检索策略**：
1. 每个笔记保留一份完整父文档（doc_type=parent）
2. 按 `##` 标题切分为多个子块（doc_type=child）
3. 搜索时对子块做语义匹配，去重后返回完整的父文档
4. 可选：通过 `[[wikilinks]]` 扩展关联笔记（1-hop）

**文件监听**：启动 Web 服务后会自动监听 Obsidian 仓库的文件变更（创建/修改/删除），2 秒防抖后增量更新索引。

## 运行测试

```bash
conda activate pytorch251
pytest tests/ -v    # 24 个测试，约 2 分钟
```

嵌入模型较大，首次运行测试会自动下载 BGE 模型到 HuggingFace 缓存目录。

## 依赖说明

| 包 | 用途 |
|----|------|
| `langchain-*` | RAG 管线编排 |
| `chromadb` | 本地向量数据库（持久化到 `chroma_data/`） |
| `sentence-transformers` | BGE 中文嵌入模型 |
| `fastapi` + `uvicorn` | Web 服务端 |
| `sse-starlette` | 流式响应（SSE） |
| `typer` + `rich` | CLI 命令行 |
| `watchdog` | 文件系统监听 |
