# 个人知识库 RAG 问答助手 — 设计文档

**日期：** 2026-06-09
**状态：** 待评审

---

## 概述

基于用户本地 Obsidian 知识库，构建一个个人 RAG（检索增强生成）问答助手。提供 CLI 命令行工具和本地 Web 聊天界面两种交互方式。用户通过自然语言提问，系统从 Obsidian 笔记中检索相关内容，结合 DeepSeek 大模型生成回答并标注引用来源。

---

## 技术选型

| 维度 | 选择 | 理由 |
|------|------|------|
| 交互形式 | CLI (Typer) + Web (Alpine.js) | 统一后端服务，两种前端消费 |
| 编程语言 | Python | RAG 生态最丰富 |
| 后端框架 | FastAPI | 轻量、高性能、SSE 原生支持、与 Typer 同作者 |
| LLM | DeepSeek API | 性价比高、openai 兼容 SDK |
| Embedding | 本地 BGE-M3 (sentence-transformers) | 数据不出本地，中文效果优秀 |
| 向量存储 | Chroma | Python 原生、增量更新、持久化 |
| RAG 框架 | LangChain 核心模块 | 学习目的，只用 langchain-core / langchain-community / langchain-text-splitters |
| 分块策略 | 父子检索 | 精准匹配 + 上下文完整性兼顾 |
| 同步方式 | watchdog 文件监听 + 手动重建命令 | 自动无感同步，兼有人工兜底 |
| CLI 框架 | Typer | 简洁、类型安全、与 FastAPI 同作者 |
| Web 前端 | Alpine.js + marked.js + SSE | 无构建工具、轻量、够用 |
| 包管理 | Python venv + pyproject.toml | 标准现代 Python 项目结构 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Obsidian 知识库                        │
│              ~/your-vault/*.md                           │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              索引管线 (Indexing Pipeline)                  │
│                                                         │
│  watchdog ──▶ ObsidianLoader ──▶ RecursiveSplitter      │
│  (文件监听)    (解析md+frontmatter)  (父子分块)            │
│                                      │                  │
│                                      ▼                  │
│                          ┌──────────────────┐           │
│                          │  BGE-M3 (本地)    │           │
│                          │  → 向量化         │           │
│                          └────────┬─────────┘           │
│                                   │                     │
│                                   ▼                     │
│                          ┌──────────────────┐           │
│                          │  Chroma          │           │
│                          │  (持久化存储)      │           │
│                          └──────────────────┘           │
└─────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI 服务 (统一后端)                       │
│                                                         │
│  POST /api/chat ──▶ RAG Chain ──▶ DeepSeek API ──▶ SSE  │
│  GET  /api/status                                     │
│  POST /api/reindex                                    │
│  GET  /api/sources/{id}                                │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│  CLI (Typer)      │   │  Web (Alpine.js)  │
│  rag ask "..."    │   │  localhost:8501   │
│  rag index --...  │   │  聊天界面          │
│  rag status       │   │                   │
└──────────────────┘   └──────────────────┘
```

### 问答数据流

```
用户问题
  → FastAPI 接收
  → 问题向量化 (BGE-M3)
  → Chroma 检索 (语义搜索 top-k 子块 + 父文档补齐)
  → 元数据过滤 (标签/文件夹可选)
  → Prompt 拼装 (系统提示 + 检索结果 + 对话历史 + 用户问题)
  → DeepSeek API (流式)
  → SSE 逐 token 返回客户端
  → 展示回答 + 引用来源
```

---

## 项目结构

```
02bankSuperpowers/
├── .venv/                   # Python 虚拟环境（gitignore）
├── .gitignore
├── pyproject.toml           # 项目配置 & 依赖
├── rag_cli/                 # CLI 入口
│   ├── __init__.py
│   └── main.py              # Typer 命令定义
├── rag_server/              # FastAPI 服务
│   ├── __init__.py
│   ├── app.py               # 应用工厂 & 路由
│   ├── chat.py              # /api/chat SSE 处理
│   └── static/              # Web 前端
│       ├── index.html       # Alpine.js 聊天界面
│       └── style.css
├── rag_core/                # 核心逻辑
│   ├── __init__.py
│   ├── indexing/            # 索引管线
│   │   ├── loader.py        # ObsidianLoader (LangChain)
│   │   ├── splitter.py      # 父子分块
│   │   ├── embedder.py      # BGE-M3 向量化
│   │   └── store.py         # Chroma 存储
│   ├── retrieval/           # 检索逻辑
│   │   ├── retriever.py     # LangChain Retriever
│   │   └── pipeline.py      # RAG Chain (LCEL)
│   ├── llm/                 # LLM 调用
│   │   └── deepseek.py      # DeepSeek ChatOpenAI
│   └── watcher.py           # 文件监听服务
├── config.py                # 全局配置
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-09-rag-knowledge-assistant-design.md
```

---

## 索引管线设计

### ObsidianLoader（自定义 LangChain Loader）

基于 `BaseLoader`，解析 Obsidian 笔记为 `List[Document]`：

- **page_content**：笔记纯文本（剥离 YAML frontmatter）
- **metadata**：
  - `source`：相对路径（如 `"Docker/Docker 网络.md"`）
  - `filename`：文件名（无扩展名）
  - `folder`：所在文件夹
  - `tags`：来自 frontmatter 的 `tags` 字段和正文 `#tag`
  - `links`：来自 `[[双向链接]]` 的链接目标
  - `mtime`：文件修改时间（ISO 格式）
  - `doc_type`：`"parent"` 或 `"child"`
  - `parent_id`：子块指向父文档的标识（子块专有）

**YAML frontmatter 处理规则：**
- `tags:` → 转为 metadata.tags 列表
- `aliases:` → 加入检索索引（别名权重等同标题）
- 其余字段保留但仅作元数据，不参与检索

### 父子分块（RecursiveSplitter）

```
一条 Obsidian 笔记
│
├── 父文档 (parent)
│   └── 整条笔记全文，不做硬性长度限制
│       metadata.doc_type = "parent"
│
└── 子文档 (children)
    ├── 按 ## 二级标题切分
    │   └── 每个 ## 段落为一个子块
    ├── 如果某个 ## 段落 > 1000 字 → RecursiveCharacterTextSplitter 继续切
    │   └── chunk_size=800, chunk_overlap=100
    └── metadata.doc_type = "child"
        metadata.parent_id = 父文档标识
```

**切分策略说明：** 不以 `#` 一级标题为切分边界。Obsidian 中一级标题通常是文件主标题，二级标题才是真正的"章节"。

### Embedding（BGE-M3）

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},        # 无 GPU 用 CPU，有则改 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)
```

- 向量维度：1024
- 首次运行自动下载模型（约 2GB），缓存到 `~/.cache/huggingface/`
- `normalize_embeddings=True` 保证余弦相似度计算准确

### Chroma 存储

两个 Collection：

| Collection | 用途 |
|------------|------|
| `rag_parents` | 存储父文档完整内容 |
| `rag_children` | 存储子块，用于语义检索匹配 |

---

## 检索与 RAG Chain 设计

### LangChain LCEL Chain

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "history": chat_history,
    }
    | prompt_template
    | llm                          # ChatOpenAI(deepseek)
    | StrOutputParser()
)
```

### 检索流程

```
用户问题
  → 问题向量化 (BGE-M3)
  → Chroma "rag_children" 语义搜索 (k=10)
  → 可选：按 metadata.tags 或 metadata.folder 预过滤
  → 子块按 parent_id 去重分组
  → 从 "rag_parents" 取完整父文档
  → 可选：[[双向链接]] 一阶扩展检索（1 跳，不递归）
  → 返回：完整父文档 + 来源标注
```

### Prompt 模板

```
System:
你是个人知识库问答助手。根据用户笔记内容回答问题。
如果笔记中没有相关信息，请明确说明，不要编造。

以下是从用户 Obsidian 知识库中检索到的相关笔记：

{context}

<对话历史>
{history}

用户问题：{question}

要求：
- 用中文回答
- 引用具体笔记时，注明来源（笔记文件名）
- 如果涉及多个笔记的观点，请分别说明
- 可以综合多篇笔记进行分析
```

### 对话历史管理

- Web 端：服务端内存存储，保留最近 10 轮对话
- CLI 端：单次会话内保留历史，`--no-history` 可关闭
- 历史窗口：拼接进 Prompt 时取最近 6 轮
- 不做持久化（个人项目，重启即清）

---

## 文件监听与同步设计

### 监听机制（watchdog）

```
watchdog Observer
  ├── 监听 Obsidian 仓库根目录（递归）
  ├── 过滤：只监听 .md 文件
  └── 忽略：.obsidian/、.trash/ 等目录

事件处理（防抖 2 秒）：
  ├── on_created  → 加载 + 分块 + 向量化 + 存入 Chroma
  ├── on_modified → 删除旧文档 + 重新索引
  └── on_deleted  → 从 Chroma 清理对应文档
```

**防抖策略：** 2 秒内连续事件合并为一个批次处理，避免批量编辑时触发 N 次重建。

### 启动全量对比

```
服务启动
  → 扫描 Obsidian 仓库 → 获取全部 .md 文件列表 + mtime
  → 对比 Chroma 中已有文档（按 source 字段匹配）
  → 新文件 → 索引 | mtime 变化 → 重建 | 已删除 → 清理 | 无变化 → 跳过
  → 启动文件监听
```

### CLI 手动操作

```bash
rag index --status       # 查看索引状态
rag index --rebuild      # 清空 Chroma，全量重建
rag index --sync         # 增量同步一次（不启动监听）
```

---

## CLI 设计

### 命令结构

```bash
rag ask "提问内容"                       # 单次问答（流式）
rag ask "提问内容" --no-history          # 不带对话历史
rag ask "提问内容" --folder "Docker"     # 限定搜索文件夹
rag ask "提问内容" --tag "python"        # 限定搜索标签

rag index --status                       # 查看索引状态
rag index --rebuild                      # 全量重建索引
rag index --sync                         # 增量同步

rag server --port 8501                   # 启动 Web 服务
rag server --port 8501 --no-watch        # 仅启动服务，不监听文件
```

### 交互特性

- `rag ask` 自动检测后端服务是否运行，未运行则以子进程启动
- 流式输出，打字机效果逐 token 打印
- 终端不支持 Markdown 渲染，做基本 ANSI 格式化
- `rag server` 启动后 fork 到后台

---

## Web 前端设计

### 页面布局

```
┌──────────────────────────────────────────────────┐
│  📚 我的知识库助手                    [⚙️ 设置]   │
│──────────────────────────────────────────────────│
│  📊 已索引 1,247 篇笔记 · 上次同步 2 分钟前        │
│──────────────────────────────────────────────────│
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ 🤖 你好，我是你的知识库助手。                 │  │
│  │ 可以问我关于你笔记中的任何问题。               │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │                          🧑 你              │  │
│  │ Docker 网络模式有哪些？                      │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ 🤖 助手                                     │  │
│  │ 根据你的笔记，Docker 网络模式有...            │  │
│  │                                            │  │
│  │ 📄 来源：Docker/Docker 网络.md              │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│──────────────────────────────────────────────────│
│  [___________________________] [🔍] [📎 筛选]    │
│  输入你的问题...                  [发送]          │
└──────────────────────────────────────────────────┘
```

### 功能特性

- **流式回答** — SSE 逐字显示
- **来源标注** — 每条回答底部列出引用笔记，可点击查看原文
- **标签/文件夹筛选** — 下拉菜单限定检索范围
- **对话历史** — 多轮对话，刷新后清空
- **状态栏** — 顶部显示索引统计和最后同步时间
- **设置面板** — 可配置 DeepSeek API Key、Obsidian 仓库路径、模型名
- **暗色/亮色模式** — 跟随系统设置

### 技术实现

- Alpine.js：消息列表状态管理、输入状态、加载态
- marked.js：Markdown 渲染
- EventSource (SSE)：流式接收
- 纯 CSS：现代简洁风格，响应式

### FastAPI 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | SSE 流式聊天（body: `{question, filters?}`） |
| `/api/status` | GET | 索引统计、同步时间 |
| `/api/reindex` | POST | 触发全量重建 |
| `/api/sources/{doc_id}` | GET | 获取笔记原文 |
| `/` | GET | 返回静态 index.html |

---

## 配置设计

### config.py 配置项

```python
class Config:
    # Obsidian
    obsidian_vault_path: str          # Obsidian 仓库路径
    obsidian_ignore_dirs: list[str]   # 忽略的目录，默认 [".obsidian", ".trash"]

    # DeepSeek
    deepseek_api_key: str             # API Key（从环境变量 DEEPSEEK_API_KEY 读取）
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Chroma
    chroma_persist_dir: str = "./chroma_data"

    # Retrieval
    retrieval_top_k: int = 10
    enable_link_expansion: bool = True  # 是否启用 [[链接]] 扩展检索

    # Server
    server_host: str = "127.0.0.1"
    server_port: int = 8501
```

---

## 错误处理策略

- **API Key 未配置**：启动时检查，未设置则给出明确提示并退出
- **Obsidian 路径不存在**：启动时检查，路径无效则提示配置
- **DeepSeek API 调用失败**：重试 1 次，仍失败则返回友好错误信息
- **Chroma 损坏**：检测到异常时提示运行 `rag index --rebuild`
- **首次启动无缓存**：提示正在下载 BGE-M3 模型，显示进度
- **文件监听异常**：日志记录，不影响问答服务

---

## 不在范围内（明确排除）

- 多用户支持
- 图片/PDF/附件等非 Markdown 文件的处理
- 对话历史的持久化存储
- 云端部署
- 移动端适配
- 笔记的自动标签/分类
- Obsidian 插件形态（可能未来考虑）
