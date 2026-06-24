# 个人简历

## 项目经验

### RAG 知识库助手 — 基于 Obsidian 的个人 RAG 问答系统

**时间**：2026 年 6 月 | **角色**：独立开发（全栈）

**项目概述**：设计并实现了一个完整的本地 RAG（检索增强生成）知识库问答助手，将个人 Obsidian 笔记转变为可语义搜索的智能知识库。支持 CLI 命令行和 Web 聊天界面两种交互方式，实现从文档加载、文本分块、向量嵌入、语义检索到 LLM 生成的完整管线。

**核心技术栈**：Python · PyTorch (GPU/CUDA) · LangChain · ChromaDB · BGE 嵌入模型 · DeepSeek API · FastAPI · Alpine.js · SSE 流式传输

---

#### 架构设计与实现

- **模块化系统架构**：设计了清晰的三层架构（`rag_core` 核心引擎 / `rag_server` Web 服务 / `rag_cli` 命令行），遵循关注点分离原则，各模块职责明确、可独立测试
- **父子分块策略**：设计了创新的文档分块策略——保留完整笔记作为父文档，按 `##` 二级标题切分为子块进行语义匹配，大段落通过 `RecursiveCharacterTextSplitter` 二次切分（800 字符窗口 + 100 字符重叠），兼顾检索精度与上下文完整性
- **父子检索管线**：实现了"子块语义检索 → parent_id 去重分组 → 完整父文档补齐 → [[双向链接]] 一阶扩展"的检索链路，基于 LangChain LCEL（LangChain Expression Language）声明式编排
- **双 Collection 向量存储**：在 ChromaDB 中分别维护 `rag_parents`（完整文档）和 `rag_children`（子块向量）两个语义隔离的 Collection，实现检索效率与内容完整性的平衡

#### 索引与同步子系统

- **Obsidian 格式解析器**：基于 LangChain `BaseLoader` 实现了完整的 Obsidian `.md` 文件加载器，支持 YAML Frontmatter 解析、`#标签` 内联提取、`[[双向链接]]` 识别、文件修改时间追踪等
- **GPU 加速嵌入**：使用 `sentence-transformers` 加载 BGE 中文嵌入模型（`bge-small-zh-v1.5`），在 CUDA GPU 上执行推理，embedding 归一化确保余弦相似度准确。采用 16 条/批次的批处理策略控制显存峰值
- **实时文件监听**：基于 `watchdog` 实现 Obsidian 仓库文件系统监听，支持文件创建/修改/删除事件的自动增量更新，2 秒防抖机制避免批量编辑时的重复索引
- **启动时增量对比**：服务启动时自动执行全量仓库扫描 → 按 source 路径和 mtime 与 Chroma 已有数据对比 → 仅索引新增/变更文件，无需全量重建

#### Web 服务与流式交互

- **FastAPI 后端服务**：实现 RESTful API（`/api/chat`、`/api/status`、`/api/reindex`、`/api/sources/{id}`），使用 lifespan 上下文管理器管理资源生命周期
- **SSE 流式传输**：采用原生 `StreamingResponse`（替代第三方 SSE 库）实现 `text/event-stream` 格式的逐 token 推送，事件类型化（thinking / token / done / error），支持实时打字机效果
- **会话管理**：服务端内存中维护多会话对话历史（最多 100 个会话），每会话保留最近 20 轮，Prompt 拼接时取最近 6 轮，防止上下文窗口溢出
- **前端 SPA**：使用 Alpine.js 实现响应式聊天界面，无需构建工具。支持 Markdown 渲染（marked.js）、SSE 流式接收、来源引用查看、设置面板（localStorage 持久化）和自动暗色模式适配
- **服务配置透传**：Web 设置面板展示服务端配置默认值（模型名、设备、检索参数），实现前后端配置一致性

#### CLI 命令行工具

- **Typer 命令体系**：实现 `rag ask`（流式问答，支持文件夹/标签过滤）、`rag index`（全量重建/增量同步/状态查看）、`rag server`（Web 服务启动）三个子命令
- **流式输出适配**：处理 Windows GBK 终端编码兼容问题，通过 `sys.stdout` 直接写入 + `UnicodeEncodeError` 异常捕获确保中文正常显示
- **Rich 美化输出**：使用 Rich 库提供彩色进度提示、表格化状态展示

#### 质量保证

- **24 个单元测试**：使用 pytest 覆盖加载器（7 个）、分块器（6 个）、向量存储（6 个）和 RAG 管线（5 个），利用 `tempfile.TemporaryDirectory` 创建临时 Obsidian 仓库的 fixture 模式
- **错误处理策略**：API Key 未配置提示、路径不存在检查、DeepSeek API 调用异常友好报错、Chroma 元数据清理（空列表/嵌套结构过滤）
- **设计文档完备**：包含完整的设计规格说明和实施计划，覆盖技术选型理由、架构图、数据流图和接口定义

---

### 技术能力总结

| 领域         | 具体技术                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------- |
| **编程语言** | Python 3.10+（类型注解、dataclass、async/await、生成器）                                          |
| **RAG/LLM**  | LangChain（LCEL、BaseLoader、BaseRetriever）、DeepSeek API（OpenAI 兼容 SDK）、Prompt Engineering |
| **向量检索** | ChromaDB（持久化、Collection 管理、元数据过滤）、BGE 中文嵌入模型、sentence-transformers          |
| **后端框架** | FastAPI（lifespan、StreamingResponse、路由）、Uvicorn                                             |
| **前端**     | Alpine.js（响应式状态管理）、SSE（EventSource API）、marked.js（Markdown 渲染）、CSS 变量暗色模式 |
| **CLI 工具** | Typer（类型安全参数解析）、Rich（终端美化输出）                                                   |
| **文件系统** | watchdog（事件监听 + 防抖）、pathlib、YAML 解析                                                   |
| **测试**     | pytest（fixture、参数化测试）、临时文件系统测试模式                                               |
| **工程化**   | pyproject.toml（依赖管理 + 构建配置）、conda 环境管理、GPU/CUDA 推理配置、.env 配置管理           |