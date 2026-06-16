# FastAPI 入门：以 RAG 知识库后端为例

> 面向 FastAPI 新手，用 `rag_server/` 的代码作为实例讲解。

---

## 1. FastAPI 是什么？

FastAPI 是一个 Python Web 框架。它的核心功能就一句话：**把 Python 函数映射到 URL 地址，让浏览器能通过 HTTP 调用它**。

```python
# 写一个函数，加一行装饰器，就成了一个 Web API
@app.get("/api/status")
async def status():
    return {"status": "ok"}
```

`GET http://127.0.0.1:8501/api/status` → 返回 `{"status":"ok"}` 的 JSON。

### 和 Flask 有什么不同？

| | Flask | FastAPI |
|---|---|---|
| 异步支持 | 需要额外插件 | **原生 async/await** |
| 数据校验 | 手动写或用插件 | **自动校验**（Pydantic） |
| API 文档 | 需要额外配置 | **自动生成** `/docs` |
| 性能 | 同步模型 | 异步高性能 |
| 类型提示 | 可选 | **一等公民** |

---

## 2. 核心概念

### 2.1 路由（Route）

把 URL 路径和函数绑定。HTTP 方法用装饰器区分：

```python
@app.get("/api/status")       # 读数据 —— 浏览器直接访问
@app.post("/api/chat")        # 提交数据 —— 表单、Ajax 请求
@app.post("/api/reindex")     # 触发操作 —— 重建索引
```

对应 `app.py` 里的 4 个路由：

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/` | 返回前端 HTML 页面 |
| GET | `/api/status` | 查索引状态、文档数 |
| POST | `/api/chat` | 发送问题，流式返回答案 |
| POST | `/api/reindex` | 重建整个向量索引 |
| GET | `/api/sources/{source}` | 查看某篇笔记原文 |

### 2.2 async / await

```python
@app.post("/api/chat")
async def chat(request: Request):      # async = 这个函数可以等待
    body = await request.json()         # await = 这里要等一下，但不阻塞其他请求
```

**关键理解**：`await` 的时候 FastAPI 不会傻等——它去处理别人的请求，等 `await` 的东西好了再回来继续。这就是"异步"：

```
传统同步:  请求A → [等数据库] → [等LLM] → 返回
           请求B → .......................排队等着.......................

FastAPI:   请求A → [等数据库] → 让出CPU → [等LLM] → 返回
           请求B → [等数据库] → [等LLM] → 返回     ← 不排队，交替执行
```

### 2.3 Lifespan（生命周期）

FastAPI 没有全局"启动时执行一次"的钩子。取而代之的是 **lifespan**：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 服务启动时执行 ===
    init_app(config)         # 初始化向量库、RAG管线、文件监控

    yield                    # ← yield 之前是启动逻辑，yield 之后是关闭逻辑

    # === 服务关闭时执行 ===
    if _watcher:
        _watcher.stop_watching()   # 停掉文件监控
```

**用 `yield` 分割启动和关闭**，这是 Python 的 `asynccontextmanager` 模式。

### 2.4 挂载静态文件

```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```

把 `rag_server/static/` 目录映射到 `/static` URL。浏览器访问 `http://127.0.0.1:8501/static/style.css` 就能拿到 CSS 文件。这等于内置了一个简单的文件服务器。

### 2.5 SSE（Server-Sent Events）

普通的 HTTP 是"请求→响应→结束"。SSE 不一样：

```
普通 HTTP:   浏览器 → POST /chat → 服务器返回完整答案 → 连接关闭
SSE:         浏览器 → POST /chat → 服务器推"R"→推"A"→推"G"→推"完成"→ 连接才关
```

FastAPI 用 `EventSourceResponse` 实现 SSE：

```python
return EventSourceResponse(event_generator())
#                          ↑ 这是一个 async generator，每 yield 一次就推一个事件给浏览器
```

对应 `chat.py` 里的生成器：

```python
async def chat_stream(pipeline, question):
    yield {"event": "thinking", "data": "正在检索笔记..."}   # 第1个事件
    for chunk in stream:
        yield {"event": "token", "data": chunk}               # 逐字推送
    yield {"event": "done", "data": full_answer}              # 结束事件
```

浏览器端用 `EventSource` API 接收：

```javascript
const source = new EventSource('/api/chat');
source.addEventListener('token', e => { 显示(e.data) });  // 每收到一个字就显示
source.addEventListener('done',  e => { 结束() });
```

### 2.6 HTTPException

```python
if not question:
    raise HTTPException(status_code=400, detail="question 不能为空")
```

FastAPI 自动把异常转成 JSON 错误响应：`{"detail": "question 不能为空"}`，状态码 400。

**常见状态码**：
- `200` OK
- `400` 请求格式不对
- `404` 找不到
- `500` 服务器内部错误

### 2.7 路径参数

```python
@app.get("/api/sources/{source:path}")   # {source:path} 匹配任意路径（含 /）
async def get_source(source: str):        # FastAPI 自动把 URL 中的 source 传进来
```

- `/api/sources/笔记/深度学习.md` → `source = "笔记/深度学习.md"`
- `{source:path}` 里的 `:path` 表示包含斜杠，不加 `:path` 斜杠会被当成路由分隔符

---

## 3. rag_server 代码详解

### 架构图

```
┌──────────────────────────────────────────────────────┐
│                     FastAPI 应用                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ app.py   │  │ chat.py  │  │ static/           │   │
│  │ 5个路由  │  │ SSE流式  │  │ index.html        │   │
│  │ lifespan │  │ 会话管理 │  │ style.css         │   │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘   │
│       │              │                                │
│       └──────┬───────┘                                │
│              ▼                                        │
│       rag_core/  ← 核心逻辑层                          │
│       ├─ indexing/embedder.py  (BGE 向量化)           │
│       ├─ indexing/store.py     (ChromaDB 存储)        │
│       ├─ retrieval/pipeline.py (RAG 问答流程)         │
│       └─ llm/deepseek.py       (DeepSeek 大模型)      │
└──────────────────────────────────────────────────────┘
```

### app.py 逐段解析

```python
# 全局变量 —— 服务启动时初始化一次，所有请求共用
_pipeline: RAGPipeline | None = None    # RAG 问答管线
_store: VectorStoreManager | None = None # 向量数据库
_watcher: VaultWatcher | None = None     # 文件监控器
```

→ **为什么用全局变量而不是每次请求都创建？** 因为向量模型加载到内存要 2GB，每次创建会 OOM。

```python
def init_app(config):
    _store = VectorStoreManager(persist_dir=...)  # ①打开 ChromaDB
    _pipeline = RAGPipeline(store=_store)          # ②创建 RAG 管线

    if cfg.obsidian_vault_path:                    # ③如果配置了笔记目录
        _watcher = VaultWatcher(...)               #    创建文件监控
        _watcher.full_sync()                       #    全量同步一次
        _watcher.start_watching()                  #    开始监听文件变动
```

→ 启动时做了三件事：**打开数据库 → 创建管线 → 同步笔记**。如果没配仓库路径，监控不启动，但服务照常运行。

```python
def create_app(config=None) -> FastAPI:
    # ... lifespan 和路由定义 ...
    return app
```

→ **工厂函数模式**：不直接创建 FastAPI 实例，而是包在函数里。好处是外部可以传自定义配置进来，测试时也能建不同的 app。

`uvicorn.run()` 的 `factory=True` 参数就是告诉 uvicorn："这个路径指向的是一个工厂函数，请调用它来获取 app"。

### chat.py 逐段解析

```python
_sessions: dict[str, list[dict]] = {}    # 内存字典，存所有会话

def get_history(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = []        # 新会话：空历史
    if len(_sessions) > 100:             # 超过100个会话
        del _sessions[oldest]             # 删最老的（防内存泄漏）
    return _sessions[session_id]
```

→ 会话历史存在内存里，服务重启就没了。对个人知识库助手来说足够，生产环境要用 Redis。

```python
async def chat_stream(pipeline, question, session_id, folder, tag):
    history = get_history(session_id)

    if folder or tag:
        stream = pipeline.ask_with_filter(question, history, folder=folder, tag=tag)
    else:
        stream = pipeline.ask(question, history)

    yield {"event": "thinking", "data": "正在检索笔记..."}

    for chunk in stream:
        full_answer += chunk
        yield {"event": "token", "data": chunk}

    history.append({"role": "user", "content": question})       # 记住用户问的
    history.append({"role": "assistant", "content": full_answer}) # 记住AI答的
    if len(history) > 20:                                        # 只保留最近10轮
        _sessions[session_id] = history[-20:]

    yield {"event": "done", "data": full_answer}
```

→ 这是 LLM 对话的标准模式：**把历史问答一起发给模型**，模型就知道上下文了。

---

## 4. 请求流程（完整）

```
用户输入 "什么是 RAG？"，点击发送
        │
        ▼
浏览器 POST /api/chat {"question":"什么是 RAG？","session_id":"default"}
        │
        ▼
app.py: chat() 函数
  ├─ 取出 question, session_id
  ├─ 检查 question 非空，否则 400
  └─ 调用 chat_stream()
        │
        ▼
chat.py: chat_stream()
  ├─ 查 session_id 获取历史记录
  ├─ 调 pipeline.ask(question, history)
  │   ├─ embedder 把问题转成向量
  │   ├─ store 在 ChromaDB 中检索相似笔记
  │   ├─ 把相关笔记 + 问题 + 历史 拼接成 prompt
  │   └─ 调 DeepSeek API 生成答案（流式）
  ├─ yield "thinking" → 浏览器显示"思考中"
  ├─ 逐字 yield token → 浏览器逐字显示
  ├─ yield "done" → 浏览器标记完成
  └─ 保存问答到历史
```

---

## 5. 关键设计要点

1. **单例初始化**：向量模型 2GB，不能每个请求都加载一次。全局变量 + lifespan 保证只加载一次
2. **工厂模式**：`create_app()` 包装创建逻辑，便于测试和自定义配置
3. **SSE 流式输出**：用户不需要等 LLM 生成完才看到结果，体验好
4. **异常兜底**：每个路由都有 try/except 或 HTTPException，出错了返回有意义的信息而不是崩溃
5. **内存会话**：对话历史存内存字典，限制 100 会话 / 每会话 10 轮，防止内存泄漏

---

## 6. 启动命令

```bash
# 开发模式
uvicorn rag_server.app:create_app --factory --reload --host 127.0.0.1 --port 8501

# 或通过 CLI
rag serve
```

启动后访问：
- Web UI：`http://127.0.0.1:8501/`
- API 文档（自动生成）：`http://127.0.0.1:8501/docs`

---

## 7. 参考资料

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [SSE 规范 MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Python asyncio 指南](https://docs.python.org/3/library/asyncio.html)
