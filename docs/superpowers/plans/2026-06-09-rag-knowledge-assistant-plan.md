# RAG 知识库问答助手 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 Obsidian 知识库的个人 RAG 问答助手，提供 CLI 和 Web 两种交互方式。

**Architecture:** FastAPI 统一后端服务 → LangChain RAG Chain（BGE-M3 Embedding + Chroma 向量库 + DeepSeek LLM）→ CLI (Typer) + Web (Alpine.js) 双前端。watchdog 监听 Obsidian 仓库实现自动索引同步。

**Tech Stack:** Python 3.11+, FastAPI, Typer, LangChain (core/community/text-splitters/openai/chroma), Chroma, sentence-transformers (BGE-M3), watchdog, Alpine.js, marked.js

---

## 文件结构

```
02bankSuperpowers/
├── .gitignore
├── pyproject.toml
├── config.py                    # 全局配置（dataclass + 环境变量）
├── rag_core/
│   ├── __init__.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── loader.py            # ObsidianLoader — 解析 .md 文件
│   │   ├── splitter.py          # 父子分块逻辑
│   │   ├── embedder.py          # BGE-M3 向量化封装
│   │   └── store.py             # Chroma 双 Collection 管理
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py         # LangChain Retriever — 父子检索
│   │   └── pipeline.py          # LCEL RAG Chain
│   ├── llm/
│   │   ├── __init__.py
│   │   └── deepseek.py          # DeepSeek ChatOpenAI 封装
│   └── watcher.py               # watchdog 文件监听 + 启动全量对比
├── rag_server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI 应用 + 路由
│   ├── chat.py                  # SSE 流式聊天端点
│   └── static/
│       ├── index.html           # Alpine.js 聊天界面
│       └── style.css            # 样式
├── rag_cli/
│   ├── __init__.py
│   └── main.py                  # Typer CLI 命令
└── tests/
    ├── __init__.py
    ├── conftest.py              # fixtures（临时 Obsidian 仓库等）
    ├── test_loader.py
    ├── test_splitter.py
    ├── test_store.py
    ├── test_retriever.py
    └── test_pipeline.py
```

---

### Task 1: 项目基础设施

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `config.py`
- Create: 所有 `__init__.py` 文件

- [ ] **Step 1: 创建 .gitignore**

```ini
# .gitignore
.venv/
__pycache__/
*.pyc
.env
chroma_data/
*.egg-info/
dist/
.pytest_cache/
```

- [ ] **Step 2: 创建 pyproject.toml**

```toml
[project]
name = "rag-assistant"
version = "0.1.0"
description = "个人知识库 RAG 问答助手"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-chroma>=0.1.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "watchdog>=5.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "sse-starlette>=2.0.0",
]

[project.scripts]
rag = "rag_cli.main:app"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["rag_core*", "rag_server*", "rag_cli*"]
```

- [ ] **Step 3: 创建 config.py**

```python
"""全局配置，从环境变量和 .env 文件读取。"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Obsidian
    obsidian_vault_path: str = field(
        default_factory=lambda: os.getenv("OBSIDIAN_VAULT_PATH", "")
    )
    obsidian_ignore_dirs: list[str] = field(
        default_factory=lambda: [".obsidian", ".trash", ".git"]
    )

    # DeepSeek
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Chroma
    chroma_persist_dir: str = "./chroma_data"

    # Retrieval
    retrieval_top_k: int = 10
    enable_link_expansion: bool = True

    # Server
    server_host: str = "127.0.0.1"
    server_port: int = 8501

    # Chunking
    child_chunk_size: int = 800
    child_chunk_overlap: int = 100
    child_max_len_before_split: int = 1000  # ## 段落超过此长度才二次切分


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config
```

- [ ] **Step 4: 创建所有 __init__.py**

```bash
mkdir -p rag_core/indexing rag_core/retrieval rag_core/llm rag_server/static rag_cli tests
```

然后为每个目录创建空的 `__init__.py`。
每个文件内容为空（或仅含 docstring）。

- [ ] **Step 5: 创建 .env.example**

```
DEEPSEEK_API_KEY=sk-your-key-here
OBSIDIAN_VAULT_PATH=C:/Users/yourname/Documents/ObsidianVault
```

- [ ] **Step 6: 创建虚拟环境并安装依赖**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

- [ ] **Step 7: 验证**

```bash
python -c "from config import get_config; print(get_config())"
```

Expected: 打印默认 Config 对象（API key 和 vault path 为空）。

- [ ] **Step 8: 提交**

```bash
git add .gitignore pyproject.toml config.py .env.example rag_core/ rag_server/ rag_cli/ tests/
git commit -m "chore: project scaffold — pyproject.toml, config, package structure"
```

---

### Task 2: Obsidian Markdown Loader

**Files:**
- Create: `rag_core/indexing/loader.py`
- Create: `tests/test_loader.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 创建测试 fixtures**

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_vault():
    """创建一个临时的 Obsidian 仓库结构用于测试。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        # 创建一条完整的笔记
        note = vault / "Docker" / "Docker 网络.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("""---
tags: [docker, network]
created: 2025-01-15
---

# Docker 网络

Docker 网络是容器之间通信的基础。

## bridge 模式

默认网络模式，容器通过 docker0 网桥通信。

## host 模式

容器直接使用宿主机网络栈，性能最好但隔离性差。

参见 [[运维/容器化实践]] 和 [[Linux/网络基础]]
""", encoding="utf-8")

        # 创建第二条笔记（无 frontmatter）
        note2 = vault / "Python" / "asyncio 笔记.md"
        note2.parent.mkdir(parents=True, exist_ok=True)
        note2.write_text("""# asyncio 笔记

## 事件循环

事件循环是 asyncio 的核心概念...

## async/await 语法

使用 async def 定义协程...
""", encoding="utf-8")

        # 创建 .obsidian 目录（应被忽略）
        obsidian_dir = vault / ".obsidian"
        obsidian_dir.mkdir(parents=True, exist_ok=True)
        (obsidian_dir / "config.json").write_text("{}")

        yield vault
```

- [ ] **Step 2: 编写 Loader 的失败测试**

```python
# tests/test_loader.py
import pytest
from rag_core.indexing.loader import ObsidianLoader


def test_loader_discovers_md_files(temp_vault):
    """Loader 应该发现仓库中所有 .md 文件（忽略 .obsidian 目录）。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "Docker/Docker 网络.md" in sources
    assert "Python/asyncio 笔记.md" in sources
    # .obsidian 下的文件不应出现
    assert not any(".obsidian" in s for s in sources)


def test_loader_extracts_frontmatter(temp_vault):
    """Loader 应该正确提取 YAML frontmatter 中的 tags。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = loader.load()

    docker_doc = [d for d in docs if "Docker 网络" in d.metadata.get("filename", "")][0]
    assert "docker" in docker_doc.metadata["tags"]
    assert "network" in docker_doc.metadata["tags"]


def test_loader_extracts_wikilinks(temp_vault):
    """Loader 应该从 [[链接]] 中提取链接列表。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = loader.load()

    docker_doc = [d for d in docs if "Docker 网络" in d.metadata.get("filename", "")][0]
    assert "运维/容器化实践" in docker_doc.metadata["links"]
    assert "Linux/网络基础" in docker_doc.metadata["links"]


def test_loader_metadata_fields(temp_vault):
    """Loader 应该为每篇文档设置正确的元数据字段。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = loader.load()

    docker_doc = [d for d in docs if "Docker 网络" in d.metadata.get("filename", "")][0]
    assert docker_doc.metadata["folder"] == "Docker"
    assert docker_doc.metadata["filename"] == "Docker 网络"
    assert "mtime" in docker_doc.metadata
    assert docker_doc.metadata["doc_type"] == "raw"  # 原始文档，尚未分块


def test_loader_strips_frontmatter_from_content(temp_vault):
    """page_content 不应包含 YAML frontmatter。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = loader.load()

    docker_doc = [d for d in docs if "Docker 网络" in d.metadata.get("filename", "")][0]
    assert "tags:" not in docker_doc.page_content
    assert "---" not in docker_doc.page_content
    assert "Docker 网络是容器之间通信的基础" in docker_doc.page_content
```

- [ ] **Step 3: 运行测试确认失败**

```bash
pytest tests/test_loader.py -v
```

Expected: 全部 FAIL（ObsidianLoader 类尚未创建/实现）。

- [ ] **Step 4: 实现 ObsidianLoader**

```python
# rag_core/indexing/loader.py
"""Obsidian Markdown 文件加载器。"""
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml
from langchain_core.documents import Document


class ObsidianLoader:
    """从 Obsidian 仓库目录加载 .md 文件为 LangChain Document 列表。

    解析 YAML frontmatter、[[双向链接]]、#标签，
    忽略指定目录（如 .obsidian、.trash）。
    """

    def __init__(
        self,
        vault_path: str,
        ignore_dirs: list[str] | None = None,
    ):
        self.vault_path = Path(vault_path)
        self.ignore_dirs = set(ignore_dirs or [])

    def load(self) -> list[Document]:
        docs = []
        for md_file in self._iter_md_files():
            doc = self._parse_file(md_file)
            if doc is not None:
                docs.append(doc)
        return docs

    def lazy_load(self) -> Iterator[Document]:
        for md_file in self._iter_md_files():
            doc = self._parse_file(md_file)
            if doc is not None:
                yield doc

    def _iter_md_files(self):
        for root, dirs, files in os.walk(self.vault_path):
            # 过滤忽略目录
            rel_dir = Path(root).relative_to(self.vault_path)
            parts = rel_dir.parts
            if any(ignored in parts for ignored in self.ignore_dirs):
                continue

            for f in files:
                if f.endswith(".md"):
                    yield Path(root) / f

    def _parse_file(self, filepath: Path) -> Document | None:
        try:
            raw_text = filepath.read_text(encoding="utf-8")
        except Exception:
            return None

        frontmatter, content = self._split_frontmatter(raw_text)

        # 提取标签
        tags = list(self._extract_frontmatter_tags(frontmatter))
        tags.extend(self._extract_inline_tags(content))

        # 提取链接
        links = list(self._extract_wikilinks(content))

        # 相对路径
        rel_path = filepath.relative_to(self.vault_path)
        folder = str(rel_path.parent) if str(rel_path.parent) != "." else ""

        mtime = datetime.fromtimestamp(
            filepath.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        return Document(
            page_content=content.strip(),
            metadata={
                "source": str(rel_path).replace("\\", "/"),
                "filename": filepath.stem,
                "folder": folder,
                "tags": tags,
                "links": links,
                "mtime": mtime,
                "doc_type": "raw",
            },
        )

    def _split_frontmatter(self, text: str) -> tuple[dict, str]:
        """分离 YAML frontmatter 和正文。"""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    fm = {}
                return fm, parts[2]
        return {}, text

    def _extract_frontmatter_tags(self, fm: dict) -> Iterator[str]:
        """从 frontmatter 中提取 tags 字段。"""
        tags_val = fm.get("tags")
        if isinstance(tags_val, list):
            yield from (str(t).strip() for t in tags_val)
        elif isinstance(tags_val, str):
            yield from (t.strip() for t in tags_val.split(","))

    def _extract_inline_tags(self, content: str) -> Iterator[str]:
        """从正文中提取 #tag 格式的标签。"""
        for match in re.finditer(r"(?<!\S)#([\w一-鿿\-/]+)", content):
            yield match.group(1)

    def _extract_wikilinks(self, content: str) -> Iterator[str]:
        """从正文中提取 [[链接]] 格式的引用。"""
        for match in re.finditer(r"\[\[([^\]]+)\]\]", content):
            link = match.group(1)
            # 去掉可能的 alias 部分：[[目标|别名]]
            if "|" in link:
                link = link.split("|")[0]
            yield link.strip()
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_loader.py -v
```

Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add rag_core/indexing/loader.py tests/test_loader.py tests/conftest.py
git commit -m "feat: ObsidianLoader — parse .md files, frontmatter, wikilinks, tags"
```

---

### Task 3: 父子分块器

**Files:**
- Create: `rag_core/indexing/splitter.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_splitter.py
import pytest
from langchain_core.documents import Document

from rag_core.indexing.splitter import parent_child_split


def test_produces_one_parent_per_document():
    """每个输入文档应该产生一个父文档。"""
    docs = [
        Document(
            page_content="# Title\n\n内容段落\n\n## 第一节\n\n第一节内容",
            metadata={"source": "test/file.md", "filename": "file", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    parents = [d for d in result if d.metadata["doc_type"] == "parent"]
    assert len(parents) == 1
    assert parents[0].page_content == "# Title\n\n内容段落\n\n## 第一节\n\n第一节内容"


def test_parent_has_child_id_list():
    """父文档的 metadata 应该包含子文档 ID 列表。"""
    docs = [
        Document(
            page_content="# A\n\nintro\n\n## Section 1\n\ncontent one\n\n## Section 2\n\ncontent two",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    parents = [d for d in result if d.metadata["doc_type"] == "parent"]
    children = [d for d in result if d.metadata["doc_type"] == "child"]

    assert len(children) >= 2
    for child in children:
        assert child.metadata["parent_id"] == "test/doc.md"


def test_splits_on_h2_headings():
    """按 ## 二级标题切分。"""
    docs = [
        Document(
            page_content="# Title\n\nintro\n\n## Section A\n\nstuff A\n\n## Section B\n\nstuff B",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]
    # intro 是第一段（标题后面的内容），Section A, Section B
    assert len(children) == 3


def test_long_section_is_further_split():
    """超过 child_max_len 的子块会被 RecursiveCharacterTextSplitter 继续切分。"""
    long_section_content = "word " * 6000  # 约 36000 字符的段落
    docs = [
        Document(
            page_content=f"# Title\n\nintro\n\n## Big Section\n\n{long_section_content}",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=800, child_chunk_overlap=100, child_max_len=1000)

    children = [d for d in result if d.metadata["doc_type"] == "child"]
    # intro + Big Section 被切分成多个块
    assert len(children) >= 3  # intro + 至少 2 个长段落切块


def test_doc_without_h2_still_splits_on_double_newline():
    """没有 ## 标题的文档，按段落边界切分。"""
    docs = [
        Document(
            page_content="# Title\n\n段落一有很多内容在这里写了很多东西\n\n段落二还有更多内容今天天气不错",
            metadata={"source": "test/doc.md", "filename": "doc", "doc_type": "raw"},
        )
    ]
    result = parent_child_split(docs, child_chunk_size=200, child_chunk_overlap=0, child_max_len=200)

    parents = [d for d in result if d.metadata["doc_type"] == "parent"]
    children = [d for d in result if d.metadata["doc_type"] == "child"]

    assert len(parents) == 1
    assert len(children) >= 2
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_splitter.py -v
```

- [ ] **Step 3: 实现 parent_child_split**

```python
# rag_core/indexing/splitter.py
"""父子分块逻辑 — 按 ## 标题拆分子块，保留完整父文档。"""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parent_child_split(
    documents: list[Document],
    child_chunk_size: int = 800,
    child_chunk_overlap: int = 100,
    child_max_len: int = 1000,
) -> list[Document]:
    """对一批文档执行父子分块。

    每个文档生成：
    - 1 个父文档（完整原始内容，doc_type="parent"）
    - N 个子文档（按 ## 切分 + 大块二次分割，doc_type="child"）

    Returns:
        包含所有父文档和子文档的列表。
    """
    result: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " "],
    )

    for doc in documents:
        source = doc.metadata.get("source", "")
        base_meta = {k: v for k, v in doc.metadata.items() if k != "doc_type"}

        # 父文档
        parent = Document(
            page_content=doc.page_content,
            metadata={
                **base_meta,
                "doc_type": "parent",
            },
        )
        result.append(parent)

        # 子文档 — 按 ## 标题切分
        sections = _split_by_h2(doc.page_content)

        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            if len(section_text) <= child_max_len:
                # 短段落直接作为一个子块
                child = Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "doc_type": "child",
                        "parent_id": source,
                    },
                )
                result.append(child)
            else:
                # 长段落二次切分
                sub_chunks = text_splitter.split_text(section_text)
                for sub_text in sub_chunks:
                    child = Document(
                        page_content=sub_text,
                        metadata={
                            **base_meta,
                            "doc_type": "child",
                            "parent_id": source,
                        },
                    )
                    result.append(child)

    return result


def _split_by_h2(text: str) -> list[str]:
    """按 ## 标题切分文本，返回各段落列表。"""
    # 用正则匹配 ## 标题行（前面不能是 ##  就是 # 必须只是两个）
    sections = []
    lines = text.split("\n")
    current_section: list[str] = []

    for line in lines:
        stripped = line.strip()
        # 匹配 ## 开头且不匹配 ###（三级标题）
        if re.match(r"^## [^#]", stripped):
            if current_section:
                sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_splitter.py -v
```

- [ ] **Step 5: 提交**

```bash
git add rag_core/indexing/splitter.py tests/test_splitter.py
git commit -m "feat: parent-child splitter — split on ## headings, recurse on long chunks"
```

---

### Task 4: Embedding 封装

**Files:**
- Create: `rag_core/indexing/embedder.py`

- [ ] **Step 1: 实现 embedder**

```python
# rag_core/indexing/embedder.py
"""BGE-M3 Embedding 封装 — 通过 LangChain 的 HuggingFaceEmbeddings 使用。"""
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import get_config


def create_embedder() -> HuggingFaceEmbeddings:
    """创建 BGE-M3 embedding 实例。

    首次调用会自动下载模型（约 2GB）到 ~/.cache/huggingface/。
    """
    config = get_config()
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
```

Embedding 是外部模型，不做单元测试。验证放在集成测试中。

- [ ] **Step 2: 提交**

```bash
git add rag_core/indexing/embedder.py
git commit -m "feat: BGE-M3 embedder wrapper via LangChain HuggingFaceEmbeddings"
```

---

### Task 5: Chroma 向量存储

**Files:**
- Create: `rag_core/indexing/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_store.py
import pytest
from langchain_core.documents import Document

from rag_core.indexing.store import VectorStoreManager


@pytest.fixture
def store_manager(tmp_path):
    """使用临时目录的 Chroma 存储管理器。"""
    persist_dir = str(tmp_path / "chroma_test")
    return VectorStoreManager(persist_dir=persist_dir)


def test_add_and_search_parents(store_manager):
    """添加父文档后能通过语义搜索找到。"""
    docs = [
        Document(
            page_content="Docker 网络模式包括 bridge、host、none 三种。",
            metadata={"source": "Docker/Docker 网络.md", "doc_type": "parent"},
        ),
    ]
    store_manager.add_parents(docs)

    results = store_manager.search_parents_by_source("Docker/Docker 网络.md")
    assert len(results) == 1
    assert results[0].page_content == docs[0].page_content


def test_add_and_semantic_search_children(store_manager):
    """子文档可以通过语义搜索找到相关内容。"""
    # 需要真实的 embedding，这里用少量文本测试
    docs = [
        Document(
            page_content="bridge 是 Docker 默认网络模式，容器之间通过 docker0 网桥通信。",
            metadata={"source": "Docker/Docker 网络.md", "doc_type": "child", "parent_id": "Docker/Docker 网络.md"},
        ),
        Document(
            page_content="Python asyncio 事件循环是异步编程的核心概念。",
            metadata={"source": "Python/asyncio 笔记.md", "doc_type": "child", "parent_id": "Python/asyncio 笔记.md"},
        ),
    ]
    store_manager.add_children(docs)

    results = store_manager.similarity_search("Docker 网络", k=2)
    assert len(results) >= 1
    # Docker 相关的应该排在前面
    assert "Docker" in results[0].page_content


def test_delete_by_source(store_manager):
    """按 source 删除文档，父文档和子文档一起删。"""
    parent = Document(
        page_content="测试内容",
        metadata={"source": "test/doc.md", "doc_type": "parent"},
    )
    child = Document(
        page_content="测试内容子块",
        metadata={"source": "test/doc.md", "doc_type": "child", "parent_id": "test/doc.md"},
    )
    store_manager.add_parents([parent])
    store_manager.add_children([child])

    store_manager.delete_by_source("test/doc.md")

    results_p = store_manager.search_parents_by_source("test/doc.md")
    assert len(results_p) == 0


def test_get_index_stats(store_manager):
    """获取索引统计信息。"""
    docs = [
        Document(page_content="doc A content", metadata={"source": "a.md", "doc_type": "parent"}),
        Document(page_content="doc B content", metadata={"source": "b.md", "doc_type": "parent"}),
    ]
    store_manager.add_parents(docs)

    stats = store_manager.get_stats()
    assert stats["parent_count"] == 2
    assert "last_sync" in stats


def test_rebuild_clears_and_readds(store_manager):
    """rebuild 清空旧数据并重新添加。"""
    docs1 = [Document(page_content="v1", metadata={"source": "a.md", "doc_type": "parent"})]
    store_manager.add_parents(docs1)

    docs2 = [Document(page_content="v2", metadata={"source": "b.md", "doc_type": "parent"})]
    store_manager.rebuild(docs2, [])

    stats = store_manager.get_stats()
    assert stats["parent_count"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_store.py -v
```

- [ ] **Step 3: 实现 VectorStoreManager**

```python
# rag_core/indexing/store.py
"""Chroma 向量存储管理 — 双 Collection（父文档 + 子块）。"""
import uuid
from datetime import datetime, timezone

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag_core.indexing.embedder import create_embedder


class VectorStoreManager:
    """管理 Chroma 中的两个 Collection：rag_parents 和 rag_children。"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._embedder = create_embedder()

        self._client = chromadb.PersistentClient(path=persist_dir)

        self._parent_store = Chroma(
            collection_name="rag_parents",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._children_store = Chroma(
            collection_name="rag_children",
            embedding_function=self._embedder,
            client=self._client,
        )

    def add_parents(self, documents: list[Document]) -> list[str]:
        """添加父文档到 rag_parents 集合。"""
        if not documents:
            return []
        ids = [f"parent_{doc.metadata.get('source', uuid.uuid4())}" for doc in documents]
        return self._parent_store.add_documents(documents, ids=ids)

    def add_children(self, documents: list[Document]) -> list[str]:
        """添加子文档到 rag_children 集合。"""
        if not documents:
            return []
        ids = [f"child_{uuid.uuid4().hex[:12]}_{doc.metadata.get('parent_id', 'unknown')}"
               for doc in documents]
        return self._children_store.add_documents(documents, ids=ids)

    def similarity_search(
        self, query: str, k: int = 10, filter_dict: dict | None = None
    ) -> list[Document]:
        """在子块中进行语义搜索。"""
        return self._children_store.similarity_search(query, k=k, filter=filter_dict)

    def search_parents_by_source(self, source: str) -> list[Document]:
        """按 source 查找父文档。"""
        result = self._parent_store.get(where={"source": source})
        if not result or not result["documents"]:
            return []
        docs = []
        for i, content in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def get_parents_by_sources(self, sources: list[str]) -> list[Document]:
        """批量按 source 获取父文档。"""
        docs = []
        for source in set(sources):
            docs.extend(self.search_parents_by_source(source))
        return docs

    def delete_by_source(self, source: str) -> None:
        """删除指定 source 的所有文档（父 + 子）。"""
        # Chroma 的 delete 按 filter 删除
        try:
            self._parent_store._collection.delete(where={"source": source})
        except Exception:
            pass
        try:
            self._children_store._collection.delete(where={"source": source})
        except Exception:
            pass

    def rebuild(self, parents: list[Document], children: list[Document]) -> None:
        """清空所有数据并重建。"""
        try:
            self._client.delete_collection("rag_parents")
        except Exception:
            pass
        try:
            self._client.delete_collection("rag_children")
        except Exception:
            pass

        # 重建 store 对象（Collection 已删除需要重建）
        self._parent_store = Chroma(
            collection_name="rag_parents",
            embedding_function=self._embedder,
            client=self._client,
        )
        self._children_store = Chroma(
            collection_name="rag_children",
            embedding_function=self._embedder,
            client=self._client,
        )

        self.add_parents(parents)
        self.add_children(children)

    def get_stats(self) -> dict:
        """获取索引统计信息。"""
        try:
            parent_count = self._parent_store._collection.count()
        except Exception:
            parent_count = 0
        try:
            child_count = self._children_store._collection.count()
        except Exception:
            child_count = 0
        return {
            "parent_count": parent_count,
            "child_count": child_count,
            "last_sync": datetime.now(timezone.utc).isoformat(),
        }
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_store.py -v
```

Expected: PASS。注意：语义搜索测试依赖 BGE-M3 模型下载（首次运行可能需要几分钟）。

- [ ] **Step 5: 提交**

```bash
git add rag_core/indexing/store.py tests/test_store.py
git commit -m "feat: VectorStoreManager — dual Chroma collections for parent-child retrieval"
```

---

### Task 6: DeepSeek LLM 封装

**Files:**
- Create: `rag_core/llm/__init__.py`
- Create: `rag_core/llm/deepseek.py`

- [ ] **Step 1: 实现 DeepSeek LLM 工厂**

```python
# rag_core/llm/deepseek.py
"""DeepSeek LLM 封装 — 通过 LangChain ChatOpenAI 调用。"""
from langchain_openai import ChatOpenAI

from config import get_config


def create_deepseek_llm(streaming: bool = True) -> ChatOpenAI:
    """创建 DeepSeek ChatOpenAI 实例。

    DeepSeek API 兼容 OpenAI SDK 格式。
    """
    config = get_config()
    if not config.deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置。请在 .env 文件或环境变量中配置。"
        )
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        streaming=streaming,
        temperature=0.3,
    )
```

- [ ] **Step 2: 提交**

```bash
git add rag_core/llm/
git commit -m "feat: DeepSeek LLM wrapper via LangChain ChatOpenAI"
```

---

### Task 7: 父子检索器

**Files:**
- Create: `rag_core/retrieval/__init__.py`
- Create: `rag_core/retrieval/retriever.py`
- Create: `tests/test_retriever.py`（集成测试风格）

- [ ] **Step 1: 实现 ParentChildRetriever**

```python
# rag_core/retrieval/retriever.py
"""父子检索器 — 子块语义检索 + 父文档补齐。"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_core.indexing.store import VectorStoreManager
from config import get_config


class ParentChildRetriever(BaseRetriever):
    """LangChain Retriever：在 rag_children 中搜索，返回完整父文档。

    检索流程：
    1. 在 rag_children 中语义搜索 top-k 个子块
    2. 按 parent_id 去重分组
    3. 从 rag_parents 取出完整父文档
    4. 可选：通过 [[链接]] 一阶扩展检索
    """

    store: VectorStoreManager
    top_k: int = 10
    enable_link_expansion: bool = True
    filter_dict: dict | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        config = get_config()
        self.top_k = config.retrieval_top_k
        self.enable_link_expansion = config.enable_link_expansion

        # Step 1: 在子块中检索
        children = self.store.similarity_search(
            query, k=self.top_k, filter_dict=self.filter_dict
        )

        if not children:
            return []

        # Step 2: 按 parent_id 去重分组
        seen_parents: set[str] = set()
        for child in children:
            pid = child.metadata.get("parent_id", "")
            if pid:
                seen_parents.add(pid)

        # Step 3: 取出完整父文档
        parent_docs = self.store.get_parents_by_sources(list(seen_parents))

        # Step 4: 可选 — 链接扩展检索
        if self.enable_link_expansion:
            linked_docs = self._expand_by_links(parent_docs)
            # 合并去重
            existing_sources = {d.metadata.get("source") for d in parent_docs}
            for ld in linked_docs:
                if ld.metadata.get("source") not in existing_sources:
                    parent_docs.append(ld)
                    existing_sources.add(ld.metadata["source"])

        return parent_docs

    def _expand_by_links(self, parent_docs: list[Document]) -> list[Document]:
        """通过 [[链接]] 一阶扩展查找关联文档。"""
        all_links: set[str] = set()
        for doc in parent_docs:
            links = doc.metadata.get("links", [])
            all_links.update(links)

        if not all_links:
            return []

        # 将链接转为 source 路径格式（链接不带 .md 后缀）
        link_sources = []
        for link in all_links:
            # 链接可能是 "Docker/Docker 基础" 这种格式
            link_src = link if link.endswith(".md") else f"{link}.md"
            link_sources.append(link_src)

        return self.store.get_parents_by_sources(link_sources)
```

- [ ] **Step 2: 编写集成测试**

```python
# tests/test_retriever.py
import pytest
from langchain_core.documents import Document

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.retriever import ParentChildRetriever


@pytest.fixture
def populated_store(tmp_path):
    """创建一个已填充数据的 store。"""
    persist_dir = str(tmp_path / "chroma_ret_test")
    store = VectorStoreManager(persist_dir=persist_dir)

    parents = [
        Document(
            page_content="Docker 网络模式：bridge 是默认模式，host 模式性能最好。",
            metadata={
                "source": "Docker/Docker 网络.md",
                "filename": "Docker 网络",
                "folder": "Docker",
                "tags": ["docker"],
                "links": ["运维/容器化实践"],
                "doc_type": "parent",
            },
        ),
        Document(
            page_content="Python asyncio：事件循环和 async/await 语法详解。",
            metadata={
                "source": "Python/asyncio.md",
                "filename": "asyncio",
                "folder": "Python",
                "tags": ["python"],
                "links": [],
                "doc_type": "parent",
            },
        ),
    ]
    children = [
        Document(
            page_content="bridge 是 Docker 默认网络模式，容器通过 docker0 网桥通信。",
            metadata={
                "source": "Docker/Docker 网络.md",
                "doc_type": "child",
                "parent_id": "Docker/Docker 网络.md",
            },
        ),
        Document(
            page_content="事件循环是 asyncio 的核心概念，负责调度协程执行。",
            metadata={
                "source": "Python/asyncio.md",
                "doc_type": "child",
                "parent_id": "Python/asyncio.md",
            },
        ),
    ]
    store.add_parents(parents)
    store.add_children(children)
    return store


def test_retriever_finds_relevant_parent(populated_store):
    """检索 'Docker 网络' 应返回 Docker 相关的父文档。"""
    retriever = ParentChildRetriever(store=populated_store, top_k=5, enable_link_expansion=False)
    docs = retriever.invoke("Docker 网络模式")

    assert len(docs) >= 1
    sources = [d.metadata["source"] for d in docs]
    assert "Docker/Docker 网络.md" in sources


def test_retriever_returns_full_parent_content(populated_store):
    """返回的文档应该是完整父文档，不是子块。"""
    retriever = ParentChildRetriever(store=populated_store, top_k=5, enable_link_expansion=False)
    docs = retriever.invoke("事件循环")

    assert len(docs) >= 1
    for doc in docs:
        assert "事件循环和 async/await" in doc.page_content
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_retriever.py -v
```

- [ ] **Step 4: 提交**

```bash
git add rag_core/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: ParentChildRetriever — semantic search on children, return full parents"
```

---

### Task 8: RAG Chain 管线

**Files:**
- Create: `rag_core/retrieval/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 实现 prompt 模板和 format_docs**

```python
# rag_core/retrieval/pipeline.py
"""RAG Chain — LCEL 管线：检索 → 格式化 → Prompt → LLM → 解析。"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from rag_core.llm.deepseek import create_deepseek_llm
from rag_core.retrieval.retriever import ParentChildRetriever
from rag_core.indexing.store import VectorStoreManager
from config import get_config


SYSTEM_PROMPT = """你是个人知识库问答助手。根据用户笔记内容回答问题。
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
- 可以综合多篇笔记进行分析"""


def _format_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为 Prompt 使用的上下文字符串。"""
    if not docs:
        return "（未找到相关笔记）"

    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"--- 笔记 {i+1}: {source} ---\n{doc.page_content}\n")
    return "\n".join(parts)


def _format_history(history: list[dict]) -> str:
    """将对话历史格式化为字符串。"""
    if not history:
        return "（无历史对话）"
    lines = []
    for turn in history[-6:]:  # 最近 6 轮
        role = "用户" if turn["role"] == "user" else "助手"
        lines.append(f"{role}：{turn['content']}")
    return "\n".join(lines)


class RAGPipeline:
    """RAG 问答管线。

    用法:
        pipeline = RAGPipeline(store)
        answer = pipeline.ask("Docker 网络模式有哪些？")
    """

    def __init__(self, store: VectorStoreManager):
        self.store = store
        self.config = get_config()

        self.retriever = ParentChildRetriever(
            store=store,
            top_k=self.config.retrieval_top_k,
            enable_link_expansion=self.config.enable_link_expansion,
        )
        self.llm = create_deepseek_llm(streaming=True)

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        self.chain = (
            {
                "context": RunnableLambda(func=self._retrieve_and_format),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(func=lambda _: ""),  # 外部注入
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_and_format(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return _format_docs(docs)

    def ask(self, question: str, history: list[dict] | None = None) -> Any:
        """执行问答，返回 LangChain stream 对象。"""
        history = history or []
        input_data = {
            "context": self._retrieve_and_format(question),
            "question": question,
            "history": _format_history(history),
        }
        return self.chain.stream(input_data)

    def ask_with_filter(
        self,
        question: str,
        history: list[dict] | None = None,
        folder: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """带过滤条件的问答。"""
        history = history or []

        # 构建 filter
        filter_dict = None
        if folder:
            filter_dict = filter_dict or {}
            filter_dict["folder"] = folder
        if tag:
            filter_dict = filter_dict or {}
            # Chroma 的 $contains 操作符用于列表字段
            filter_dict["tags"] = {"$contains": tag}

        # 保存原始 filter 并临时替换
        original_filter = self.retriever.filter_dict
        self.retriever.filter_dict = filter_dict
        try:
            input_data = {
                "context": self._retrieve_and_format(question),
                "question": question,
                "history": _format_history(history),
            }
            return self.chain.stream(input_data)
        finally:
            self.retriever.filter_dict = original_filter
```

- [ ] **Step 2: 编写测试（Mock DeepSeek API）**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline, _format_docs, _format_history


def test_format_docs():
    docs = [
        Document(
            page_content="bridge 是默认网络模式。",
            metadata={"source": "Docker/Docker 网络.md"},
        ),
    ]
    result = _format_docs(docs)
    assert "Docker/Docker 网络.md" in result
    assert "bridge 是默认网络模式" in result


def test_format_docs_empty():
    assert "未找到相关笔记" in _format_docs([])


def test_format_history():
    history = [
        {"role": "user", "content": "什么是 Docker？"},
        {"role": "assistant", "content": "Docker 是一个容器化平台。"},
    ]
    result = _format_history(history)
    assert "什么是 Docker？" in result
    assert "容器化平台" in result


def test_format_history_empty():
    assert "无历史对话" in _format_history([])


def test_format_history_truncates_to_6_turns():
    """历史应该只保留最近 6 轮（12 条消息）。"""
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"问题 {i}"})
        history.append({"role": "assistant", "content": f"回答 {i}"})

    result = _format_history(history)
    # 只保留最近 6 轮 = 最后 12 条中的最近 6 个 turn
    assert "问题 0" not in result
    assert "问题 19" in result
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 4: 提交**

```bash
git add rag_core/retrieval/pipeline.py tests/test_pipeline.py
git commit -m "feat: RAG pipeline — LCEL chain with DeepSeek, parent-child retrieval"
```

---

### Task 9: 文件监听服务

**Files:**
- Create: `rag_core/watcher.py`

- [ ] **Step 1: 实现文件监听器**

```python
# rag_core/watcher.py
"""Obsidian 仓库文件监听 — watchdog 监控 .md 文件变更，自动更新 Chroma 索引。"""
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from rag_core.indexing.loader import ObsidianLoader
from rag_core.indexing.splitter import parent_child_split
from rag_core.indexing.store import VectorStoreManager
from config import get_config

logger = logging.getLogger(__name__)


class VaultSyncHandler(FileSystemEventHandler):
    """处理 Obsidian 仓库文件变更事件，增量更新 Chroma。"""

    def __init__(self, store: VectorStoreManager, vault_path: str, ignore_dirs: list[str]):
        self.store = store
        self.vault_path = vault_path
        self.ignore_dirs = ignore_dirs
        self._pending: dict[str, str] = {}  # source -> event_type
        self._debounce_seconds = 2.0
        self._last_event_time = 0.0

        self.config = get_config()

    def on_created(self, event):
        if self._should_handle(event):
            self._schedule("created", event.src_path)

    def on_modified(self, event):
        if self._should_handle(event):
            self._schedule("modified", event.src_path)

    def on_deleted(self, event):
        if self._should_handle(event):
            self._schedule("deleted", event.src_path)

    def _should_handle(self, event) -> bool:
        """判断是否应该处理此事件。"""
        if event.is_directory:
            return False
        if not event.src_path.endswith(".md"):
            return False
        # 检查是否在忽略目录中
        try:
            rel = Path(event.src_path).relative_to(self.vault_path)
        except ValueError:
            return False
        return not any(ignored in rel.parts for ignored in self.ignore_dirs)

    def _schedule(self, event_type: str, path: str):
        """将事件加入待处理队列并防抖。"""
        try:
            rel = Path(path).relative_to(self.vault_path)
            source = str(rel).replace("\\", "/")
        except ValueError:
            return

        self._pending[source] = event_type
        self._last_event_time = time.time()

    def process_pending(self):
        """处理所有待处理的文件变更（由外部定时器或手动调用）。"""
        if not self._pending:
            return

        now = time.time()
        if now - self._last_event_time < self._debounce_seconds:
            return  # 防抖中

        pending = dict(self._pending)
        self._pending.clear()

        loaded_docs = []
        deleted_sources = []

        for source, event_type in pending.items():
            if event_type == "deleted":
                deleted_sources.append(source)
            else:
                # created 或 modified → 重新加载
                filepath = Path(self.vault_path) / source
                if filepath.exists():
                    loader = ObsidianLoader(
                        vault_path=self.vault_path,
                        ignore_dirs=list(self.ignore_dirs),
                    )
                    # 只解析这一个文件
                    doc = loader._parse_file(filepath)
                    if doc is not None:
                        loaded_docs.append(doc)

        # 先删除
        for source in deleted_sources:
            self.store.delete_by_source(source)
            logger.info(f"已从索引中删除: {source}")

        # 再添加（先删旧的再添加新的）
        for doc in loaded_docs:
            source = doc.metadata["source"]
            self.store.delete_by_source(source)

            # 分块
            all_docs = parent_child_split(
                [doc],
                child_chunk_size=self.config.child_chunk_size,
                child_chunk_overlap=self.config.child_chunk_overlap,
                child_max_len=self.config.child_max_len_before_split,
            )
            parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
            children = [d for d in all_docs if d.metadata["doc_type"] == "child"]

            self.store.add_parents(parents)
            self.store.add_children(children)
            logger.info(f"已更新索引: {source}")


class VaultWatcher:
    """Obsidian 仓库文件监听器。

    启动时执行全量对比，然后持续监听变更。
    """

    def __init__(
        self,
        store: VectorStoreManager,
        vault_path: str,
        ignore_dirs: list[str] | None = None,
    ):
        self.store = store
        self.vault_path = vault_path
        self.ignore_dirs = ignore_dirs or [".obsidian", ".trash", ".git"]
        self.config = get_config()
        self._observer: Observer | None = None

    def full_sync(self) -> dict:
        """全量对比 — 扫描仓库并增量更新索引。

        Returns:
            包含同步统计的字典。
        """
        vault = Path(self.vault_path)
        if not vault.exists():
            raise FileNotFoundError(f"Obsidian 仓库路径不存在: {self.vault_path}")

        # 扫描当前仓库中的所有 .md 文件
        loader = ObsidianLoader(
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        current_docs = loader.load()

        current_sources = {doc.metadata["source"] for doc in current_docs}

        # 获取 Chroma 中已有的父文档
        stats = self.store.get_stats()

        new_count = 0
        updated_count = 0

        for doc in current_docs:
            source = doc.metadata["source"]
            existing = self.store.search_parents_by_source(source)

            if not existing:
                # 新文件
                self._index_document(doc)
                new_count += 1
            else:
                # 检查 mtime 是否变化
                existing_mtime = existing[0].metadata.get("mtime", "")
                if existing_mtime != doc.metadata.get("mtime", ""):
                    self.store.delete_by_source(source)
                    self._index_document(doc)
                    updated_count += 1

        # 清理已删除的文件
        # Chroma 中不再存在于仓库的文档
        # （这个操作比较昂贵，主要在 --rebuild 时做）
        # 增量同步时不做全量清理，由文件监听处理删除事件

        logger.info(
            f"同步完成: {new_count} 篇新增, {updated_count} 篇更新, "
            f"总计 {len(current_docs)} 篇笔记"
        )

        return {
            "total": len(current_docs),
            "new": new_count,
            "updated": updated_count,
        }

    def rebuild(self) -> dict:
        """全量重建索引。"""
        vault = Path(self.vault_path)
        if not vault.exists():
            raise FileNotFoundError(f"Obsidian 仓库路径不存在: {self.vault_path}")

        loader = ObsidianLoader(
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        docs = loader.load()

        all_docs = []
        for doc in docs:
            split_docs = parent_child_split(
                [doc],
                child_chunk_size=self.config.child_chunk_size,
                child_chunk_overlap=self.config.child_chunk_overlap,
                child_max_len=self.config.child_max_len_before_split,
            )
            all_docs.extend(split_docs)

        parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
        children = [d for d in all_docs if d.metadata["doc_type"] == "child"]

        self.store.rebuild(parents, children)

        logger.info(f"全量重建完成: {len(parents)} 篇父文档, {len(children)} 个子块")

        return {
            "total_parents": len(parents),
            "total_children": len(children),
        }

    def start_watching(self):
        """启动文件监听（后台线程）。"""
        event_handler = VaultSyncHandler(
            store=self.store,
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        self._observer = Observer()
        self._observer.schedule(event_handler, self.vault_path, recursive=True)
        self._observer.start()
        logger.info(f"开始监听 Obsidian 仓库: {self.vault_path}")

        # 启动定时防抖处理线程
        import threading
        def _process_loop():
            while self._observer and self._observer.is_alive():
                time.sleep(1)
                try:
                    event_handler.process_pending()
                except Exception as e:
                    logger.error(f"处理文件变更时出错: {e}")

        threading.Thread(target=_process_loop, daemon=True).start()

    def stop_watching(self):
        """停止文件监听。"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def _index_document(self, doc):
        """索引单篇文档。"""
        all_docs = parent_child_split(
            [doc],
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
            child_max_len=self.config.child_max_len_before_split,
        )
        parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
        children = [d for d in all_docs if d.metadata["doc_type"] == "child"]
        self.store.add_parents(parents)
        self.store.add_children(children)
```

- [ ] **Step 2: 提交**

```bash
git add rag_core/watcher.py
git commit -m "feat: VaultWatcher — watchdog file monitoring, full sync, incremental updates"
```

---

### Task 10: FastAPI 服务

**Files:**
- Create: `rag_server/__init__.py`
- Create: `rag_server/app.py`
- Create: `rag_server/chat.py`

- [ ] **Step 1: 实现 chat.py（SSE 端点）**

```python
# rag_server/chat.py
"""聊天端点 — SSE 流式问答。"""
import json
import asyncio
from typing import AsyncIterator

from sse_starlette.sse import EventSourceResponse
from fastapi import Request

from rag_core.retrieval.pipeline import RAGPipeline

# 会话内存存储: session_id -> history list
_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    # 清理旧会话（保留最近 100 个 session）
    if len(_sessions) > 100:
        oldest = list(_sessions.keys())[0]
        del _sessions[oldest]
    return _sessions[session_id]


async def chat_stream(
    pipeline: RAGPipeline,
    question: str,
    session_id: str = "default",
    folder: str | None = None,
    tag: str | None = None,
) -> AsyncIterator[str]:
    """SSE 流式返回答案。"""
    history = get_history(session_id)

    try:
        if folder or tag:
            stream = pipeline.ask_with_filter(question, history, folder=folder, tag=tag)
        else:
            stream = pipeline.ask(question, history)

        full_answer = ""
        # 发送 thinking 事件
        yield _sse_event("thinking", {"message": "正在检索笔记..."})

        for chunk in stream:
            if chunk:
                full_answer += chunk
                yield _sse_event("token", {"text": chunk})

        # 保存对话历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        if len(history) > 20:  # 保留最近 10 轮
            _sessions[session_id] = history[-20:]

        # 发送完成事件
        yield _sse_event("done", {"full_text": full_answer})

    except Exception as e:
        yield _sse_event("error", {"message": str(e)})


def _sse_event(event: str, data: dict) -> str:
    """构建 SSE 事件字符串。"""
    return json.dumps({"event": event, "data": data}, ensure_ascii=False)
```

- [ ] **Step 2: 实现 app.py（FastAPI 应用）**

```python
# rag_server/app.py
"""FastAPI 应用 — RAG 知识库助手后端服务。"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pathlib import Path

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from rag_server.chat import chat_stream, get_history
from config import get_config, Config

# 全局状态
_pipeline: RAGPipeline | None = None
_store: VectorStoreManager | None = None
_watcher: VaultWatcher | None = None


def init_app(config: Config | None = None):
    """初始化全局组件。"""
    global _pipeline, _store, _watcher
    cfg = config or get_config()

    _store = VectorStoreManager(persist_dir=cfg.chroma_persist_dir)
    _pipeline = RAGPipeline(store=_store)

    if cfg.obsidian_vault_path:
        _watcher = VaultWatcher(
            store=_store,
            vault_path=cfg.obsidian_vault_path,
            ignore_dirs=list(cfg.obsidian_ignore_dirs),
        )
        try:
            _watcher.full_sync()
        except FileNotFoundError:
            pass  # 仓库路径无效，稍后由用户配置

        try:
            _watcher.start_watching()
        except Exception:
            pass


def create_app(config: Config | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""
    cfg = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_app(config)
        yield
        if _watcher:
            _watcher.stop_watching()

    app = FastAPI(title="RAG 知识库助手", version="0.1.0", lifespan=lifespan)

    # 静态文件
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        """返回聊天界面。"""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"message": "RAG 知识库助手 API 服务运行中"}, status_code=200)

    @app.post("/api/chat")
    async def chat(request: Request):
        """SSE 流式聊天端点。"""
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空")

        session_id = body.get("session_id", "default")
        folder = body.get("folder")
        tag = body.get("tag")

        if _pipeline is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

        async def event_generator():
            async for event_str in chat_stream(
                _pipeline, question, session_id=session_id, folder=folder, tag=tag
            ):
                yield event_str

        return EventSourceResponse(event_generator())

    @app.get("/api/status")
    async def status():
        """返回索引状态。"""
        if _store is None:
            return JSONResponse({"status": "not_initialized"}, status_code=200)

        stats = _store.get_stats()
        return JSONResponse({
            "status": "ok",
            "index": stats,
            "vault_path": cfg.obsidian_vault_path,
        })

    @app.post("/api/reindex")
    async def reindex():
        """触发全量重建索引。"""
        if _watcher is None:
            raise HTTPException(status_code=400, detail="未配置 Obsidian 仓库路径")

        try:
            result = _watcher.rebuild()
            return JSONResponse({"status": "ok", **result})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sources/{source:path}")
    async def get_source(source: str):
        """获取笔记原文。"""
        if _store is None:
            raise HTTPException(status_code=500, detail="服务未初始化")

        docs = _store.search_parents_by_source(source)
        if not docs:
            raise HTTPException(status_code=404, detail=f"未找到笔记: {source}")

        doc = docs[0]
        return JSONResponse({
            "source": doc.metadata.get("source"),
            "content": doc.page_content,
            "metadata": doc.metadata,
        })

    return app


def run_server(host: str = "127.0.0.1", port: int = 8501, no_watch: bool = False):
    """启动 uvicorn 服务器。"""
    import uvicorn

    if no_watch and _watcher:
        _watcher.stop_watching()

    uvicorn.run(
        "rag_server.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="info",
    )
```

- [ ] **Step 3: 提交**

```bash
git add rag_server/app.py rag_server/chat.py
git commit -m "feat: FastAPI server — SSE chat endpoint, status, reindex, source lookup"
```

---

### Task 11: Web 前端

**Files:**
- Create: `rag_server/static/index.html`
- Create: `rag_server/static/style.css`

- [ ] **Step 1: 创建 style.css**

```css
/* rag_server/static/style.css */
:root {
    --bg: #ffffff;
    --bg-secondary: #f5f5f5;
    --text: #1a1a1a;
    --text-secondary: #666666;
    --border: #e0e0e0;
    --accent: #4f46e5;
    --accent-hover: #4338ca;
    --user-bubble: #e8e5ff;
    --assistant-bubble: #f5f5f5;
    --danger: #dc2626;
    --radius: 12px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1a1a2e;
        --bg-secondary: #16213e;
        --text: #e0e0e0;
        --text-secondary: #a0a0a0;
        --border: #2a2a4a;
        --accent: #6366f1;
        --user-bubble: #2a2550;
        --assistant-bubble: #1e1e3a;
    }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-secondary);
}

.header h1 { font-size: 18px; }

.status-bar {
    font-size: 12px;
    color: var(--text-secondary);
}

/* Chat area */
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 14px;
}

.message.user {
    align-self: flex-end;
    background: var(--user-bubble);
}

.message.assistant {
    align-self: flex-start;
    background: var(--assistant-bubble);
    border: 1px solid var(--border);
}

.message .sources {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-secondary);
}

.message .sources a {
    color: var(--accent);
    text-decoration: none;
    cursor: pointer;
}

.message .sources a:hover { text-decoration: underline; }

/* Typing indicator */
.typing-indicator {
    align-self: flex-start;
    padding: 12px 16px;
    color: var(--text-secondary);
    font-size: 14px;
    font-style: italic;
}

/* Input area */
.input-area {
    padding: 12px 20px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 8px;
    background: var(--bg-secondary);
}

.input-area input {
    flex: 1;
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 24px;
    font-size: 14px;
    background: var(--bg);
    color: var(--text);
    outline: none;
}

.input-area input:focus { border-color: var(--accent); }

.input-area button {
    padding: 10px 20px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 24px;
    font-size: 14px;
    cursor: pointer;
}

.input-area button:hover { background: var(--accent-hover); }
.input-area button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Settings panel */
.settings-panel {
    position: fixed;
    top: 0; right: 0;
    width: 360px;
    height: 100vh;
    background: var(--bg);
    border-left: 1px solid var(--border);
    padding: 24px;
    z-index: 100;
    overflow-y: auto;
}

.settings-panel h2 {
    font-size: 16px;
    margin-bottom: 16px;
}

.settings-panel label {
    display: block;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    margin-top: 12px;
}

.settings-panel input, .settings-panel select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    background: var(--bg-secondary);
    color: var(--text);
}

.settings-panel button {
    width: 100%;
    margin-top: 16px;
    padding: 10px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}

/* Overlay */
.overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.3);
    z-index: 99;
}

/* Markdown content styling */
.message.assistant p { margin-bottom: 8px; }
.message.assistant p:last-child { margin-bottom: 0; }
.message.assistant code {
    background: var(--border);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}
.message.assistant pre {
    background: var(--border);
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
}
.message.assistant ul, .message.assistant ol { padding-left: 20px; }
```

- [ ] **Step 2: 创建 index.html**

```html
<!-- rag_server/static/index.html -->
<!DOCTYPE html>
<html lang="zh-CN" x-data="chatApp()">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 知识库助手</title>
    <link rel="stylesheet" href="/static/style.css">
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <h1>📚 我的知识库助手</h1>
        <div>
            <span class="status-bar" x-text="statusText">连接中...</span>
            <button @click="showSettings = !showSettings" style="background:none;border:none;cursor:pointer;font-size:18px;margin-left:12px;">⚙️</button>
        </div>
    </header>

    <!-- Chat messages -->
    <div class="chat-container" x-ref="chatContainer" id="chatContainer">
        <template x-if="messages.length === 0">
            <div class="message assistant">
                🤖 你好，我是你的知识库助手。<br>可以问我关于你笔记中的任何问题。
            </div>
        </template>

        <template x-for="msg in messages" :key="msg.id">
            <div class="message" :class="msg.role">
                <div x-html="renderMarkdown(msg.content)"></div>
                <div class="sources" x-show="msg.sources && msg.sources.length > 0">
                    📄 来源：
                    <template x-for="src in msg.sources">
                        <a @click="viewSource(src)" x-text="src" style="margin-right:12px;"></a>
                    </template>
                </div>
            </div>
        </template>

        <div class="typing-indicator" x-show="loading" x-text="typingText">...</div>
    </div>

    <!-- Input area -->
    <div class="input-area">
        <input
            type="text"
            x-model="question"
            placeholder="输入你的问题..."
            @keydown.enter="sendMessage"
            :disabled="loading"
        />
        <button @click="sendMessage" :disabled="loading || !question.trim()">发送</button>
    </div>

    <!-- Settings overlay -->
    <template x-if="showSettings">
        <div>
            <div class="overlay" @click="showSettings = false"></div>
            <div class="settings-panel">
                <h2>⚙️ 设置</h2>
                <label>Obsidian 仓库路径</label>
                <input type="text" x-model="settings.vaultPath" placeholder="C:/Users/.../ObsidianVault">

                <label>DeepSeek API Key</label>
                <input type="password" x-model="settings.apiKey" placeholder="sk-...">

                <label>检索数量 (top-k)</label>
                <input type="number" x-model="settings.topK" min="1" max="50">

                <button @click="saveSettings()">保存设置</button>
                <button @click="rebuildIndex()" style="background:var(--danger);margin-top:8px;">🔄 重建索引</button>
            </div>
        </div>
    </template>

    <script>
    function chatApp() {
        return {
            messages: [],
            question: '',
            loading: false,
            typingText: '',
            showSettings: false,
            sessionId: 'web_' + Date.now(),
            settings: {
                vaultPath: localStorage.getItem('vaultPath') || '',
                apiKey: localStorage.getItem('apiKey') || '',
                topK: parseInt(localStorage.getItem('topK') || '10'),
            },
            statusText: '连接中...',

            async init() {
                await this.fetchStatus();
                setInterval(() => this.fetchStatus(), 60000);
            },

            async fetchStatus() {
                try {
                    const res = await fetch('/api/status');
                    const data = await res.json();
                    if (data.status === 'ok' && data.index) {
                        this.statusText = `已索引 ${data.index.parent_count} 篇笔记 · 最近同步 ${new Date(data.index.last_sync).toLocaleTimeString('zh-CN')}`;
                    }
                } catch (e) {
                    this.statusText = '服务未连接';
                }
            },

            async sendMessage() {
                const q = this.question.trim();
                if (!q || this.loading) return;

                const userMsg = { id: Date.now(), role: 'user', content: q, sources: [] };
                this.messages.push(userMsg);
                this.question = '';
                this.loading = true;
                this.typingText = '🔍 正在检索笔记...';
                this.scrollToBottom();

                const assistantMsg = { id: Date.now() + 1, role: 'assistant', content: '', sources: [] };
                this.messages.push(assistantMsg);

                try {
                    const resp = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            question: q,
                            session_id: this.sessionId,
                        }),
                    });

                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || '';

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const event = JSON.parse(line.slice(6));
                                    // event 可能是 SSE 格式 {event, data}
                                    const payload = typeof event === 'object' ? (event.data || event) : event;

                                    if (payload.event === 'thinking' || event.event === 'thinking') {
                                        this.typingText = (event.data || payload).message || '...';
                                    } else if (payload.event === 'token' || event.event === 'token') {
                                        this.typingText = '';
                                        assistantMsg.content += (event.data || payload).text || '';
                                    } else if (payload.event === 'done' || event.event === 'done') {
                                        this.loading = false;
                                    } else if (payload.event === 'error' || event.event === 'error') {
                                        assistantMsg.content = '⚠️ 出错了：' + ((event.data || payload).message || '');
                                        this.loading = false;
                                    }
                                } catch (e) {
                                    // ignore parse errors for partial chunks
                                }
                            }
                        }
                        this.scrollToBottom();
                    }
                } catch (e) {
                    assistantMsg.content = '⚠️ 网络错误：' + e.message;
                    this.loading = false;
                }

                this.loading = false;
                this.typingText = '';
                this.scrollToBottom();
            },

            scrollToBottom() {
                this.$nextTick(() => {
                    const container = this.$refs.chatContainer;
                    if (container) container.scrollTop = container.scrollHeight;
                });
            },

            renderMarkdown(text) {
                if (!text) return '';
                try {
                    return marked.parse(text);
                } catch (e) {
                    return text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                }
            },

            async viewSource(source) {
                try {
                    const resp = await fetch('/api/sources/' + encodeURIComponent(source));
                    const data = await resp.json();
                    alert(`[${data.source}]\n\n${data.content.substring(0, 1000)}...`);
                } catch (e) {
                    alert('无法加载笔记：' + e.message);
                }
            },

            async saveSettings() {
                localStorage.setItem('vaultPath', this.settings.vaultPath);
                localStorage.setItem('apiKey', this.settings.apiKey);
                localStorage.setItem('topK', this.settings.topK);
                this.showSettings = false;
                alert('设置已保存（部分设置需要重启服务生效）');
            },

            async rebuildIndex() {
                if (!confirm('确定要重建整个索引吗？这可能需要几分钟。')) return;
                try {
                    const resp = await fetch('/api/reindex', { method: 'POST' });
                    const data = await resp.json();
                    alert(`索引重建完成：${data.total_parents} 篇父文档，${data.total_children} 个子块`);
                    await this.fetchStatus();
                } catch (e) {
                    alert('重建失败：' + e.message);
                }
            },
        };
    }
    </script>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add rag_server/static/
git commit -m "feat: Web frontend — Alpine.js chat UI with SSE streaming, settings, dark mode"
```

---

### Task 12: CLI 入口

**Files:**
- Create: `rag_cli/__init__.py`
- Create: `rag_cli/main.py`

- [ ] **Step 1: 实现 CLI**

```python
# rag_cli/main.py
"""CLI 入口 — Typer 命令：rag ask, rag index, rag server。"""
import subprocess
import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from rag_core.indexing.store import VectorStoreManager
from rag_core.retrieval.pipeline import RAGPipeline
from rag_core.watcher import VaultWatcher
from config import get_config, Config

app = typer.Typer(
    name="rag",
    help="个人知识库 RAG 问答助手",
    no_args_is_help=True,
)
console = Console()


def _ensure_server_running(config: Config) -> bool:
    """检查后端服务是否在运行，未运行则自动启动。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((config.server_host, config.server_port))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        s.close()
        return False


@app.command()
def ask(
    question: str = typer.Argument(..., help="你的问题"),
    no_history: bool = typer.Option(False, "--no-history", help="不使用对话历史"),
    folder: Optional[str] = typer.Option(None, "--folder", help="限定搜索文件夹"),
    tag: Optional[str] = typer.Option(None, "--tag", help="限定搜索标签"),
):
    """向知识库提问（流式输出）。"""
    config = get_config()

    if not config.obsidian_vault_path:
        console.print("[red]❌ 请先设置 OBSIDIAN_VAULT_PATH 环境变量或 .env 文件[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]📚 仓库: {config.obsidian_vault_path}[/dim]")

    # 初始化
    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    pipeline = RAGPipeline(store=store)

    # 增量同步
    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )
    try:
        result = watcher.full_sync()
        console.print(f"[dim]📊 索引: {result['total']} 篇笔记[/dim]")
    except FileNotFoundError:
        console.print(f"[red]❌ Obsidian 仓库路径不存在: {config.obsidian_vault_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]🔍 正在检索...[/dim]\n")

    history: list[dict] = [] if no_history else []

    try:
        full_answer = ""
        if folder or tag:
            stream = pipeline.ask_with_filter(question, history, folder=folder, tag=tag)
        else:
            stream = pipeline.ask(question, history)

        for chunk in stream:
            if chunk:
                full_answer += chunk
                console.print(chunk, end="")

        console.print("\n")

        if not no_history:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})

    except Exception as e:
        console.print(f"\n[red]❌ 错误: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="全量重建索引"),
    sync: bool = typer.Option(False, "--sync", help="增量同步一次"),
    status: bool = typer.Option(False, "--status", help="查看索引状态"),
):
    """管理知识库索引。"""
    config = get_config()

    if status:
        store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
        stats = store.get_stats()
        console.print(f"📊 索引状态:")
        console.print(f"   父文档数: {stats['parent_count']}")
        console.print(f"   子块数:   {stats['child_count']}")
        console.print(f"   最近同步: {stats['last_sync']}")
        return

    if not config.obsidian_vault_path:
        console.print("[red]❌ 请先设置 OBSIDIAN_VAULT_PATH[/red]")
        raise typer.Exit(code=1)

    store = VectorStoreManager(persist_dir=config.chroma_persist_dir)
    watcher = VaultWatcher(
        store=store,
        vault_path=config.obsidian_vault_path,
        ignore_dirs=list(config.obsidian_ignore_dirs),
    )

    if rebuild:
        console.print("🔄 正在全量重建索引...")
        result = watcher.rebuild()
        console.print(f"[green]✅ 索引重建完成: {result['total_parents']} 篇父文档, {result['total_children']} 个子块[/green]")
    elif sync:
        console.print("🔄 正在增量同步...")
        result = watcher.full_sync()
        console.print(f"[green]✅ 同步完成: {result['total']} 篇笔记 ({result['new']} 新增, {result['updated']} 更新)[/green]")


@app.command()
def server(
    port: int = typer.Option(8501, "--port", help="服务端口"),
    host: str = typer.Option("127.0.0.1", "--host", help="绑定地址"),
    no_watch: bool = typer.Option(False, "--no-watch", help="不启动文件监听"),
):
    """启动 Web 服务。"""
    config = get_config()
    config.server_port = port
    config.server_host = host

    console.print(f"🚀 启动 RAG 知识库助手服务...")
    console.print(f"   Web 界面: http://{host}:{port}")
    console.print(f"   API:      http://{host}:{port}/api/chat")

    from rag_server.app import run_server
    run_server(host=host, port=port, no_watch=no_watch)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: 提交**

```bash
git add rag_cli/main.py
git commit -m "feat: CLI — Typer commands: rag ask, rag index, rag server"
```

---

### Task 13: 集成验证与收尾

- [ ] **Step 1: 验证 import 链**

```bash
python -c "from rag_core.indexing.loader import ObsidianLoader; print('OK loader')"
python -c "from rag_core.indexing.splitter import parent_child_split; print('OK splitter')"
python -c "from rag_core.indexing.embedder import create_embedder; print('OK embedder')"
python -c "from rag_core.indexing.store import VectorStoreManager; print('OK store')"
python -c "from rag_core.retrieval.retriever import ParentChildRetriever; print('OK retriever')"
python -c "from rag_core.retrieval.pipeline import RAGPipeline; print('OK pipeline')"
python -c "from rag_core.watcher import VaultWatcher; print('OK watcher')"
python -c "from rag_core.llm.deepseek import create_deepseek_llm; print('OK LLM')"
```

- [ ] **Step 2: 运行所有单元测试**

```bash
pytest tests/ -v
```

- [ ] **Step 3: 验证 CLI help**

```bash
python -m rag_cli.main --help
```

Expected: 显示 Typer 生成的帮助信息，包含 ask、index、server 三个子命令。

- [ ] **Step 4: 创建 .env 文件供用户配置**

```bash
cp .env.example .env
```

提示用户编辑 .env 填入 DeepSeek API Key 和 Obsidian 仓库路径。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: complete RAG knowledge assistant — all modules integrated"
```
