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


@pytest.fixture
def temp_vault_with_tags():
    """创建一个包含内联标签的 Obsidian 仓库。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        note = vault / "tags 测试.md"
        note.write_text("""---
tags: [frontmatter_tag]
---

# 标签测试

这里有一个 #inline_tag 在内文中。

还有一个 #python/async 的嵌套标签。
""", encoding="utf-8")
        yield vault
