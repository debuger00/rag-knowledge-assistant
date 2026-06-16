# Python 命令行入口：console_scripts 与 pip install 原理

> 讲解 `rag ask` 这种命令是怎么来的，和 `python main.py` 的等价关系。

---

## 1. 三种启动方式等价

以本项目为例，以下三种写法完全等价：

```bash
# 方式1：直接敲命令（需要先 pip install）
rag ask "什么是RAG?"

# 方式2：用 python -m 运行模块
python -m rag_cli.main ask "什么是RAG?"

# 方式3：直接运行 .py 文件
python rag_cli/main.py ask "什么是RAG?"
```

三种方式最终都是：**找到 `rag_cli/main.py` 里的 `app` 对象，调用 `app()`**。

区别在于方式1不需要 cd 到项目目录，在任何地方都能用。

---

## 2. 谁生成了 rag.exe？

答案是 `pyproject.toml` 里的这段配置：

```toml
[project.scripts]
rag = "rag_cli.main:app"
#  ↑         ↑         ↑
#  命令名    模块路径    对象名
```

当你运行 `pip install -e .` 时，setuptools 做了两件事：

1. 在 Python 环境的 `Scripts/` 目录下生成 `rag.exe`（Windows）或 `rag` 脚本（Linux/Mac）
2. 把这个目录加到系统 PATH 里（如果不在 PATH 中则命令行找不到 `rag`）

**生成的文件在哪？**

```powershell
# Windows（本项目为例）
E:\program\agent\02bankSuperpowers\.venv\Scripts\rag.exe   # 约 108KB

# Linux / Mac
~/project/.venv/bin/rag                         # 几 KB 的 shell 脚本
```

---

## 3. rag.exe 里是什么？

本质上就是一个极简的启动器，等价于：

```python
# rag.exe 内部做的事（简化版）
import sys
from rag_cli.main import app

if __name__ == "__main__":
    sys.exit(app())   # app 是 Typer 实例，app() 解析命令行参数
```

它做的事情：
1. 从 `sys.argv` 拿到命令行参数（如 `["rag", "ask", "什么是RAG"]`）
2. 导入 `rag_cli.main` 模块中的 `app` 对象
3. 调用 `app()`，Typer 内部解析参数，匹配子命令，执行对应函数

---

## 4. 格式详解

```
rag = "rag_cli.main:app"
 ↑         ↑            ↑
 命令名    模块路径      对象名（用冒号分隔）
```

| 你想敲的命令 | pyproject.toml 写法 | 含义 |
|---|---|---|
| `rag` | `rag = "rag_cli.main:app"` | 调 `rag_cli.main` 的 `app` |
| `pytest` | `pytest = "pytest:console_main"` | 调 `pytest` 包的 `console_main` |
| `black` | `black = "black:patched_main"` | 调 `black` 包的 `patched_main` |
| `uvicorn` | `uvicorn = "uvicorn.main:main"` | 调 `uvicorn.main` 模块的 `main` |

**对象只要是 callable 就行**，可以是函数，也可以是 Typer/Click 实例（它们内部实现了 `__call__`）。

---

## 5. editable 模式（-e）

```bash
pip install -e .
```

`-e` 是 `--editable` 的缩写。它的作用是：

```
普通安装 (pip install .)：
    把代码复制到 site-packages/  →  改代码后必须重新 pip install

Editable 安装 (pip install -e .)：
    在 site-packages/ 里放一个"链接"指向你的项目目录  →  改代码立即生效
```

**这对开发阶段极为重要** —— 你改了 `rag_cli/main.py`，不需要重新 pip install，直接再敲 `rag ask` 就能看到效果。

验证方法：看 site-packages 里的 `.pth` 文件：

```powershell
# 安装后这里会多一个指向项目目录的路径文件
ls .venv/Lib/site-packages/*.egg-link       # 或 *.pth
cat .venv/Lib/site-packages/rag-assistant.egg-link
# 内容: E:\program\agent\02bankSuperpowers
```

---

## 6. 完整流程图

```
pip install -e .
     │
     ├─→ ① 在 site-packages/ 创建 rag-assistant.egg-link
     │     内容: E:\program\agent\02bankSuperpowers
     │     Python import 时顺着这个链接找到你的代码
     │
     ├─→ ② 读 pyproject.toml [project.scripts]
     │     rag = "rag_cli.main:app"
     │     生成 .venv\Scripts\rag.exe
     │
     └─→ ③ 现在你可以在终端敲 rag 了：
     │
     │     > rag ask "什么是RAG？"
     │       │
     │       ▼
     │     .venv\Scripts\rag.exe
     │       │
     │       ▼
     │     from rag_cli.main import app    ← site-packages 的 egg-link 让 Python 找到它
     │       │
     │       ▼
     │     app()                            ← Typer 解析参数、执行子命令
     │       │
     │       ▼
     │     ask() 函数执行                    ← 流式输出答案
```

---

## 7. 常见问题

### Q: 为什么我装了 pip install -e . 但敲 rag 报 "command not found"？

A: `.venv/Scripts/` 没在 PATH 里。需要先激活虚拟环境：

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

激活后 `which rag` 就能看到路径。

### Q: 如果多项目都叫 rag 怎么办？

A: 哪个虚拟环境被激活，就用哪个。不激活虚拟环境时用的是全局 Python 的 Scripts。

### Q: 怎么取消这个命令？

```bash
pip uninstall rag-assistant   # 会删掉 rag.exe 和 egg-link
```

### Q: 我之前的项目都是 python main.py，没有 pyproject.toml，怎么加？

如果你用的是 `argparse`（标准库），可以改成 console_scripts 格式：

```toml
# pyproject.toml
[project.scripts]
my-tool = "my_package.cli:main"      # main 是个函数，不是 Typer 实例
```

然后在代码里确保 `main()` 函数解析命令行参数并执行。Typer/Click 只是让这个过程更优雅，原理相同。

---

## 8. 总结

| 概念 | 一句话 |
|---|---|
| `[project.scripts]` | pyproject.toml 里定义命令入口的配置段 |
| `rag = "rag_cli.main:app"` | 冒号左边=模块路径，右边=要调用的对象 |
| `pip install -e .` | 生成 `rag.exe` + egg-link，改代码即时生效 |
| 等价关系 | `rag ask "xxx"` = `python -m rag_cli.main ask "xxx"` |
| Typer 的角色 | 让定义子命令和参数更简洁，但它不是必须的——函数也行 |
