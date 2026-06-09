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
