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
    assert docker_doc.metadata["doc_type"] == "raw"


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


def test_loader_inline_tags(temp_vault_with_tags):
    """Loader 应该同时提取 frontmatter 和内联标签。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault_with_tags),
        ignore_dirs=[],
    )
    docs = loader.load()
    assert len(docs) == 1

    tags = docs[0].metadata["tags"]
    assert "frontmatter_tag" in tags
    assert "inline_tag" in tags
    assert "python/async" in tags


def test_lazy_load_yields_documents(temp_vault):
    """lazy_load 应该逐个产出 Document。"""
    loader = ObsidianLoader(
        vault_path=str(temp_vault),
        ignore_dirs=[".obsidian", ".trash"],
    )
    docs = list(loader.lazy_load())
    assert len(docs) == 2
