"""父子分块逻辑，生成可读且可稳定复现的证据 anchor。"""
import hashlib
import re
import unicodedata
from langchain_core.documents import Document


INDEX_VERSION = 2


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
    for doc in documents:
        source = doc.metadata.get("source", "")
        # 复制除 doc_type 外的元数据
        base_meta = {k: v for k, v in doc.metadata.items() if k != "doc_type"}

        # 父文档
        parent = Document(
            page_content=doc.page_content,
            metadata={
                **base_meta,
                "doc_type": "parent",
                "index_version": INDEX_VERSION,
            },
        )
        result.append(parent)

        # 子文档按 Markdown 标题切分；标题会作为可读 section_title。
        sections = _split_by_heading(doc.page_content)
        used_anchors: set[str] = set()

        for heading, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            base_anchor = _section_anchor(source, heading, section_text)
            base_anchor = _unique_anchor(
                base_anchor, source, heading, section_text, used_anchors
            )

            if len(section_text) <= child_max_len:
                child = Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "doc_type": "child",
                        "parent_id": source,
                        "anchor": base_anchor,
                        "heading": heading,
                        "section_title": heading,
                        "chunk_index": 0,
                        "index_version": INDEX_VERSION,
                    },
                )
                result.append(child)
            else:
                # 长段落二次切分
                sub_chunks = _split_long_text(
                    section_text, child_chunk_size, child_chunk_overlap
                )
                for chunk_index, sub_text in enumerate(sub_chunks):
                    chunk_anchor = (
                        f"{base_anchor}-{_stable_hash(source, heading, sub_text)}"
                    )
                    child = Document(
                        page_content=sub_text,
                        metadata={
                            **base_meta,
                            "doc_type": "child",
                            "parent_id": source,
                            "anchor": chunk_anchor,
                            "heading": heading,
                            "section_title": heading,
                            "chunk_index": chunk_index,
                            "index_version": INDEX_VERSION,
                        },
                    )
                    result.append(child)

    return result


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切分，并使用最近标题作为段落标题。"""
    sections: list[tuple[str, str]] = []
    current_section: list[str] = []
    current_heading = ""

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", stripped)
        if match:
            if current_section:
                sections.append((current_heading, "\n".join(current_section)))
            current_heading = match.group(1).strip()
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append((current_heading, "\n".join(current_section)))

    return sections


# 旧名称保留给可能存在的内部调用方。
_split_by_h2 = _split_by_heading


def _slugify(value: str) -> str:
    """Create a stable Markdown-style anchor while retaining CJK characters."""
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff -]", "", normalized)
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized


def _stable_hash(*values: str) -> str:
    payload = "\0".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按自然边界切长文本，避免为基础切块加载嵌入模型依赖。"""
    chunks: list[str] = []
    start = 0
    separators = ("\n\n", "\n", "。", ".", " ")
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            lower_bound = start + max(chunk_size // 2, 1)
            best_end = -1
            for separator in separators:
                position = text.rfind(separator, lower_bound, end)
                if position >= 0:
                    best_end = max(best_end, position + len(separator))
            if best_end > start:
                end = best_end
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _section_anchor(source: str, heading: str, content: str) -> str:
    heading_slug = _slugify(heading)
    if heading_slug:
        return heading_slug

    first_text = next(
        (
            re.sub(r"^[#>*`\-\s]+", "", line).strip()
            for line in content.splitlines()
            if re.sub(r"^[#>*`\-\s]+", "", line).strip()
        ),
        "段落",
    )
    summary = _slugify(first_text[:24]) or "段落"
    return f"{summary}-{_stable_hash(source, content)}"


def _unique_anchor(
    anchor: str,
    source: str,
    heading: str,
    content: str,
    used: set[str],
) -> str:
    if anchor not in used:
        used.add(anchor)
        return anchor
    candidate = f"{anchor}-{_stable_hash(source, heading, content)}"
    suffix_length = 8
    while candidate in used:
        suffix_length += 2
        digest = hashlib.sha256(
            f"{source}\0{heading}\0{content}".encode("utf-8")
        ).hexdigest()[:suffix_length]
        candidate = f"{anchor}-{digest}"
    used.add(candidate)
    return candidate
