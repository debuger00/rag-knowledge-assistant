"""父子分块逻辑 — 按 ## 标题拆分子块，保留完整父文档。"""
import re
import unicodedata
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
        # 复制除 doc_type 外的元数据
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
        used_anchors: dict[str, int] = {}

        for heading, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            base_anchor = _unique_anchor(_slugify(heading), used_anchors)

            if len(section_text) <= child_max_len:
                child = Document(
                    page_content=section_text,
                    metadata={
                        **base_meta,
                        "doc_type": "child",
                        "parent_id": source,
                        "anchor": base_anchor,
                        "heading": heading,
                        "chunk_index": 0,
                    },
                )
                result.append(child)
            else:
                # 长段落二次切分
                sub_chunks = text_splitter.split_text(section_text)
                for chunk_index, sub_text in enumerate(sub_chunks):
                    child = Document(
                        page_content=sub_text,
                        metadata={
                            **base_meta,
                            "doc_type": "child",
                            "parent_id": source,
                            "anchor": (
                                base_anchor
                                if chunk_index == 0
                                else f"{base_anchor}-part-{chunk_index + 1}"
                            ),
                            "heading": heading,
                            "chunk_index": chunk_index,
                        },
                    )
                    result.append(child)

    return result


def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """按 ## 标题切分文本，返回各段落列表。

    以 ## 开头的行（但非 ### 开头）作为切分边界。
    不将 # 一级标题作为切分边界——Obsidian 中一级标题通常是文件标题。
    """
    sections: list[tuple[str, str]] = []
    current_section: list[str] = []
    current_heading = "document-start"

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        # 匹配 ## 开头但非 ### 开头
        if re.match(r"^## [^#]", stripped):
            if current_section:
                sections.append((current_heading, "\n".join(current_section)))
            current_heading = stripped[3:].strip()
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append((current_heading, "\n".join(current_section)))

    return sections


def _slugify(value: str) -> str:
    """Create a stable Markdown-style anchor while retaining CJK characters."""
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff -]", "", normalized)
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or "document-start"


def _unique_anchor(anchor: str, used: dict[str, int]) -> str:
    count = used.get(anchor, 0)
    used[anchor] = count + 1
    return anchor if count == 0 else f"{anchor}-{count + 1}"
