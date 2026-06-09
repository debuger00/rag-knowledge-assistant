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

        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            if len(section_text) <= child_max_len:
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
    """按 ## 标题切分文本，返回各段落列表。

    以 ## 开头的行（但非 ### 开头）作为切分边界。
    不将 # 一级标题作为切分边界——Obsidian 中一级标题通常是文件标题。
    """
    sections: list[str] = []
    current_section: list[str] = []

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        # 匹配 ## 开头但非 ### 开头
        if re.match(r"^## [^#]", stripped):
            if current_section:
                sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections
