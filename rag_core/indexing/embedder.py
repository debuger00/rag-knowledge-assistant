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
