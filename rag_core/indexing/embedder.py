"""Hugging Face embedding adapter."""
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_config


def create_embedder() -> HuggingFaceEmbeddings:
    """创建 BGE-M3 embedding 实例。

    首次调用会自动下载模型（约 2GB）到 ~/.cache/huggingface/。
    """
    config = get_config()
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={
            "device": config.embedding_device,
            "local_files_only": config.embedding_local_files_only,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,          # 控制内部 batch，降低内存峰值
        },
    )
