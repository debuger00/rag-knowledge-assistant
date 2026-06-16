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
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cuda"

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
    child_max_len_before_split: int = 2000


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config
