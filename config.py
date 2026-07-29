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
        default_factory=lambda: os.getenv(
            "OBSIDIAN_VAULT_PATH", "./documents"
        )
    )
    obsidian_ignore_dirs: list[str] = field(
        default_factory=lambda: [".obsidian", ".trash", ".git"]
    )

    # OpenAI-compatible LLM gateway. DeepSeek is the development placeholder.
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
    )

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )
    embedding_local_files_only: bool = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_LOCAL_FILES_ONLY", "false"
        ).lower() in {"1", "true", "yes"}
    )

    # Chroma
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    )

    # Retrieval
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10"))
    )
    retrieval_score_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")
        )
    )
    enable_link_expansion: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_LINK_EXPANSION", "false"
        ).lower() in {"1", "true", "yes"}
    )

    # History
    history_dir: str = "./chat_history"

    # Server
    server_host: str = field(
        default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0")
    )
    server_port: int = field(
        default_factory=lambda: int(os.getenv("SERVER_PORT", "8501"))
    )

    # Chunking
    child_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "800"))
    )
    child_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))
    )
    child_max_len_before_split: int = field(
        default_factory=lambda: int(os.getenv("CHILD_MAX_LEN", "2000"))
    )

    # Compatibility aliases for code using the old DeepSeek-specific names.
    @property
    def deepseek_api_key(self) -> str:
        return self.llm_api_key

    @property
    def deepseek_model(self) -> str:
        return self.llm_model

    @property
    def deepseek_base_url(self) -> str:
        return self.llm_base_url


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config
