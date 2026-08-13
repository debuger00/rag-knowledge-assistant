"""Application configuration loaded from YAML, with secrets from .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

_YAML_FIELDS = {
    "documents": {
        "path": "obsidian_vault_path",
        "ignore_dirs": "obsidian_ignore_dirs",
    },
    "llm": {
        "model": "llm_model",
        "base_url": "llm_base_url",
    },
    "embedding": {
        "model": "embedding_model",
        "device": "embedding_device",
        "local_files_only": "embedding_local_files_only",
    },
    "storage": {
        "chroma_dir": "chroma_persist_dir",
        "history_dir": "history_dir",
    },
    "retrieval": {
        "top_k": "retrieval_top_k",
        "score_threshold": "retrieval_score_threshold",
        "max_citations": "rag_max_citations",
        "max_retry": "rag_max_retry",
        "require_citations": "rag_require_citations",
        "enable_link_expansion": "enable_link_expansion",
    },
    "server": {
        "host": "server_host",
        "port": "server_port",
    },
    "chunking": {
        "chunk_size": "child_chunk_size",
        "chunk_overlap": "child_chunk_overlap",
        "max_len_before_split": "child_max_len_before_split",
    },
    "graph": {
        "enabled": "graph_enabled",
        "db_path": "graph_db_path",
        "max_hops": "graph_max_hops",
        "max_seed_nodes": "graph_max_seed_nodes",
        "max_neighbors": "graph_max_neighbors",
        "graph_weight": "graph_weight",
        "entity_extraction": "graph_entity_extraction",
        "community_detection": "graph_community_detection",
        "community_reports": "graph_community_reports",
    },
}


@dataclass
class Config:
    # The only environment-backed application setting is the secret.
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")
        )
    )

    obsidian_vault_path: str = "./documents"
    obsidian_ignore_dirs: list[str] = field(
        default_factory=lambda: [".obsidian", ".trash", ".git"]
    )
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_local_files_only: bool = False
    chroma_persist_dir: str = "./chroma_data"
    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.35
    rag_max_citations: int = 5
    rag_max_retry: int = 1
    rag_require_citations: bool = True
    enable_link_expansion: bool = False
    history_dir: str = "./chat_history"
    server_host: str = "0.0.0.0"
    server_port: int = 8501
    child_chunk_size: int = 800
    child_chunk_overlap: int = 100
    child_max_len_before_split: int = 2000
    graph_enabled: bool = True
    graph_db_path: str = "./graph_data/graph.sqlite3"
    graph_max_hops: int = 2
    graph_max_seed_nodes: int = 10
    graph_max_neighbors: int = 30
    graph_weight: float = 0.25
    graph_entity_extraction: bool = False
    graph_community_detection: bool = False
    graph_community_reports: bool = False

    def __post_init__(self) -> None:
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval.top_k 必须大于 0")
        if not 0 <= self.retrieval_score_threshold <= 1:
            raise ValueError("retrieval.score_threshold 必须在 0 到 1 之间")
        if self.rag_max_citations < 1:
            raise ValueError("retrieval.max_citations 必须大于 0")
        if self.rag_max_retry < 0:
            raise ValueError("retrieval.max_retry 必须大于等于 0")
        if not 1 <= self.server_port <= 65535:
            raise ValueError("server.port 必须在 1 到 65535 之间")
        if self.child_chunk_size < 1:
            raise ValueError("chunking.chunk_size 必须大于 0")
        if not 0 <= self.child_chunk_overlap < self.child_chunk_size:
            raise ValueError(
                "chunking.chunk_overlap 必须大于等于 0 且小于 chunk_size"
            )
        if self.child_max_len_before_split < self.child_chunk_size:
            raise ValueError(
                "chunking.max_len_before_split 不能小于 chunk_size"
            )
        if self.graph_max_hops < 1:
            raise ValueError("graph.max_hops 必须大于 0")
        if self.graph_max_seed_nodes < 1:
            raise ValueError("graph.max_seed_nodes 必须大于 0")
        if self.graph_max_neighbors < 1:
            raise ValueError("graph.max_neighbors 必须大于 0")
        if not 0 <= self.graph_weight <= 1:
            raise ValueError("graph.graph_weight 必须在 0 到 1 之间")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config.yaml 顶层必须是对象")

        values: dict[str, Any] = {}
        unknown_sections = set(raw) - set(_YAML_FIELDS)
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(f"config.yaml 包含未知配置段: {names}")

        for section, fields in _YAML_FIELDS.items():
            section_data = raw.get(section, {})
            if not isinstance(section_data, dict):
                raise ValueError(f"config.yaml 的 {section} 必须是对象")

            forbidden = {"api_key", "key", "secret"} & set(section_data)
            if forbidden:
                raise ValueError("密钥禁止写入 config.yaml，请使用 .env")

            unknown_fields = set(section_data) - set(fields)
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise ValueError(f"config.yaml 的 {section} 包含未知字段: {names}")

            for yaml_name, value in section_data.items():
                values[fields[yaml_name]] = value

        env_overrides: dict[str, tuple[str, Any]] = {
            "RAG_MIN_RETRIEVAL_SCORE": (
                "retrieval_score_threshold", float
            ),
            "RAG_MAX_CITATIONS": ("rag_max_citations", int),
            "RAG_MAX_RETRY": ("rag_max_retry", int),
            "RAG_REQUIRE_CITATIONS": (
                "rag_require_citations",
                lambda value: value.lower() in {"1", "true", "yes", "on"},
            ),
        }
        for env_name, (field_name, converter) in env_overrides.items():
            if env_name in os.environ:
                values[field_name] = converter(os.environ[env_name])

        return cls(**values)

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
        _config = Config.from_yaml()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config
