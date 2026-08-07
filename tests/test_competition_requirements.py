from config import Config
from rag_core.watcher import VaultWatcher


def test_yaml_configuration_and_environment_secret(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm:
  base_url: https://gateway.example/v1
  model: competition-model
embedding:
  device: cpu
retrieval:
  score_threshold: 0.42
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "secret-from-env")

    config = Config.from_yaml(config_file)

    assert config.llm_base_url == "https://gateway.example/v1"
    assert config.llm_model == "competition-model"
    assert config.llm_api_key == "secret-from-env"
    assert config.embedding_device == "cpu"
    assert config.retrieval_score_threshold == 0.42


def test_yaml_rejects_api_key(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  api_key: must-not-be-here\n",
        encoding="utf-8",
    )

    try:
        Config.from_yaml(config_file)
    except ValueError as exc:
        assert "密钥禁止写入" in str(exc)
    else:
        raise AssertionError("config.yaml 中的密钥必须被拒绝")


def test_full_sync_removes_files_deleted_while_service_was_stopped(tmp_path):
    class FakeStore:
        def __init__(self):
            self.deleted = []

        def list_parent_sources(self):
            return {"removed.md"}

        def delete_by_source(self, source):
            self.deleted.append(source)

    store = FakeStore()
    watcher = VaultWatcher(store=store, vault_path=str(tmp_path))

    result = watcher.full_sync()

    assert store.deleted == ["removed.md"]
    assert result["deleted"] == 1
