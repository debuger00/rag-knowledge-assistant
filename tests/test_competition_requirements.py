from config import Config
from rag_core.watcher import VaultWatcher


def test_gateway_and_cpu_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("LLM_MODEL", "competition-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.42")

    config = Config()

    assert config.llm_base_url == "https://gateway.example/v1"
    assert config.llm_model == "competition-model"
    assert config.embedding_device == "cpu"
    assert config.retrieval_score_threshold == 0.42


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
