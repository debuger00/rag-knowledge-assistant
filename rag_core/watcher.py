"""Obsidian 仓库文件监听 — watchdog 监控 .md 文件变更，自动更新 Chroma 索引。"""
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from rag_core.indexing.loader import ObsidianLoader
from rag_core.indexing.splitter import INDEX_VERSION, parent_child_split
from rag_core.indexing.store import VectorStoreManager
from config import get_config

logger = logging.getLogger(__name__)


class VaultSyncHandler(FileSystemEventHandler):
    """处理 Obsidian 仓库文件变更事件，增量更新 Chroma。"""

    def __init__(self, store: VectorStoreManager, vault_path: str, ignore_dirs: list[str]):
        self.store = store
        self.vault_path = vault_path
        self.ignore_dirs = ignore_dirs
        self._pending: dict[str, str] = {}
        self._debounce_seconds = 2.0
        self._last_event_time = 0.0

        self.config = get_config()

    def on_created(self, event):
        if self._should_handle(event):
            self._schedule("created", event.src_path)

    def on_modified(self, event):
        if self._should_handle(event):
            self._schedule("modified", event.src_path)

    def on_deleted(self, event):
        if self._should_handle(event):
            self._schedule("deleted", event.src_path)

    def _should_handle(self, event) -> bool:
        if event.is_directory:
            return False
        if not event.src_path.endswith(".md"):
            return False
        try:
            rel = Path(event.src_path).relative_to(self.vault_path)
        except ValueError:
            return False
        return not any(ignored in rel.parts for ignored in self.ignore_dirs)

    def _schedule(self, event_type: str, path: str):
        try:
            rel = Path(path).relative_to(self.vault_path)
            source = str(rel).replace("\\", "/")
        except ValueError:
            return

        self._pending[source] = event_type
        self._last_event_time = time.time()

    def process_pending(self):
        """处理所有待处理的文件变更（由外部定时器调用）。"""
        if not self._pending:
            return

        now = time.time()
        if now - self._last_event_time < self._debounce_seconds:
            return

        pending = dict(self._pending)
        self._pending.clear()

        loaded_docs = []
        deleted_sources = []

        for source, event_type in pending.items():
            if event_type == "deleted":
                deleted_sources.append(source)
            else:
                filepath = Path(self.vault_path) / source
                if filepath.exists():
                    loader = ObsidianLoader(
                        vault_path=self.vault_path,
                        ignore_dirs=list(self.ignore_dirs),
                    )
                    doc = loader._parse_file(filepath)
                    if doc is not None:
                        loaded_docs.append(doc)

        for source in deleted_sources:
            self.store.delete_by_source(source)
            logger.info(f"已从索引中删除: {source}")

        for doc in loaded_docs:
            source = doc.metadata["source"]
            self.store.delete_by_source(source)

            all_docs = parent_child_split(
                [doc],
                child_chunk_size=self.config.child_chunk_size,
                child_chunk_overlap=self.config.child_chunk_overlap,
                child_max_len=self.config.child_max_len_before_split,
            )
            parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
            children = [d for d in all_docs if d.metadata["doc_type"] == "child"]

            self.store.add_parents(parents)
            self.store.add_children(children)
            logger.info(f"已更新索引: {source}")


class VaultWatcher:
    """Obsidian 仓库文件监听器。

    启动时执行全量对比，然后持续监听变更。
    """

    def __init__(
        self,
        store: VectorStoreManager,
        vault_path: str,
        ignore_dirs: list[str] | None = None,
    ):
        self.store = store
        self.vault_path = vault_path
        self.ignore_dirs = ignore_dirs or [".obsidian", ".trash", ".git"]
        self.config = get_config()
        self._observer: Observer | None = None

    def full_sync(self) -> dict:
        """全量对比 — 扫描仓库并增量更新索引。"""
        vault = Path(self.vault_path)
        if not vault.exists():
            raise FileNotFoundError(f"Obsidian 仓库路径不存在: {self.vault_path}")

        loader = ObsidianLoader(
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        current_docs = loader.load()
        current_sources = {
            str(doc.metadata["source"]) for doc in current_docs
        }

        new_count = 0
        updated_count = 0
        deleted_count = 0

        for doc in current_docs:
            source = doc.metadata["source"]
            existing = self.store.search_parents_by_source(source)

            if not existing:
                self._index_document(doc)
                new_count += 1
            else:
                existing_mtime = existing[0].metadata.get("mtime", "")
                existing_version = existing[0].metadata.get("index_version")
                if (
                    existing_mtime != doc.metadata.get("mtime", "")
                    or existing_version != INDEX_VERSION
                ):
                    self.store.delete_by_source(source)
                    self._index_document(doc)
                    updated_count += 1

        stale_sources = self.store.list_parent_sources() - current_sources
        for source in stale_sources:
            self.store.delete_by_source(source)
            deleted_count += 1

        logger.info(
            f"同步完成: {new_count} 篇新增, {updated_count} 篇更新, "
            f"{deleted_count} 篇删除, 总计 {len(current_docs)} 篇文档"
        )

        return {
            "total": len(current_docs),
            "new": new_count,
            "updated": updated_count,
            "deleted": deleted_count,
        }

    def rebuild(self) -> dict:
        """全量重建索引。"""
        vault = Path(self.vault_path)
        if not vault.exists():
            raise FileNotFoundError(f"Obsidian 仓库路径不存在: {self.vault_path}")

        loader = ObsidianLoader(
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        print("  [1/3] 加载笔记...", flush=True)
        docs = loader.load()
        print(f"  [1/3] 加载完成: {len(docs)} 篇笔记", flush=True)

        print("  [2/3] 文本切分...", flush=True)
        all_docs = []
        for idx, doc in enumerate(docs):
            split_docs = parent_child_split(
                [doc],
                child_chunk_size=self.config.child_chunk_size,
                child_chunk_overlap=self.config.child_chunk_overlap,
                child_max_len=self.config.child_max_len_before_split,
            )
            all_docs.extend(split_docs)
            if (idx + 1) % 100 == 0:
                print(f"  [2/3] 切分进度: {idx + 1}/{len(docs)}", flush=True)
        print(f"  [2/3] 切分完成: {len(all_docs)} 个文档块", flush=True)

        parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
        children = [d for d in all_docs if d.metadata["doc_type"] == "child"]
        print(f"  [3/3] 向量嵌入: {len(parents)} 父文档 + {len(children)} 子块", flush=True)

        self.store.rebuild(parents, children)

        logger.info(f"全量重建完成: {len(parents)} 篇父文档, {len(children)} 个子块")

        return {
            "total_parents": len(parents),
            "total_children": len(children),
        }

    def start_watching(self):
        """启动文件监听（后台线程）。"""
        event_handler = VaultSyncHandler(
            store=self.store,
            vault_path=self.vault_path,
            ignore_dirs=list(self.ignore_dirs),
        )
        self._observer = Observer()
        self._observer.schedule(event_handler, self.vault_path, recursive=True)
        self._observer.start()
        logger.info(f"开始监听 Obsidian 仓库: {self.vault_path}")

        import threading
        def _process_loop():
            while self._observer and self._observer.is_alive():
                time.sleep(1)
                try:
                    event_handler.process_pending()
                except Exception as e:
                    logger.error(f"处理文件变更时出错: {e}")

        threading.Thread(target=_process_loop, daemon=True).start()

    def stop_watching(self):
        """停止文件监听。"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def _index_document(self, doc):
        """索引单篇文档。"""
        all_docs = parent_child_split(
            [doc],
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
            child_max_len=self.config.child_max_len_before_split,
        )
        parents = [d for d in all_docs if d.metadata["doc_type"] == "parent"]
        children = [d for d in all_docs if d.metadata["doc_type"] == "child"]
        self.store.add_parents(parents)
        self.store.add_children(children)
