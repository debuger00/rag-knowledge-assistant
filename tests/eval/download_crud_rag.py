"""Download the official CRUD-RAG evaluation split and optional background corpus."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "CRUD_RAG"
RAW_ROOT = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main"
SPLIT_FILE = "data/crud_split/split_merged.json"
CORPUS_FILES = tuple(
    f"data/80000_docs/documents_dup_part_{major}_part_{minor}"
    for major in range(1, 16)
    for minor in range(1, 4)
) + tuple(
    f"data/80000_docs/documents_hallu.txt_part_{minor}"
    for minor in range(1, 4)
)


def download_file(relative_path: str, output_root: Path, retries: int = 5) -> Path:
    destination = output_root / relative_path
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_ROOT}/{relative_path}"
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "CRUD-RAG-eval-adapter/1.0"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise OSError("downloaded file is empty")
            partial.replace(destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(attempt * 2, 10))
    raise RuntimeError(f"failed to download {relative_path}")


def download(output_root: Path, include_corpus: bool, workers: int = 4) -> list[Path]:
    paths = [SPLIT_FILE, *(CORPUS_FILES if include_corpus else ())]
    completed: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(download_file, relative, output_root): relative
            for relative in paths
        }
        for index, future in enumerate(as_completed(futures), 1):
            relative = futures[future]
            result = future.result()
            completed.append(result)
            print(f"[{index}/{len(paths)}] {relative} ({result.stat().st_size:,} bytes)", flush=True)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official CRUD-RAG data files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-corpus", action="store_true",
        help="Also download all 48 shards from data/80000_docs.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    files = download(args.output.resolve(), args.include_corpus, args.workers)
    print(f"Downloaded or verified {len(files)} file(s) under {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
