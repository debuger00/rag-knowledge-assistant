"""Download Microsoft GraphRAG benchmarking datasets from the official repository."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "microsoft-graphrag-benchmarking"
PROFILE_PATH = Path(__file__).resolve().parent / "datasets" / "microsoft_graphrag_profiles.json"
RAW_ROOT = "https://raw.githubusercontent.com/microsoft/graphrag-benchmarking-datasets/main/data"


def profiles() -> dict[str, dict]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def download_file(name: str, output: Path, retries: int = 5) -> Path:
    destination = output / name
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    output.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_ROOT}/{quote(name)}"
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "Microsoft-GraphRAG-eval-adapter/1.0"})
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
    raise RuntimeError(f"failed to download {name}")


def archive_file_count(path: Path) -> int:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return sum(not item.is_dir() for item in archive.infolist())
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            return sum(item.isfile() for item in archive.getmembers())
    raise ValueError(f"Unsupported archive format: {path}")


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def download(output: Path) -> dict[str, dict[str, int | str]]:
    catalog = profiles()
    names = sorted({item[key] for item in catalog.values() for key in ("archive", "questions")})
    for index, name in enumerate(names, 1):
        path = download_file(name, output)
        print(f"[{index}/{len(names)}] {name} ({path.stat().st_size:,} bytes)", flush=True)
    result = {}
    for name, profile in catalog.items():
        question_count = csv_row_count(output / profile["questions"])
        if question_count != profile["expected_questions"]:
            raise ValueError(
                f"{name}: expected {profile['expected_questions']} questions, got {question_count}"
            )
        result[name] = {
            "questions": question_count,
            "archive_files": archive_file_count(output / profile["archive"]),
            "recommended_mode": profile["recommended_mode"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Microsoft GraphRAG benchmark data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = download(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
