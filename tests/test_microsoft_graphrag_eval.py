import csv
import io
import json
import tarfile
import zipfile

from tests.eval.run_microsoft_graphrag_eval import prepare_profile


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question_text"])
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_hotpot_profile_maps_question_index_to_context(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    archive_path = data / "hotpot.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for index in (0, 2):
            content = f"context {index}".encode()
            info = tarfile.TarInfo(f"input/test_{index}.txt")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    _write_csv(data / "questions.csv", [
        {"question_id": "q0", "question_text": "question zero"},
        {"question_id": "q1", "question_text": "question one"},
    ])
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({"hotpotqa": {
        "display_name": "test", "archive": "hotpot.tar.gz",
        "questions": "questions.csv", "corpus_kind": "hotpot_per_question",
        "expected_questions": 2, "recommended_mode": "local",
        "evaluation_mode": "paired_context_recall", "has_paired_source": True,
    }}), encoding="utf-8")
    monkeypatch.setattr("tests.eval.run_microsoft_graphrag_eval.PROFILE_PATH", profile_path)

    manifest = prepare_profile("hotpotqa", data, tmp_path / "work")
    cases = [json.loads(line) for line in open(manifest["cases"], encoding="utf-8")]

    assert manifest["document_count"] == 2
    assert cases[1]["expected_sources"] == ["contexts/test_2.md"]
    assert "context 2" in (tmp_path / "work/hotpotqa/vault/contexts/test_2.md").read_text()


def test_prepare_podcast_profile_merges_parts(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    with zipfile.ZipFile(data / "podcasts.zip", "w") as archive:
        archive.writestr("input/Episode-One-part2.txt", "second")
        archive.writestr("input/Episode-One-part1.txt", "first")
    _write_csv(data / "questions.csv", [
        {"question_id": "q", "question_text": "global question"},
    ])
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({"kevin_scott": {
        "display_name": "test", "archive": "podcasts.zip",
        "questions": "questions.csv", "corpus_kind": "podcast_parts",
        "expected_questions": 1, "recommended_mode": "global",
        "evaluation_mode": "retrieval_diagnostic", "has_paired_source": False,
    }}), encoding="utf-8")
    monkeypatch.setattr("tests.eval.run_microsoft_graphrag_eval.PROFILE_PATH", profile_path)

    manifest = prepare_profile("kevin_scott", data, tmp_path / "work")
    markdown = next((tmp_path / "work/kevin_scott/vault").rglob("*.md")).read_text()

    assert manifest["document_count"] == 1
    assert manifest["global_pipeline_status"] == "required_not_implemented"
    assert markdown.index("first") < markdown.index("second")
