from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "codegraph_to_gephi.py"
SPEC = importlib.util.spec_from_file_location("codegraph_to_gephi", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
                file_path TEXT, language TEXT, start_line INTEGER, end_line INTEGER
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY, source TEXT, target TEXT, kind TEXT,
                metadata TEXT, line INTEGER, col INTEGER, provenance TEXT
            );
        """)
        connection.executemany(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("n1", "function", "调用者", "模块.调用者", "目录\\中文.py", "python", 1, 3),
                ("n2", "method", "callee & helper", "", "x.py", "python", 4, 8),
                ("n3", "class", "Isolated", "Isolated", "x.py", "python", 10, 12),
            ],
        )
        connection.executemany(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(1, "n1", "n2", "calls", None, 2, 4, "test & static"),
             (2, "n1", "missing", "calls", None, 3, 4, "test")],
        )


def test_export_is_consistent_utf8_and_valid_graphml(tmp_path: Path) -> None:
    db = tmp_path / "codegraph.db"
    output = tmp_path / "out"
    make_db(db)
    args = exporter.build_parser().parse_args(["--db", str(db), "--output", str(output), "--preset", "call"])
    assert exporter.run(args) == 0
    root = ET.parse(output / "codegraph.graphml").getroot()
    namespace = {"g": exporter.GRAPHML_NS}
    node_ids = {node.attrib["id"] for node in root.findall(".//g:node", namespace)}
    graph_edges = root.findall(".//g:edge", namespace)
    assert node_ids == {"n1", "n2"}
    assert all(edge.attrib["source"] in node_ids and edge.attrib["target"] in node_ids for edge in graph_edges)
    with (output / "nodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Label"] == "模块.调用者"
    assert rows[0]["FilePath"] == "目录\\中文.py"
    with (output / "edges.csv").open(encoding="utf-8", newline="") as handle:
        edge_rows = list(csv.DictReader(handle))
    assert edge_rows[0]["Type"] == "Directed"
    assert edge_rows[0]["Kind"] == "calls"
    assert "Skipped invalid edges: 1" in (output / "summary.txt").read_text(encoding="utf-8")


def test_empty_edge_kind_returns_clear_error(tmp_path: Path, capsys: object) -> None:
    db = tmp_path / "codegraph.db"
    make_db(db)
    args = exporter.build_parser().parse_args(["--db", str(db), "--output", str(tmp_path / "out"), "--edges", "unknown"])
    assert exporter.run(args) == 2
    captured = capsys.readouterr()
    assert "No matching edges found." in captured.err
    assert "calls" in captured.err
    assert not (tmp_path / "out").exists()
