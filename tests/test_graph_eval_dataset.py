from tests.eval.run_graph_eval import FIXTURE_ROOT, run


def test_controlled_graph_dataset_has_expected_size():
    assert len(list(FIXTURE_ROOT.rglob("*.md"))) == 27


def test_controlled_graph_structure_and_retrieval_benchmark():
    report = run(output_dir=None)
    structure = report["structure"]
    retrieval = report["retrieval"]

    assert structure["passed"], structure["errors"]
    assert structure["resolved_link_precision"] == 1.0
    assert structure["resolved_link_recall"] == 1.0
    assert retrieval["local_source_recall_at_5"] == 1.0
    assert retrieval["recall_delta"] >= 0.9
    assert retrieval["graph_harm_count"] == 0
    assert retrieval["safe_negative_pass_rate"] == 1.0
    assert retrieval["forbidden_leak_count"] == 0
