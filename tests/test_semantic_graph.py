from langchain_core.documents import Document
from types import SimpleNamespace

from config import Config
from rag_core.graph.builder import rebuild_structure_graph
from rag_core.graph.communities import build_communities, generate_community_reports
from rag_core.graph.semantic import SemanticGraphIndexer
from rag_core.graph.extractor import LLMGraphExtractor
from rag_core.graph.semantic_models import GraphExtraction
from rag_core.graph.store import GraphStore


ENTITY_TYPES = ("person", "organization", "geo")


def _document(source: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "filename": source[:-3],
            "folder": "",
            "tags": [],
            "doc_type": "raw",
        },
    )


class FakeExtractor:
    model_id = "fake-model"
    prompt_hash = "fake-prompt-v1"
    extractor_version = 1
    entity_types = ENTITY_TYPES

    def __init__(self):
        self.calls = 0

    def extract(self, text: str) -> GraphExtraction:
        self.calls += 1
        if "张三" in text:
            return GraphExtraction.from_dict({
                "entities": [
                    {
                        "name": "张三", "type": "person",
                        "description": "张三是ABC科技公司的CEO",
                        "evidence_quote": "张三是ABC科技公司的CEO",
                        "confidence": 0.95,
                    },
                    {
                        "name": "ABC科技公司", "type": "organization",
                        "description": "ABC科技公司由张三担任CEO",
                        "aliases": ["ABC公司"],
                        "evidence_quote": "ABC科技公司",
                        "confidence": 0.95,
                    },
                ],
                "relationships": [{
                    "source": "张三", "target": "ABC公司",
                    "description": "张三担任ABC科技公司的CEO",
                    "predicate": "CEO_OF", "strength": 9,
                    "evidence_quote": "张三是ABC科技公司的CEO",
                    "confidence": 0.95,
                }],
            }, text=text, entity_types=ENTITY_TYPES, min_confidence=0.65)
        return GraphExtraction.from_dict({
            "entities": [
                {
                    "name": "ABC科技公司", "type": "organization",
                    "description": "ABC科技公司总部位于上海并收购XYZ公司",
                    "evidence_quote": "ABC科技公司",
                    "confidence": 0.95,
                },
                {
                    "name": "上海", "type": "geo",
                    "description": "ABC科技公司的总部所在地",
                    "evidence_quote": "上海", "confidence": 0.95,
                },
                {
                    "name": "XYZ公司", "type": "organization",
                    "description": "被ABC科技公司收购的公司",
                    "evidence_quote": "XYZ公司", "confidence": 0.95,
                },
            ],
            "relationships": [
                {
                    "source": "ABC科技公司", "target": "上海",
                    "description": "ABC科技公司总部位于上海",
                    "predicate": "LOCATED_IN", "strength": 8,
                    "evidence_quote": "ABC科技公司总部位于上海",
                    "confidence": 0.95,
                },
                {
                    "source": "ABC科技公司", "target": "XYZ公司",
                    "description": "ABC科技公司收购XYZ公司",
                    "predicate": "ACQUIRED", "strength": 9,
                    "evidence_quote": "ABC科技公司收购XYZ公司",
                    "confidence": 0.95,
                },
            ],
        }, text=text, entity_types=ENTITY_TYPES, min_confidence=0.65)


class FakeSummarizer:
    model_id = "fake-model"
    prompt_hash = "fake-summary-v1"

    def __init__(self):
        self.calls = 0

    def summarize(self, kind: str, descriptions: list[str]) -> str:
        self.calls += 1
        return f"SUMMARY[{kind}]: " + " | ".join(descriptions)


class FakeCommunityReporter:
    model_id = "fake-model"
    prompt_hash = "fake-community-v1"

    def __init__(self):
        self.calls = 0

    def report(self, context):
        self.calls += 1
        return {
            "title": "公司治理社区",
            "summary": "围绕ABC科技公司形成的实体关系社区。",
            "findings": [{"summary": "治理", "explanation": "张三担任CEO"}],
            "rank": 8.0,
        }


def _config(tmp_path) -> Config:
    return Config(
        graph_db_path=str(tmp_path / "graph.sqlite3"),
        child_chunk_size=200,
        child_chunk_overlap=20,
        child_max_len_before_split=500,
        graph_entity_types=list(ENTITY_TYPES),
        graph_extraction_concurrency=1,
    )


def test_semantic_index_merges_entities_and_preserves_evidence(tmp_path):
    config = _config(tmp_path)
    store = GraphStore(config.graph_db_path)
    documents = [
        _document("people.md", "# 人物\n\n张三是ABC科技公司的CEO。"),
        _document(
            "company.md",
            "# 公司\n\nABC科技公司总部位于上海。ABC科技公司收购XYZ公司。",
        ),
    ]
    rebuild_structure_graph(
        store, documents,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        child_max_len=config.child_max_len_before_split,
    )
    extractor = FakeExtractor()
    result = SemanticGraphIndexer(store, extractor, config).build(documents)

    snapshot = store.snapshot()
    entities = [node for node in snapshot["nodes"] if node["type"] == "entity"]
    relationships = [
        edge for edge in snapshot["edges"] if edge["type"] == "RELATED_TO"
    ]
    mentions = [edge for edge in snapshot["edges"] if edge["type"] == "MENTIONS"]

    assert result["documents_indexed"] == 2
    assert extractor.calls == 2
    assert len(entities) == 4
    assert len(relationships) == 3
    assert len(mentions) == 5
    abc = next(node for node in entities if node["name"] == "ABC科技公司")
    assert abc["metadata"]["frequency"] == 2
    assert "CEO" in abc["description"] and "上海" in abc["description"]
    assert store.get_stats()["semantic_edge_evidence_count"] == 3
    assert {hit.source for hit in store.expand_entities([abc["id"]])} == {
        "people.md", "company.md"
    }

    second = SemanticGraphIndexer(store, extractor, config).build(
        documents, changed_only=True
    )
    assert second["documents_skipped"] == 2
    assert second["llm_calls"] == 0
    assert extractor.calls == 2


def test_semantic_cache_and_source_pruning_keep_shared_entity(tmp_path):
    config = _config(tmp_path)
    store = GraphStore(config.graph_db_path)
    people = _document("people.md", "# 人物\n\n张三是ABC科技公司的CEO。")
    company = _document(
        "company.md",
        "# 公司\n\nABC科技公司总部位于上海。ABC科技公司收购XYZ公司。",
    )
    documents = [people, company]
    rebuild_structure_graph(
        store, documents,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        child_max_len=config.child_max_len_before_split,
    )
    extractor = FakeExtractor()
    indexer = SemanticGraphIndexer(store, extractor, config)
    indexer.build(documents)

    cached = indexer.build(documents, changed_only=False)
    assert cached["cache_hits"] == 2
    assert cached["llm_calls"] == 0
    assert extractor.calls == 2

    pruned = indexer.build([people], changed_only=True)
    entities = store.list_entities()
    assert pruned["pruned_sources"] == 1
    assert {entity["name"] for entity in entities} == {"张三", "ABC科技公司"}
    assert store.get_stats()["semantic_edge_evidence_count"] == 1

    # A structural refresh must not erase the semantic layer.
    rebuild_structure_graph(
        store, [people],
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        child_max_len=config.child_max_len_before_split,
    )
    assert {entity["name"] for entity in store.list_entities()} == {
        "张三", "ABC科技公司"
    }


def test_description_summaries_are_applied_and_cached(tmp_path):
    config = _config(tmp_path)
    store = GraphStore(config.graph_db_path)
    documents = [
        _document("people.md", "# 人物\n\n张三是ABC科技公司的CEO。"),
        _document(
            "company.md",
            "# 公司\n\nABC科技公司总部位于上海。ABC科技公司收购XYZ公司。",
        ),
    ]
    rebuild_structure_graph(
        store, documents,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        child_max_len=config.child_max_len_before_split,
    )
    extractor = FakeExtractor()
    summarizer = FakeSummarizer()
    indexer = SemanticGraphIndexer(
        store, extractor, config, summarizer=summarizer
    )

    first = indexer.build(documents)
    abc = next(value for value in store.list_entities() if value["name"] == "ABC科技公司")
    assert first["summary_llm_calls"] == 1
    assert abc["description"].startswith("SUMMARY[entity]")

    second = indexer.build(documents, changed_only=False)
    assert second["summary_llm_calls"] == 0
    assert second["summary_cache_hits"] == 1
    assert summarizer.calls == 1


def test_invalid_or_low_confidence_evidence_is_rejected():
    extraction = GraphExtraction.from_dict({
        "entities": [
            {
                "name": "ABC", "type": "organization",
                "description": "valid", "evidence_quote": "ABC",
                "confidence": 0.9,
            },
            {
                "name": "XYZ", "type": "organization",
                "description": "hallucinated", "evidence_quote": "not present",
                "confidence": 0.9,
            },
            {
                "name": "上海", "type": "geo",
                "description": "low", "evidence_quote": "上海",
                "confidence": 0.2,
            },
        ],
        "relationships": [],
    }, text="ABC位于上海", entity_types=ENTITY_TYPES, min_confidence=0.65)

    assert [entity.name for entity in extraction.entities] == ["ABC"]


def test_leiden_communities_and_reports_are_persisted_and_cached(tmp_path):
    config = _config(tmp_path)
    config.graph_community_max_cluster_size = 10
    documents = [
        _document("people.md", "# 人物\n\n张三是ABC科技公司的CEO。"),
        _document(
            "company.md",
            "# 公司\n\nABC科技公司总部位于上海。ABC科技公司收购XYZ公司。",
        ),
    ]
    store = GraphStore(config.graph_db_path)
    rebuild_structure_graph(
        store, documents,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        child_max_len=config.child_max_len_before_split,
    )
    SemanticGraphIndexer(store, FakeExtractor(), config).build(documents)

    community_stats = build_communities(store, config)
    contexts = store.community_contexts()
    assert community_stats["community_count"] >= 1
    assert sum(len(value["entities"]) for value in contexts) >= 4
    assert any(
        edge["type"] == "IN_COMMUNITY" for edge in store.snapshot()["edges"]
    )

    reporter = FakeCommunityReporter()
    first = generate_community_reports(store, reporter)
    second = generate_community_reports(store, reporter)
    assert first["community_report_llm_calls"] == len(contexts)
    assert second["community_report_llm_calls"] == 0
    assert second["community_report_cache_hits"] == len(contexts)
    assert reporter.calls == len(contexts)


def test_llm_extractor_retries_invalid_json_and_accepts_fenced_json(tmp_path):
    prompt = tmp_path / "extract.txt"
    prompt.write_text(
        "types={{entity_types}}\n<TEXT_UNIT>\n{{text}}\n</TEXT_UNIT>",
        encoding="utf-8",
    )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt_value):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="not-json")
            return SimpleNamespace(content="""```json
{"entities":[{"name":"ABC","type":"organization","description":"公司","aliases":[],"evidence_quote":"ABC","confidence":0.9}],"relationships":[]}
```""")

    config = Config(
        graph_extraction_prompt=str(prompt),
        graph_entity_types=["organization"],
        graph_extraction_max_retries=1,
        graph_min_confidence=0.65,
    )
    llm = FakeLLM()
    extraction = LLMGraphExtractor(config, llm=llm).extract("ABC是一家公司")

    assert llm.calls == 2
    assert [value.name for value in extraction.entities] == ["ABC"]
