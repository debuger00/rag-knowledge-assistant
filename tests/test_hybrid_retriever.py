from langchain_core.documents import Document

from config import Config
from rag_core.graph.models import GraphHit
from rag_core.retrieval.hybrid import HybridGraphRetriever


def test_hybrid_retriever_adds_grounded_chunks_from_graph_sources():
    seed = Document(
        page_content="seed evidence",
        metadata={"source": "a.md", "anchor": "a"},
    )
    related = Document(
        page_content="related evidence",
        metadata={"source": "b.md", "anchor": "b"},
    )

    class Basic:
        def retrieve_with_scores(self, query, filter_dict=None):
            return [(seed, 0.8)]

    class Graph:
        def expand_sources(self, seeds, **kwargs):
            return [GraphHit("b.md", 0.9, ("seed", "target"))]

    class Store:
        def similarity_search_by_sources(self, query, sources, **kwargs):
            assert sources == ["b.md"]
            return [(related, 0.7)]

    retriever = object.__new__(HybridGraphRetriever)
    retriever.store = Store()
    retriever.graph_store = Graph()
    retriever.basic = Basic()
    retriever.config = Config(
        retrieval_score_threshold=0.35,
        graph_weight=0.25,
        graph_max_hops=2,
        graph_max_seed_nodes=10,
        graph_max_neighbors=30,
    )
    retriever.last_trace = {}

    results = retriever.retrieve_with_scores("question")

    assert {doc.metadata["source"] for doc, _ in results} == {"a.md", "b.md"}
    assert related.metadata["graph_path"] == ["seed", "target"]
    assert retriever.last_trace["expanded_sources"] == 1
