from langchain_core.documents import Document

from config import Config
from rag_core.retrieval.global_search import GlobalCommunityRetriever


def test_global_retriever_maps_community_reports_back_to_grounded_chunks():
    report = Document(
        page_content="company community",
        metadata={"community_id": "community:1"},
    )
    evidence = Document(
        page_content="grounded evidence",
        metadata={"source": "company.md", "anchor": "company"},
    )

    class Store:
        def similarity_search_communities(self, query, k):
            return [(report, 0.9)]

        def similarity_search_by_sources(self, query, sources, **kwargs):
            assert sources == ["company.md"]
            return [(evidence, 0.8)]

    class Graph:
        def sources_for_communities(self, community_ids, limit):
            assert community_ids == ["community:1"]
            return ["company.md"]

    retriever = object.__new__(GlobalCommunityRetriever)
    retriever.store = Store()
    retriever.graph_store = Graph()
    retriever.config = Config(
        retrieval_top_k=10,
        retrieval_score_threshold=0.35,
        graph_max_seed_nodes=10,
        graph_max_neighbors=30,
    )
    retriever.last_trace = {}

    result = retriever.retrieve_with_scores("question")

    assert result == [(evidence, 0.8250000000000001)]
    assert evidence.metadata["community_ids"] == ["community:1"]
    assert retriever.last_trace["community_hits"] == 1
