from rag.models.chunk import Chunk
from rag.models.search_result import SearchResult
from rag.retrievers.similarity_retriever import SimilarityRetriever


class FakeEmbedder:

    def __init__(self):
        self.last_query = None

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query

        return [1.0, 0.0, 0.0]


class FakeVectorStore:

    def __init__(self, results):
        self.results = results
        self.last_query_vector = None
        self.last_k = None

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[SearchResult]:

        self.last_query_vector = query_vector
        self.last_k = k

        return self.results


def create_result(
    content: str,
    score: float,
) -> SearchResult:

    chunk = Chunk(
        content=content,
        metadata={
            "source": "test.txt",
            "page": 1,
        },
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_retriever_embeds_query():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=[]
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    query = "What is artificial intelligence?"

    retriever.retrieve(query)

    assert (
        embedder.last_query
        == query
    )


def test_retriever_sends_query_vector_to_vector_store():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=[]
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever.retrieve(
        "What is AI?"
    )

    assert (
        vector_store.last_query_vector
        == [1.0, 0.0, 0.0]
    )


def test_retriever_uses_default_top_k():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=[]
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=3,
    )

    retriever.retrieve(
        "What is AI?"
    )

    assert vector_store.last_k == 3


def test_retriever_allows_top_k_override():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=[]
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=5,
    )

    retriever.retrieve(
        "What is AI?",
        top_k=2,
    )

    assert vector_store.last_k == 2


def test_retriever_filters_by_min_score():
    results = [
        create_result(
            "Highly relevant chunk",
            0.90,
        ),
        create_result(
            "Relevant chunk",
            0.75,
        ),
        create_result(
            "Weak chunk",
            0.40,
        ),
        create_result(
            "Irrelevant chunk",
            0.20,
        ),
    ]

    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=results
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        min_score=0.70,
    )

    filtered_results = retriever.retrieve(
        "What is AI?"
    )

    assert len(filtered_results) == 2

    assert (
        filtered_results[0].score
        == 0.90
    )

    assert (
        filtered_results[1].score
        == 0.75
    )


def test_retriever_allows_min_score_override():
    results = [
        create_result("Chunk 1", 0.90),
        create_result("Chunk 2", 0.75),
        create_result("Chunk 3", 0.60),
    ]

    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=results
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        min_score=0.50,
    )

    filtered_results = retriever.retrieve(
        "What is AI?",
        min_score=0.80,
    )

    assert len(filtered_results) == 1

    assert (
        filtered_results[0].score
        == 0.90
    )


def test_retriever_returns_vector_store_results_when_all_pass():
    results = [
        create_result("Chunk 1", 0.90),
        create_result("Chunk 2", 0.80),
        create_result("Chunk 3", 0.70),
    ]

    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=results
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        min_score=0.00,
    )

    retrieved = retriever.retrieve(
        "What is AI?"
    )

    assert retrieved == results


def test_retriever_returns_empty_when_no_results():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore(
        results=[]
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is AI?"
    )

    assert results == []