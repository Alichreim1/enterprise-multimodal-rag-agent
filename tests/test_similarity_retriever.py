from unittest.mock import MagicMock

from rag.retrievers.similarity_retriever import SimilarityRetriever
from rag.models.search_result import SearchResult
from rag.models.chunk import Chunk


def test_retrieve():

    embedder = MagicMock()

    vector_store = MagicMock()

    embedder.embed_query.return_value = [0.1, 0.2]

    vector_store.search.return_value = [

        SearchResult(

            chunk=Chunk(

                content="AI",

                metadata={},

            ),

            score=0.95,

        )

    ]

    retriever = SimilarityRetriever(

        embedder=embedder,

        vector_store=vector_store,

    )

    results = retriever.retrieve(
        "AI"
    )

    assert len(results) == 1

    assert results[0].score == 0.95

    embedder.embed_query.assert_called_once()

    vector_store.search.assert_called_once()