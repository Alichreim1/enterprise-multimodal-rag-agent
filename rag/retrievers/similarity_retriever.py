import logging

from rag.embeddings.base_embedder import BaseEmbedder
from rag.models.search_result import SearchResult
from rag.retrievers.base_retriever import BaseRetriever
from rag.vectorstores.base_vector_store import BaseVectorStore


logger = logging.getLogger(__name__)


class SimilarityRetriever(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        min_score: float = 0.00,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:

        top_k = (
            self.top_k
            if top_k is None
            else top_k
        )

        min_score = (
            self.min_score
            if min_score is None
            else min_score
        )

        print()
        print("=" * 60)
        print("RETRIEVER DEBUG")
        print("Query:", query)
        print("top_k:", top_k)
        print("min_score:", min_score)
        print("=" * 60)

        query_vector = self.embedder.embed_query(
            query
        )

        results = self.vector_store.search(
            query_vector=query_vector,
            k=top_k,
        )

        print()
        print("RAW RESULTS:", len(results))

        for i, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(f"Result {i}")
            print("Score:", result.score)
            print(
                "Source:",
                result.chunk.metadata.get("source"),
            )
            print(
                "Page:",
                result.chunk.metadata.get("page"),
            )

            if result.chunk.is_image:
                print(
                    "Image:",
                    result.chunk.image_path,
                )
            else:
                print(
                    "Content:",
                    result.chunk.content[:300],
                )

        filtered_results = [
            result
            for result in results
            if result.score >= min_score
        ]

        print()
        print(
            "FILTERED RESULTS:",
            len(filtered_results),
        )

        return filtered_results
