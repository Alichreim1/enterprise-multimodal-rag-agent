import logging

from rag.models.multimodal_search_result import (
    MultimodalSearchResult,
)


logger = logging.getLogger(__name__)


class MultimodalRetriever:

    def __init__(
        self,
        text_retriever,
        image_retriever,
        text_top_k: int = 5,
        image_top_k: int = 3,
    ):
        self.text_retriever = (
            text_retriever
        )

        self.image_retriever = (
            image_retriever
        )

        self.text_top_k = text_top_k
        self.image_top_k = image_top_k

    def retrieve(
        self,
        query: str,
    ) -> MultimodalSearchResult:

        query = query.strip()

        if not query:
            raise ValueError(
                "Retrieval query cannot "
                "be empty."
            )

        logger.info(
            "Running multimodal retrieval "
            "for query: %s",
            query,
        )

        # Keep the scores separate because BGE
        # and CLIP use different embedding spaces.
        text_results = (
            self.text_retriever.retrieve(
                query=query,
                top_k=self.text_top_k,
            )
        )

        image_results = (
            self.image_retriever.retrieve(
                query=query,
                top_k=self.image_top_k,
            )
        )

        result = MultimodalSearchResult(
            text_results=text_results,
            image_results=image_results,
        )

        logger.info(
            "Multimodal retrieval completed. "
            "Text results: %d, "
            "Image results: %d",
            len(text_results),
            len(image_results),
        )

        return result