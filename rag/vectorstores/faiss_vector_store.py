import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.models.embedded_chunk import EmbeddedChunk
from rag.models.search_result import SearchResult
from rag.vectorstores.base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):

    def __init__(self, dimension: int):

        self.index = faiss.IndexFlatIP(dimension)

        self.embedded_chunks: list[EmbeddedChunk] = []

    def add(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:

        if not embedded_chunks:
            return

        vectors = np.asarray(
            [chunk.vector for chunk in embedded_chunks],
            dtype=np.float32,
        )

        self.index.add(vectors)

        self.embedded_chunks.extend(
            embedded_chunks
        )

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[SearchResult]:

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            [query_vector],
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:
                continue

            results.append(

                SearchResult(

                    chunk=self.embedded_chunks[index].chunk,

                    score=float(score),

                )

            )

        return results

    @property
    def size(self) -> int:

        return self.index.ntotal

    def clear(self) -> None:

        self.index.reset()

        self.embedded_chunks.clear()

    def save(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with open(
            metadata_path,
            "wb",
        ) as file:

            pickle.dump(
                self.embedded_chunks,
                file,
            )

    def load(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "rb",
        ) as file:

            self.embedded_chunks = pickle.load(
                file
            )

    def exists(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> bool:

        return (
            index_path.exists()
            and metadata_path.exists()
        )