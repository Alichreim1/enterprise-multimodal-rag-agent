import json
from pathlib import Path

from rag.memory.memory_extractor import (
    MemoryExtractor,
)
from rag.memory.models import MemoryItem
from rag.memory.vector_memory import (
    VectorMemory,
)


class MemoryManager:

    SEMANTIC_THRESHOLD = 0.90

    def __init__(
        self,
        extractor: MemoryExtractor,
        vector_memory: VectorMemory,
        index_path: Path,
        metadata_path: Path,
        reasoning_llm,
    ):
        self.extractor = extractor
        self.vector_memory = vector_memory
        self.reasoning_llm = reasoning_llm

        self.index_path = index_path
        self.metadata_path = metadata_path

    # ==================================================
    # JSON Parsing
    # ==================================================

    @staticmethod
    def _parse_json_response(
        response: str,
    ) -> dict | None:
        if not isinstance(response, str):
            return None

        cleaned_response = response.strip()

        if cleaned_response.startswith("```"):
            lines = cleaned_response.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            cleaned_response = "\n".join(
                lines
            ).strip()

        try:
            data = json.loads(
                cleaned_response
            )
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return None

        if not isinstance(data, dict):
            return None

        return data

    # ==================================================
    # Remember
    # ==================================================

    def remember(
        self,
        text: str,
    ) -> list[MemoryItem]:
        print(
            "\n================ MEMORY REMEMBER "
            "================"
        )

        print(f"Input text: {text}")

        memories = self.extractor.extract(
            text
        )

        print(
            f"[MEMORY] Extracted memories: "
            f"{len(memories)}"
        )

        for memory in memories:
            print(
                f"[MEMORY] Extracted: "
                f"type={memory.memory_type} | "
                f"content={memory.content}"
            )

        if not memories:
            print(
                "[MEMORY] Nothing extracted."
            )

            print(
                "================================"
                "=================\n"
            )

            return []

        new_memories = []

        for memory in memories:
            if self._is_duplicate(memory):
                print(
                    f"[MEMORY] Duplicate skipped: "
                    f"{memory.content}"
                )

                continue

            new_memories.append(memory)

        print(
            f"[MEMORY] New memories to save: "
            f"{len(new_memories)}"
        )

        if not new_memories:
            print(
                "[MEMORY] Nothing new to save."
            )

            print(
                "================================"
                "=================\n"
            )

            return []

        self.vector_memory.add_many(
            new_memories
        )

        print(
            f"[MEMORY] Added "
            f"{len(new_memories)} "
            f"memory(s) to VectorMemory."
        )

        self.vector_memory.save(
            index_path=self.index_path,
            metadata_path=(
                self.metadata_path
            ),
        )

        print(
            f"[MEMORY] Saved successfully. "
            f"Total memory size: "
            f"{self.vector_memory.size}"
        )

        print(
            "================================"
            "=================\n"
        )

        return new_memories

    # ==================================================
    # Duplicate Detection
    # ==================================================

    def _is_duplicate(
        self,
        memory: MemoryItem,
    ) -> bool:
        results = (
            self.vector_memory
            .retrieve_with_scores(
                query=memory.content,
                k=3,
            )
        )

        if not results:
            return False

        new_content = (
            memory.content.strip()
        )

        for result in results:
            existing_content = (
                result.chunk.content.strip()
            )

            score = result.score

            # Exact duplicate
            if existing_content == new_content:
                return True

            # Similarity is not high enough
            # to require an LLM comparison.
            if (
                score
                < self.SEMANTIC_THRESHOLD
            ):
                continue

            print(
                "[MEMORY] Possible semantic "
                "duplicate:"
            )

            print(
                f"    Existing: "
                f"{existing_content}"
            )

            print(
                f"    New:      "
                f"{new_content}"
            )

            print(
                f"    Score:    "
                f"{score:.4f}"
            )

            if self._semantic_duplicate(
                existing_content,
                new_content,
            ):
                return True

        return False

    # ==================================================
    # Semantic Duplicate
    # ==================================================

    def _semantic_duplicate(
        self,
        existing: str,
        new: str,
    ) -> bool:
        prompt = f"""
You are a memory deduplication system.

Determine whether the NEW memory represents the
same user information as the EXISTING memory.

The memories may use different wording.

Return "duplicate" only when they describe the same
underlying user fact, preference, goal, project,
or interest.

Return "different" when they contain different
values, entities, choices, or facts.

Rules:

1. Similar sentence structure does not mean duplicate.
2. Different values mean different memories.
3. Do not infer information that is not stated.
4. Ignore minor wording differences.

Examples:

"User prefers Python" and
"User prefers Java" are different.

"User prefers Python" and
"User's favorite programming language is Python"
are duplicates.

EXISTING MEMORY:

{existing}

NEW MEMORY:

{new}

Return ONLY valid JSON:

{{
    "result": "duplicate"
}}

or:

{{
    "result": "different"
}}
"""

        response = (
            self.reasoning_llm.generate(
                prompt
            )
        )

        data = self._parse_json_response(
            response
        )

        if data is None:
            return False

        return (
            data.get("result")
            == "duplicate"
        )

    # ==================================================
    # Recall
    # ==================================================

    def recall(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemoryItem]:
        return self.vector_memory.retrieve(
            query=query,
            k=k,
        )

    # ==================================================
    # Relevant Memory Recall
    # ==================================================

    def recall_relevant(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryItem]:
        print(
            "\n================ MEMORY RECALL "
            "================"
        )

        print(f"Query: {query}")
        print(f"Top K: {k}")

        candidates = (
            self.vector_memory.retrieve(
                query=query,
                k=k,
            )
        )

        print(
            f"[MEMORY RECALL] "
            f"Candidates retrieved: "
            f"{len(candidates)}"
        )

        if not candidates:
            print(
                "[MEMORY RECALL] "
                "No candidates found."
            )

            print(
                "================================"
                "================\n"
            )

            return []

        for index, memory in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"[MEMORY RECALL] "
                f"Candidate {index}: "
                f"type={memory.memory_type} | "
                f"content={memory.content}"
            )

        # One LLM request for all candidates.
        relevant_memories = (
            self._select_relevant_memories(
                candidates=candidates,
                query=query,
            )
        )

        relevant_ids = {
            id(memory)
            for memory in relevant_memories
        }

        for memory in candidates:
            is_relevant = (
                id(memory) in relevant_ids
            )

            print(
                f"[MEMORY RECALL] "
                f"Relevance: {is_relevant} | "
                f"memory={memory.content}"
            )

        print(
            f"[MEMORY RECALL] "
            f"Relevant memories: "
            f"{len(relevant_memories)}"
        )

        for memory in relevant_memories:
            print(
                f"[MEMORY RECALL] "
                f"Selected: "
                f"{memory.content}"
            )

        print(
            "================================"
            "================\n"
        )

        return relevant_memories

    # ==================================================
    # Batch Memory Relevance
    # ==================================================

    def _select_relevant_memories(
        self,
        candidates: list[MemoryItem],
        query: str,
    ) -> list[MemoryItem]:
        """
        Evaluate all candidate memories with one
        reasoning LLM request.

        Every candidate is evaluated independently.
        """

        if not candidates:
            return []

        candidate_data = [
            {
                "id": index,
                "memory_type": (
                    memory.memory_type
                ),
                "content": memory.content,
            }
            for index, memory in enumerate(
                candidates,
                start=1,
            )
        ]

        candidates_json = json.dumps(
            candidate_data,
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
You are a strict memory relevance evaluator.

Evaluate every candidate memory against the user's
question.

A memory is relevant only when that individual memory
directly provides information needed to answer at least
one part of the question.

Rules:

1. Evaluate every memory independently.
2. Do not combine memories to infer new information.
3. Do not infer, assume, or invent information.
4. The person, subject, or entity must match.
5. "my", "I", and "me" refer to the user.
6. The requested attribute must match the memory.
7. A shared topic alone is not sufficient.
8. Do not use outside or world knowledge.
9. A memory that answers one part of a multi-part
   question is relevant.
10. Return only the IDs of relevant memories.
11. If no memory is relevant, return an empty list.
12. Do not include explanations.

USER QUESTION:

{query}

CANDIDATE MEMORIES:

{candidates_json}

Return ONLY valid JSON:

{{
    "relevant_ids": [1, 3]
}}

If no candidate is relevant:

{{
    "relevant_ids": []
}}
"""

        response = (
            self.reasoning_llm.generate(
                prompt
            )
        )

        data = self._parse_json_response(
            response
        )

        if data is None:
            print(
                "[MEMORY RECALL] Invalid batch "
                "relevance response."
            )

            return []

        relevant_ids = data.get(
            "relevant_ids",
            [],
        )

        if not isinstance(
            relevant_ids,
            list,
        ):
            print(
                "[MEMORY RECALL] relevant_ids "
                "is not a list."
            )

            return []

        valid_ids = {
            memory_id
            for memory_id in relevant_ids
            if (
                isinstance(memory_id, int)
                and not isinstance(
                    memory_id,
                    bool,
                )
                and 1
                <= memory_id
                <= len(candidates)
            )
        }

        return [
            memory
            for index, memory in enumerate(
                candidates,
                start=1,
            )
            if index in valid_ids
        ]

    # ==================================================
    # Single Memory Relevance
    # ==================================================

    def _memory_answers_query(
        self,
        memory: str,
        query: str,
    ) -> bool:
        """
        Kept for backward compatibility and focused
        single-memory tests.

        recall_relevant() now uses the batch evaluator.
        """

        prompt = f"""
You are a strict memory relevance evaluator.

Determine whether the MEMORY can directly answer any
part of the USER QUESTION.

Rules:

1. The memory must contain the actual information.
2. Do not infer, assume, or invent information.
3. The subject, entity, or person must match.
4. Do not transfer attributes between entities.
5. "my", "I", and "me" refer to the user.
6. The requested attribute must match the memory.
7. Shared topic alone is not enough.
8. Do not use world knowledge.
9. If the memory directly answers one part of a
   multi-part question, return "relevant".
10. Otherwise return "not_relevant".

MEMORY:

{memory}

USER QUESTION:

{query}

Return ONLY valid JSON:

{{
    "result": "relevant"
}}

or:

{{
    "result": "not_relevant"
}}
"""

        response = (
            self.reasoning_llm.generate(
                prompt
            )
        )

        data = self._parse_json_response(
            response
        )

        if data is None:
            return False

        return (
            data.get("result")
            == "relevant"
        )

    # ==================================================
    # Persistence
    # ==================================================

    def load(self) -> None:
        if not (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):
            return

        self.vector_memory.load(
            index_path=self.index_path,
            metadata_path=(
                self.metadata_path
            ),
        )

    def save(self) -> None:
        self.vector_memory.save(
            index_path=self.index_path,
            metadata_path=(
                self.metadata_path
            ),
        )

    # ==================================================
    # Size
    # ==================================================

    @property
    def size(self) -> int:
        return self.vector_memory.size