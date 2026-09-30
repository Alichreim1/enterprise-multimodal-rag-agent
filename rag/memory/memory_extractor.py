import json
import logging

from rag.llms.base_llm import BaseLLM
from rag.memory.models import MemoryItem


logger = logging.getLogger(__name__)


class MemoryExtractor:
    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    def extract(
        self,
        text: str,
    ) -> list[MemoryItem]:

        prompt = self._build_prompt(text)

        response = self.llm.generate(prompt)

        print(
            "\n================ MEMORY EXTRACTOR ================"
        )
        print(f"Raw response: {response}")
        print("===================================================\n")

        return self._parse_response(response)

    def _build_prompt(
        self,
        text: str,
    ) -> str:

        return f"""
You are a memory extraction system.

You are given a confirmed user statement that has already
been identified as information worth considering for memory.

Your job is to extract the user information contained in
this statement and convert it into a reusable memory.

Do NOT treat the statement as a question or task.

Store information only if it is:

1. About the user.
2. Explicitly stated by the user.
3. Likely to remain useful in future conversations.
4. A preference, fact, goal, project, or interest.

Possible Memory type definitions:

- preference:
  Something the user likes, dislikes, prefers, or considers
  their favorite.

- fact:
  A stable factual statement about the user.

- goal:
  Something the user wants to achieve.

- project:
  Something the user is currently building or working on.

- interest:
  A topic or activity the user is interested in.

Important classification rule:

If the user says "favorite", "prefer", "like", or "dislike",
use "preference" rather than "interest".

Rewrite the information as a short standalone statement
that can be understood without the original conversation.

Examples:

Statement:
"My favorite programming language is Python."

Output:
[
    {{
        "content": "User's favorite programming language is Python.",
        "memory_type": "preference"
    }}
]

Statement:
"I am building a RAG agent in Python."

Output:
[
    {{
        "content": "User is building a RAG agent in Python.",
        "memory_type": "project"
    }}
]

Statement:
"I enjoy basketball."

Output:
[
    {{
        "content": "User enjoys basketball.",
        "memory_type": "interest"
    }}
]

Statement:
"I prefer Python over Java."

Output:
[
    {{
        "content": "User prefers Python over Java.",
        "memory_type": "preference"
    }}
]

Return ONLY valid JSON.

STATEMENT:
{text}
"""

    def _parse_response(
        self,
        response: str,
    ) -> list[MemoryItem]:

        try:
            data = json.loads(response)

        except json.JSONDecodeError:
            logger.warning(
                "Memory extractor returned invalid JSON: %s",
                response,
            )
            return []

        if not isinstance(data, list):
            return []

        memories = []

        for item in data:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            memory_type = item.get("memory_type")

            if not content:
                continue

            if not memory_type:
                continue

            memories.append(
                MemoryItem(
                    id=None,
                    content=str(content).strip(),
                    memory_type=str(memory_type).strip(),
                )
            )

        return memories
