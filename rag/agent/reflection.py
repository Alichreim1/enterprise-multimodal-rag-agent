import json
import re

from rag.llms.base_llm import BaseLLM
from rag.models.reflection import ReflectionDecision


class Reflection:

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def reflect(
        self,
        task: str,
        result,
    ) -> ReflectionDecision:

        prompt = self._build_prompt(
            task,
            result,
        )

        raw_result = self.llm.generate(prompt)

        data = self._parse_response(raw_result)

        return self._build_decision(data)

    def _build_prompt(
        self,
        task: str,
        result,
    ) -> str:

        return f"""
You are the reflection component of an AI agent.

Your job is to evaluate whether the result obtained for the CURRENT TASK
is sufficient to consider the task completed.

You do NOT solve the task.
You do NOT generate a final user-facing answer.
You only evaluate the result.

Return valid JSON only.

Output format:

{{
  "completed": true,
  "reason": "The result directly answers the task."
}}

Rules:

1. Focus only on the CURRENT TASK.
2. Compare the task with the obtained result.
3. Set "completed" to true only when the result sufficiently satisfies the task.
4. Set "completed" to false when the result is missing, irrelevant, incomplete, or does not answer the requested information.
5. Do not assume information that is not present in the result.
6. If the result is exactly "I don't know based on the provided context.",
   treat the task as completed because the retrieval attempt has reached
   a valid final negative result.
7. Do not solve the task yourself.
8. Do not generate a final answer.
9. Return valid JSON only.
10. Do not add explanations outside the JSON.


CURRENT TASK:
{task}

RESULT:
{result}
"""

    def _parse_response(
        self,
        raw_result: str,
    ) -> dict:

        cleaned = raw_result.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Reflection returned invalid JSON: {raw_result}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Reflection response must be a JSON object."
            )

        if "completed" not in data:
            raise ValueError(
                "Reflection response must contain 'completed'."
            )

        return data

    def _build_decision(
        self,
        data: dict,
    ) -> ReflectionDecision:

        completed = data.get("completed")
        reason = data.get("reason")

        if not isinstance(completed, bool):
            raise ValueError(
                "Reflection 'completed' must be a boolean."
            )

        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                "Reflection reason must be a string or None."
            )

        return ReflectionDecision(
            completed=completed,
            reason=reason,
        )