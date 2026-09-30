
import json

from rag.llms.base_llm import BaseLLM
from rag.memory.models import MemoryItem
from rag.models.result import AgentResult
from rag.models.task import AgentTask


class FinalAnswerGenerator:

    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    def generate(
        self,
        question: str,
        tasks: list[AgentTask],
        memories: list[MemoryItem],
    ) -> str:

        prompt = self._build_prompt(
            question=question,
            tasks=tasks,
            memories=memories,
        )

        return self.llm.generate(prompt)

    def generate_stream(
        self,
        question: str,
        tasks: list[AgentTask],
        memories: list[MemoryItem],
    ):

        prompt = self._build_prompt(
            question=question,
            tasks=tasks,
            memories=memories,
        )

        yield from self.llm.generate_stream(
            prompt
        )

    def _build_prompt(
        self,
        question: str,
        tasks: list[AgentTask],
        memories: list[MemoryItem],
    ) -> str:

        task_results = []

        for task in tasks:

            if task.result is None:
                continue

            formatted_result = self._format_task_result(
                task.result
            )

            task_results.append(
                f"""
Task:
{task.description}

Result:
{formatted_result}
"""
            )

        task_results_text = "\n".join(
            task_results
        )

        memory_context = []

        for memory in memories:
            memory_context.append(
                f"- {memory.content}"
            )

        memories_text = "\n".join(
            memory_context
        )

        if not memories_text:
            memories_text = (
                "No relevant memories were found."
            )

        return f"""
You are the final answer generator of an AI agent.

Your job is to turn the available information into
a natural, concise answer for the user.

You do NOT perform new retrieval.
You do NOT perform calculations.
You do NOT use tools.
You do NOT search for information.
You do NOT invent information.

You may use two sources of information:

1. TASK RESULTS

These contain results produced by the agent while solving
the user's current request.

2. RELEVANT MEMORIES

These contain previously stored information about the user.
Use them only when they are relevant to the user's question.

Rules:

1. Answer the user's original question.
2. Use only information contained in the task results
   and relevant memories.
3. Use memories only when they directly help answer the question.
4. Do not treat memories as general knowledge.
5. Do not invent or assume information that is not provided.
6. Combine task results and memories naturally when both are relevant.
7. If the requested information is unavailable, communicate that naturally.
8. When a successful task result contains a concrete value
   explicitly requested by the user, include that value in
   the final answer.
9. Concrete requested values include email addresses, URLs,
   names, dates, calculation results, and identifiers.
10. When the user asks to find contact information and then
    perform an action using it, include both the contact
    information and the outcome of the action.
11. Do not mention internal components such as Planner, Executor,
    Reflection, RAG, Memory, AgentTask, or the agent architecture.
12. Do not repeat the user's question.
13. Keep the answer concise and natural.
14. Return only the final answer.
15. Do not return JSON.
16.Return plain text only ,Do not use Markdown formatting ,When task results contain both a raw URL and a formatted version, use the raw URL.

USER QUESTION:
{question}

RELEVANT MEMORIES:
{memories_text}

TASK RESULTS:
{task_results_text}
"""

    def _format_task_result(
        self,
        result,
    ) -> str:

        # Multi-Agent structured result
        if isinstance(result, AgentResult):

            data_text = json.dumps(
                result.data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            parts = [
                f"Status: {result.status}",
                f"Data:\n{data_text}",
            ]

            if result.message:
                parts.append(
                    f"Message: {result.message}"
                )

            return "\n".join(parts)

        # Legacy result
        return str(result)

