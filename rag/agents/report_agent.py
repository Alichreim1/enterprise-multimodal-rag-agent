import json
import re

from rag.agents.base import BaseAgent
from rag.llms.base_llm import BaseLLM
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class ReportAgent(BaseAgent):

    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    @property
    def name(self) -> str:
        return "report_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        # -----------------------------------------
        # Validate Dependencies
        # -----------------------------------------

        if not context.dependency_results:
            return AgentResult(
                status="failure",
                message=(
                    "No dependency results "
                    "are available for the report."
                ),
            )

        failed_dependencies = [
            dependency_id
            for dependency_id, result
            in context.dependency_results.items()
            if result.status != "success"
        ]

        if failed_dependencies:
            return AgentResult(
                status="failure",
                data={
                    "failed_dependencies": (
                        failed_dependencies
                    ),
                },
                message=(
                    "One or more report "
                    "dependencies failed."
                ),
            )

        # -----------------------------------------
        # Build Structured Dependency Context
        # -----------------------------------------

        dependency_data = {
            str(dependency_id): result.data
            for dependency_id, result
            in context.dependency_results.items()
        }

        dependency_text = json.dumps(
            dependency_data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        # -----------------------------------------
        # Generate Report
        # -----------------------------------------

        prompt = self._build_prompt(
            task_description=(
                context.task_description
            ),
            dependency_text=dependency_text,
        )

        try:
            raw_result = self.llm.generate(
                prompt
            )

        except Exception as exc:
            return AgentResult(
                status="failure",
                message=(
                    "Report generation failed: "
                    f"{exc}"
                ),
            )

        report_data = self._parse_result(
            raw_result
        )

        if report_data is None:
            return AgentResult(
                status="failure",
                message=(
                    "Could not parse the "
                    "generated report."
                ),
            )

        return AgentResult(
            status="success",
            data=report_data,
            message=(
                "Report generated successfully."
            ),
        )

    # ==================================================
    # Prompt
    # ==================================================

    def _build_prompt(
        self,
        task_description: str,
        dependency_text: str,
    ) -> str:

        return f"""
You are a report-writing agent.

Your job is to create the report requested in the
current task using only the dependency results.

Rules:

1. Follow the report task exactly.
2. Use only facts contained in dependency results.
3. Do not search for new information.
4. Do not use tools.
5. Do not invent missing facts.
6. Keep the report concise unless the task requests
   detailed content.
7. Use a clear and natural title.
8. Return only valid JSON.
9. Do not add Markdown code fences.

Required format:

{{
  "title": "...",
  "content": "..."
}}

Report task:
{task_description}

Dependency results:
{dependency_text}
"""

    # ==================================================
    # Parse Result
    # ==================================================

    def _parse_result(
        self,
        raw_result: str,
    ) -> dict | None:

        cleaned = raw_result.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        title = data.get("title")
        content = data.get("content")

        if (
            not isinstance(title, str)
            or not title.strip()
        ):
            return None

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            return None

        return {
            "title": title.strip(),
            "content": content.strip(),
        }