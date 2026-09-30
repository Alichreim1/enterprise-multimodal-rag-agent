import json

from rag.agents.base import BaseAgent
from rag.llms.base_llm import BaseLLM
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class EmailAgent(BaseAgent):

    def __init__(
        self,
        email_tool,
        llm: BaseLLM,
    ):
        self.email_tool = email_tool
        self.llm = llm

    @property
    def name(self) -> str:
        return "email_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        if not context.dependency_results:
            return AgentResult(
                status="failure",
                message=(
                    "No dependency results "
                    "are available."
                ),
            )

        # -----------------------------------------
        # Validate Dependencies
        # -----------------------------------------

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
                    "One or more dependency "
                    "tasks failed."
                ),
            )

        # -----------------------------------------
        # Find Recipient In All Dependencies
        # -----------------------------------------

        recipient = None

        for result in context.dependency_results.values():
            email = result.data.get("email")

            if email:
                recipient = email
                break

        if not recipient:
            return AgentResult(
                status="failure",
                message=(
                    "Email was not found in "
                    "dependency results."
                ),
            )

        # -----------------------------------------
        # Extract Subject And Body
        # -----------------------------------------

        email_content = self._extract_email_content(
            task_description=context.task_description,
            dependency_results=context.dependency_results,
        )

        if email_content is None:
            return AgentResult(
                status="failure",
                message=(
                    "Could not extract "
                    "email content."
                ),
            )

        subject = email_content["subject"]
        body = email_content["body"]

        # -----------------------------------------
        # Send Email
        # -----------------------------------------

        try:
            tool_result = self.email_tool.run(
                to=recipient,
                subject=subject,
                body=body,
            )

        except Exception as exc:
            return AgentResult(
                status="failure",
                data={
                    "recipient": recipient,
                    "subject": subject,
                    "body": body,
                },
                message=(
                    f"Email sending failed: {exc}"
                ),
            )

        # -----------------------------------------
        # Return Structured Result
        # -----------------------------------------

        return AgentResult(
            status="success",
            data={
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "tool_result": tool_result,
            },
            message="Email sent successfully.",
        )

    def _extract_email_content(
        self,
        task_description: str,
        dependency_results: dict,
    ) -> dict | None:

        prompt = f"""
Extract the email subject and body from the task below.

Use the dependency results as additional context.

Return ONLY valid JSON.

Required format:
{{
    "subject": "...",
    "body": "..."
}}

Task:
{task_description}

Dependency Results:
{dependency_results}
"""

        raw_result = self.llm.generate(prompt)

        try:
            data = json.loads(raw_result.strip())
        except json.JSONDecodeError:
            return None

        subject = data.get("subject")
        body = data.get("body")

        if not subject or not body:
            return None

        return {
            "subject": subject,
            "body": body,
        }
