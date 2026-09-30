import json
import re

from rag.llms.base_llm import BaseLLM
from rag.models.planner import PlannerDecision


class Planner:

    ALLOWED_ACTIONS = {
        "retrieve",
        "tool",
        "finish",
    }

    def __init__(
        self,
        llm: BaseLLM,
        tool_manager,
    ):
        self.llm = llm
        self.tool_manager = tool_manager

    def plan(
        self,
        task: str,
    ) -> PlannerDecision:

        prompt = self._build_prompt(task)

        raw_result = self.llm.generate(prompt)

        data = self._parse_response(raw_result)

        return self._build_decision(data)

    def _build_prompt(
        self,
        task: str,
    ) -> str:

        tool_descriptions = "\n".join(
            [
                f"- {schema['function']['name']}: "
                f"{schema['function']['description']}\n"
                f"  Parameters: "
                f"{schema['function']['parameters']}"
                for schema in self.tool_manager.schemas
            ]
        )

        return f"""
You are the planning component of an AI agent.

Your job is to decide the next action required to solve the CURRENT TASK.

You do NOT solve the task.
You only decide what action should be taken.

Allowed actions:

- retrieve: when information must be retrieved from the knowledge system
- tool: when a tool is required
- finish: ONLY when the current task has already been completed

Available tools:

{tool_descriptions}

IMPORTANT:

The CURRENT TASK is a new task that has not yet been completed unless
there is explicit evidence that it is already completed.

A task that asks for information is NOT completed just because you
understand the question or know what the answer would be.

Rules:

1. Focus only on the CURRENT TASK.
2. Do not solve the task yourself.
3. Do not perform calculations yourself.
4. Do not generate the final answer.
5. Use "retrieve" for information-seeking tasks that require the
   knowledge system.
6. Use "tool" when one of the available tools is required.
7. NEVER use "finish" for a new task that has not been executed yet.
8. Use "finish" ONLY when there is explicit evidence that the current
   task is already completed and no further action is required.
9. Do not use "finish" simply because the task is easy.
10. Do not use "finish" simply because you know the answer.
11. For a task asking to find, identify, retrieve, look up, or obtain
    information from the knowledge system, use "retrieve".
12. For a task requiring a calculation, use the appropriate calculation tool.
13. For a task requiring the current date or time, use the datetime tool.
14. For a task requiring an external action, use the appropriate available tool.
15. When using a tool, provide the exact tool name from the available tools and the required arguments.
16. If a tool requires no arguments, use an empty object {{}}.
17. Return valid JSON only.
18. Do not add explanations outside the JSON.

For a retrieve action, use:

{{
  "action": "retrieve",
  "reason": "The task requires information retrieval.",
  "tool_name": null,
  "arguments": {{}}
}}

For a tool action, use:

{{
  "action": "tool",
  "reason": "The task requires a tool.",
  "tool_name": "calculator",
  "arguments": {{
    "expression": "25 * 8"
  }}
}}

For a tool that requires no arguments, use:

{{
  "action": "tool",
  "reason": "The task requires the datetime tool.",
  "tool_name": "datetime",
  "arguments": {{}}
}}

For a finish action, use:

{{
  "action": "finish",
  "reason": "The task is already completed.",
  "tool_name": null,
  "arguments": {{}}
}}

Examples:

CURRENT TASK:
Identify the capital of France.

Correct decision:

{{
  "action": "retrieve",
  "reason": "The task requires information retrieval.",
  "tool_name": null,
  "arguments": {{}}
}}

CURRENT TASK:
Calculate 235 multiplied by 3.

Correct decision:

{{
  "action": "tool",
  "reason": "The task requires a calculation.",
  "tool_name": "calculator",
  "arguments": {{
    "expression": "235 * 3"
  }}
}}

CURRENT TASK:
What is the current date and time?

Correct decision:

{{
  "action": "tool",
  "reason": "The task requires the datetime tool.",
  "tool_name": "datetime",
  "arguments": {{}}
}}

CURRENT TASK:
Find Ali's GitHub account or profile.

Correct decision:

{{
  "action": "retrieve",
  "reason": "The task requires information retrieval.",
  "tool_name": null,
  "arguments": {{}}
}}

CURRENT TASK:
{task}
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
                f"Planner returned invalid JSON: {raw_result}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Planner response must be a JSON object."
            )

        if "action" not in data:
            raise ValueError(
                "Planner response must contain 'action'."
            )

        return data

    def _build_decision(
        self,
        data: dict,
    ) -> PlannerDecision:

        action = data.get("action")
        reason = data.get("reason")
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})

        if action not in self.ALLOWED_ACTIONS:
            raise ValueError(
                f"Invalid planner action: {action}"
            )

        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                "Planner reason must be a string or None."
            )

        if tool_name is not None and not isinstance(tool_name, str):
            raise ValueError(
                "Planner tool_name must be a string or None."
            )

        if not isinstance(arguments, dict):
            raise ValueError(
                "Planner arguments must be a dictionary."
            )

        if action == "tool":

            if not tool_name:
                raise ValueError(
                    "Tool action requires 'tool_name'."
                )

        else:

            if tool_name is not None:
                raise ValueError(
                    f"{action} action must not specify a tool."
                )

            if arguments:
                raise ValueError(
                    f"{action} action must not specify arguments."
                )

        return PlannerDecision(
            action=action,
            reason=reason,
            tool_name=tool_name,
            arguments=arguments,
        )