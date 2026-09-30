import json
import re

from rag.llms.base_llm import BaseLLM
from rag.models.task import AgentTask


class TaskDecomposer:

    ALLOWED_TASK_TYPES = {
        "retrieve",
        "web_search",
        "report",
        "email",
        "tool",
        "statement",
    }

    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    def decompose(
        self,
        question: str,
    ) -> list[AgentTask]:

        prompt = self._build_prompt(question)

        raw_result = self.llm.generate(prompt)

        data = self._parse_response(raw_result)

        return self._build_tasks(data)

    # ==================================================
    # Prompt
    # ==================================================

    def _build_prompt(
        self,
        question: str,
    ) -> str:

        return f"""
You are the task decomposition component of a
multi-agent system.

Your job is to decompose the user's request into
small, ordered tasks.

You do NOT execute or solve any task.

For every task return:

- id: a unique integer starting from 1
- description: the exact task that must be executed
- task_type: one of:
  "retrieve", "web_search", "report",
  "email", "tool", or "statement"
- is_statement: true only when the user is providing
  information rather than requesting information or
  an action
- depends_on: a list containing the IDs of earlier
  tasks whose results are required by this task

==================================================
TASK TYPES
==================================================

1. RETRIEVE

Use "retrieve" for information expected to exist in
the uploaded document or internal knowledge system.

Information about a person or organization contained
in an uploaded document must use "retrieve".

Examples include:

- Email addresses
- Phone numbers
- LinkedIn profiles
- GitHub profiles
- Personal websites
- Education
- Employment history
- Experience
- Skills
- Projects
- Company policies
- Document facts

Words such as "link", "URL", "profile", "LinkedIn",
"GitHub", or "website" do NOT automatically mean
web_search.

If the requested information can reasonably be
answered from the uploaded document, prefer
"retrieve" over "web_search".

2. WEB_SEARCH

Use "web_search" only when:

- The user explicitly requests searching the web,
  internet, or online sources.
- The user asks for current, recent, latest, live,
  or changing external information.
- The requested information is clearly outside the
  uploaded document or internal knowledge system.

Examples include:

- Latest stable software version
- Current news
- Current product prices
- Recent public announcements
- Live external information
- Explicit online searches

Do not use "web_search" merely because the requested
information is a link or public profile.

3. REPORT

Use "report" when information from earlier tasks must
be summarized, organized, rewritten, compared, or
converted into a report.

A report task must depend on every earlier task whose
information it uses.

The report task must not perform new retrieval or web
search.

4. EMAIL

Use "email" when an email must be sent.

The email task must depend on:

- The task that found the recipient's email address.
- The web_search task if raw web research is being
  emailed directly.
- The report task if a generated report is being
  emailed.
- Every other task whose output is required to compose
  or send the email.

Preserve an explicitly provided email subject and body
inside the email task description.

5. TOOL

Use "tool" for deterministic utility operations such as:

- Mathematical calculations
- Current date or time
- Other available non-email utility tools

Do not use web_search for the current date or time.

6. STATEMENT

Use "statement" only when the user is providing personal
information or a fact without asking a question or
requesting an action.

For statement tasks:

- task_type must be "statement"
- is_statement must be true

For all other task types:

- is_statement must be false

==================================================
DEPENDENCY RULES
==================================================

1. Every task must have a unique integer ID.

2. Task IDs must start from 1 and follow execution order.

3. A task may depend only on an earlier task.

4. Use an empty depends_on list when the task does not
   require another task's result.

5. Do not add a dependency merely because another task
   appears earlier.

6. A report must depend on the research tasks whose
   information it uses.

7. An email must depend on the task that provides the
   recipient's email address.

8. If an email sends a generated report, it must depend
   on both:
   - the recipient retrieval task
   - the report task

9. If an email directly sends web research without a
   report task, it must depend on:
   - the web_search task
   - the recipient retrieval task

10. Do not create circular dependencies.

==================================================
GENERAL RULES
==================================================

1. Do not solve the tasks.

2. Do not answer the user's question.

3. Do not retrieve information.

4. Do not search the web.

5. Do not write the requested report.

6. Do not send an email.

7. Do not perform calculations.

8. Preserve the user's original meaning.

9. Preserve exact mathematical expressions.

10. Preserve names, dates, email subjects, email bodies,
    report requirements, and search queries.

11. Split the request only when separate execution steps
    are actually required.

12. Do not create unnecessary tasks.

13. Do not create a report task unless the user requests
    a report, summary, comparison, rewrite, or organized
    content.

14. Do not create an email task unless the user asks to
    send an email.

15. Return valid JSON only.

16. Do not add Markdown code fences.

17. Do not add explanations before or after the JSON.

==================================================
OUTPUT FORMAT
==================================================

{{
  "tasks": [
    {{
      "id": 1,
      "description": "...",
      "task_type": "retrieve",
      "is_statement": false,
      "depends_on": []
    }}
  ]
}}

==================================================
EXAMPLE 1: DOCUMENT PROFILE LINK
==================================================

User request:
What is Ali's LinkedIn link?

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Find Ali's LinkedIn link in the uploaded document.",
      "task_type": "retrieve",
      "is_statement": false,
      "depends_on": []
    }}
  ]
}}

==================================================
EXAMPLE 2: EXPLICIT WEB PROFILE SEARCH
==================================================

User request:
Search the web for Ali's latest LinkedIn post.

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Search the web for Ali's latest LinkedIn post.",
      "task_type": "web_search",
      "is_statement": false,
      "depends_on": []
    }}
  ]
}}

==================================================
EXAMPLE 3: DOCUMENT RETRIEVAL AND EMAIL
==================================================

User request:
Find Ali's email and send him an email with subject
"Meeting" and body "The meeting is tomorrow at 5 PM."

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Find Ali's email in the uploaded document.",
      "task_type": "retrieve",
      "is_statement": false,
      "depends_on": []
    }},
    {{
      "id": 2,
      "description": "Send Ali an email with subject Meeting and body: The meeting is tomorrow at 5 PM.",
      "task_type": "email",
      "is_statement": false,
      "depends_on": [1]
    }}
  ]
}}

==================================================
EXAMPLE 4: WEB RESEARCH AND EMAIL
==================================================

User request:
Find the latest stable Python version from the web,
find Ali's email in the uploaded document, and email
the result to Ali with subject "Python Update".

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Find the latest stable Python version from the web.",
      "task_type": "web_search",
      "is_statement": false,
      "depends_on": []
    }},
    {{
      "id": 2,
      "description": "Find Ali's email in the uploaded document.",
      "task_type": "retrieve",
      "is_statement": false,
      "depends_on": []
    }},
    {{
      "id": 3,
      "description": "Send Ali an email containing the latest stable Python version with subject Python Update.",
      "task_type": "email",
      "is_statement": false,
      "depends_on": [1, 2]
    }}
  ]
}}

==================================================
EXAMPLE 5: COMPLETE REPORT FLOW
==================================================

User request:
Find the latest stable Python version from the web,
write a short report about it, find Ali's email in the
uploaded document, and email the report to Ali with
subject "Python Report".

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Find the latest stable Python version from the web.",
      "task_type": "web_search",
      "is_statement": false,
      "depends_on": []
    }},
    {{
      "id": 2,
      "description": "Find Ali's email in the uploaded document.",
      "task_type": "retrieve",
      "is_statement": false,
      "depends_on": []
    }},
    {{
      "id": 3,
      "description": "Write a short report about the latest stable Python version.",
      "task_type": "report",
      "is_statement": false,
      "depends_on": [1]
    }},
    {{
      "id": 4,
      "description": "Send the generated report to Ali with subject Python Report.",
      "task_type": "email",
      "is_statement": false,
      "depends_on": [2, 3]
    }}
  ]
}}

==================================================
EXAMPLE 6: CALCULATION
==================================================

User request:
What is 235 multiplied by 3?

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Calculate 235 multiplied by 3.",
      "task_type": "tool",
      "is_statement": false,
      "depends_on": []
    }}
  ]
}}

==================================================
EXAMPLE 7: DATE AND TIME
==================================================

User request:
What is the current date and time?

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "Get the current date and time.",
      "task_type": "tool",
      "is_statement": false,
      "depends_on": []
    }}
  ]
}}

==================================================
EXAMPLE 8: USER STATEMENT
==================================================

User request:
My favorite programming language is Python.

Output:

{{
  "tasks": [
    {{
      "id": 1,
      "description": "The user's favorite programming language is Python.",
      "task_type": "statement",
      "is_statement": true,
      "depends_on": []
    }}
  ]
}}

==================================================
CURRENT USER REQUEST
==================================================

{question}
"""

    # ==================================================
    # Parse Response
    # ==================================================

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
                "TaskDecomposer returned "
                f"invalid JSON: {raw_result}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "TaskDecomposer response must "
                "be a JSON object."
            )

        if "tasks" not in data:
            raise ValueError(
                "TaskDecomposer response must "
                "contain 'tasks'."
            )

        if not isinstance(data["tasks"], list):
            raise ValueError(
                "'tasks' must be a list."
            )

        if not data["tasks"]:
            raise ValueError(
                "'tasks' cannot be empty."
            )

        return data

    # ==================================================
    # Build Tasks
    # ==================================================

    def _build_tasks(
        self,
        data: dict,
    ) -> list[AgentTask]:

        tasks = []
        known_task_ids = set()

        for index, item in enumerate(data["tasks"]):

            if not isinstance(item, dict):
                raise ValueError(
                    "Each task must be a JSON object."
                )

            # Backward compatibility with older
            # tests that do not include task IDs.
            task_id = item.get("id", index)

            description = item.get("description")
            task_type = item.get("task_type")
            is_statement = item.get("is_statement", False)
            depends_on = item.get("depends_on", [])

            # ------------------------------------------
            # Validate ID
            # ------------------------------------------

            if not isinstance(task_id, int):
                raise ValueError(
                    "Task id must be an integer."
                )

            if task_id in known_task_ids:
                raise ValueError(
                    f"Duplicate task id: {task_id}"
                )

            # ------------------------------------------
            # Validate Description
            # ------------------------------------------

            if (
                not isinstance(description, str)
                or not description.strip()
            ):
                raise ValueError(
                    "Task description must be "
                    "a non-empty string."
                )

            # ------------------------------------------
            # Validate Type
            # ------------------------------------------

            if task_type not in self.ALLOWED_TASK_TYPES:
                raise ValueError(
                    f"Invalid task_type: {task_type}"
                )

            # ------------------------------------------
            # Validate Statement Flag
            # ------------------------------------------

            if not isinstance(is_statement, bool):
                raise ValueError(
                    "is_statement must be a boolean."
                )

            if (
                task_type == "statement"
                and not is_statement
            ):
                raise ValueError(
                    "A statement task must have "
                    "is_statement=true."
                )

            if (
                task_type != "statement"
                and is_statement
            ):
                raise ValueError(
                    "Only statement tasks may have "
                    "is_statement=true."
                )

            # ------------------------------------------
            # Validate Dependencies
            # ------------------------------------------

            if not isinstance(depends_on, list):
                raise ValueError(
                    "depends_on must be a list."
                )

            if not all(
                isinstance(dependency_id, int)
                for dependency_id in depends_on
            ):
                raise ValueError(
                    "Every dependency id must "
                    "be an integer."
                )

            if len(depends_on) != len(set(depends_on)):
                raise ValueError(
                    f"Task {task_id} contains "
                    "duplicate dependencies."
                )

            for dependency_id in depends_on:

                if dependency_id not in known_task_ids:
                    raise ValueError(
                        f"Task {task_id} depends "
                        f"on unknown or future task: "
                        f"{dependency_id}"
                    )

            # ------------------------------------------
            # Create Task
            # ------------------------------------------

            task = AgentTask(
                id=task_id,
                description=description.strip(),
                task_type=task_type,
                is_statement=is_statement,
                depends_on=depends_on,
            )

            tasks.append(task)

            known_task_ids.add(task_id)

        return tasks