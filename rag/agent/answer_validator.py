from rag.agent.state import AgentState


class AnswerValidator:

    def validate(
        self,
        state: AgentState,
        answer: str,
    ) -> dict:

        if not isinstance(answer, str):
            return {
                "valid": False,
                "reason": "Final answer must be a string.",
            }

        answer = answer.strip()

        if not answer:
            return {
                "valid": False,
                "reason": "Final answer is empty.",
            }

        # ==================================================
        # Validate completed tasks
        # ==================================================

        completed_tasks = [
            task
            for task in state.tasks
            if task.completed and not task.failed
        ]

        for task in completed_tasks:

            if not task.result:
                return {
                    "valid": False,
                    "reason": (
                        f"Task {task.id} is completed "
                        "but has no result."
                    ),
                }

            # ----------------------------------------------
            # Tool task
            # ----------------------------------------------

            if task.task_type == "tool":

                expected_result = str(
                    task.result
                ).strip()

                if expected_result not in answer:
                    return {
                        "valid": False,
                        "reason": (
                            f"Task {task.id} result "
                            f"'{expected_result}' "
                            "is missing from the final answer."
                        ),
                    }

            # ----------------------------------------------
            # Retrieval task
            # ----------------------------------------------

            elif task.task_type == "retrieve":

                if not self._retrieval_result_supported(
                    task.result,
                    task.description,
                    answer,
                ):
                    return {
                        "valid": False,
                        "reason": (
                            f"Task {task.id} is not "
                            "represented in the final answer."
                        ),
                    }

        # ==================================================
        # Validate failed tasks
        # ==================================================

        failed_tasks = [
            task
            for task in state.tasks
            if task.failed
        ]

        # A failed task must NOT be represented
        # as a successful factual answer.
        #
        # We do not perform aggressive semantic checking
        # here. The main responsibility is ensuring that
        # completed tasks are not lost.

        return {
            "valid": True,
            "reason": "All completed tasks are represented.",
        }

    # ======================================================
    # Retrieval validation
    # ======================================================

    def _retrieval_result_supported(
        self,
        result: str,
        description: str,
        answer: str,
    ) -> bool:

        if not result:
            return False

        # --------------------------------------------------
        # Extract explicit values from retrieval result
        # --------------------------------------------------

        lines = result.splitlines()

        candidates = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # GitHub / LinkedIn / URLs
            if "http://" in line or "https://" in line:
                candidates.append(line)

            # Explicit field-style values
            if ":" in line:
                key, value = line.split(
                    ":",
                    1,
                )

                key = key.strip().lower()
                value = value.strip()

                if value:
                    candidates.append(value)

        # --------------------------------------------------
        # Check explicit values
        # --------------------------------------------------

        for candidate in candidates:

            if candidate in answer:
                return True

        return False