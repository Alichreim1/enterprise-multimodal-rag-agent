import logging
from time import perf_counter

from rag.agent.executor import Executor
from rag.agent.final_answer import FinalAnswerGenerator
from rag.agent.planner import Planner
from rag.agent.reflection import Reflection
from rag.agent.state import AgentState
from rag.agent.stream_event import StreamEvent
from rag.agent.task_decomposer import TaskDecomposer
from rag.observability.performance_tracker import PerformanceTracker


logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(
        self,
        task_decomposer: TaskDecomposer,
        planner: Planner,
        executor: Executor,
        reflection: Reflection,
        final_answer_generator: FinalAnswerGenerator,
        memory_manager,
        supervisor=None,
        max_iterations: int = 10,
        performance_tracker_factory=PerformanceTracker,
    ):
        self.task_decomposer = task_decomposer
        self.planner = planner
        self.executor = executor
        self.reflection = reflection
        self.final_answer_generator = final_answer_generator
        self.memory_manager = memory_manager
        self.supervisor = supervisor
        self.max_iterations = max_iterations
        self.performance_tracker_factory = (
            performance_tracker_factory
        )
        self.last_performance: dict = {}

    # =========================================================
    # UPDATE RAG PIPELINE
    # =========================================================

    def update_rag_pipeline(
        self,
        rag_pipeline,
    ) -> None:
        """
        Replace the active RAG pipeline everywhere
        it is used by the legacy and Multi-Agent flows.
        """

        # Legacy Executor
        self.executor.rag = rag_pipeline

        # Multi-Agent ResearchAgent
        if (
            self.supervisor is not None
            and self.supervisor.can_handle("retrieve")
        ):
            research_agent = self.supervisor.registry.get(
                "retrieve",
            )

            research_agent.rag = rag_pipeline

    # =========================================================
    # NORMAL ASK
    # =========================================================

    def ask(
        self,
        question: str,
    ) -> str:
        tracker = self.performance_tracker_factory()

        tracker.start()

        # ==================================================
        # Memory Recall
        # ==================================================

        with tracker.measure("memory_recall"):
            memories = self.memory_manager.recall_relevant(
                query=question,
                k=5,
            )

        # ==================================================
        # Task Decomposition
        # ==================================================

        with tracker.measure("task_decomposition"):
            tasks = self.task_decomposer.decompose(
                question,
            )

        print("\n=== TASKS ===")

        for task in tasks:
            print(
                f"[{task.id}] {task.description} | "
                f"type={task.task_type} | "
                f"statement={task.is_statement} | "
                f"depends_on={task.depends_on}"
            )

        # ==================================================
        # Agent State
        # ==================================================

        state = AgentState(
            question=question,
            tasks=tasks,
            memories=memories,
        )

        # ==================================================
        # Agent Loop
        # ==================================================

        while not state.finished:
            if state.current_task_index >= len(state.tasks):
                state.finished = True
                break

            if state.iteration >= self.max_iterations:
                raise RuntimeError(
                    "Maximum agent iterations reached.",
                )

            task = state.tasks[state.current_task_index]

            print(
                "\n================ AGENT TASK "
                "================"
            )

            print(
                f"Task index: "
                f"{state.current_task_index}"
            )
            print(f"Task ID: {task.id}")
            print(f"Iteration: {state.iteration}")
            print(f"Description: {task.description}")
            print(f"Type: {task.task_type}")
            print(f"Statement: {task.is_statement}")
            print(f"Depends on: {task.depends_on}")

            print(
                "============================================="
            )

            # ==============================================
            # Multi-Agent Task
            # ==============================================

            if (
                self.supervisor is not None
                and self.supervisor.can_handle(
                    task.task_type,
                )
            ):
                with tracker.measure(
                    "supervisor_execution",
                ):
                    result = (
                        self.supervisor.execute_current_task(
                            state,
                        )
                    )

                print(
                    f"[SUPERVISOR] "
                    f"task={task.description} | "
                    f"status={result.status}"
                )

                if result.status != "success":
                    state.finished = True

                    print(
                        f"[SUPERVISOR] Task failed: "
                        f"{result.message}"
                    )

                    break

                continue

            # ==============================================
            # Statement Task
            # ==============================================

            if task.is_statement:
                with tracker.measure("memory_save"):
                    self.memory_manager.remember(
                        task.description,
                    )

                task.completed = True
                task.result = task.description

                state.current_task_index += 1

                continue

            # ==============================================
            # Legacy Task
            # ==============================================

            task.attempts += 1
            state.iteration += 1

            # ----------------------------------------------
            # Planner
            # ----------------------------------------------

            with tracker.measure("planner"):
                decision = self.planner.plan(
                    task.description,
                )

            print(
                f"[PLANNER] "
                f"task={task.description} | "
                f"action={decision.action}"
            )

            # ----------------------------------------------
            # Executor
            # ----------------------------------------------

            with tracker.measure("legacy_executor"):
                result = self.executor.execute(
                    task,
                    decision,
                )

            if hasattr(result, "answer"):
                task.result = result.answer
            else:
                task.result = str(result)

            print(
                f"[EXECUTOR] "
                f"task={task.description} | "
                f"result={task.result}"
            )

            # ----------------------------------------------
            # Reflection
            # ----------------------------------------------

            with tracker.measure("reflection"):
                reflection_decision = (
                    self.reflection.reflect(
                        task.description,
                        task.result,
                    )
                )

            task.completed = reflection_decision.completed
            task.reflection = reflection_decision.reason

            print(
                f"[REFLECTION] "
                f"completed={task.completed} | "
                f"reason={task.reflection}"
            )

            if task.completed:
                state.current_task_index += 1

        # ==================================================
        # Final Answer
        # ==================================================

        print(
            f"[AGENT] "
            f"iteration={state.iteration}, "
            f"current_task_index="
            f"{state.current_task_index}, "
            f"finished={state.finished}"
        )

        with tracker.measure("final_answer_generation"):
            final_answer = (
                self.final_answer_generator.generate(
                    question=state.question,
                    tasks=state.tasks,
                    memories=state.memories,
                )
            )

        tracker.stop()

        self.last_performance = tracker.snapshot()

        logger.info(
            "Complete agent performance: %s",
            self.last_performance,
        )

        print()
        print("=== AGENT PERFORMANCE ===")
        print(self.last_performance)

        return final_answer

    # =========================================================
    # STREAM EVENT HELPER
    # =========================================================

    def _event(
        self,
        event_type: str,
        message: str = None,
        task_id: int = None,
        result=None,
        metadata: dict = None,
    ) -> dict:
        return StreamEvent(
            type=event_type,
            message=message,
            task_id=task_id,
            result=result,
            metadata=metadata or {},
        ).to_dict()

    # =========================================================
    # STREAM
    # =========================================================

    def stream(
        self,
        question: str,
    ):
        agent_started_at = perf_counter()

        # -----------------------------------------------------
        # Memory Recall
        # -----------------------------------------------------

        yield self._event(
            "memory_recall_started",
            message="Recalling relevant memories...",
        )

        memories = self.memory_manager.recall_relevant(
            query=question,
            k=5,
        )

        yield self._event(
            "memory_recall_completed",
            message="Memory recall completed.",
            metadata={
                "count": len(memories),
            },
        )

        # -----------------------------------------------------
        # Task Decomposition
        # -----------------------------------------------------

        yield self._event(
            "task_decomposition_started",
            message=(
                "Decomposing the question "
                "into tasks..."
            ),
        )

        tasks = self.task_decomposer.decompose(
            question,
        )

        yield self._event(
            "tasks_created",
            message=f"{len(tasks)} task(s) created.",
            metadata={
                "count": len(tasks),
                "tasks": [
                    {
                        "id": task.id,
                        "description": task.description,
                        "task_type": task.task_type,
                        "is_statement": task.is_statement,
                        "depends_on": task.depends_on,
                    }
                    for task in tasks
                ],
            },
        )

        # -----------------------------------------------------
        # Agent State
        # -----------------------------------------------------

        state = AgentState(
            question=question,
            tasks=tasks,
            memories=memories,
        )

        # -----------------------------------------------------
        # Agent Loop
        # -----------------------------------------------------

        while not state.finished:
            if state.current_task_index >= len(state.tasks):
                state.finished = True
                break

            if state.iteration >= self.max_iterations:
                yield self._event(
                    "agent_error",
                    message=(
                        "Maximum agent iterations "
                        "reached."
                    ),
                    metadata={
                        "iteration": state.iteration,
                        "max_iterations": (
                            self.max_iterations
                        ),
                    },
                )

                raise RuntimeError(
                    "Maximum agent iterations reached.",
                )

            task = state.tasks[state.current_task_index]

            # -------------------------------------------------
            # Task Started
            # -------------------------------------------------

            yield self._event(
                "task_started",
                message=task.description,
                task_id=task.id,
                metadata={
                    "task_index": (
                        state.current_task_index
                    ),
                    "task_type": task.task_type,
                    "is_statement": task.is_statement,
                    "depends_on": task.depends_on,
                    "iteration": state.iteration,
                },
            )

            # -------------------------------------------------
            # Multi-Agent Task
            # -------------------------------------------------

            if (
                self.supervisor is not None
                and self.supervisor.can_handle(
                    task.task_type,
                )
            ):
                yield self._event(
                    "execution_started",
                    message=(
                        "Executing task with "
                        "specialized agent..."
                    ),
                    task_id=task.id,
                    metadata={
                        "mode": "multi_agent",
                        "task_type": task.task_type,
                    },
                )

                result = (
                    self.supervisor.execute_current_task(
                        state,
                    )
                )

                event_result = (
                    result.message
                    or result.data.get("answer")
                    or str(result.data)
                )

                yield self._event(
                    "execution_completed",
                    message=(
                        "Specialized agent execution "
                        "completed."
                    ),
                    task_id=task.id,
                    result=event_result,
                    metadata={
                        "mode": "multi_agent",
                        "status": result.status,
                        "data": result.data,
                    },
                )

                if result.status == "success":
                    yield self._event(
                        "task_completed",
                        message=(
                            result.message
                            or (
                                "Task completed "
                                "successfully."
                            )
                        ),
                        task_id=task.id,
                        result=event_result,
                        metadata={
                            "mode": "multi_agent",
                            "status": result.status,
                        },
                    )

                    continue

                state.finished = True

                yield self._event(
                    "agent_error",
                    message=(
                        result.message
                        or (
                            "Specialized agent "
                            "task failed."
                        )
                    ),
                    task_id=task.id,
                    metadata={
                        "mode": "multi_agent",
                        "status": result.status,
                    },
                )

                break

            # -------------------------------------------------
            # Statement Task
            # -------------------------------------------------

            if task.is_statement:
                yield self._event(
                    "memory_save_started",
                    message=(
                        "Saving statement "
                        "to memory..."
                    ),
                    task_id=task.id,
                )

                self.memory_manager.remember(
                    task.description,
                )

                task.completed = True
                task.result = task.description

                yield self._event(
                    "memory_saved",
                    message="Statement saved to memory.",
                    task_id=task.id,
                    result=task.result,
                )

                yield self._event(
                    "task_completed",
                    message="Statement task completed.",
                    task_id=task.id,
                    result=task.result,
                )

                state.current_task_index += 1

                continue

            # -------------------------------------------------
            # Increment Attempts / Iteration
            # -------------------------------------------------

            task.attempts += 1
            state.iteration += 1

            # -------------------------------------------------
            # Planner
            # -------------------------------------------------

            yield self._event(
                "planning_started",
                message="Planning task execution...",
                task_id=task.id,
            )

            decision = self.planner.plan(
                task.description,
            )

            yield self._event(
                "planning_completed",
                message=(
                    f"Planner selected: "
                    f"{decision.action}"
                ),
                task_id=task.id,
                metadata={
                    "action": decision.action,
                },
            )

            # -------------------------------------------------
            # Executor
            # -------------------------------------------------

            yield self._event(
                "execution_started",
                message="Executing task...",
                task_id=task.id,
                metadata={
                    "action": decision.action,
                    "mode": "legacy",
                },
            )

            result = self.executor.execute(
                task,
                decision,
            )

            if hasattr(result, "answer"):
                task.result = result.answer
            else:
                task.result = str(result)

            yield self._event(
                "execution_completed",
                message="Task execution completed.",
                task_id=task.id,
                result=task.result,
                metadata={
                    "mode": "legacy",
                },
            )

            # -------------------------------------------------
            # Reflection
            # -------------------------------------------------

            yield self._event(
                "reflection_started",
                message=(
                    "Checking whether the task "
                    "was completed..."
                ),
                task_id=task.id,
            )

            reflection_decision = (
                self.reflection.reflect(
                    task.description,
                    task.result,
                )
            )

            task.completed = reflection_decision.completed
            task.reflection = reflection_decision.reason

            yield self._event(
                "reflection_completed",
                message=task.reflection,
                task_id=task.id,
                metadata={
                    "completed": task.completed,
                    "reason": task.reflection,
                },
            )

            # -------------------------------------------------
            # Next Task / Retry
            # -------------------------------------------------

            if task.completed:
                yield self._event(
                    "task_completed",
                    message=(
                        "Task completed "
                        "successfully."
                    ),
                    task_id=task.id,
                    result=task.result,
                )

                state.current_task_index += 1

            else:
                yield self._event(
                    "task_retry",
                    message=(
                        "Task was not completed. "
                        "Retrying..."
                    ),
                    task_id=task.id,
                    metadata={
                        "attempts": task.attempts,
                        "reason": task.reflection,
                    },
                )

        # -----------------------------------------------------
        # Final Answer
        # -----------------------------------------------------

        yield self._event(
            "final_answer_started",
            message="Generating final answer...",
        )

        final_answer_chunks = []

        for chunk in (
            self.final_answer_generator.generate_stream(
                question=state.question,
                tasks=state.tasks,
                memories=state.memories,
            )
        ):
            final_answer_chunks.append(chunk)

            yield self._event(
                "final_answer_chunk",
                result=chunk,
            )

        final_answer = "".join(
            final_answer_chunks
        )

        yield self._event(
            "final_answer",
            message="Final answer generated.",
            result=final_answer,
        )

        # -----------------------------------------------------
        # Agent Finished
        # -----------------------------------------------------

        elapsed_seconds = round(
            perf_counter() - agent_started_at,
            2,
        )

        print(
            f"\n[AGENT] Worked in "
            f"{elapsed_seconds} seconds."
        )

        yield self._event(
            "agent_finished",
            message=(
                f"Agent finished in "
                f"{elapsed_seconds} seconds."
            ),
            metadata={
                "iteration": state.iteration,
                "current_task_index": (
                    state.current_task_index
                ),
                "finished": state.finished,
                "duration_seconds": (
                    elapsed_seconds
                ),
            },
        )
