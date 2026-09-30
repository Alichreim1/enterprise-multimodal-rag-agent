from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter


class PerformanceTracker:
    """
    Measures total execution time and individual stages.

    One tracker instance should be created for each
    request or agent execution.
    """

    def __init__(
        self,
        clock: Callable[[], float] = perf_counter,
    ):
        self.clock = clock

        self._started_at: float | None = None
        self._finished_at: float | None = None

        self._active_stages: dict[
            str,
            float,
        ] = {}

        self._stage_durations: dict[
            str,
            float,
        ] = {}

        self._stage_counts: dict[
            str,
            int,
        ] = {}

    # ==================================================
    # Total Execution
    # ==================================================

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError(
                "PerformanceTracker has already started."
            )

        self._started_at = self.clock()

    def stop(self) -> float:
        if self._started_at is None:
            raise RuntimeError(
                "PerformanceTracker has not started."
            )

        if self._finished_at is not None:
            raise RuntimeError(
                "PerformanceTracker has already stopped."
            )

        if self._active_stages:
            active_names = ", ".join(
                self._active_stages
            )

            raise RuntimeError(
                "Cannot stop PerformanceTracker while "
                f"stages are active: {active_names}"
            )

        self._finished_at = self.clock()

        return self.total_ms

    # ==================================================
    # Individual Stages
    # ==================================================

    def start_stage(
        self,
        name: str,
    ) -> None:
        self._validate_stage_name(name)

        if name in self._active_stages:
            raise RuntimeError(
                f"Stage is already active: {name}"
            )

        self._active_stages[name] = self.clock()

    def stop_stage(
        self,
        name: str,
    ) -> float:
        self._validate_stage_name(name)

        if name not in self._active_stages:
            raise RuntimeError(
                f"Stage is not active: {name}"
            )

        started_at = self._active_stages.pop(
            name
        )

        duration_seconds = max(
            0.0,
            self.clock() - started_at,
        )

        self._stage_durations[name] = (
            self._stage_durations.get(
                name,
                0.0,
            )
            + duration_seconds
        )

        self._stage_counts[name] = (
            self._stage_counts.get(
                name,
                0,
            )
            + 1
        )

        return duration_seconds * 1000

    # ==================================================
    # Context Manager
    # ==================================================

    @contextmanager
    def measure(
        self,
        name: str,
    ):
        self.start_stage(name)

        try:
            yield
        finally:
            self.stop_stage(name)

    # ==================================================
    # Results
    # ==================================================

    @property
    def total_ms(self) -> float:
        if self._started_at is None:
            return 0.0

        finished_at = (
            self._finished_at
            if self._finished_at is not None
            else self.clock()
        )

        return max(
            0.0,
            finished_at - self._started_at,
        ) * 1000

    def get_stage_ms(
        self,
        name: str,
    ) -> float:
        return (
            self._stage_durations.get(
                name,
                0.0,
            )
            * 1000
        )

    def snapshot(self) -> dict:
        stages = {}

        for name, duration_seconds in (
            self._stage_durations.items()
        ):
            count = self._stage_counts[name]

            total_ms = (
                duration_seconds * 1000
            )

            stages[name] = {
                "total_ms": round(
                    total_ms,
                    3,
                ),
                "count": count,
                "average_ms": round(
                    total_ms / count,
                    3,
                ),
            }

        return {
            "total_ms": round(
                self.total_ms,
                3,
            ),
            "finished": (
                self._finished_at is not None
            ),
            "stages": stages,
        }

    # ==================================================
    # Validation
    # ==================================================

    @staticmethod
    def _validate_stage_name(
        name: str,
    ) -> None:
        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError(
                "Stage name must be a "
                "non-empty string."
            )