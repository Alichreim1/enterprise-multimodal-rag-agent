import pytest

from rag.observability.performance_tracker import (
    PerformanceTracker,
)


class FakeClock:

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_tracker_measures_total_and_stage():

    clock = FakeClock(
        [
            10.0,   # tracker.start
            10.1,   # retrieval starts
            10.35,  # retrieval stops
            11.0,   # tracker.stop
        ]
    )

    tracker = PerformanceTracker(
        clock=clock
    )

    tracker.start()

    with tracker.measure("retrieval"):
        pass

    total_ms = tracker.stop()

    result = tracker.snapshot()

    assert total_ms == pytest.approx(
        1000.0
    )

    assert result["finished"] is True

    assert result["total_ms"] == pytest.approx(
        1000.0
    )

    assert result["stages"]["retrieval"] == {
        "total_ms": 250.0,
        "count": 1,
        "average_ms": 250.0,
    }


def test_repeated_stage_is_accumulated():

    clock = FakeClock(
        [
            0.0,  # first start
            0.1,  # first stop
            0.2,  # second start
            0.5,  # second stop
        ]
    )

    tracker = PerformanceTracker(
        clock=clock
    )

    with tracker.measure("agent_execution"):
        pass

    with tracker.measure("agent_execution"):
        pass

    result = tracker.snapshot()

    stage = result["stages"][
        "agent_execution"
    ]

    assert stage["total_ms"] == pytest.approx(
        400.0
    )

    assert stage["count"] == 2

    assert stage["average_ms"] == pytest.approx(
        200.0
    )


def test_stage_stops_when_exception_occurs():

    clock = FakeClock(
        [
            5.0,
            5.2,
        ]
    )

    tracker = PerformanceTracker(
        clock=clock
    )

    with pytest.raises(
        ValueError,
        match="test error",
    ):
        with tracker.measure("failing_stage"):
            raise ValueError("test error")

    result = tracker.snapshot()

    assert result["stages"][
        "failing_stage"
    ]["total_ms"] == pytest.approx(
        200.0
    )


def test_duplicate_active_stage_is_rejected():

    clock = FakeClock([1.0])

    tracker = PerformanceTracker(
        clock=clock
    )

    tracker.start_stage("retrieval")

    with pytest.raises(
        RuntimeError,
        match="already active",
    ):
        tracker.start_stage("retrieval")


def test_stop_rejects_active_stage():

    clock = FakeClock(
        [
            1.0,  # tracker start
            1.1,  # stage start
        ]
    )

    tracker = PerformanceTracker(
        clock=clock
    )

    tracker.start()
    tracker.start_stage("retrieval")

    with pytest.raises(
        RuntimeError,
        match="stages are active",
    ):
        tracker.stop()


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_stage_name_is_rejected(
    invalid_name,
):

    tracker = PerformanceTracker()

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        tracker.start_stage(
            invalid_name
        )