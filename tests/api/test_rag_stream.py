import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_rag_agent
from api.routes.rag import router


class FakeStreamingAgent:

    def __init__(self):
        self.received_question = None

    def stream(self, question: str):

        self.received_question = question

        yield {
            "type": "tasks_created",
            "message": "2 task(s) created.",
            "task_id": None,
            "result": None,
            "metadata": {
                "count": 2,
            },
        }

        yield {
            "type": "execution_completed",
            "message": (
                "Research completed successfully."
            ),
            "task_id": 1,
            "result": (
                "Research completed successfully."
            ),
            "metadata": {
                "mode": "multi_agent",
                "status": "success",
                "data": {
                    "email": "ali@example.com",
                },
            },
        }

        yield {
            "type": "execution_completed",
            "message": (
                "Email sent successfully."
            ),
            "task_id": 2,
            "result": "Email sent successfully.",
            "metadata": {
                "mode": "multi_agent",
                "status": "success",
                "data": {
                    "recipient": (
                        "ali@example.com"
                    ),
                },
            },
        }

        yield {
            "type": "final_answer",
            "message": (
                "Final answer generated."
            ),
            "task_id": None,
            "result": (
                "Ali's email is ali@example.com. "
                "The email was sent successfully."
            ),
            "metadata": {},
        }

        yield {
            "type": "agent_finished",
            "message": "Agent finished.",
            "task_id": None,
            "result": None,
            "metadata": {
                "finished": True,
            },
        }


def test_rag_stream_endpoint():

    app = FastAPI()

    app.include_router(router)

    fake_agent = FakeStreamingAgent()

    app.dependency_overrides[
        get_rag_agent
    ] = lambda: fake_agent

    client = TestClient(app)

    response = client.post(
        "/rag/stream",
        json={
            "question": (
                "Find Ali's email and "
                "send him an email."
            ),
        },
    )

    assert response.status_code == 200

    assert (
        "application/x-ndjson"
        in response.headers["content-type"]
    )

    events = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]

    assert fake_agent.received_question == (
        "Find Ali's email and send him "
        "an email."
    )

    assert [
        event["type"]
        for event in events
    ] == [
        "tasks_created",
        "execution_completed",
        "execution_completed",
        "final_answer",
        "agent_finished",
    ]

    assert (
        events[1]["metadata"]["mode"]
        == "multi_agent"
    )

    assert (
        events[1]["metadata"]["data"]["email"]
        == "ali@example.com"
    )

    assert (
        events[2]
        ["metadata"]["data"]["recipient"]
        == "ali@example.com"
    )

    assert (
        events[3]["result"]
        == (
            "Ali's email is ali@example.com. "
            "The email was sent successfully."
        )
    )

    assert (
        events[-1]["metadata"]["finished"]
        is True
    )


class FailingStreamingAgent:

    def stream(self, question: str):
        yield {
            "type": "task_started",
            "message": "Starting task.",
            "task_id": 1,
            "result": None,
            "metadata": {},
        }

        raise RuntimeError(
            "Test streaming failure."
        )


def test_rag_stream_returns_error_event():

    app = FastAPI()

    app.include_router(router)

    app.dependency_overrides[
        get_rag_agent
    ] = lambda: FailingStreamingAgent()

    client = TestClient(app)

    response = client.post(
        "/rag/stream",
        json={
            "question": "Test question",
        },
    )

    assert response.status_code == 200

    events = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]

    assert events[0]["type"] == "task_started"
    assert events[-1]["type"] == "agent_error"

    assert events[-1]["message"] == (
        "Failed to generate answer."
    )    