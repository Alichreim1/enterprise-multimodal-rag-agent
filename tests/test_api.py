from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"


def test_ask():

    response = client.post(
        "/rag/ask",
        json={
            "question": "What is RAG?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True

    assert "answer" in body


def test_empty_question():

    response = client.post(
        "/rag/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422