from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.routes.conversations import router
from rag.database.connection import Base
from rag.database.connection import (
    get_database_session,
)


engine = create_engine(
    "sqlite://",
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)

TestingSession = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)

Base.metadata.create_all(
    bind=engine,
)


def override_database_session():
    session = TestingSession()

    try:
        yield session
    finally:
        session.close()


app = FastAPI()

app.include_router(router)

app.dependency_overrides[
    get_database_session
] = override_database_session

client = TestClient(app)


def test_conversation_api_flow():
    conversation_id = (
        "test-conversation-api-flow"
    )

    create_response = client.post(
        "/conversations",
        json={
            "id": conversation_id,
            "title": "GitHub Chat",
        },
    )

    assert create_response.status_code == 201
    assert create_response.json()["id"] == (
        conversation_id
    )
    assert create_response.json()["messages"] == []

    user_response = client.post(
        (
            f"/conversations/{conversation_id}"
            "/messages"
        ),
        json={
            "id": "user-message-1",
            "role": "user",
            "content": (
                "What is Ali's GitHub link?"
            ),
        },
    )

    assert user_response.status_code == 201
    assert user_response.json()["role"] == "user"

    assistant_response = client.post(
        (
            f"/conversations/{conversation_id}"
            "/messages"
        ),
        json={
            "id": "assistant-message-1",
            "role": "assistant",
            "content": (
                "https://github.com/Alichreim1/"
            ),
        },
    )

    assert assistant_response.status_code == 201

    get_response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert get_response.status_code == 200

    conversation = get_response.json()

    assert len(conversation["messages"]) == 2
    assert conversation["messages"][0][
        "role"
    ] == "user"
    assert conversation["messages"][1][
        "role"
    ] == "assistant"

    update_response = client.patch(
        f"/conversations/{conversation_id}",
        json={
            "title": "Updated GitHub Chat",
        },
    )

    assert update_response.status_code == 200
    assert update_response.json()["title"] == (
        "Updated GitHub Chat"
    )

    list_response = client.get(
        "/conversations"
    )

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    delete_response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert delete_response.status_code == 204

    missing_response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert missing_response.status_code == 404