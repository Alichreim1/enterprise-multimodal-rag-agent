from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from rag.database.connection import Base
from rag.database.conversation_repository import (
    ConversationRepository,
)


def create_repository():
    engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(
        bind=engine,
    )

    TestingSession = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    session = TestingSession()

    return (
        session,
        ConversationRepository(session),
    )


def test_create_and_get_conversation():
    session, repository = create_repository()

    try:
        conversation = (
            repository.create_conversation(
                title="Test Chat",
            )
        )

        stored = repository.get_conversation(
            conversation.id,
        )

        assert stored is not None
        assert stored.title == "Test Chat"
        assert stored.messages == []

    finally:
        session.close()


def test_create_conversation_with_client_id():
    session, repository = create_repository()

    try:
        conversation = (
            repository.create_conversation(
                conversation_id=(
                    "frontend-conversation-id"
                ),
                title="Frontend Chat",
            )
        )

        assert conversation.id == (
            "frontend-conversation-id"
        )

    finally:
        session.close()


def test_add_messages_to_conversation():
    session, repository = create_repository()

    try:
        conversation = (
            repository.create_conversation()
        )

        user_message = repository.add_message(
            conversation_id=conversation.id,
            role="user",
            content="Hello",
        )

        assistant_message = (
            repository.add_message(
                conversation_id=conversation.id,
                role="assistant",
                content="Hi",
            )
        )

        stored = repository.get_conversation(
            conversation.id,
        )

        assert user_message is not None
        assert assistant_message is not None
        assert stored is not None
        assert len(stored.messages) == 2
        assert stored.messages[0].content == "Hello"
        assert stored.messages[1].content == "Hi"

    finally:
        session.close()


def test_add_message_to_missing_conversation():
    session, repository = create_repository()

    try:
        result = repository.add_message(
            conversation_id="missing-id",
            role="user",
            content="Hello",
        )

        assert result is None

    finally:
        session.close()


def test_list_conversations():
    session, repository = create_repository()

    try:
        first = repository.create_conversation(
            title="First",
        )

        second = repository.create_conversation(
            title="Second",
        )

        repository.add_message(
            conversation_id=first.id,
            role="user",
            content="Update first",
        )

        conversations = (
            repository.list_conversations()
        )

        assert len(conversations) == 2
        assert conversations[0].id == first.id
        assert conversations[1].id == second.id

    finally:
        session.close()


def test_update_conversation_title():
    session, repository = create_repository()

    try:
        conversation = (
            repository.create_conversation()
        )

        updated = repository.update_title(
            conversation_id=conversation.id,
            title="Updated title",
        )

        assert updated is not None
        assert updated.title == "Updated title"

    finally:
        session.close()


def test_delete_conversation():
    session, repository = create_repository()

    try:
        conversation = (
            repository.create_conversation()
        )

        deleted = (
            repository.delete_conversation(
                conversation.id
            )
        )

        stored = repository.get_conversation(
            conversation.id
        )

        assert deleted is True
        assert stored is None

    finally:
        session.close()