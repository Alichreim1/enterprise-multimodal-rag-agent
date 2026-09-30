from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from rag.database.connection import Base
from rag.database.models import Conversation
from rag.database.models import Message


def test_conversation_and_messages_are_persisted():
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

    with TestingSession() as session:
        conversation = Conversation(
            title="Ali GitHub",
        )

        conversation.messages.extend(
            [
                Message(
                    role="user",
                    content=(
                        "What is Ali's GitHub link?"
                    ),
                ),
                Message(
                    role="assistant",
                    content=(
                        "https://github.com/Alichreim1/"
                    ),
                ),
            ]
        )

        session.add(conversation)
        session.commit()

        conversation_id = conversation.id

    with TestingSession() as session:
        stored = session.get(
            Conversation,
            conversation_id,
        )

        assert stored is not None
        assert stored.title == "Ali GitHub"
        assert len(stored.messages) == 2

        assert stored.messages[0].role == "user"
        assert stored.messages[0].content == (
            "What is Ali's GitHub link?"
        )

        assert stored.messages[1].role == "assistant"
        assert stored.messages[1].content == (
            "https://github.com/Alichreim1/"
        )


def test_deleting_conversation_deletes_messages():
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

    with TestingSession() as session:
        conversation = Conversation(
            title="Delete Test",
        )

        message = Message(
            role="user",
            content="Hello",
        )

        conversation.messages.append(message)

        session.add(conversation)
        session.commit()

        message_id = message.id

        session.delete(conversation)
        session.commit()

        assert session.get(
            Message,
            message_id,
        ) is None