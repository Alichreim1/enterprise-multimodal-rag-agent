from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload

from rag.database.models import Conversation
from rag.database.models import Message


class ConversationRepository:

    def __init__(
        self,
        session: Session,
    ):
        self.session = session

    # ==================================================
    # Create Conversation
    # ==================================================

    def create_conversation(
        self,
        title: str = "New Chat",
        conversation_id: str | None = None,
    ) -> Conversation:
        conversation_data = {
            "title": title,
        }

        if conversation_id is not None:
            conversation_data["id"] = conversation_id

        conversation = Conversation(
            **conversation_data,
        )

        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)

        return conversation

    # ==================================================
    # Get Conversation
    # ==================================================

    def get_conversation(
        self,
        conversation_id: str,
    ) -> Conversation | None:
        statement = (
            select(Conversation)
            .options(
                selectinload(
                    Conversation.messages,
                )
            )
            .where(
                Conversation.id
                == conversation_id
            )
        )

        return self.session.scalar(statement)

    # ==================================================
    # List Conversations
    # ==================================================

    def list_conversations(
        self,
    ) -> list[Conversation]:
        statement = (
            select(Conversation)
            .options(
                selectinload(
                    Conversation.messages,
                )
            )
            .order_by(
                Conversation.updated_at.desc(),
            )
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    # ==================================================
    # Update Title
    # ==================================================

    def update_title(
        self,
        conversation_id: str,
        title: str,
    ) -> Conversation | None:
        conversation = self.session.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            return None

        conversation.title = title

        self.session.commit()
        self.session.refresh(conversation)

        return conversation

    # ==================================================
    # Add Message
    # ==================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        message_id: str | None = None,
    ) -> Message | None:
        conversation = self.session.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            return None

        message_data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }

        if message_id is not None:
            message_data["id"] = message_id

        message = Message(
            **message_data,
        )

        self.session.add(message)
        self.session.flush()

        # Force updated_at to change when a message is added.
        conversation.updated_at = (
            message.created_at
        )

        self.session.commit()
        self.session.refresh(message)

        return message

    # ==================================================
    # Delete Conversation
    # ==================================================

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        conversation = self.session.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            return False

        self.session.delete(conversation)
        self.session.commit()

        return True