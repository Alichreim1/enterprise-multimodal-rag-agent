from sqlalchemy import inspect

from rag.database.connection import engine
from rag.database.connection import (
    create_database,
)


def test_create_database_creates_chat_tables():
    create_database()

    inspector = inspect(engine)

    table_names = inspector.get_table_names()

    assert "conversations" in table_names
    assert "messages" in table_names