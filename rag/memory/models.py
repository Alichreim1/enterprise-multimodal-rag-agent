from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryItem:

    id: str | None
    content: str
    memory_type: str
    created_at: datetime | None = None