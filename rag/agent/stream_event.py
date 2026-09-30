from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StreamEvent:
    type: str
    message: Optional[str] = None
    task_id: Optional[int] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "task_id": self.task_id,
            "result": self.result,
            "metadata": self.metadata,
        }