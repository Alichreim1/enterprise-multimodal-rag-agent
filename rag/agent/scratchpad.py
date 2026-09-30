from dataclasses import dataclass, field


@dataclass
class Scratchpad:

    entries: list[str] = field(
        default_factory=list
    )

    def add(
        self,
        entry: str,
    ) -> None:

        self.entries.append(entry)

    def to_text(self) -> str:

        if not self.entries:
            return "No previous actions."

        return "\n\n".join(
            self.entries
        )