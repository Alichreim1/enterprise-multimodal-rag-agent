from pathlib import Path


class PromptLoader:

    def __init__(
        self,
        templates_dir: str | Path | None = None,
    ):

        if templates_dir is None:
            templates_dir = (
                Path(__file__).parent.parent
                / "templates"
            )

        self.templates_dir = Path(templates_dir)
        self._cache: dict[str, str] = {}

    def load(
        self,
        template_name: str,
    ) -> str:

        if template_name in self._cache:
            return self._cache[template_name]

        template_path = (
            self.templates_dir
            / f"{template_name}.txt"
        )

        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}"
            )

        template = template_path.read_text(
            encoding="utf-8"
        )

        self._cache[template_name] = template

        return template