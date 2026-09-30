from types import SimpleNamespace

from rag.llms.ollama_vision_llm import (
    OllamaVisionLLM,
)
from rag.models.multimodal_prompt import (
    MultimodalPrompt,
)


def test_ollama_vision_llm(
    monkeypatch,
):

    captured = {}

    def fake_chat(
        model,
        messages,
    ):
        captured["model"] = model
        captured["messages"] = messages

        return SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    "The chart shows "
                    "increasing sales."
                )
            )
        )

    monkeypatch.setattr(
        (
            "rag.llms."
            "ollama_vision_llm."
            "ollama.chat"
        ),
        fake_chat,
    )

    llm = OllamaVisionLLM(
        model="test-vision-model",
    )

    prompt = MultimodalPrompt(
        text="What does the chart show?",
        image_paths=[
            "images/chart.png",
        ],
    )

    answer = (
        llm.generate_multimodal(
            prompt
        )
    )

    assert answer == (
        "The chart shows "
        "increasing sales."
    )

    assert captured["model"] == (
        "test-vision-model"
    )

    message = captured[
        "messages"
    ][0]

    assert message["content"] == (
        "What does the chart show?"
    )

    assert message["images"] == [
        "images/chart.png"
    ]