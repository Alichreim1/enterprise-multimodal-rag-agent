import json

from rag.llms.ollama_llm import OllamaLLM


class FakeMessage:

    def __init__(
        self,
        content="",
        tool_calls=None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeResponse:

    def __init__(self, message):
        self.message = message


class FakeFunction:

    def __init__(
        self,
        name,
        arguments,
    ):
        self.name = name
        self.arguments = arguments


class FakeToolCall:

    def __init__(
        self,
        name,
        arguments,
    ):
        self.function = FakeFunction(
            name=name,
            arguments=arguments,
        )


def test_generate_returns_llm_content(
    monkeypatch,
):

    def fake_chat(
        model,
        messages,
    ):
        return FakeResponse(
            FakeMessage(
                content="This is the generated answer."
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    result = llm.generate(
        "What is AI?"
    )

    assert (
        result
        == "This is the generated answer."
    )


def test_generate_sends_prompt_to_llm(
    monkeypatch,
):

    captured = {}

    def fake_chat(
        model,
        messages,
    ):
        captured["model"] = model
        captured["messages"] = messages

        return FakeResponse(
            FakeMessage(
                content="Answer"
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    prompt = "Explain RAG."

    llm.generate(prompt)

    assert captured["model"] == "test-model"

    assert (
        captured["messages"]
        == [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )


def test_decide_returns_finish_when_no_tool_is_called(
    monkeypatch,
):

    def fake_chat(
        model,
        messages,
        tools=None,
    ):
        return FakeResponse(
            FakeMessage(
                content="The answer is 42."
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    result = llm.decide(
        "What is the answer?"
    )

    assert result == {
        "action": "finish",
        "answer": "The answer is 42.",
    }


def test_decide_returns_tool_call(
    monkeypatch,
):

    def fake_chat(
        model,
        messages,
        tools=None,
    ):
        return FakeResponse(
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        name="calculator",
                        arguments={
                            "a": 10,
                            "b": 5,
                        },
                    )
                ],
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    result = llm.decide(
        "Calculate 10 + 5"
    )

    assert result == {
        "action": "tool",
        "tool": "calculator",
        "arguments": {
            "a": 10,
            "b": 5,
        },
    }


def test_decide_parses_string_arguments(
    monkeypatch,
):

    def fake_chat(
        model,
        messages,
        tools=None,
    ):
        return FakeResponse(
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        name="calculator",
                        arguments=json.dumps(
                            {
                                "a": 10,
                                "b": 5,
                            }
                        ),
                    )
                ],
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    result = llm.decide(
        "Calculate 10 + 5"
    )

    assert result["action"] == "tool"

    assert result["tool"] == "calculator"

    assert result["arguments"] == {
        "a": 10,
        "b": 5,
    }


def test_decide_passes_tool_schemas(
    monkeypatch,
):

    captured = {}

    fake_tool_manager = type(
        "FakeToolManager",
        (),
        {
            "schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator"
                    },
                }
            ]
        },
    )()

    def fake_chat(
        model,
        messages,
        tools=None,
    ):
        captured["tools"] = tools

        return FakeResponse(
            FakeMessage(
                content="Done"
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model",
        tool_manager=fake_tool_manager,
    )

    llm.decide(
        "Calculate something"
    )

    assert (
        captured["tools"]
        == fake_tool_manager.schemas
    )


def test_decide_works_without_tool_manager(
    monkeypatch,
):

    captured = {}

    def fake_chat(
        model,
        messages,
        tools=None,
    ):
        captured["tools"] = tools

        return FakeResponse(
            FakeMessage(
                content="Final answer"
            )
        )

    monkeypatch.setattr(
        "rag.llms.ollama_llm.chat",
        fake_chat,
    )

    llm = OllamaLLM(
        model="test-model"
    )

    result = llm.decide(
        "Hello"
    )

    assert result == {
        "action": "finish",
        "answer": "Final answer",
    }

    assert captured["tools"] is None