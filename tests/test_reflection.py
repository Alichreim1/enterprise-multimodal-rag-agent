from rag.agent.reflection import Reflection


class FakeLLM:

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_reflection_completed():

    llm = FakeLLM(
        """
        {
            "completed": true,
            "reason": "The result directly answers the task."
        }
        """
    )

    reflection = Reflection(llm)

    decision = reflection.reflect(
        "What is Ali's GitHub?",
        "Ali's GitHub is https://github.com/Alichreim1/."
    )

    assert decision.completed is True
    assert decision.reason is not None


def test_reflection_not_completed():

    llm = FakeLLM(
        """
        {
            "completed": false,
            "reason": "The requested information is not available."
        }
        """
    )

    reflection = Reflection(llm)

    decision = reflection.reflect(
        "What is Ali's favorite programming language?",
        "The requested information could not be found in the documents."
    )

    assert decision.completed is False


def test_reflection_markdown_json():

    llm = FakeLLM(
        """
        ```json
        {
            "completed": true,
            "reason": "The result contains the requested information."
        }
        ```
        """
    )

    reflection = Reflection(llm)

    decision = reflection.reflect(
        "What is Ali's GitHub?",
        "Ali's GitHub is https://github.com/Alichreim1/."
    )

    assert decision.completed is True


def test_reflection_invalid_json():

    llm = FakeLLM(
        "not valid json"
    )

    reflection = Reflection(llm)

    try:
        reflection.reflect(
            "What is Ali's GitHub?",
            "Ali's GitHub is ..."
        )

        assert False

    except ValueError as exc:
        assert "invalid JSON" in str(exc)