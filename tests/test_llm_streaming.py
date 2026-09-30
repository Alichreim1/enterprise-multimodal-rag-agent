from rag.llms.ollama_llm import OllamaLLM


def test_ollama_generate_stream():

    llm = OllamaLLM(
        model="qwen3:8B"
    )

    chunks = list(
        llm.generate_stream(
            "Say hello in one short sentence."
        )
    )

    assert chunks
    assert all(
        isinstance(chunk, str)
        for chunk in chunks
    )

    text = "".join(chunks)

    assert text.strip()