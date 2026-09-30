from rag.llms.ollama_llm import OllamaLLM
from rag.memory.memory_extractor import MemoryExtractor


llm = OllamaLLM(
    model="qwen3:8b"
)

extractor = MemoryExtractor(llm)


tests = [

    # ==============================
    # Preference
    # ==============================

    "I prefer Python over Java.",

    "My favorite programming language is Python.",

    "I don't like JavaScript.",


    # ==============================
    # Project
    # ==============================

    "I am currently building a RAG agent.",

    "I am working on a recommendation system.",


    # ==============================
    # Interest
    # ==============================

    "I am interested in artificial intelligence.",

    "I enjoy playing basketball.",


    # ==============================
    # Goal
    # ==============================

    "I want to build a Jarvis-like AI assistant.",

    "My goal is to become a machine learning engineer.",


    # ==============================
    # Fact
    # ==============================

    "I live in Lebanon.",

    "I am a computer science graduate.",


    # ==============================
    # Should NOT be remembered
    # ==============================

    "What is 25 * 8?",

    "Can you calculate 100 * 20?",

    "Tell me what Python is.",

]


for text in tests:

    print("\n")
    print("=" * 70)
    print("USER:")
    print(text)

    memories = extractor.extract(text)

    print("\nEXTRACTED MEMORIES:")

    if not memories:
        print("[]")
        continue

    for memory in memories:

        print(
            memory.memory_type,
            "->",
            memory.content,
        )