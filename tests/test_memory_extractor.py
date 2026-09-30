from rag.llms.ollama_llm import OllamaLLM
from rag.memory.memory_extractor import MemoryExtractor


llm = OllamaLLM(
    model="qwen2.5:3b"
)

extractor = MemoryExtractor(llm)


tests = [
    "My favorite programming language is Python.",
    "I am building a RAG agent in Python.",
    "What is 25 * 8?",
    "I like basketball.",
    "Can you calculate 100 * 20?",

    "I used Python yesterday.",
    "I have been using Python for three years.",
    "I prefer Python because I am more comfortable with it.",
    "My project is currently using FastAPI.",
    "I want to build a Jarvis-like AI assistant.",
    "I am learning machine learning.",
    "What do you think about Python?",
    "Please calculate 500 * 20.",
    "Today I am working on the RAG agent.",
]


for text in tests:

    print("\nUSER:")
    print(text)

    memories = extractor.extract(text)

    print("\nMEMORIES:")

    for memory in memories:
        print(
            memory.memory_type,
            "->",
            memory.content,
        )