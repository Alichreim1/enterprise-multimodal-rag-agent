const API_BASE_URL = "http://127.0.0.1:8000";


export async function askQuestion(question) {

    const response = await fetch(`${API_BASE_URL}/rag/ask`, {
        method: "POST",

        headers: {
            "Content-Type": "application/json",
        },

        body: JSON.stringify({
            question,
        }),
    });

    if (!response.ok) {
        throw new Error("Failed to get answer.");
    }

    return await response.json();
}