const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL ||
    "http://127.0.0.1:8000";


// ==================================================
// Request Helper
// ==================================================

async function apiRequest(
    path,
    options = {}
) {
    const response = await fetch(
        `${API_BASE_URL}${path}`,
        {
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {}),
            },
        }
    );

    if (response.status === 204) {
        return null;
    }

    let data = null;

    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        throw new Error(
            data?.detail ||
                `Request failed with status ${response.status}.`
        );
    }

    return data;
}


// ==================================================
// Normalize Conversation
// ==================================================

export function normalizeConversation(
    conversation
) {
    return {
        id: conversation.id,
        title: conversation.title,
        createdAt: conversation.created_at,
        updatedAt: conversation.updated_at,

        messages: (
            conversation.messages || []
        ).map((message) => ({
            id: message.id,
            role: message.role,
            content: message.content,

            // Runtime UI state is not stored yet.
            tasks: [],
            progress: [],
            status: "",
            loading: false,

            createdAt: message.created_at,
        })),
    };
}


// ==================================================
// List Conversations
// ==================================================

export async function getConversations() {
    const conversations = await apiRequest(
        "/conversations"
    );

    return conversations.map(
        normalizeConversation
    );
}


// ==================================================
// Get One Conversation
// ==================================================

export async function getConversation(
    conversationId
) {
    const conversation = await apiRequest(
        `/conversations/${conversationId}`
    );

    return normalizeConversation(
        conversation
    );
}


// ==================================================
// Create Conversation
// ==================================================

export async function createConversation({
    id,
    title = "New Chat",
}) {
    const conversation = await apiRequest(
        "/conversations",
        {
            method: "POST",
            body: JSON.stringify({
                id,
                title,
            }),
        }
    );

    return normalizeConversation(
        conversation
    );
}


// ==================================================
// Update Conversation Title
// ==================================================

export async function updateConversationTitle(
    conversationId,
    title
) {
    const conversation = await apiRequest(
        `/conversations/${conversationId}`,
        {
            method: "PATCH",
            body: JSON.stringify({
                title,
            }),
        }
    );

    return normalizeConversation(
        conversation
    );
}


// ==================================================
// Add Message
// ==================================================

export async function saveMessage(
    conversationId,
    message
) {
    return apiRequest(
        `/conversations/${conversationId}/messages`,
        {
            method: "POST",
            body: JSON.stringify({
                id: message.id,
                role: message.role,
                content: message.content,
            }),
        }
    );
}


// ==================================================
// Delete Conversation
// ==================================================

export async function deleteConversation(
    conversationId
) {
    await apiRequest(
        `/conversations/${conversationId}`,
        {
            method: "DELETE",
        }
    );
}