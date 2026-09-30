import {
    useEffect,
    useRef,
    useState,
} from "react";

import ChatInput from "../components/ChatInput";
import ChatMessages from "../components/ChatMessages";
import Sidebar from "../components/Sidebar";

import {
    createConversation,
    deleteConversation,
    getConversations,
    saveMessage,
    updateConversationTitle,
} from "../services/conversationApi";

const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL ||
    "http://127.0.0.1:8000";

function ChatPage() {
    // ==================================================
    // Conversations
    // ==================================================

    const [conversations, setConversations] =
        useState([]);

    const [
        activeConversation,
        setActiveConversation,
    ] = useState(null);

    // ==================================================
    // Loading / Errors
    // ==================================================

    const [
        conversationsLoading,
        setConversationsLoading,
    ] = useState(true);

    const [
        conversationError,
        setConversationError,
    ] = useState("");

    const [loading, setLoading] =
        useState(false);

    // ==================================================
    // Refs
    // ==================================================

    const abortControllerRef = useRef(null);
    const messagesEndRef = useRef(null);
    const uploadInputRef = useRef(null);

    // ==================================================
    // Multi-File Upload
    // ==================================================

    const [selectedFiles, setSelectedFiles] =
        useState([]);

    const [uploading, setUploading] =
        useState(false);

    const [uploadMessage, setUploadMessage] =
        useState("");

    // ==================================================
    // Active Conversation
    // ==================================================

    const activeConversationData =
        conversations.find(
            (conversation) =>
                conversation.id ===
                activeConversation
        );

    const messages =
        activeConversationData?.messages || [];

    // ==================================================
    // Helpers
    // ==================================================

    const createConversationTitle = (
        question
    ) => {
        const trimmedQuestion =
            question.trim();

        if (trimmedQuestion.length > 30) {
            return `${trimmedQuestion.slice(
                0,
                30
            )}...`;
        }

        return trimmedQuestion;
    };

    // ==================================================
    // Load Stored Conversations
    // ==================================================

    useEffect(() => {
        let mounted = true;

        const loadConversations =
            async () => {
                setConversationsLoading(true);
                setConversationError("");

                try {
                    const storedConversations =
                        await getConversations();

                    if (!mounted) {
                        return;
                    }

                    setConversations(
                        storedConversations
                    );

                    if (
                        storedConversations.length >
                        0
                    ) {
                        setActiveConversation(
                            storedConversations[0].id
                        );
                    }
                } catch (error) {
                    console.error(
                        "Failed to load conversations:",
                        error
                    );

                    if (mounted) {
                        setConversationError(
                            error.message ||
                                "Failed to load conversations."
                        );
                    }
                } finally {
                    if (mounted) {
                        setConversationsLoading(
                            false
                        );
                    }
                }
            };

        loadConversations();

        return () => {
            mounted = false;
        };
    }, []);

    // ==================================================
    // Auto Scroll
    // ==================================================

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({
            behavior: "smooth",
        });
    }, [messages]);

    // ==================================================
    // Streaming Event Messages
    // ==================================================

    const getEventMessage = (event) => {
        switch (event.type) {
            case "memory_recall_started":
                return "🧠 Checking memory...";

            case "memory_recall_completed":
                return "🧠 Memory check completed.";

            case "task_decomposition_started":
                return "🔍 Understanding your question...";

            case "tasks_created":
                return "📋 Tasks created.";

            case "task_started":
                return (
                    event.message ||
                    "⚙️ Working on a task..."
                );

            case "memory_save_started":
                return "💾 Saving information...";

            case "memory_saved":
                return "💾 Information saved.";

            case "planning_started":
                return "🧭 Planning the next step...";

            case "planning_completed":
                return "🧭 Plan ready.";

            case "execution_started":
                return event.metadata?.mode ===
                    "multi_agent"
                    ? "🤖 Specialized agent is working..."
                    : "⚙️ Executing...";

            case "execution_completed":
                return event.metadata?.mode ===
                    "multi_agent"
                    ? (
                          event.message ||
                          "✅ Specialized agent completed."
                      )
                    : "✅ Execution completed.";

            case "reflection_started":
                return "🔎 Checking the result...";

            case "reflection_completed":
                return "🔎 Result checked.";

            case "task_completed":
                return "✅ Task completed.";

            case "task_retry":
                return "🔄 Retrying the task...";

            case "final_answer_started":
                return "✍️ Preparing the final answer...";

            case "final_answer_chunk":
                return "";

            case "final_answer":
                return "✅ Final answer ready.";

            case "agent_finished": {
                const duration =
                    event.metadata?.duration_seconds;

                if (
                    duration !== null &&
                    duration !== undefined
                ) {
                    return `🏁 Done. Worked in ${duration} seconds.`;
                }

                return "🏁 Done.";
            }

            case "agent_error":
                return (
                    event.message ||
                    "❌ Something went wrong."
                );

            default:
                return "";
        }
    };

    // ==================================================
    // Update Assistant Message
    // ==================================================

    const updateAssistantMessage = (
        conversationId,
        assistantId,
        updater
    ) => {
        setConversations(
            (previousConversations) =>
                previousConversations.map(
                    (conversation) => {
                        if (
                            conversation.id !==
                            conversationId
                        ) {
                            return conversation;
                        }

                        return {
                            ...conversation,

                            messages:
                                conversation.messages.map(
                                    (message) => {
                                        if (
                                            message.id !==
                                            assistantId
                                        ) {
                                            return message;
                                        }

                                        return updater(
                                            message
                                        );
                                    }
                                ),
                        };
                    }
                )
        );
    };

    // ==================================================
    // Update Task From Event
    // ==================================================

    const updateTaskFromEvent = (
        conversationId,
        assistantId,
        event
    ) => {
        if (
            event.type === "tasks_created" &&
            Array.isArray(
                event.metadata?.tasks
            )
        ) {
            updateAssistantMessage(
                conversationId,
                assistantId,
                (message) => ({
                    ...message,

                    tasks:
                        event.metadata.tasks.map(
                            (task) => ({
                                id: task.id,
                                description:
                                    task.description,
                                taskType:
                                    task.task_type,
                                dependsOn:
                                    task.depends_on ||
                                    [],
                                status: "pending",
                                result: null,
                                data: null,
                                mode: null,
                            })
                        ),
                })
            );

            return;
        }

        if (
            event.task_id === null ||
            event.task_id === undefined
        ) {
            return;
        }

        updateAssistantMessage(
            conversationId,
            assistantId,
            (message) => {
                const taskId = event.task_id;

                const taskExists =
                    message.tasks.some(
                        (task) =>
                            task.id === taskId
                    );

                const currentTasks = taskExists
                    ? message.tasks
                    : [
                          ...message.tasks,
                          {
                              id: taskId,
                              description:
                                  event.message ||
                                  "Task",
                              taskType:
                                  event.metadata
                                      ?.task_type ||
                                  null,
                              dependsOn:
                                  event.metadata
                                      ?.depends_on ||
                                  [],
                              status: "pending",
                              result: null,
                              data: null,
                              mode: null,
                          },
                      ];

                const updateTask = (
                    updates
                ) =>
                    currentTasks.map((task) =>
                        task.id === taskId
                            ? {
                                  ...task,
                                  ...updates,
                              }
                            : task
                    );

                switch (event.type) {
                    case "task_started":
                        return {
                            ...message,

                            tasks: updateTask({
                                description:
                                    event.message,

                                taskType:
                                    event.metadata
                                        ?.task_type,

                                dependsOn:
                                    event.metadata
                                        ?.depends_on ||
                                    [],

                                status: "working",
                            }),
                        };

                    case "execution_started":
                        return {
                            ...message,

                            tasks: updateTask({
                                status: "working",

                                mode:
                                    event.metadata
                                        ?.mode ||
                                    "legacy",
                            }),
                        };

                    case "execution_completed":
                        return {
                            ...message,

                            tasks: updateTask({
                                result:
                                    event.result,

                                data:
                                    event.metadata
                                        ?.data ||
                                    null,

                                mode:
                                    event.metadata
                                        ?.mode ||
                                    "legacy",
                            }),
                        };

                    case "task_completed":
                        return {
                            ...message,

                            tasks: updateTask({
                                status: "completed",

                                result:
                                    event.result,
                            }),
                        };

                    case "task_retry":
                        return {
                            ...message,

                            tasks: updateTask({
                                status: "retrying",
                            }),
                        };

                    case "agent_error":
                        return {
                            ...message,

                            tasks: updateTask({
                                status: "failed",

                                result:
                                    event.message ||
                                    "Task failed.",
                            }),
                        };

                    default:
                        return message;
                }
            }
        );
    };

    // ==================================================
    // Update Progress
    // ==================================================

    const updateProgress = (
        conversationId,
        assistantId,
        event
    ) => {
        const progressMessages = {
            memory_recall_started:
                "Checking memory...",

            memory_recall_completed:
                "Memory check completed.",

            task_decomposition_started:
                "Understanding the question...",

            tasks_created:
                "Tasks created.",

            planning_started:
                "Planning the next step...",

            planning_completed:
                "Plan ready.",

            execution_started:
                "Executing...",

            execution_completed:
                "Execution completed.",

            reflection_started:
                "Checking the result...",

            reflection_completed:
                "Result checked.",

            final_answer_started:
                "Preparing the final answer...",

            final_answer:
                "Final answer ready.",

            agent_finished:
                "Done.",
        };

        let progressMessage =
            progressMessages[event.type];

        if (event.type === "agent_finished") {
            const duration =
                event.metadata?.duration_seconds;

            progressMessage =
                duration !== null &&
                duration !== undefined
                    ? `Done in ${duration} seconds.`
                    : "Done.";
        }

        if (!progressMessage) {
            return;
        }

        updateAssistantMessage(
            conversationId,
            assistantId,
            (assistant) => {
                const existingIndex =
                    assistant.progress.findIndex(
                        (item) =>
                            item.type ===
                            event.type
                    );

                const newItem = {
                    type: event.type,

                    message: progressMessage,

                    completed:
                        event.type !==
                        "final_answer_started",
                };

                if (existingIndex !== -1) {
                    return {
                        ...assistant,

                        progress:
                            assistant.progress.map(
                                (
                                    item,
                                    index
                                ) =>
                                    index ===
                                    existingIndex
                                        ? newItem
                                        : item
                            ),
                    };
                }

                return {
                    ...assistant,

                    progress: [
                        ...assistant.progress,
                        newItem,
                    ],
                };
            }
        );
    };

    // ==================================================
    // New Conversation
    // ==================================================

    const handleNewChat = async () => {
        if (
            loading ||
            conversationsLoading
        ) {
            return;
        }

        const conversationId =
            crypto.randomUUID();

        try {
            setConversationError("");

            const conversation =
                await createConversation({
                    id: conversationId,
                    title: "New Chat",
                });

            setConversations(
                (previousConversations) => [
                    conversation,
                    ...previousConversations,
                ]
            );

            setActiveConversation(
                conversation.id
            );
        } catch (error) {
            console.error(
                "Failed to create conversation:",
                error
            );

            setConversationError(
                error.message ||
                    "Failed to create conversation."
            );
        }
    };

    // ==================================================
    // Select Conversation
    // ==================================================

    const handleSelectConversation = (
        conversationId
    ) => {
        if (loading) {
            return;
        }

        setActiveConversation(
            conversationId
        );
    };

    // ==================================================
    // Delete Conversation
    // ==================================================

    const handleDeleteConversation = async (
        conversationId
    ) => {
        if (loading) {
            return;
        }

        try {
            setConversationError("");

            await deleteConversation(
                conversationId
            );

            setConversations(
                (previousConversations) =>
                    previousConversations.filter(
                        (conversation) =>
                            conversation.id !==
                            conversationId
                    )
            );

            if (
                activeConversation ===
                conversationId
            ) {
                setActiveConversation(null);
            }
        } catch (error) {
            console.error(
                "Failed to delete conversation:",
                error
            );

            setConversationError(
                error.message ||
                    "Failed to delete conversation."
            );
        }
    };

    // ==================================================
    // Stop Generating
    // ==================================================

    const handleStop = () => {
        abortControllerRef.current?.abort();
    };

    // ==================================================
    // Upload Multiple Documents And Images
    // ==================================================

    const handleUpload = async () => {
        if (
            selectedFiles.length === 0 ||
            uploading ||
            loading
        ) {
            return;
        }

        setUploading(true);
        setUploadMessage("");

        try {
            const formData = new FormData();

            selectedFiles.forEach((file) => {
                formData.append(
                    "files",
                    file
                );
            });

            const response = await fetch(
                `${API_BASE_URL}/documents/upload`,
                {
                    method: "POST",
                    body: formData,
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail ||
                        "Upload failed."
                );
            }

            const knowledgeBase =
                data.knowledge_base;

            setUploadMessage(
                `✅ ${data.uploaded_count} file(s) uploaded. ` +
                    `Knowledge base: ` +
                    `${knowledgeBase.document_count} file(s), ` +
                    `${knowledgeBase.text_chunk_count} text chunk(s), ` +
                    `${knowledgeBase.image_chunk_count} image chunk(s).`
            );

            setSelectedFiles([]);

            if (uploadInputRef.current) {
                uploadInputRef.current.value =
                    "";
            }
        } catch (error) {
            console.error(
                "Upload failed:",
                error
            );

            setUploadMessage(
                `❌ ${
                    error.message ||
                    "Upload failed."
                }`
            );
        } finally {
            setUploading(false);
        }
    };

    // ==================================================
    // Prepare Conversation For Question
    // ==================================================

    const prepareConversation = async (
        question
    ) => {
        const title =
            createConversationTitle(
                question
            );

        // ------------------------------------------
        // Create Conversation When None Is Active
        // ------------------------------------------

        if (!activeConversation) {
            const conversationId =
                crypto.randomUUID();

            const conversation =
                await createConversation({
                    id: conversationId,
                    title,
                });

            setConversations(
                (previousConversations) => [
                    conversation,
                    ...previousConversations,
                ]
            );

            setActiveConversation(
                conversationId
            );

            return conversationId;
        }

        // ------------------------------------------
        // Update New Chat Title
        // ------------------------------------------

        const currentConversation =
            conversations.find(
                (conversation) =>
                    conversation.id ===
                    activeConversation
            );

        if (
            currentConversation?.title ===
            "New Chat"
        ) {
            await updateConversationTitle(
                activeConversation,
                title
            );

            setConversations(
                (previousConversations) =>
                    previousConversations.map(
                        (conversation) =>
                            conversation.id ===
                            activeConversation
                                ? {
                                      ...conversation,
                                      title,
                                  }
                                : conversation
                    )
            );
        }

        return activeConversation;
    };

    // ==================================================
    // Persist Assistant Answer
    // ==================================================

    const persistAssistantAnswer = async (
        conversationId,
        assistantId,
        answer
    ) => {
        if (!answer?.trim()) {
            return;
        }

        try {
            await saveMessage(
                conversationId,
                {
                    id: assistantId,
                    role: "assistant",
                    content: answer,
                }
            );
        } catch (error) {
            console.error(
                "Failed to save assistant answer:",
                error
            );

            setConversationError(
                error.message ||
                    "The answer was generated but could not be saved."
            );
        }
    };

    // ==================================================
    // Send Question
    // ==================================================

    const handleAsk = async (
        newQuestion
    ) => {
        if (
            !newQuestion.trim() ||
            loading
        ) {
            return;
        }

        setConversationError("");

        let conversationId;

        try {
            conversationId =
                await prepareConversation(
                    newQuestion
                );
        } catch (error) {
            console.error(
                "Failed to prepare conversation:",
                error
            );

            setConversationError(
                error.message ||
                    "Failed to create conversation."
            );

            return;
        }

        // ------------------------------------------
        // User Message
        // ------------------------------------------

        const userMessage = {
            id: crypto.randomUUID(),
            role: "user",
            content: newQuestion.trim(),
        };

        try {
            await saveMessage(
                conversationId,
                userMessage
            );
        } catch (error) {
            console.error(
                "Failed to save user message:",
                error
            );

            setConversationError(
                error.message ||
                    "Failed to save your message."
            );

            return;
        }

        // ------------------------------------------
        // Assistant Message
        // ------------------------------------------

        const assistantId =
            crypto.randomUUID();

        const assistantMessage = {
            id: assistantId,
            role: "assistant",
            content: "",
            tasks: [],
            progress: [],
            status: "",
            loading: true,
        };

        setConversations(
            (previousConversations) =>
                previousConversations.map(
                    (conversation) => {
                        if (
                            conversation.id !==
                            conversationId
                        ) {
                            return conversation;
                        }

                        return {
                            ...conversation,

                            messages: [
                                ...conversation.messages,
                                userMessage,
                                assistantMessage,
                            ],
                        };
                    }
                )
        );

        setLoading(true);

        const controller =
            new AbortController();

        abortControllerRef.current =
            controller;

        let finalAnswerText = "";
        let assistantSaved = false;

        // ==================================================
        // Streaming Request
        // ==================================================

        try {
            const response = await fetch(
                `${API_BASE_URL}/rag/stream`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        question:
                            newQuestion.trim(),
                    }),

                    signal:
                        controller.signal,
                }
            );

            if (!response.ok) {
                throw new Error(
                    "Failed to connect to the API."
                );
            }

            if (!response.body) {
                throw new Error(
                    "Streaming is not supported."
                );
            }

            const reader =
                response.body.getReader();

            const decoder =
                new TextDecoder();

            let buffer = "";

            // ==================================================
            // Read Stream
            // ==================================================

            while (true) {
                const { value, done } =
                    await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(
                    value,
                    {
                        stream: true,
                    }
                );

                const lines =
                    buffer.split("\n");

                buffer =
                    lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim()) {
                        continue;
                    }

                    let event;

                    try {
                        event =
                            JSON.parse(line);
                    } catch (error) {
                        console.error(
                            "Failed to parse stream event:",
                            line,
                            error
                        );

                        continue;
                    }

                    console.log(
                        "STREAM EVENT:",
                        event
                    );

                    updateTaskFromEvent(
                        conversationId,
                        assistantId,
                        event
                    );

                    updateProgress(
                        conversationId,
                        assistantId,
                        event
                    );

                    const statusMessage =
                        getEventMessage(event);

                    if (statusMessage) {
                        updateAssistantMessage(
                            conversationId,
                            assistantId,
                            (assistant) => ({
                                ...assistant,

                                status:
                                    statusMessage,
                            })
                        );
                    }

                    // --------------------------------------
                    // Final Answer Chunk
                    // --------------------------------------

                    if (
                        event.type ===
                        "final_answer_chunk"
                    ) {
                        const chunk =
                            event.result || "";

                        finalAnswerText +=
                            chunk;

                        updateAssistantMessage(
                            conversationId,
                            assistantId,
                            (assistant) => ({
                                ...assistant,

                                content:
                                    assistant.content +
                                    chunk,
                            })
                        );
                    }

                    // --------------------------------------
                    // Final Answer Complete
                    // --------------------------------------

                    if (
                        event.type ===
                        "final_answer"
                    ) {
                        const completeAnswer =
                            event.result ||
                            finalAnswerText;

                        finalAnswerText =
                            completeAnswer;

                        updateAssistantMessage(
                            conversationId,
                            assistantId,
                            (assistant) => ({
                                ...assistant,

                                content:
                                    completeAnswer ||
                                    assistant.content,

                                status:
                                    "✅ Final answer ready.",
                            })
                        );

                        if (
                            !assistantSaved &&
                            completeAnswer.trim()
                        ) {
                            await persistAssistantAnswer(
                                conversationId,
                                assistantId,
                                completeAnswer
                            );

                            assistantSaved =
                                true;
                        }
                    }

                    // --------------------------------------
                    // Agent Error
                    // --------------------------------------

                    if (
                        event.type ===
                        "agent_error"
                    ) {
                        updateAssistantMessage(
                            conversationId,
                            assistantId,
                            (assistant) => ({
                                ...assistant,

                                loading: false,

                                status:
                                    event.message ||
                                    "❌ Something went wrong.",
                            })
                        );
                    }

                    // --------------------------------------
                    // Agent Finished
                    // --------------------------------------

                    if (
                        event.type ===
                        "agent_finished"
                    ) {
                        updateAssistantMessage(
                            conversationId,
                            assistantId,
                            (assistant) => ({
                                ...assistant,
                                loading: false,
                            })
                        );
                    }
                }
            }
        } catch (error) {
            if (
                error.name ===
                "AbortError"
            ) {
                console.log(
                    "Generation stopped by user."
                );

                updateAssistantMessage(
                    conversationId,
                    assistantId,
                    (assistant) => ({
                        ...assistant,

                        loading: false,

                        status:
                            "⏹️ Generation stopped.",
                    })
                );
            } else {
                console.error(error);

                updateAssistantMessage(
                    conversationId,
                    assistantId,
                    (assistant) => ({
                        ...assistant,

                        loading: false,

                        status:
                            "❌ Failed to connect to the API.",
                    })
                );
            }
        } finally {
            updateAssistantMessage(
                conversationId,
                assistantId,
                (assistant) => ({
                    ...assistant,
                    loading: false,
                })
            );

            setLoading(false);

            abortControllerRef.current =
                null;
        }
    };

    // ==================================================
    // Selected File Names
    // ==================================================

    const selectedFileText = (() => {
        if (selectedFiles.length === 0) {
            return "No documents or images selected";
        }

        if (selectedFiles.length === 1) {
            return selectedFiles[0].name;
        }

        return (
            `${selectedFiles.length} files selected: ` +
            selectedFiles
                .map((file) => file.name)
                .join(", ")
        );
    })();

    // ==================================================
    // Render
    // ==================================================

    return (
        <div className="app-layout">
            <Sidebar
                conversations={conversations}
                activeConversation={
                    activeConversation
                }
                onNewChat={handleNewChat}
                onSelectConversation={
                    handleSelectConversation
                }
                onDeleteConversation={
                    handleDeleteConversation
                }
            />

            <div className="chat-page">
                <header className="chat-header">
                    <h1>Enterprise Multimodal Multi-Agent RAG</h1>

                    <div className="document-upload">
                        <input
                            ref={uploadInputRef}
                            id="document-upload"
                            type="file"
                            multiple
                            accept={
                                ".pdf,.docx,.txt," +
                                ".png,.jpg,.jpeg,.webp"
                            }
                            onChange={(event) => {
                                const files =
                                    Array.from(
                                        event.target
                                            .files ||
                                            []
                                    );

                                setSelectedFiles(
                                    files
                                );

                                setUploadMessage(
                                    ""
                                );
                            }}
                            disabled={
                                uploading ||
                                loading
                            }
                        />

                        <label
                            htmlFor="document-upload"
                            className="choose-file-button"
                        >
                            Choose files
                        </label>

                        <span className="selected-file-name">
                            {selectedFileText}
                        </span>

                        <button
                            type="button"
                            onClick={
                                handleUpload
                            }
                            disabled={
                                selectedFiles.length ===
                                    0 ||
                                uploading ||
                                loading
                            }
                        >
                            {uploading
                                ? "Uploading..."
                                : selectedFiles.length >
                                    1
                                  ? `Upload ${selectedFiles.length} files`
                                  : "Upload"}
                        </button>

                        {uploadMessage && (
                            <span className="upload-message">
                                {uploadMessage}
                            </span>
                        )}
                    </div>

                    {conversationsLoading && (
                        <div className="conversation-database-status">
                            Loading conversations...
                        </div>
                    )}

                    {conversationError && (
                        <div className="conversation-database-error">
                            {conversationError}
                        </div>
                    )}
                </header>

                <main className="chat-container">
                    <ChatMessages
                        messages={messages}
                    />

                    <div ref={messagesEndRef} />
                </main>

                <div className="chat-input-container">
                    <ChatInput
                        onAsk={handleAsk}
                        onStop={handleStop}
                        loading={loading}
                    />
                </div>
            </div>
        </div>
    );
}

export default ChatPage;
