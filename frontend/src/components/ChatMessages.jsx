import { useState } from "react";


function ChatMessages({ messages }) {
    if (!messages.length) {
        return null;
    }

    return (
        <div className="messages">
            {messages.map((message) => (
                <Message
                    key={message.id}
                    message={message}
                />
            ))}
        </div>
    );
}


function getTaskTypeLabel(taskType) {
    switch (taskType) {
        case "retrieve":
            return "Research";

        case "email":
            return "Email";

        case "tool":
            return "Tool";

        case "statement":
            return "Memory";

        default:
            return "Task";
    }
}


function TaskStatus({ status }) {
    switch (status) {
        case "pending":
            return (
                <span>
                    ○ Pending
                </span>
            );

        case "working":
            return (
                <span>
                    ⟳ Working...
                </span>
            );

        case "retrying":
            return (
                <span>
                    ⟳ Retrying...
                </span>
            );

        case "completed":
            return (
                <span>
                    ✓ Completed
                </span>
            );

        case "failed":
            return (
                <span>
                    ✕ Failed
                </span>
            );

        default:
            return null;
    }
}


function Message({ message }) {
    const [showTasks, setShowTasks] =
        useState(true);

    const isUser = message.role === "user";

    return (
        <div
            className={
                isUser
                    ? "message-row user-row"
                    : "message-row assistant-row"
            }
        >
            <div
                className={
                    isUser
                        ? "message-bubble user-bubble"
                        : "message-bubble assistant-bubble"
                }
            >
                {isUser && (
                    <div>
                        <div className="message-role">
                            You
                        </div>

                        <p className="message-content">
                            {message.content}
                        </p>
                    </div>
                )}

                {!isUser && (
                    <div>
                        <div className="message-role">
                            Agent
                        </div>

                        {message.tasks?.length > 0 && (
                            <div className="tasks-section">
                                <button
                                    type="button"
                                    className="tasks-toggle"
                                    onClick={() =>
                                        setShowTasks(
                                            (current) =>
                                                !current
                                        )
                                    }
                                >
                                    <span>
                                        {showTasks
                                            ? "▼"
                                            : "▶"}
                                    </span>

                                    <span>
                                        {message.tasks.length}{" "}
                                        {message.tasks.length === 1
                                            ? "Task"
                                            : "Tasks"}
                                    </span>
                                </button>

                                {showTasks && (
                                    <div className="tasks">
                                        {message.tasks.map(
                                            (task, index) => {
                                                const taskResult =
                                                    typeof task.result ===
                                                    "string"
                                                        ? task.result
                                                        : null;

                                                return (
                                                    <div
                                                        key={task.id}
                                                        className={
                                                            `task task-${task.status}`
                                                        }
                                                    >
                                                        <div className="task-title">
                                                            <span>
                                                                Task{" "}
                                                                {index + 1}
                                                            </span>

                                                            <span className="task-type">
                                                                {getTaskTypeLabel(
                                                                    task.taskType
                                                                )}
                                                            </span>
                                                        </div>

                                                        <div className="task-description">
                                                            {task.description}
                                                        </div>

                                                        {task.dependsOn
                                                            ?.length > 0 && (
                                                            <div className="task-dependencies">
                                                                Depends on:{" "}
                                                                {task.dependsOn
                                                                    .map(
                                                                        (
                                                                            dependencyId
                                                                        ) =>
                                                                            `Task ${dependencyId}`
                                                                    )
                                                                    .join(
                                                                        ", "
                                                                    )}
                                                            </div>
                                                        )}

                                                        <div className="task-status">
                                                            <TaskStatus
                                                                status={
                                                                    task.status
                                                                }
                                                            />
                                                        </div>

                                                        {taskResult &&
                                                            task.status ===
                                                                "completed" && (
                                                                <div className="task-result">
                                                                    {
                                                                        taskResult
                                                                    }
                                                                </div>
                                                            )}
                                                    </div>
                                                );
                                            }
                                        )}
                                    </div>
                                )}
                            </div>
                        )}

                        {message.progress?.length > 0 && (
                            <div className="agent-progress">
                                {message.progress.map(
                                    (item, index) => (
                                        <div
                                            key={`${item.type}-${index}`}
                                            className={
                                                item.completed
                                                    ? "progress-item completed"
                                                    : "progress-item active"
                                            }
                                        >
                                            <span className="progress-icon">
                                                {item.completed
                                                    ? "✓"
                                                    : "⟳"}
                                            </span>

                                            <span>
                                                {item.message}
                                            </span>
                                        </div>
                                    )
                                )}
                            </div>
                        )}

                        {message.content && (
                            <div className="final-answer">
                                {message.content}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}


export default ChatMessages;