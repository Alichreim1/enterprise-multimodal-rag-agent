function Sidebar({
    conversations,
    activeConversation,
    onNewChat,
    onSelectConversation,
    onDeleteConversation,
}) {

    return (

        <aside className="sidebar">

            <button
                className="new-chat-button"
                onClick={onNewChat}
            >
                + New Chat
            </button>


            <div className="conversation-list">

                {conversations.map(
                    (conversation) => (

                        <div
                            key={conversation.id}
                            className={
                                conversation.id ===
                                activeConversation
                                    ? "conversation-item-wrapper active"
                                    : "conversation-item-wrapper"
                            }
                        >

                            <button
                                className={
                                    conversation.id ===
                                    activeConversation
                                        ? "conversation-item active"
                                        : "conversation-item"
                                }
                                onClick={() =>
                                    onSelectConversation(
                                        conversation.id
                                    )
                                }
                            >
                                {conversation.title}
                            </button>


                            <button
                                className="delete-conversation-button"
                                onClick={() =>
                                    onDeleteConversation(
                                        conversation.id
                                    )
                                }
                            >
                                🗑️
                            </button>

                        </div>

                    )
                )}

            </div>

        </aside>
    );
}

export default Sidebar;