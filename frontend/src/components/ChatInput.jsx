import {
    useEffect,
    useRef,
    useState,
} from "react";


function ChatInput({
    onAsk,
    onStop,
    loading,
}) {
    // ==================================================
    // Input State
    // ==================================================

    const [question, setQuestion] =
        useState("");

    // ==================================================
    // Voice State
    // ==================================================

    const [listening, setListening] =
        useState(false);

    const [
        speechSupported,
        setSpeechSupported,
    ] = useState(true);

    const [
        voiceError,
        setVoiceError,
    ] = useState("");

    // ==================================================
    // Voice References
    // ==================================================

    const recognitionRef = useRef(null);

    const baseQuestionRef = useRef("");

    const silenceTimerRef = useRef(null);

    const ignoreSpeechResultsRef =
        useRef(false);

    // ==================================================
    // Silence Timer
    // ==================================================

    const clearSilenceTimer = () => {
        if (!silenceTimerRef.current) {
            return;
        }

        clearTimeout(
            silenceTimerRef.current
        );

        silenceTimerRef.current = null;
    };

    // ==================================================
    // Browser Support
    // ==================================================

    useEffect(() => {
        const SpeechRecognition =
            window.SpeechRecognition ||
            window.webkitSpeechRecognition;

        setSpeechSupported(
            Boolean(SpeechRecognition)
        );
    }, []);

    // ==================================================
    // Stop Voice While Agent Is Generating
    // ==================================================

    useEffect(() => {
        if (
            loading &&
            recognitionRef.current
        ) {
            ignoreSpeechResultsRef.current =
                true;

            clearSilenceTimer();

            recognitionRef.current.abort();
        }
    }, [loading]);

    // ==================================================
    // Cleanup
    // ==================================================

    useEffect(() => {
        return () => {
            ignoreSpeechResultsRef.current =
                true;

            clearSilenceTimer();

            if (recognitionRef.current) {
                recognitionRef.current.abort();
                recognitionRef.current = null;
            }
        };
    }, []);

    // ==================================================
    // Start Listening
    // ==================================================

    const startListening = () => {
        const SpeechRecognition =
            window.SpeechRecognition ||
            window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            setSpeechSupported(false);

            setVoiceError(
                "Voice input is not supported "
                + "by this browser."
            );

            return;
        }

        if (recognitionRef.current) {
            return;
        }

        setVoiceError("");

        ignoreSpeechResultsRef.current =
            false;

        const recognition =
            new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.lang =
            navigator.language || "en-US";

        baseQuestionRef.current =
            question.trim();

        // ------------------------------------------
        // Started
        // ------------------------------------------

        recognition.onstart = () => {
            setListening(true);
            setVoiceError("");

            clearSilenceTimer();

            silenceTimerRef.current =
                setTimeout(() => {
                    ignoreSpeechResultsRef.current =
                        true;

                    setVoiceError(
                        "No speech was detected. "
                        + "Please try again."
                    );

                    if (
                        recognitionRef.current
                    ) {
                        recognitionRef.current.abort();
                    }
                }, 8000);
        };

        // ------------------------------------------
        // Speech Result
        // ------------------------------------------

        recognition.onresult = (event) => {
            if (
                ignoreSpeechResultsRef.current
            ) {
                return;
            }

            let transcript = "";

            for (
                let index = 0;
                index < event.results.length;
                index += 1
            ) {
                transcript +=
                    event.results[index][0]
                        .transcript;
            }

            if (transcript.trim()) {
                clearSilenceTimer();
                setVoiceError("");
            }

            const baseQuestion =
                baseQuestionRef.current;

            const combinedQuestion = [
                baseQuestion,
                transcript.trim(),
            ]
                .filter(Boolean)
                .join(" ");

            setQuestion(
                combinedQuestion
            );
        };

        // ------------------------------------------
        // Recognition Error
        // ------------------------------------------

        recognition.onerror = (event) => {
            clearSilenceTimer();

            if (event.error === "aborted") {
                return;
            }

            const errorMessages = {
                "not-allowed":
                    "Microphone permission was denied.",

                "service-not-allowed":
                    "Voice recognition service "
                    + "is not allowed.",

                "audio-capture":
                    "No microphone was detected.",

                "no-speech":
                    "No speech was detected. "
                    + "Please try again.",

                network:
                    "Voice recognition network error.",
            };

            setVoiceError(
                errorMessages[event.error] ||
                    "Voice recognition failed."
            );
        };

        // ------------------------------------------
        // Recognition Ended
        // ------------------------------------------

        recognition.onend = () => {
            clearSilenceTimer();

            setListening(false);

            if (
                recognitionRef.current
                === recognition
            ) {
                recognitionRef.current = null;
            }
        };

        recognitionRef.current =
            recognition;

        try {
            recognition.start();
        } catch (error) {
            console.error(
                "Could not start voice input:",
                error
            );

            clearSilenceTimer();

            recognitionRef.current = null;

            setListening(false);

            setVoiceError(
                "Could not start voice input."
            );
        }
    };

    // ==================================================
    // Stop Listening
    // ==================================================

    const stopListening = () => {
        if (!recognitionRef.current) {
            return;
        }

        clearSilenceTimer();

        recognitionRef.current.stop();
    };

    // ==================================================
    // Voice Button
    // ==================================================

    const handleVoiceButton = () => {
        if (listening) {
            stopListening();
            return;
        }

        startListening();
    };

    // ==================================================
    // Submit
    // ==================================================

    const handleSubmit = (event) => {
        event.preventDefault();

        const trimmedQuestion =
            question.trim();

        if (
            !trimmedQuestion ||
            loading
        ) {
            return;
        }

        if (recognitionRef.current) {
            ignoreSpeechResultsRef.current =
                true;

            clearSilenceTimer();

            recognitionRef.current.abort();
        }

        onAsk(trimmedQuestion);

        setQuestion("");
        setVoiceError("");
    };

    // ==================================================
    // Keyboard
    // ==================================================

    const handleKeyDown = (event) => {
        /*
         * Enter = send
         * Shift + Enter = new line
         */

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            handleSubmit(event);
        }
    };

    // ==================================================
    // Render
    // ==================================================

    return (
        <form
            className="chat-input-form"
            onSubmit={handleSubmit}
        >
            <div className="chat-input-row">
                <textarea
                    value={question}
                    placeholder={
                        listening
                            ? "Listening..."
                            : "Ask a question..."
                    }
                    onChange={(event) =>
                        setQuestion(
                            event.target.value
                        )
                    }
                    onKeyDown={handleKeyDown}
                    disabled={loading}
                    rows={2}
                />

                {speechSupported &&
                    !loading && (
                        <button
                            type="button"
                            className={
                                listening
                                    ? (
                                        "voice-button "
                                        + "listening"
                                    )
                                    : "voice-button"
                            }
                            onClick={
                                handleVoiceButton
                            }
                            aria-label={
                                listening
                                    ? "Stop voice input"
                                    : "Start voice input"
                            }
                            title={
                                listening
                                    ? "Stop listening"
                                    : "Use voice input"
                            }
                        >
                            {listening
                                ? "⏹️"
                                : "🎙️"}
                        </button>
                    )}

                {loading ? (
                    <button
                        type="button"
                        onClick={onStop}
                    >
                        Stop
                    </button>
                ) : (
                    <button
                        type="submit"
                        disabled={
                            !question.trim()
                        }
                    >
                        Send
                    </button>
                )}
            </div>

            {listening && (
                <div className="voice-status">
                    <span className="listening-dot" />

                    <span>
                        Listening… Speak now.
                    </span>
                </div>
            )}

            {voiceError && (
                <div className="voice-error">
                    {voiceError}
                </div>
            )}

            {!speechSupported && (
                <div className="voice-error">
                    Voice input is not supported
                    by this browser. Use Chrome
                    or Edge.
                </div>
            )}
        </form>
    );
}


export default ChatInput;