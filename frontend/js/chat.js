(function () {
    const messagesEl = document.getElementById("chat-messages");
    const inputEl    = document.getElementById("chat-input");
    const sendBtn    = document.getElementById("chat-send-btn");
    const useContext = document.getElementById("use-context");

    let history = [];

    sendBtn.addEventListener("click", () => sendMessage());
    inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    async function sendMessage() {
        const text = inputEl.value.trim();
        if (!text) return;

        appendMessage("user", text);
        inputEl.value    = "";
        sendBtn.disabled = true;

        const bubbleEl = appendStreamBubble();
        let   fullText = "";

        try {
            const context = useContext.checked ? getDiagnosisContext() : "";

            const response = await fetch("http://localhost:8000/api/chat/stream", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({
                    message:           text,
                    history:           history,
                    diagnosis_context: context,
                }),
            });

            const reader  = response.body.getReader();
            const decoder = new TextDecoder();
            let   buffer  = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;

                    let data;
                    try {
                        data = JSON.parse(line.slice(6));
                    } catch {
                        continue;
                    }

                    if (data.token) {
                        fullText += data.token;
                        bubbleEl.innerHTML = formatText(fullText);
                        messagesEl.scrollTop = messagesEl.scrollHeight;
                    }

                    if (data.done) {
                        history = data.history || history;
                    }
                }
            }

        } catch (err) {
            bubbleEl.innerHTML = "Xəta: " + escapeHtml(err.message);
        } finally {
            sendBtn.disabled = false;
            inputEl.focus();
        }
    }

    function appendMessage(role, text) {
        const isUser = role === "user";
        const div    = document.createElement("div");
        div.className = `chat-msg ${role}`;
        div.innerHTML = `
            ${!isUser ? `<div class="msg-avatar"><i class="ph ph-robot"></i></div>` : ""}
            <div class="msg-bubble">${formatText(text)}</div>
            ${isUser  ? `<div class="msg-avatar"><i class="ph ph-user"></i></div>`  : ""}
        `;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function appendStreamBubble() {
        const div = document.createElement("div");
        div.className = "chat-msg assistant";
        div.innerHTML = `
            <div class="msg-avatar"><i class="ph ph-robot"></i></div>
            <div class="msg-bubble stream-bubble">
                <span class="cursor-blink">▍</span>
            </div>
        `;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div.querySelector(".msg-bubble");
    }

    function formatText(text) {
        return escapeHtml(text)
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g,     "<em>$1</em>");
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\n\n/g, "<br><br>")
            .replace(/\n/g,   "<br>");
    }
})();