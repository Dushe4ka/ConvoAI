const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// Генерация session_id (для сохранения истории)
const sessionId = localStorage.getItem("session_id") || Math.random().toString(36).substr(2, 9);
localStorage.setItem("session_id", sessionId);

// Функция отправки сообщения
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user-message");
    userInput.value = "";

    addMessage("...", "ai-message", true);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message }),
        });

        const data = await response.json();
        removeLoading();
        addMessage(data.response, "ai-message");
    } catch (error) {
        console.error("Ошибка:", error);
        removeLoading();
        addMessage("Ошибка сервера 😢", "ai-message");
    }
}

// Функция добавления сообщений
function addMessage(text, className, isLoading = false) {
    const msg = document.createElement("div");
    msg.classList.add("message", className);
    if (isLoading) msg.classList.add("loading");
    msg.innerHTML = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Удаление "..."
function removeLoading() {
    const loading = document.querySelector(".loading");
    if (loading) loading.remove();
}

// Кнопка "Отправить"
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Загрузка истории сообщений
async function loadChatHistory() {
    try {
        const response = await fetch(`/history/${sessionId}`);
        const history = await response.json();
        history.forEach(msg => {
            addMessage(msg.message, msg.role === "user" ? "user-message" : "ai-message");
        });
    } catch (error) {
        console.error("Ошибка загрузки истории:", error);
    }
}

window.onload = loadChatHistory;
