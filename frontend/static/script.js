// Получение session_id из localStorage или генерация нового
let sessionId = localStorage.getItem("session_id") || generateSessionId();
localStorage.setItem("session_id", sessionId);

document.addEventListener("DOMContentLoaded", () => {
    if (!sessionId) {
        alert("Не удалось найти идентификатор сессии.");
        return;
    }

    document.getElementById("send-button").addEventListener("click", sendMessage);
    document.getElementById("clear-chat-button").addEventListener("click", clearChatHistory);
    document.getElementById("upload-button").addEventListener("click", uploadFile);
    document.getElementById("user-input").addEventListener("keydown", handleEnterKey);

    loadChatHistory(); // Загрузка истории при открытии страницы
});

// Генерация нового session_id
function generateSessionId() {
    return Math.random().toString(36).substr(2, 9);
}

// Отправка сообщения
async function sendMessage() {
    const userInput = document.getElementById("user-input").value.trim();
    if (userInput === "") return;

    const chatHistory = document.getElementById("chat-history");
    chatHistory.innerHTML += `<div class="user"><strong>Вы:</strong> ${userInput}</div>`;
    document.getElementById("user-input").value = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: userInput,
            }),
        });

        const data = await response.json();
        chatHistory.innerHTML += `<div class="ai"><strong>ИИ:</strong> ${data.response}</div>`;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    } catch (error) {
        console.error("Ошибка:", error);
        chatHistory.innerHTML += `<div class="ai error"><strong>Ошибка:</strong> Не удалось получить ответ.</div>`;
    }
}

// Обработчик Enter
function handleEnterKey(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
}

// Очистка истории чата
async function clearChatHistory() {
    if (!confirm("Вы уверены, что хотите очистить историю чата?")) return;

    try {
        const response = await fetch(`/clear_chat/${sessionId}`, { method: "DELETE" });
        const data = await response.json();
        alert(data.message);
        document.getElementById("chat-history").innerHTML = "";
    } catch (error) {
        console.error("Ошибка очистки чата:", error);
        alert("Ошибка при очистке истории чата.");
    }
}

// Загрузка истории чата при открытии страницы
async function loadChatHistory() {
    try {
        const response = await fetch(`/history/${sessionId}`);
        const history = await response.json();

        const chatHistory = document.getElementById("chat-history");
        history.forEach(msg => {
            chatHistory.innerHTML += `<div class="${msg.role === "user" ? "user" : "ai"}">
                <strong>${msg.role === "user" ? "Вы" : "ИИ"}:</strong> ${msg.message}
            </div>`;
        });
        chatHistory.scrollTop = chatHistory.scrollHeight;
    } catch (error) {
        console.error("Ошибка загрузки истории:", error);
    }
}

// Загрузка файла
async function uploadFile() {
    const fileInput = document.getElementById("file-upload");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);

    try {
        const response = await fetch("/upload_file/", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        alert(data.message);
        fileInput.value = "";
    } catch (error) {
        console.error("Ошибка загрузки файла:", error);
        alert("Ошибка при загрузке файла.");
    }
}
