const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const newChatBtn = document.getElementById("newChatBtn"); // Кнопка "Новый чат"
const clearChatBtn = document.getElementById("clearChatBtn"); // Кнопка "Очистить историю"

// Генерация session_id (если нет в localStorage)
let sessionId = localStorage.getItem("session_id") || generateSessionId();
localStorage.setItem("session_id", sessionId);

// Функция генерации нового session_id
function generateSessionId() {
    return Math.random().toString(36).substr(2, 9);
}

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

// Функция загрузки файла
uploadBtn.addEventListener("click", async () => {
    if (fileInput.files.length === 0) {
        alert("Выберите файл перед загрузкой.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    addMessage("Загрузка файла...", "ai-message", true);
    uploadBtn.disabled = true;

    try {
        let response = await fetch("/upload_file", {
            method: "POST",
            body: formData
        });

        let result = await response.json();
        removeLoading();
        alert(result.message);
        addMessage("Файл успешно загружен! ✅", "ai-message");
    } catch (error) {
        removeLoading();
        alert("Ошибка загрузки файла.");
        console.error("Ошибка:", error);
        addMessage("Ошибка загрузки файла ❌", "ai-message");
    }

    uploadBtn.disabled = false;
});

// --- Кнопка "Очистить историю" ---
clearChatBtn.addEventListener("click", async () => {
    if (!confirm("Вы действительно хотите очистить историю чата?")) return;

    try {
        await fetch(`/chat/${sessionId}`, { method: "DELETE" }); // Исправленный эндпоинт
        chatBox.innerHTML = ""; // Очистка визуального интерфейса
        alert("История чата очищена!");
    } catch (error) {
        console.error("Ошибка очистки истории:", error);
        alert("Ошибка при очистке истории.");
    }
});

// --- Кнопка "Новый чат" ---
newChatBtn.addEventListener("click", async () => {
    if (!confirm("Вы действительно хотите создать новый чат?")) return;

    try {
        const response = await fetch("/new_chat"); // Исправленный эндпоинт
        const data = await response.json();

        sessionId = data.session_id;
        localStorage.setItem("session_id", sessionId);
        chatBox.innerHTML = ""; // Очистка чата в интерфейсе

        alert(`Создан новый чат (ID: ${sessionId})`);
    } catch (error) {
        console.error("Ошибка создания нового чата:", error);
        alert("Ошибка при создании нового чата.");
    }
});
