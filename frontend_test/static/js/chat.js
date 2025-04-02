const chatHistoryElement = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const chatInputElement = document.getElementById('chat-input');
const newChatButton = document.getElementById('new-chat-button');
// const clearHistoryButton = document.getElementById('clear-history-button');
const chatErrorElement = document.getElementById('chat-error');

let currentSessionId = null;

function displayChatMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', sender === 'user' ? 'user' : 'bot');

    const senderSpan = document.createElement('strong');
    senderSpan.textContent = sender === 'user' ? 'Вы' : 'Бот';
    messageDiv.appendChild(senderSpan);

    // Простая обработка Markdown (например, для **bold** и *italic*)
    // Для более сложного Markdown нужна библиотека
    let formattedMessage = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedMessage = formattedMessage.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Обработка новых строк
    formattedMessage = formattedMessage.replace(/\n/g, '<br>');

    const messageContent = document.createElement('p');
    messageContent.innerHTML = formattedMessage; // Используем innerHTML для отрендеренного markdown/br
    messageDiv.appendChild(messageContent);


    chatHistoryElement.appendChild(messageDiv);
    // Прокрутка вниз
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
}

function displayChatError(message) {
    chatErrorElement.textContent = message;
    chatErrorElement.style.display = 'block';
}

 function hideChatError() {
     chatErrorElement.style.display = 'none';
 }

async function loadChatHistory(sessionId) {
    if (!sessionId) return;
     hideChatError();
    try {
         console.log(`Loading history for session: ${sessionId}`);
        const history = await api.get(`/history/${sessionId}`);
        chatHistoryElement.innerHTML = ''; // Очистить перед загрузкой
        history.forEach(entry => {
            displayChatMessage('user', entry.user_message);
            if (entry.ai_response) {
                displayChatMessage('bot', entry.ai_response);
            }
        });
         console.log(`History loaded for session: ${sessionId}`);
    } catch (error) {
        console.error('Failed to load chat history:', error);
        displayChatError('Не удалось загрузить историю чата.');
         // Возможно, сессия истекла или невалидна, начать новую?
         // await startNewChat();
    }
}

async function sendMessage(message) {
    if (!currentSessionId) {
        console.warn("No active session ID. Starting new chat.");
         await startNewChat(); // Попробуем начать новый чат, если сессии нет
         if (!currentSessionId) { // Если и после этого нет, то ошибка
            displayChatError("Не удалось начать чат. Попробуйте обновить страницу.");
            return;
         }
    }
    hideChatError();
    displayChatMessage('user', message);
    chatInputElement.disabled = true; // Блокируем ввод во время ожидания ответа

    try {
        const response = await api.post('/chat', {
            session_id: currentSessionId,
            message: message
        });
        if (response.response) {
            displayChatMessage('bot', response.response);
        } else {
            console.warn("No response text from bot.");
             // Можно показать сообщение "Бот не ответил"
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        displayChatError(error.message || 'Не удалось отправить сообщение.');
    } finally {
        chatInputElement.disabled = false; // Разблокируем ввод
         chatInputElement.focus();
    }
}

async function startNewChat() {
    hideChatError();
    try {
        const data = await api.get('/new_chat');
        if (data.session_id) {
            currentSessionId = data.session_id;
            chatHistoryElement.innerHTML = ''; // Очистить историю на экране
            chatInputElement.value = ''; // Очистить поле ввода
            console.log(`Started new chat session: ${currentSessionId}`);
            // Сохраняем ID сессии, чтобы восстановить при перезагрузке
             sessionStorage.setItem('currentChatSessionId', currentSessionId);
             return true;
        }
         displayChatError('Не удалось получить ID новой сессии.');
         return false;
    } catch (error) {
        console.error('Failed to create new chat:', error);
        displayChatError('Не удалось создать новый чат.');
         return false;
    }
}

// --- Инициализация и обработчики ---
if (chatForm) {
    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const message = chatInputElement.value.trim();
        if (message) {
            await sendMessage(message);
            chatInputElement.value = ''; // Очистить поле ввода после отправки
        }
    });
}

if (newChatButton) {
    newChatButton.addEventListener('click', async () => {
         await startNewChat();
    });
}

 // Попытка восстановить ID сессии при загрузке
 function initializeChat() {
    const savedSessionId = sessionStorage.getItem('currentChatSessionId');
    if (savedSessionId) {
        currentSessionId = savedSessionId;
        loadChatHistory(currentSessionId);
    } else {
        // Если нет сохраненной сессии, можно сразу начать новую
        // startNewChat(); // Или ждать, пока пользователь начнет сам
        console.log("No saved chat session found.");
    }
 }