document.addEventListener('DOMContentLoaded', function() {
    // Элементы интерфейса
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileUpload = document.getElementById('file-upload');
    const uploadStatus = document.getElementById('upload-status');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatSessions = document.getElementById('chat-sessions');
    const categorySelect = document.getElementById('category-select');
    const dateFrom = document.getElementById('date-from');
    const dateTo = document.getElementById('date-to');

    // Текущая сессия
    let currentSessionId = null;
    let categories = [];

    // Инициализация
    initApp();

    async function initApp() {
        await loadCategories();
        setupEventListeners();
        createNewChat();
    }
    // Загрузка категорий
    async function loadCategories() {
        try {
            const response = await fetch('/api/categories');
            if (response.ok) {
                const data = await response.json();
                // Убедимся, что получаем массив категорий
                if (Array.isArray(data.categories)) {
                    categories = data.categories;
                    updateCategoryFilter();
                } else {
                    console.error('Некорректный формат категорий:', data);
                }
            }
        } catch (error) {
            console.error('Ошибка загрузки категорий:', error);
        }
    }

    // Обновление фильтра категорий
    function updateCategoryFilter() {
        categorySelect.innerHTML = ''; // Очищаем список

        // Добавляем опцию "Все категории"
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Все категории';
        categorySelect.appendChild(defaultOption);

        // Добавляем категории из списка
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            categorySelect.appendChild(option);
        });

    console.log('Категории загружены:', categories); // Для отладки
}

    function setupEventListeners() {
        // Отправка сообщения
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Загрузка файлов
        uploadBtn.addEventListener('click', () => fileUpload.click());
        fileUpload.addEventListener('change', handleFileUpload);

        // Управление чатами
        newChatBtn.addEventListener('click', createNewChat);

        // Фильтры
        categorySelect.addEventListener('change', applyFilters);
        dateFrom.addEventListener('change', applyFilters);
        dateTo.addEventListener('change', applyFilters);
    }

    function applyFilters() {
        if (currentSessionId) {
            loadChatHistory();
        }
    }

    async function createNewChat() {
        try {
            const response = await fetch('/api/new_chat');
            if (response.ok) {
                const data = await response.json();
                currentSessionId = data.session_id;
                chatMessages.innerHTML = '';
                resetFilters();
            }
        } catch (error) {
            console.error('Ошибка создания чата:', error);
            showError('Не удалось создать новый чат');
        }
    }

    function resetFilters() {
        categorySelect.value = '';
        dateFrom.value = '';
        dateTo.value = '';
    }

    async function loadChatHistory() {
        if (!currentSessionId) return;

        try {
            const response = await fetch(`/api/history/${currentSessionId}`);
            if (response.ok) {
                const messages = await response.json();
                renderChatMessages(messages);
            }
        } catch (error) {
            console.error('Ошибка загрузки истории:', error);
        }
    }

    function renderChatMessages(messages) {
        chatMessages.innerHTML = '';
        messages.forEach(msg => {
            addMessage(msg.role, msg.message);
        });
        scrollToBottom();
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message || !currentSessionId) return;

        addMessage('user', message);
        userInput.value = '';

        try {
            const filters = {
                category: categorySelect.value || null,
                date_from: dateFrom.value || null,
                date_to: dateTo.value || null
            };

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: message,
                    filters: filters
                })
            });

            if (response.ok) {
                const data = await response.json();
                addMessage('ai', data.response, data.metadata);
            } else {
                throw new Error('Ошибка сервера');
            }
        } catch (error) {
            console.error('Ошибка отправки:', error);
            addMessage('ai', 'Ошибка обработки запроса. Попробуйте позже.');
        }

        scrollToBottom();
    }

    function addMessage(role, content, metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;

        let metaHTML = '';
        if (metadata) {
            if (metadata.category) {
                metaHTML += `<div class="message-meta">Категория: ${metadata.category}</div>`;
            }
            if (metadata.upload_dates) {
                const {start, end} = metadata.upload_dates;
                metaHTML += `<div class="message-meta">Период загрузки: ${start || 'любой'} — ${end || 'любой'}</div>`;
            }
            if (metadata.sources?.length > 0) {
                metaHTML += `<div class="message-meta">Источники: ${metadata.sources.join(', ')}</div>`;
            }
        }

        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            ${metaHTML}
        `;

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    async function handleFileUpload() {
        if (!fileUpload.files.length) {
            showStatus('Файл не выбран', 'error');
            return;
        }

        showStatus('Загрузка...', 'info');

        try {
            const formData = new FormData();
            Array.from(fileUpload.files).forEach(file => {
                formData.append('files', file); // Убедитесь, что FastAPI ожидает 'file'
            });

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Ошибка сервера');
            }

            const data = await response.json();
            showStatus(`Успешно загружено: ${data.filename}`, 'success');
            await loadCategories();  // Обновляем категории после загрузки
        } catch (error) {
            console.error('Ошибка загрузки:', error);
            showStatus(`Ошибка: ${error.message}`, 'error');
        } finally {
            fileUpload.value = ''; // Сброс выбора файла
        }
    }



    function showStatus(text, type) {
        uploadStatus.textContent = text;
        uploadStatus.className = type;
        // Автоматически скрываем сообщение через 5 секунд
        if (type === 'success') {
            setTimeout(() => {
                if (uploadStatus.textContent === text) {
                    uploadStatus.textContent = '';
                }
            }, 5000);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message message-error';
        errorDiv.textContent = message;
        chatMessages.appendChild(errorDiv);
        scrollToBottom();
    }
});