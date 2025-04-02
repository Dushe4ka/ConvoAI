// script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Элементы ---
    const chatList = document.getElementById('chat-list');
    const createChatBtn = document.getElementById('create-chat-btn');
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const currentChatNameHeader = document.getElementById('current-chat-name');
    const typingIndicator = document.getElementById('typing-indicator');
    const uploadIndicator = document.getElementById('upload-indicator');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // --- Состояние приложения ---
    let currentChatId = null; // ID текущего активного чата
    let isLoadingHistory = false;
    let isSendingMessage = false;
    let isUploading = false;
    let generalChatId = null; // Сохраним ID общего чата

    // --- Инициализация ---
    loadChats();
    adjustTextareaHeight();

    // --- Вспомогательные функции ---

    /**
     * Обертка для fetch API
     * @param {string} url - URL эндпоинта
     * @param {object} options - Опции для fetch (method, headers, body, etc.)
     * @returns {Promise<any>} - Распарсенный JSON ответ
     * @throws {Error} - В случае ошибки сети или ответа сервера с ошибкой
     */
    async function fetchAPI(url, options = {}) {
        // Добавляем стандартные заголовки
        options.headers = {
            'Accept': 'application/json',
            ...options.headers // Позволяет переопределить или добавить заголовки
        };
        // Не добавляем 'Content-Type': 'application/json' для FormData
        if (!(options.body instanceof FormData)) {
            options.headers['Content-Type'] = 'application/json';
        }

        try {
            const response = await fetch(url, options);

            if (!response.ok) {
                let errorData;
                try {
                    // Пытаемся получить детали ошибки из JSON ответа
                    errorData = await response.json();
                } catch (e) {
                    // Если ответ не JSON или пустой
                    errorData = { detail: response.statusText || `Ошибка ${response.status}` };
                }
                // Формируем сообщение об ошибке
                const errorMessage = errorData.detail || `Ошибка сервера: ${response.status}`;
                console.error(`API Error ${response.status}: ${errorMessage}`, errorData);
                throw new Error(errorMessage);
            }

            // Обработка пустого ответа (например, для DELETE)
            if (response.status === 204 || response.headers.get('content-length') === '0') {
                return null; // Или вернуть { success: true }
            }

            // Пытаемся распарсить JSON
            const data = await response.json();
            return data;

        } catch (error) {
            console.error('Fetch API error:', error);
            // Перевыбрасываем ошибку или обрабатываем её специфично
            throw error; // Передаем ошибку дальше для обработки в вызывающем коде
        }
    }

    /** Отображает сообщение в окне чата */
    function renderMessage(role, messageContent, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role); // role: 'user', 'ai', 'system', 'error'

        // Обработка Markdown для AI сообщений
        if (role === 'ai') {
            try {
                // Используем Marked.js для преобразования Markdown в HTML
                // Используем DOMPurify для очистки HTML перед вставкой
                const dirtyHtml = marked.parse(messageContent);
                messageDiv.innerHTML = DOMPurify.sanitize(dirtyHtml);
            } catch (e) {
                console.error("Ошибка обработки Markdown:", e);
                // В случае ошибки вставляем текст как есть (безопасный вариант)
                messageDiv.textContent = messageContent;
            }
        } else {
             // Для других ролей просто вставляем текст (безопасно)
            messageDiv.textContent = messageContent;
        }


        // Добавляем временную метку, если она есть
        if (timestamp) {
            const timeSpan = document.createElement('span');
            timeSpan.classList.add('timestamp');
            try {
                // Форматируем время для отображения
                timeSpan.textContent = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } catch { /* Игнорируем ошибки парсинга даты */ }
            messageDiv.appendChild(timeSpan);
        }

        chatMessages.appendChild(messageDiv);
        scrollToBottom(); // Прокручиваем вниз после добавления сообщения
    }

    /** Прокручивает область сообщений вниз */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /** Очищает область сообщений */
    function clearMessages() {
        chatMessages.innerHTML = '';
    }

    /** Управляет видимостью индикатора загрузки/печати */
    function showTypingIndicator(show) {
        typingIndicator.style.display = show ? 'block' : 'none';
        // Блокируем ввод во время ожидания ответа AI
        messageInput.disabled = show;
        sendBtn.disabled = show;
        uploadBtn.disabled = show;
    }

     /** Управляет видимостью индикатора загрузки файлов */
     function showUploadIndicator(show, message = 'Загрузка файлов...') {
        isUploading = show; // Обновляем состояние
        uploadIndicator.textContent = message;
        uploadIndicator.style.display = show ? 'block' : 'none';
        // Блокируем ввод во время загрузки файлов
        messageInput.disabled = show;
        sendBtn.disabled = show;
        uploadBtn.disabled = show;
    }


    /** Автоматически изменяет высоту textarea */
    function adjustTextareaHeight() {
        messageInput.style.height = 'auto'; // Сброс высоты
        let scrollHeight = messageInput.scrollHeight;
        // Учитываем border и padding, если box-sizing: border-box
        let style = window.getComputedStyle(messageInput);
        let borderTop = parseInt(style.borderTopWidth, 10);
        let borderBottom = parseInt(style.borderBottomWidth, 10);
        scrollHeight += borderTop + borderBottom;

        // Ограничиваем максимальную высоту
        const maxHeight = 100;
        if (scrollHeight > maxHeight) {
            messageInput.style.height = `${maxHeight}px`;
            messageInput.style.overflowY = 'auto'; // Показываем скроллбар, если нужно
        } else {
            messageInput.style.height = `${scrollHeight}px`;
            messageInput.style.overflowY = 'hidden'; // Скрываем скроллбар
        }
    }

    // --- Функции Загрузки и Отображения Данных ---

    /** Загружает и отображает список чатов */
    async function loadChats() {
        chatList.innerHTML = '<li class="loading-chats">Загрузка чатов...</li>';
        try {
            const data = await fetchAPI('/api/chats');
            renderChatList(data.chats);
            // Находим и сохраняем ID общего чата
            const general = data.chats.find(chat => chat.is_general);
            if (general) {
                generalChatId = general.session_id;
            }

            // Если текущий чат не выбран или удален, выбираем общий чат
            if (!currentChatId || !data.chats.some(chat => chat.session_id === currentChatId)) {
                 if(generalChatId) {
                    handleSelectChat(generalChatId);
                 } else if (data.chats.length > 0) {
                     // Если нет общего, выбираем первый в списке
                     handleSelectChat(data.chats[0].session_id);
                 } else {
                     // Если вообще нет чатов
                     handleSelectChat(null);
                 }
            } else {
                // Обновляем выделение активного чата (на случай если список перезагрузился)
                 highlightActiveChat(currentChatId);
                 // Обновляем имя в шапке
                 const currentChatData = data.chats.find(chat => chat.session_id === currentChatId);
                 if (currentChatData) {
                    currentChatNameHeader.textContent = currentChatData.name;
                    clearHistoryBtn.style.display = 'inline-block'; // Показываем кнопку очистки
                 }
            }

        } catch (error) {
            console.error('Ошибка загрузки чатов:', error);
            chatList.innerHTML = '<li class="loading-chats error">Не удалось загрузить чаты</li>';
            // Отображаем ошибку в основном окне
            clearMessages();
            renderMessage('error', `Ошибка загрузки списка чатов: ${error.message}`);
        }
    }

    /** Отображает список чатов в DOM */
    function renderChatList(chats) {
        chatList.innerHTML = ''; // Очищаем список
        if (!chats || chats.length === 0) {
            chatList.innerHTML = '<li class="loading-chats">Нет доступных чатов</li>';
            return;
        }

        chats.forEach(chat => {
            const li = document.createElement('li');
            li.dataset.sessionId = chat.session_id;
            li.title = `ID: ${chat.session_id}\nСоздан: ${chat.created_at ? new Date(chat.created_at).toLocaleString() : 'N/A'}`;

            const nameSpan = document.createElement('span');
            nameSpan.classList.add('chat-name');
            nameSpan.textContent = chat.name;
            li.appendChild(nameSpan);

            // Добавляем иконку замка для общего чата
            if (chat.is_general) {
                li.classList.add('general-chat');
                const lockIcon = document.createElement('i');
                lockIcon.classList.add('fas', 'fa-lock', 'fa-xs');
                lockIcon.style.marginLeft = '5px';
                lockIcon.style.opacity = '0.6';
                nameSpan.appendChild(lockIcon);
            }

            // Кнопка удаления для пользовательских чатов
            if (chat.can_delete) {
                const deleteBtn = document.createElement('button');
                deleteBtn.classList.add('delete-chat-btn');
                deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
                deleteBtn.title = 'Удалить этот чат';
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation(); // Предотвращаем выбор чата при клике на удаление
                    handleDeleteChat(chat.session_id, chat.name);
                });
                li.appendChild(deleteBtn);
            }

            // Обработчик выбора чата
            li.addEventListener('click', () => {
                handleSelectChat(chat.session_id);
            });

            chatList.appendChild(li);
        });
    }

     /** Выделяет активный чат в списке */
     function highlightActiveChat(sessionId) {
        const items = chatList.querySelectorAll('li');
        items.forEach(item => {
            if (item.dataset.sessionId === sessionId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    /** Загружает историю выбранного чата */
    async function loadChatHistory(sessionId) {
         if (isLoadingHistory) return; // Предотвращаем двойную загрузку
         if (!sessionId) {
             clearMessages();
             renderMessage('system', 'Чат не выбран.');
             currentChatNameHeader.textContent = 'Нет чата';
             clearHistoryBtn.style.display = 'none';
             return;
         }

        isLoadingHistory = true;
        clearMessages();
        renderMessage('system', 'Загрузка истории...'); // Индикатор загрузки

        try {
            const history = await fetchAPI(`/api/history/${sessionId}`);
            clearMessages(); // Очищаем сообщение "Загрузка..."
            if (history && history.length > 0) {
                history.forEach(msg => renderMessage(msg.role, msg.message, msg.created_at));
            } else {
                renderMessage('system', 'История чата пуста.');
            }
        } catch (error) {
            console.error(`Ошибка загрузки истории для чата ${sessionId}:`, error);
            clearMessages();
            renderMessage('error', `Не удалось загрузить историю чата: ${error.message}`);
        } finally {
            isLoadingHistory = false;
             scrollToBottom(); // Прокрутка вниз после загрузки
        }
    }

    // --- Обработчики Событий ---

    /** Обработчик клика на кнопку "Создать чат" */
    async function handleCreateChat() {
        const chatName = prompt("Введите имя для нового чата (необязательно):", "");
        // Пользователь нажал "Отмена"
        if (chatName === null) return;

        try {
            // Используем FormData для отправки имени
            const formData = new FormData();
            if (chatName.trim()) {
                formData.append('name', chatName.trim());
            }

            // Показываем индикатор ожидания
            createChatBtn.disabled = true;
            createChatBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            const newChat = await fetchAPI('/api/chats', {
                method: 'POST',
                body: formData // Отправляем FormData
            });

            console.log('Создан новый чат:', newChat);
            await loadChats(); // Перезагружаем список чатов
            // Автоматически выбираем только что созданный чат
            if (newChat && newChat.session_id) {
                 handleSelectChat(newChat.session_id);
            }

        } catch (error) {
            console.error('Ошибка создания чата:', error);
            alert(`Не удалось создать чат: ${error.message}`);
        } finally {
             // Восстанавливаем кнопку
             createChatBtn.disabled = false;
             createChatBtn.innerHTML = '<i class="fas fa-plus"></i>';
        }
    }

    /** Обработчик выбора чата */
    function handleSelectChat(sessionId) {
        if (currentChatId === sessionId) return; // Не выбираем уже выбранный чат

        currentChatId = sessionId;
        console.log(`Выбран чат: ${currentChatId}`);

        // Обновляем UI
        highlightActiveChat(sessionId);

        // Обновляем заголовок чата
        const selectedLi = chatList.querySelector(`li[data-session-id="${sessionId}"] .chat-name`);
        currentChatNameHeader.textContent = selectedLi ? selectedLi.textContent : 'Загрузка...';

        // Показываем или скрываем кнопку очистки
        clearHistoryBtn.style.display = sessionId ? 'inline-block' : 'none';

        // Загружаем историю для выбранного чата
        loadChatHistory(sessionId);

         // Сбрасываем состояние ввода/загрузки
         showTypingIndicator(false);
         showUploadIndicator(false);
         messageInput.disabled = !sessionId; // Блокируем ввод, если чат не выбран
         sendBtn.disabled = !sessionId;
         uploadBtn.disabled = !sessionId;
    }


    /** Обработчик клика на кнопку "Удалить чат" */
    async function handleDeleteChat(sessionId, chatName) {
        if (!confirm(`Вы уверены, что хотите удалить чат "${chatName}"?\nЭто действие необратимо и удалит всю историю и загруженные файлы этого чата.`)) {
            return;
        }

        try {
             // Показываем индикатор удаления (можно добавить спиннер на сам элемент списка)
             const listItem = chatList.querySelector(`li[data-session-id="${sessionId}"]`);
             if(listItem) {
                 const deleteBtn = listItem.querySelector('.delete-chat-btn');
                 if (deleteBtn) deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
             }


            await fetchAPI(`/api/chats/${sessionId}`, { method: 'DELETE' });
            console.log(`Чат ${sessionId} удален.`);

            // Если удалили текущий активный чат, переключаемся на общий
            if (currentChatId === sessionId) {
                currentChatId = null; // Сбрасываем текущий ID
            }
             await loadChats(); // Перезагружаем список и автоматически выбираем общий/первый

        } catch (error) {
            console.error(`Ошибка удаления чата ${sessionId}:`, error);
            alert(`Не удалось удалить чат: ${error.message}`);
             // Восстанавливаем кнопку удаления, если элемент еще существует
             const listItem = chatList.querySelector(`li[data-session-id="${sessionId}"]`);
             if(listItem) {
                 const deleteBtn = listItem.querySelector('.delete-chat-btn');
                 if (deleteBtn) deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
             }
        }
    }

    /** Обработчик отправки сообщения */
    async function handleSendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText || !currentChatId || isSendingMessage) {
            return;
        }

        isSendingMessage = true;
        showTypingIndicator(true); // Показываем индикатор "AI думает..."

        // Отображаем сообщение пользователя сразу
        renderMessage('user', messageText, new Date().toISOString()); // Используем текущее время
        messageInput.value = ''; // Очищаем поле ввода
        adjustTextareaHeight(); // Корректируем высоту поля ввода

        try {
            const response = await fetchAPI('/api/chat', {
                method: 'POST',
                body: JSON.stringify({
                    session_id: currentChatId,
                    message: messageText
                    // Сюда можно добавить filters, если они будут на фронте
                })
            });

            // Отображаем ответ AI
            renderMessage('ai', response.response, new Date().toISOString());
             // Можно отобразить метаданные ответа
             console.log("AI Response Metadata:", response.metadata);


        } catch (error) {
            console.error('Ошибка отправки сообщения:', error);
            renderMessage('error', `Не удалось отправить сообщение: ${error.message}`);
        } finally {
            isSendingMessage = false;
            showTypingIndicator(false); // Скрываем индикатор
             messageInput.focus(); // Возвращаем фокус в поле ввода
        }
    }

     /** Обработчик клика на кнопку загрузки файлов */
     function handleUploadButtonClick() {
        if (!currentChatId || isUploading) return;
        fileInput.click(); // Открываем стандартное окно выбора файлов
    }

    /** Обработчик изменения в поле выбора файлов */
    async function handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0 || !currentChatId || isUploading) {
            return;
        }

        showUploadIndicator(true, `Загрузка ${files.length} файлов...`);

        const formData = new FormData();
        formData.append('session_id', currentChatId);
        for (const file of files) {
            formData.append('files', file); // Используем 'files' как имя поля для списка файлов
        }

        try {
            const response = await fetchAPI('/api/upload', {
                method: 'POST',
                body: formData
                // Заголовки Content-Type для FormData устанавливаются браузером автоматически
            });

            console.log('Результат загрузки файлов:', response);

            // Отображаем результат загрузки в чате
            let successCount = 0;
            let errorCount = 0;
            let skippedCount = 0;
            let resultsHtml = '<b>Результаты загрузки:</b><ul>';

            if (response && response.results) {
                 response.results.forEach(res => {
                     resultsHtml += `<li>${res.filename}: `;
                     if (res.status === 'success') {
                         resultsHtml += `<span style="color: var(--success-color);">Успех</span>`;
                         if(res.details) resultsHtml += ` (Чанков: ${res.details.chunks_indexed}, Категория: ${res.details.detected_category || '?'})`;
                         successCount++;
                     } else if (res.status === 'skipped') {
                         resultsHtml += `<span style="color: var(--text-secondary);">Пропущено</span>`;
                          if(res.details) resultsHtml += ` (${res.details.message})`;
                         skippedCount++;
                     } else if (res.status === 'warning') {
                         resultsHtml += `<span style="color: orange;">Предупреждение</span> (${res.error || 'Неизвестная проблема'})`;
                         // Не увеличиваем errorCount
                    } else {
                         resultsHtml += `<span style="color: var(--error-color);">Ошибка</span> (${res.error || 'Неизвестная ошибка'})`;
                         errorCount++;
                     }
                     resultsHtml += '</li>';
                 });
            } else {
                resultsHtml += '<li>Нет детальной информации о результатах.</li>';
            }
            resultsHtml += '</ul>';

             renderMessage('system', resultsHtml); // Отображаем сводку как системное сообщение

        } catch (error) {
            console.error('Ошибка загрузки файлов:', error);
            renderMessage('error', `Не удалось загрузить файлы: ${error.message}`);
        } finally {
            showUploadIndicator(false);
             // Сбрасываем значение input[type=file], чтобы можно было выбрать те же файлы снова
             event.target.value = null;
        }
    }

    /** Обработчик клика на кнопку очистки истории */
    async function handleClearHistory() {
         if (!currentChatId || isLoadingHistory || isSendingMessage) return;

         if (!confirm(`Вы уверены, что хотите очистить всю историю сообщений в чате "${currentChatNameHeader.textContent}"? \nЭто действие необратимо.`)) {
             return;
         }

         clearHistoryBtn.disabled = true;
         clearHistoryBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

         try {
            await fetchAPI(`/api/chat/${currentChatId}`, { method: 'DELETE' });
            clearMessages(); // Очищаем сообщения на фронтенде
            renderMessage('system', 'История чата очищена.');
             console.log(`История чата ${currentChatId} очищена.`);
             // Память на сервере тоже очистится

         } catch(error) {
            console.error(`Ошибка очистки истории чата ${currentChatId}:`, error);
             renderMessage('error', `Не удалось очистить историю: ${error.message}`);
         } finally {
            clearHistoryBtn.disabled = false;
            clearHistoryBtn.innerHTML = '<i class="fas fa-eraser"></i>';
         }
    }


    // --- Назначение Обработчиков ---
    createChatBtn.addEventListener('click', handleCreateChat);
    sendBtn.addEventListener('click', handleSendMessage);
    uploadBtn.addEventListener('click', handleUploadButtonClick);
    fileInput.addEventListener('change', handleFileUpload);
    clearHistoryBtn.addEventListener('click', handleClearHistory);

    // Отправка по Enter в поле ввода (Shift+Enter для новой строки)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Предотвращаем перенос строки
            handleSendMessage();
        }
    });

     // Динамическое изменение высоты textarea
     messageInput.addEventListener('input', adjustTextareaHeight);


}); // Конец DOMContentLoaded