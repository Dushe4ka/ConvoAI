// script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Элементы ---
    const chatList = document.getElementById('chat-list');
    const createChatBtn = document.getElementById('create-chat-btn');
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn'); // Кнопка-скрепка
    const fileInput = document.getElementById('file-input');  // Единственный инпут
    const currentChatNameHeader = document.getElementById('current-chat-name');
    const typingIndicator = document.getElementById('typing-indicator');
    const uploadIndicator = document.getElementById('upload-indicator');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // --- Состояние приложения ---
    let currentChatId = null;
    let isLoadingHistory = false;
    let isSendingMessage = false;
    let isUploading = false;
    let generalChatId = null;

     // --- Константы для типов файлов ---
    const ALLOWED_TEXT_EXTENSIONS = ['.pdf', '.docx', '.txt'];
    const ALLOWED_TEXT_MIMES = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
        // Добавьте другие MIME типы для txt если нужно
    ];
    const ALLOWED_IMAGE_MIMES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
    const MAX_IMAGE_SIZE = 15 * 1024 * 1024; // 15MB
    // Макс. размер для текстовых файлов (из config.py, если нужно будет проверять)
    // const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB

    // --- Инициализация ---
    loadChats();
    adjustTextareaHeight();

    // --- Вспомогательные функции ---

    /** Обертка для fetch API */
    async function fetchAPI(url, options = {}) {
        options.headers = {
            'Accept': 'application/json',
            ...options.headers
        };
        // FormData Content-Type устанавливается браузером автоматически
        if (!(options.body instanceof FormData)) {
            options.headers['Content-Type'] = 'application/json';
        }
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                let errorData;
                try { errorData = await response.json(); } catch (e) { errorData = { detail: response.statusText || `Ошибка ${response.status}` }; }
                const errorMessage = errorData.detail || `Ошибка сервера: ${response.status}`;
                console.error(`API Error ${response.status}: ${errorMessage}`, errorData);
                throw new Error(errorMessage);
            }
            // Обработка пустого ответа или ответа без контента
            if (response.status === 204 || response.headers.get('content-length') === '0') {
                 return null; // Или можно вернуть { success: true }
            }
            // Пытаемся распарсить JSON
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Fetch API error:', error);
            throw error; // Передаем ошибку дальше
        }
    }

    /** Отображает сообщение в окне чата */
    function renderMessage(role, messageContent, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role);

        // Используем innerHTML только для доверенного контента или после очистки
        if (role === 'ai') {
            try {
                const dirtyHtml = marked.parse(messageContent);
                // Вставляем очищенный HTML
                messageDiv.innerHTML = DOMPurify.sanitize(dirtyHtml, {USE_PROFILES: {html: true}}); // Разрешаем базовые HTML теги
            } catch (e) {
                console.error("Ошибка обработки Markdown:", e);
                messageDiv.textContent = messageContent; // Безопасный вариант
            }
        } else if (role === 'system') {
             // Для системных сообщений, где мы используем <b> и <i>, тоже нужна очистка
             messageDiv.innerHTML = DOMPurify.sanitize(messageContent, {USE_PROFILES: {html: true}});
        }
        else {
            messageDiv.textContent = messageContent; // Для user и error
        }

        if (timestamp) {
            const timeSpan = document.createElement('span');
            timeSpan.classList.add('timestamp');
            try {
                timeSpan.textContent = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } catch { /* Игнорируем ошибки */ }
            // Добавляем timestamp после контента
             messageDiv.appendChild(timeSpan);
        }

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    /** Прокручивает область сообщений вниз */
    function scrollToBottom() {
        // Небольшая задержка может помочь, если контент рендерится асинхронно
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);
    }

    /** Очищает область сообщений */
    function clearMessages() {
        chatMessages.innerHTML = '';
    }

    /** Управляет видимостью индикатора печати AI и блокировкой ввода */
    function showTypingIndicator(show) {
        isSendingMessage = show; // Обновляем состояние отправки
        typingIndicator.style.display = show ? 'block' : 'none';
        // Блокируем/разблокируем элементы в зависимости от обоих состояний (отправка И загрузка)
        const disableInputs = isSendingMessage || isUploading;
        messageInput.disabled = disableInputs;
        sendBtn.disabled = disableInputs;
        uploadBtn.disabled = disableInputs;
        clearHistoryBtn.disabled = disableInputs; // Блокируем и очистку истории
    }

     /** Управляет видимостью индикатора загрузки и блокировкой ввода */
     function showUploadIndicator(show, message = 'Загрузка...') {
        isUploading = show; // Обновляем состояние загрузки
        uploadIndicator.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${message}`;
        uploadIndicator.style.display = show ? 'block' : 'none';
        // Блокируем/разблокируем элементы в зависимости от обоих состояний (отправка И загрузка)
        const disableInputs = isSendingMessage || isUploading;
        messageInput.disabled = disableInputs;
        sendBtn.disabled = disableInputs;
        uploadBtn.disabled = disableInputs;
        clearHistoryBtn.disabled = disableInputs; // Блокируем и очистку истории
    }

    /** Автоматически изменяет высоту textarea */
    function adjustTextareaHeight() {
        messageInput.style.height = 'auto';
        let scrollHeight = messageInput.scrollHeight;
        let style = window.getComputedStyle(messageInput);
        let borderTop = parseInt(style.borderTopWidth, 10);
        let borderBottom = parseInt(style.borderBottomWidth, 10);
        scrollHeight += borderTop + borderBottom;
        const maxHeight = 100;
        if (scrollHeight > maxHeight) {
            messageInput.style.height = `${maxHeight}px`;
            messageInput.style.overflowY = 'auto';
        } else {
            messageInput.style.height = `${scrollHeight}px`;
            messageInput.style.overflowY = 'hidden';
        }
    }

    // --- Функции Загрузки и Отображения Данных ---

    /** Загружает и отображает список чатов */
    async function loadChats() {
        chatList.innerHTML = '<li class="loading-chats">Загрузка чатов...</li>';
        try {
            const data = await fetchAPI('/api/chats');
            renderChatList(data.chats);
            const general = data.chats.find(chat => chat.is_general);
            if (general) generalChatId = general.session_id;
            // Автоматический выбор чата при загрузке или если текущий удален
            const currentChatExists = data.chats.some(chat => chat.session_id === currentChatId);
            if (!currentChatId || !currentChatExists) {
                if (generalChatId) handleSelectChat(generalChatId);
                else if (data.chats.length > 0) handleSelectChat(data.chats[0].session_id);
                else handleSelectChat(null); // Нет чатов
            } else {
                highlightActiveChat(currentChatId); // Обновить выделение
                const currentChatData = data.chats.find(chat => chat.session_id === currentChatId);
                if (currentChatData) { // Обновить имя в шапке
                    currentChatNameHeader.textContent = currentChatData.name;
                    clearHistoryBtn.style.display = currentChatData.is_general ? 'none' : 'inline-block';
                }
            }
        } catch (error) {
            console.error('Ошибка загрузки чатов:', error);
            chatList.innerHTML = '<li class="loading-chats error">Не удалось загрузить чаты</li>';
            clearMessages();
            renderMessage('error', `Ошибка загрузки списка чатов: ${error.message}`);
        }
    }

    /** Отображает список чатов в DOM */
    function renderChatList(chats) {
        chatList.innerHTML = '';
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
            if (chat.is_general) {
                li.classList.add('general-chat');
                const lockIcon = document.createElement('i');
                lockIcon.classList.add('fas', 'fa-lock', 'fa-xs');
                lockIcon.style.marginLeft = '5px'; lockIcon.style.opacity = '0.6';
                nameSpan.appendChild(lockIcon);
            }
            if (chat.can_delete) {
                const deleteBtn = document.createElement('button');
                deleteBtn.classList.add('delete-chat-btn');
                deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
                deleteBtn.title = 'Удалить этот чат';
                deleteBtn.addEventListener('click', (e) => { e.stopPropagation(); handleDeleteChat(chat.session_id, chat.name); });
                li.appendChild(deleteBtn);
            }
            li.addEventListener('click', () => { handleSelectChat(chat.session_id); });
            chatList.appendChild(li);
        });
    }

     /** Выделяет активный чат в списке */
     function highlightActiveChat(sessionId) {
        const items = chatList.querySelectorAll('li');
        items.forEach(item => {
            item.classList.toggle('active', item.dataset.sessionId === sessionId);
        });
    }

    /** Загружает историю выбранного чата */
    async function loadChatHistory(sessionId) {
        if (isLoadingHistory) return;
        if (!sessionId) {
            clearMessages(); renderMessage('system', 'Чат не выбран.');
            currentChatNameHeader.textContent = 'Нет чата';
            clearHistoryBtn.style.display = 'none'; return;
        }
        isLoadingHistory = true; clearMessages(); renderMessage('system', 'Загрузка истории...');
        try {
            const history = await fetchAPI(`/api/history/${sessionId}`);
            clearMessages();
            if (history && history.length > 0) {
                history.forEach(msg => renderMessage(msg.role, msg.message, msg.created_at));
            } else {
                renderMessage('system', 'История чата пуста.');
            }
        } catch (error) {
            console.error(`Ошибка загрузки истории для чата ${sessionId}:`, error);
            clearMessages(); renderMessage('error', `Не удалось загрузить историю чата: ${error.message}`);
        } finally {
            isLoadingHistory = false; scrollToBottom();
        }
    }

    // --- Обработчики Событий ---

    /** Обработчик клика на кнопку "Создать чат" */
    async function handleCreateChat() {
        if (isUploading || isSendingMessage) return; // Не создавать во время других операций
        const chatName = prompt("Введите имя для нового чата (необязательно):", "");
        if (chatName === null) return;
        const buttonIcon = createChatBtn.innerHTML; // Сохраняем иконку
        try {
            const formData = new FormData();
            if (chatName.trim()) formData.append('name', chatName.trim());
            createChatBtn.disabled = true;
            createChatBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; // Показываем спиннер
            const newChat = await fetchAPI('/api/chats', { method: 'POST', body: formData });
            console.log('Создан новый чат:', newChat);
            await loadChats();
            if (newChat && newChat.session_id) handleSelectChat(newChat.session_id); // Выбираем новый чат
        } catch (error) {
            console.error('Ошибка создания чата:', error);
            alert(`Не удалось создать чат: ${error.message}`);
        } finally {
            createChatBtn.disabled = false;
            createChatBtn.innerHTML = buttonIcon; // Восстанавливаем иконку
        }
    }

    /** Обработчик выбора чата */
    function handleSelectChat(sessionId) {
        if (currentChatId === sessionId || isLoadingHistory) return; // Не выбираем тот же или во время загрузки
        currentChatId = sessionId;
        console.log(`Выбран чат: ${currentChatId}`);
        highlightActiveChat(sessionId);
        const selectedLi = chatList.querySelector(`li[data-session-id="${sessionId}"]`);
        currentChatNameHeader.textContent = selectedLi ? selectedLi.querySelector('.chat-name').textContent : 'Загрузка...';
        const isGeneral = selectedLi ? selectedLi.classList.contains('general-chat') : true;
        clearHistoryBtn.style.display = sessionId && !isGeneral ? 'inline-block' : 'none';
        loadChatHistory(sessionId);
        // Сбрасываем состояния и доступность инпутов
        showTypingIndicator(false);
        showUploadIndicator(false);
        const disableInputs = !sessionId; // Отключить если чат не выбран
        messageInput.disabled = disableInputs;
        sendBtn.disabled = disableInputs;
        uploadBtn.disabled = disableInputs;
        clearHistoryBtn.disabled = disableInputs || isGeneral; // Отключить очистку если не выбран или общий
    }

    /** Обработчик клика на кнопку "Удалить чат" */
    async function handleDeleteChat(sessionId, chatName) {
        if (isUploading || isSendingMessage) return; // Не удалять во время других операций
        if (!confirm(`Вы уверены, что хотите удалить чат "${chatName}"?\nЭто действие необратимо и удалит всю историю и связанные данные этого чата.`)) {
            return;
        }
        const listItem = chatList.querySelector(`li[data-session-id="${sessionId}"]`);
        const deleteBtn = listItem ? listItem.querySelector('.delete-chat-btn') : null;
        const originalIcon = deleteBtn ? deleteBtn.innerHTML : '';
        try {
            if (deleteBtn) {
                 deleteBtn.disabled = true; // Блокируем кнопку на время удаления
                 deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            }
            await fetchAPI(`/api/chats/${sessionId}`, { method: 'DELETE' });
            console.log(`Чат ${sessionId} удален.`);
            // Если удалили текущий, сбрасываем ID и перезагружаем список
            // Перезагрузка автоматически выберет общий или первый чат
            if (currentChatId === sessionId) currentChatId = null;
            await loadChats();
        } catch (error) {
            console.error(`Ошибка удаления чата ${sessionId}:`, error);
            alert(`Не удалось удалить чат: ${error.message}`);
             if (deleteBtn) { // Восстанавливаем кнопку при ошибке
                 deleteBtn.disabled = false;
                 deleteBtn.innerHTML = originalIcon;
            }
        }
    }

    /** Обработчик отправки сообщения */
    async function handleSendMessage() {
        const messageText = messageInput.value.trim();
        // Дополнительная проверка на активные операции
        if (!messageText || !currentChatId || isSendingMessage || isUploading) {
            return;
        }
        showTypingIndicator(true); // Блокирует инпуты
        renderMessage('user', messageText, new Date().toISOString());
        messageInput.value = '';
        adjustTextareaHeight();
        try {
            const response = await fetchAPI('/api/chat', {
                method: 'POST',
                body: JSON.stringify({ session_id: currentChatId, message: messageText })
            });
            renderMessage('ai', response.response, new Date().toISOString());
            console.log("AI Response Metadata:", response.metadata);
        } catch (error) {
            console.error('Ошибка отправки сообщения:', error);
            renderMessage('error', `Не удалось отправить сообщение: ${error.message}`);
        } finally {
            showTypingIndicator(false); // Разблокирует инпуты
             // Возвращаем фокус только если не идет загрузка файла
             if (!isUploading) {
                 messageInput.focus();
             }
        }
    }

    /** Обработчик клика на кнопку-скрепку (#upload-btn) */
     function handleUploadButtonClick() {
        // Не позволяем инициировать загрузку, если уже что-то грузится или отправляется
        if (!currentChatId || isUploading || isSendingMessage) return;
        fileInput.click(); // Триггерим единственный input
    }


    /** ПЕРЕПИСАННЫЙ: Обработчик изменения в ЕДИНСТВЕННОМ поле выбора файлов */
    async function handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0 || !currentChatId || isUploading || isSendingMessage) {
            event.target.value = null; return;
        }

        showUploadIndicator(true, `Обработка ${files.length} файлов...`);

        const textFiles = [];
        const imageFiles = [];
        const skippedFilesInfo = []; // Собираем информацию о пропущенных файлах

        // 1. Разделяем файлы на группы и проверяем
        for (const file of files) {
            const fileName = file.name;
            const fileExt = '.' + fileName.split('.').pop().toLowerCase();
            const mimeType = file.type;
            let isText = ALLOWED_TEXT_EXTENSIONS.includes(fileExt) || ALLOWED_TEXT_MIMES.includes(mimeType);
            let isImage = ALLOWED_IMAGE_MIMES.includes(mimeType);

            if (isImage) {
                if (file.size > MAX_IMAGE_SIZE) {
                    skippedFilesInfo.push({ filename: fileName, status: 'skipped', error: 'Слишком большой размер (>15MB)' });
                } else {
                    imageFiles.push(file);
                }
            } else if (isText) {
                 // Добавить проверку MAX_FILE_SIZE если нужно
                 textFiles.push(file);
            } else {
                skippedFilesInfo.push({ filename: fileName, status: 'skipped', error: `Неподдерживаемый тип (${mimeType || fileExt})` });
            }
        }

        console.log(`К загрузке: ${textFiles.length} текст., ${imageFiles.length} изображений.`);
        let overallResults = [...skippedFilesInfo]; // Начинаем с уже пропущенных

        // --- Функция для отображения результатов одного файла ---
        function addResultToList(result) {
             overallResults.push(result);
             // Можно добавить логику немедленного отображения статуса файла, если нужно
        }

        try {
            // 2. Загрузка текстовых файлов (если есть)
            if (textFiles.length > 0) {
                showUploadIndicator(true, `Загрузка ${textFiles.length} текст. файлов...`);
                const textFormData = new FormData();
                textFormData.append('session_id', currentChatId);
                textFiles.forEach(file => textFormData.append('files', file));
                try {
                    const response = await fetchAPI('/api/upload', { method: 'POST', body: textFormData });
                    console.log('Результат загрузки текстовых файлов:', response);
                    if (response && response.results) {
                        // Добавляем результаты из ответа сервера
                        overallResults = overallResults.concat(response.results);
                    } else {
                        textFiles.forEach(f => addResultToList({ filename: f.name, status: 'error', error: 'Нет ответа от сервера (текст)'}));
                    }
                } catch (error) {
                    console.error('Ошибка загрузки текстовых файлов:', error);
                    textFiles.forEach(f => addResultToList({ filename: f.name, status: 'error', error: `Ошибка сети (текст): ${error.message}`}));
                    renderMessage('error', `Ошибка при загрузке текстовых файлов: ${error.message}`);
                }
            }

            // 3. Загрузка изображений (если есть) - ПОСЛЕДОВАТЕЛЬНО
            if (imageFiles.length > 0) {
                 showUploadIndicator(true, `Загрузка ${imageFiles.length} изображений (0/${imageFiles.length})...`);
                 for (let i = 0; i < imageFiles.length; i++) {
                     const imageFile = imageFiles[i];
                     // Проверяем, не отменил ли пользователь операцию (например, сменой чата)
                     if (isUploading === false) {
                          console.log("Загрузка изображений прервана.");
                          addResultToList({ filename: imageFile.name, status: 'skipped', error: 'Операция прервана' });
                          continue; // Пропускаем оставшиеся
                     }

                     showUploadIndicator(true, `Загрузка ${imageFiles.length} изображений (${i + 1}/${imageFiles.length}): ${imageFile.name}...`);
                     const imageFormData = new FormData();
                     imageFormData.append('session_id', currentChatId);
                     imageFormData.append('image_file', imageFile);
                     try {
                         const response = await fetchAPI('/api/upload/image', { method: 'POST', body: imageFormData });
                         console.log(`Результат загрузки изображения ${imageFile.name}:`, response);
                         addResultToList({
                             filename: response.filename || imageFile.name,
                             status: 'success',
                             details: { message: "Изображение загружено", description_preview: response.description_preview },
                             error: null
                         });
                     } catch (error) {
                         console.error(`Ошибка загрузки изображения ${imageFile.name}:`, error);
                         addResultToList({
                             filename: imageFile.name,
                             status: 'error',
                             error: `Ошибка: ${error.message}`
                         });
                         renderMessage('error', `Не удалось загрузить "${imageFile.name}": ${error.message}`);
                     }
                 }
            }

             // 4. Отображение итоговой сводки в чате
             if (overallResults.length > 0) {
                 // Сортируем результаты по имени файла для наглядности
                 overallResults.sort((a, b) => a.filename.localeCompare(b.filename));

                let resultsHtml = `<b>Результаты обработки ${files.length} выбранных файлов:</b><ul>`;
                overallResults.forEach(res => {
                     resultsHtml += `<li>${DOMPurify.sanitize(res.filename)}: `; // Очищаем имя файла
                     const statusText = res.status || 'unknown';
                     const errorText = res.error ? ` (${DOMPurify.sanitize(res.error)})` : ''; // Очищаем ошибку

                     switch(statusText) {
                         case 'success':
                             resultsHtml += `<span style="color: var(--success-color);">Успех</span>`;
                             if (res.details && res.details.message) resultsHtml += ` (${DOMPurify.sanitize(res.details.message)})`;
                             break;
                         case 'skipped':
                              resultsHtml += `<span style="color: var(--text-secondary);">Пропущено</span>${errorText}`;
                              break;
                         case 'warning':
                              resultsHtml += `<span style="color: orange;">Предупреждение</span>${errorText}`;
                              break;
                         default: // error или unknown
                              resultsHtml += `<span style="color: var(--error-color);">Ошибка</span>${errorText}`;
                              break;
                     }
                     resultsHtml += '</li>';
                 });
                 resultsHtml += '</ul>';
                 renderMessage('system', resultsHtml);
             } else if (files.length > 0){
                 // Если были выбраны файлы, но все были пропущены
                 renderMessage('system', 'Не найдено подходящих файлов для загрузки среди выбранных.');
             }

        } catch (error) {
             console.error('Общая ошибка обработки файлов:', error);
             renderMessage('error', `Произошла ошибка при обработке файлов: ${error.message}`);
        } finally {
            showUploadIndicator(false); // Скрываем индикатор после всех операций
            event.target.value = null; // ВСЕГДА сбрасываем input в конце
             // Возвращаем фокус в поле ввода, если не идет отправка сообщения
             if (!isSendingMessage) {
                  messageInput.focus();
             }
        }
    }


    /** Обработчик клика на кнопку очистки истории */
    async function handleClearHistory() {
         if (!currentChatId || isLoadingHistory || isSendingMessage || isUploading) return;
         const selectedLi = chatList.querySelector(`li[data-session-id="${currentChatId}"]`);
         const chatName = selectedLi ? selectedLi.querySelector('.chat-name').textContent : 'текущий чат';
         if (selectedLi && selectedLi.classList.contains('general-chat')) {
              renderMessage('system', "Историю общего чата очищать нельзя.");
              return;
         }
         if (!confirm(`Вы уверены, что хотите очистить всю историю сообщений в чате "${chatName}"? \nЭто действие необратимо.`)) {
             return;
         }
         const buttonIcon = clearHistoryBtn.innerHTML;
         clearHistoryBtn.disabled = true;
         clearHistoryBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
         try {
            await fetchAPI(`/api/chat/${currentChatId}`, { method: 'DELETE' });
            clearMessages();
            renderMessage('system', 'История чата очищена.');
            console.log(`История чата ${currentChatId} очищена.`);
         } catch(error) {
            console.error(`Ошибка очистки истории чата ${currentChatId}:`, error);
             renderMessage('error', `Не удалось очистить историю: ${error.message}`);
         } finally {
            clearHistoryBtn.disabled = false;
            clearHistoryBtn.innerHTML = buttonIcon;
         }
    }

    // --- Назначение Обработчиков ---
    createChatBtn.addEventListener('click', handleCreateChat);
    sendBtn.addEventListener('click', handleSendMessage);
    clearHistoryBtn.addEventListener('click', handleClearHistory);
    uploadBtn.addEventListener('click', handleUploadButtonClick); // Кнопка-скрепка
    fileInput.addEventListener('change', handleFileUpload);      // Единственный input

    // Обработчик для Enter
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); handleSendMessage();
        }
    });
    // Динамическая высота textarea
    messageInput.addEventListener('input', adjustTextareaHeight);

}); // Конец DOMContentLoaded