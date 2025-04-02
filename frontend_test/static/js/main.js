// Получаем ссылки на все основные "виды"
const views = {
    login: document.getElementById('login-view'),
    app: document.getElementById('app-view'),
    profile: document.getElementById('profile-view'),
    admin: document.getElementById('admin-view')
};
const appHeader = document.getElementById('app-header');
const profileLink = document.getElementById('profile-link');
const adminLink = document.getElementById('admin-link');
const uploadForm = document.getElementById('upload-form');
const uploadStatus = document.getElementById('upload-status');
const uploadProgress = document.getElementById('upload-progress');
const uploadErrorElement = document.getElementById('upload-error');


// Функция для отображения только одного вида и скрытия остальных
function showView(viewId) {
    // Сначала скрыть все виды
    Object.values(views).forEach(view => view.style.display = 'none');
    // Показать нужный вид
    if (views[viewId]) {
        views[viewId].style.display = 'block';
    } else {
        console.error(`View with id ${viewId} not found.`);
    }

    // Управление видимостью хедера
    if (viewId === 'login') {
        appHeader.style.display = 'none';
    } else {
        appHeader.style.display = 'block';
    }
}

// Функция для отображения статуса загрузки
function displayUploadStatus(message, isError = false) {
    uploadStatus.textContent = message;
    uploadStatus.style.color = isError ? 'var(--pico-color-red-500)' : 'inherit';
     uploadErrorElement.style.display = isError ? 'block' : 'none';
     if (isError) uploadErrorElement.textContent = message;
     else uploadErrorElement.style.display = 'none';
 }

 function hideUploadMessages() {
    uploadStatus.textContent = '';
     uploadErrorElement.style.display = 'none';
    uploadProgress.style.display = 'none';
 }

async function handleFileUpload(event) {
     event.preventDefault();
     hideUploadMessages();
     const fileInput = document.getElementById('file');
     const file = fileInput.files[0];

     if (!file) {
         displayUploadStatus('Пожалуйста, выберите файл.', true);
         return;
     }

     const formData = new FormData();
     formData.append('file', file); // Имя 'file' должно совпадать с ожидаемым на бэкенде

     uploadProgress.style.display = 'block';
     uploadProgress.value = 0;
     displayUploadStatus('Загрузка...');

     try {
         // Используем api.upload, который не устанавливает Content-Type вручную
          const response = await api.upload('/upload', formData, {
             // Можно добавить обработчик прогресса, если нужно, но fetch API не поддерживает его из коробки
             // Для прогресса обычно используют XMLHttpRequest или библиотеки типа Axios
         });

         uploadProgress.value = 100; // Показываем завершение
         displayUploadStatus(`Файл "${response.filename}" успешно загружен. Категория: ${response.category}.`);
         fileInput.value = ''; // Сбросить поле выбора файла
         setTimeout(hideUploadMessages, 5000); // Скрыть сообщение через 5 секунд

     } catch (error) {
         console.error('File upload failed:', error);
         displayUploadStatus(`Ошибка загрузки: ${error.message}`, true);
         uploadProgress.style.display = 'none';
     }
 }


// --- Инициализация приложения ---
async function initializeApp() {
    if (isLoggedIn()) {
         // Пользователь вошел в систему
         showView('app'); // Показываем основной интерфейс (чат и загрузку)
         initializeChat(); // Загружаем историю чата или начинаем новую сессию

         // Проверяем права администратора, чтобы показать/скрыть ссылку
        try {
            const profile = await api.get('/profile'); // Заодно проверим токен
            if (profile.is_admin) {
                adminLink.style.display = 'block';
            } else {
                adminLink.style.display = 'none';
            }
        } catch (error) {
             // Если не удалось получить профиль (например, токен истек), выходим
            console.error("Token might be invalid or expired.");
            logoutUser(); // Вызовет перезагрузку и покажет логин
            return; // Прерываем инициализацию
         }


    } else {
        // Пользователь не вошел
        showView('login');
        adminLink.style.display = 'none'; // Скрыть админ-ссылку на всякий случай
    }
}

// --- Обработчики навигации ---
if (profileLink) {
    profileLink.addEventListener('click', async (event) => {
        event.preventDefault();
        const loaded = await loadProfileData(); // Загружаем данные перед показом
         if (loaded) {
            showView('profile');
         } else {
             // Если не удалось загрузить профиль (токен невалиден?), остаемся где были или выходим
         }
    });
}

if (adminLink) {
    adminLink.addEventListener('click', async (event) => {
        event.preventDefault();
        showView('admin');
        await initializeAdminPanel(); // Загружаем данные для админки
    });
}

// Обработчик формы загрузки
 if (uploadForm) {
     uploadForm.addEventListener('submit', handleFileUpload);
 }


// --- Запуск при загрузке страницы ---
document.addEventListener('DOMContentLoaded', initializeApp);