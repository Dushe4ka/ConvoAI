// frontend_test/static/js/api.js

const BASE_URL = '/api'; // Базовый URL вашего API

// Функция для получения токена аутентификации из localStorage
function getAuthToken() {
    return localStorage.getItem('authToken');
}

// --- Основная функция для выполнения запросов к API ---
// endpoint: Часть URL после /api (например, '/chat', '/auth/login')
// options: Объект настроек для fetch (method, headers, body и т.д.)
async function request(endpoint, options = {}) {
    // Корректно формируем полный URL с использованием шаблонной строки
    const url = `${BASE_URL}${endpoint}`;

    const token = getAuthToken();
    const headers = {
        // По умолчанию ожидаем JSON, если не указано иное
        'Content-Type': 'application/json',
        // Распространяем любые пользовательские заголовки из options.headers
        ...options.headers,
    };

    // Добавляем заголовок авторизации, если токен есть
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    // Удаляем 'Content-Type' для GET/DELETE запросов или если тело не передается,
    // чтобы браузер мог установить его сам, если нужно (особенно для FormData)
    if (!options.body && !(options.method === 'POST' || options.method === 'PUT' || options.method === 'PATCH')) {
         delete headers['Content-Type'];
    }

    try {
        const response = await fetch(url, {
            ...options, // Распространяем метод, тело и другие опции fetch
            headers,    // Передаем сформированные заголовки
        });

        // Проверяем, успешен ли ответ (статус 2xx)
        if (!response.ok) {
            // Пытаемся прочитать тело ответа как JSON для получения деталей ошибки
            let errorData = { detail: `HTTP error! status: ${response.status}` }; // Значение по умолчанию
            try {
                 // response.clone() нужен, если вы захотите прочитать тело еще раз позже
                errorData = await response.json();
            } catch (e) {
                // Если тело не JSON или пустое, используем текст статуса
                errorData = { detail: response.statusText || `HTTP error! status: ${response.status}` };
            }
            console.error("API Error Response:", errorData);
            // Создаем ошибку с деталями из ответа сервера или текстом статуса
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        // Обрабатываем ответы без содержимого (например, 204 No Content)
        if (response.status === 204 || response.headers.get('Content-Length') === '0') {
            return null; // Возвращаем null или можно { success: true }
        }

        // Для всех остальных успешных ответов парсим тело как JSON
        return await response.json();

    } catch (error) {
        // Ловим как ошибки сети (fetch не удался), так и ошибки из блока !response.ok
        console.error('Fetch API request error:', error);
        // Перебрасываем ошибку дальше, чтобы ее можно было обработать в вызывающем коде
        throw error;
    }
}

// --- Хелпер для загрузки файлов (использует FormData) ---
async function uploadFile(endpoint, formData, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const token = getAuthToken();
    const headers = {
        // ВАЖНО: НЕ устанавливаем 'Content-Type' вручную для FormData,
        // браузер сам установит его в 'multipart/form-data' с правильным boundary
        ...options.headers,
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    try {
       const response = await fetch(url, {
           method: 'POST', // Загрузка обычно POST
           body: formData, // Передаем объект FormData
           headers,
           ...options // Другие опции fetch, если нужны
       });

        if (!response.ok) {
           let errorData = { detail: `Upload HTTP error! status: ${response.status}` };
           try {
               errorData = await response.json();
           } catch (e) {
                errorData = { detail: response.statusText || `Upload HTTP error! status: ${response.status}` };
           }
            console.error("API Upload Error Response:", errorData);
           throw new Error(errorData.detail || `Upload HTTP error! status: ${response.status}`);
        }

        // Ответ при загрузке тоже может быть JSON
        return await response.json();

    } catch (error) {
        console.error('Fetch API upload error:', error);
        throw error;
    }
}


// --- Создаем и экспортируем объект 'api' ---
// Этот объект будет доступен в других скриптах, если api.js загружен первым
const api = {
    // Оборачиваем вызовы request в удобные методы
    get: (endpoint, options = {}) => request(endpoint, { ...options, method: 'GET' }),
    post: (endpoint, body, options = {}) => request(endpoint, { ...options, method: 'POST', body: JSON.stringify(body) }),
    put: (endpoint, body, options = {}) => request(endpoint, { ...options, method: 'PUT', body: JSON.stringify(body) }),
    delete: (endpoint, options = {}) => request(endpoint, { ...options, method: 'DELETE' }),
    // Отдельный метод для загрузки файлов
    upload: (endpoint, formData, options = {}) => uploadFile(endpoint, formData, options)
};

// Убедимся, что объект `api` создан и доступен
console.log("api.js loaded, 'api' object created:", api); // Для отладки в консоли браузера