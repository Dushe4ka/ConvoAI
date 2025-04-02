const loginForm = document.getElementById('login-form');
const logoutButton = document.getElementById('logout-button');
const loginErrorElement = document.getElementById('login-error');

function saveToken(token) {
    localStorage.setItem('authToken', token);
}

function clearToken() {
    localStorage.removeItem('authToken');
}

function isLoggedIn() {
    return !!getAuthToken();
}

async function loginUser(username, password) {
    try {
        const data = await api.post('/auth/login', { username, password });
        if (data.access_token) {
            saveToken(data.access_token);
            loginErrorElement.style.display = 'none';
            return true;
        }
         displayLoginError('Не удалось получить токен доступа.');
         return false;
    } catch (error) {
         console.error("Login failed:", error);
         displayLoginError(error.message || 'Ошибка входа. Проверьте логин и пароль.');
         return false;
    }
}

function logoutUser() {
    clearToken();
    // Дополнительно можно вызвать API endpoint для инвалидации токена на сервере, если он есть
    location.reload(); // Перезагружаем страницу для сброса состояния
}

function displayLoginError(message) {
     loginErrorElement.textContent = message;
     loginErrorElement.style.display = 'block';
 }

 function hideLoginError() {
     loginErrorElement.style.display = 'none';
 }

// --- Инициализация и обработчики ---
if (loginForm) {
    loginForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        hideLoginError();
        const username = loginForm.username.value;
        const password = loginForm.password.value;
        const success = await loginUser(username, password);
        if (success) {
            // В main.js будет логика переключения на основной вид
             await initializeApp(); // Вызовем инициализацию из main.js
        }
    });
}

if (logoutButton) {
    logoutButton.addEventListener('click', (event) => {
        event.preventDefault();
        logoutUser();
    });
}