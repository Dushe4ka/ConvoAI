const adminView = document.getElementById('admin-view');
const userListBody = document.getElementById('user-list-body');
const registerForm = document.getElementById('register-form');
const backToChatFromAdminButton = document.getElementById('back-to-chat-from-admin-button');
const adminErrorElement = document.getElementById('admin-error');
const registerMessage = document.getElementById('register-message');
const registerErrorElement = document.getElementById('register-error');

function displayAdminError(message) {
    adminErrorElement.textContent = message;
    adminErrorElement.style.display = 'block';
}

 function hideAdminError() {
     adminErrorElement.style.display = 'none';
 }

 function displayRegisterMessage(message) {
    registerMessage.textContent = message;
    registerMessage.style.display = 'block';
    registerErrorElement.style.display = 'none'; // Скрыть ошибку
    setTimeout(() => registerMessage.style.display = 'none', 3000); // Скрыть через 3 сек
}

function displayRegisterError(message) {
    registerErrorElement.textContent = message;
    registerErrorElement.style.display = 'block';
    registerMessage.style.display = 'none'; // Скрыть успех
}

function hideRegisterMessages() {
    registerMessage.style.display = 'none';
    registerErrorElement.style.display = 'none';
}

async function loadUsers() {
    hideAdminError();
    try {
        const users = await api.get('/admin/users');
        userListBody.innerHTML = ''; // Очистить таблицу
        users.forEach(user => {
            const row = userListBody.insertRow();
            row.insertCell().textContent = user.id;
            row.insertCell().textContent = user.username;
            row.insertCell().textContent = user.full_name || '-';
            row.insertCell().textContent = user.is_admin ? 'Да' : 'Нет';

            const actionsCell = row.insertCell();
            // Не даем удалить самого себя (если логика админа это позволяет)
            // Лучше проверять на стороне бэкенда, но можно и тут добавить проверку
            // const currentUser = await api.get('/profile'); // Нужно будет получить текущего юзера
            // if (currentUser.username !== user.username) {
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Удалить';
                deleteButton.classList.add('secondary', 'outline'); // Стили Pico
                deleteButton.onclick = () => deleteUser(user.id, user.username); // Передаем ID и имя для подтверждения
                actionsCell.appendChild(deleteButton);
            // }
        });
    } catch (error) {
        console.error('Failed to load users:', error);
         displayAdminError('Не удалось загрузить список пользователей. У вас есть права администратора?');
        // Возможно, нужно скрыть админ-панель
        // showView('app-view');
    }
}

async function deleteUser(userId, username) {
    if (confirm(`Вы уверены, что хотите удалить пользователя "${username}" (ID: ${userId})?`)) {
         hideAdminError();
        try {
            await api.delete(`/admin/users/${userId}`);
            // alert('Пользователь удален.'); // Простое уведомление
            loadUsers(); // Обновить список после удаления
        } catch (error) {
            console.error('Failed to delete user:', error);
            displayAdminError(error.message || `Не удалось удалить пользователя ${username}.`);
        }
    }
}

 async function registerUser(event) {
    event.preventDefault();
    hideRegisterMessages();
    const formData = new FormData(registerForm);
    const userData = {
        username: formData.get('username'),
        password: formData.get('password'),
        full_name: formData.get('full_name') || null,
        birth_date: formData.get('birth_date') || null,
        is_admin: formData.get('is_admin') === 'on' // Checkbox value is 'on' if checked
    };

    try {
        const response = await api.post('/auth/register', userData);
        displayRegisterMessage(response.message || "Пользователь успешно зарегистрирован!");
        registerForm.reset(); // Очистить форму
        loadUsers(); // Обновить список пользователей
    } catch (error) {
        console.error("Registration failed:", error);
        displayRegisterError(error.message || "Ошибка регистрации пользователя.");
    }
}

// --- Инициализация и обработчики ---
if (registerForm) {
    registerForm.addEventListener('submit', registerUser);
}

if (backToChatFromAdminButton) {
    backToChatFromAdminButton.addEventListener('click', () => {
         // Функция showView должна быть определена в main.js
         showView('app-view');
     });
}

// Функция для инициализации админ-панели (вызывается из main.js)
async function initializeAdminPanel() {
    await loadUsers();
}