const profileView = document.getElementById('profile-view');
const profileForm = document.getElementById('profile-form');
const profileUsername = document.getElementById('profile-username');
const profileFullName = document.getElementById('profile-full-name');
const profileBirthDate = document.getElementById('profile-birth-date');
const profilePassword = document.getElementById('profile-password');
const profileMessage = document.getElementById('profile-message');
const profileErrorElement = document.getElementById('profile-error');
const backToChatButton = document.getElementById('back-to-chat-button');

function displayProfileMessage(message) {
    profileMessage.textContent = message;
    profileMessage.style.display = 'block';
    profileErrorElement.style.display = 'none'; // Скрыть сообщение об ошибке
    setTimeout(() => profileMessage.style.display = 'none', 3000); // Скрыть через 3 сек
}

function displayProfileError(message) {
    profileErrorElement.textContent = message;
    profileErrorElement.style.display = 'block';
    profileMessage.style.display = 'none'; // Скрыть сообщение об успехе
}

function hideProfileMessages() {
    profileMessage.style.display = 'none';
    profileErrorElement.style.display = 'none';
}


async function loadProfileData() {
    hideProfileMessages();
    try {
        const data = await api.get('/profile');
        profileUsername.value = data.username;
        profileFullName.value = data.full_name || '';
         // Форматирование даты для input type="date" (YYYY-MM-DD)
        profileBirthDate.value = data.birth_date ? data.birth_date.split('T')[0] : '';
        profilePassword.value = ''; // Очищаем поле пароля при загрузке
         return true;
    } catch (error) {
        console.error('Failed to load profile:', error);
        displayProfileError('Не удалось загрузить данные профиля.');
        // Возможно, токен истек, перенаправить на логин?
         // logoutUser(); // Определена в auth.js
         return false;
    }
}

async function updateProfile() {
    hideProfileMessages();
    const updateData = {
        full_name: profileFullName.value || null, // Отправляем null, если пусто
        birth_date: profileBirthDate.value || null,
        password: profilePassword.value || null, // Отправляем null, если пусто
    };
    // Удаляем ключи с null значениями, если API их не ожидает пустыми
     // Object.keys(updateData).forEach(key => updateData[key] == null && delete updateData[key]);

    // Фильтруем null значения, чтобы не отправлять их, если API этого не хочет
    const filteredUpdateData = Object.fromEntries(
         Object.entries(updateData).filter(([_, v]) => v !== null && v !== '')
    );


    if (Object.keys(filteredUpdateData).length === 0) {
        displayProfileMessage("Нет данных для обновления.");
        return;
    }

     // Особая обработка пароля - отправляем, только если он не пустой
     if (updateData.password) {
         filteredUpdateData.password = updateData.password;
     } else {
        // Убедимся что поле password не отправляется, если пустое
         delete filteredUpdateData.password;
     }


    try {
        const response = await api.put('/profile', filteredUpdateData);
        displayProfileMessage(response.message || 'Профиль успешно обновлен!');
        profilePassword.value = ''; // Очищаем поле пароля после успешного обновления
    } catch (error) {
        console.error('Failed to update profile:', error);
        displayProfileError(error.message || 'Не удалось обновить профиль.');
    }
}

// --- Инициализация и обработчики ---
if (profileForm) {
    profileForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        await updateProfile();
    });
}

if (backToChatButton) {
     backToChatButton.addEventListener('click', () => {
         // Функция showView должна быть определена в main.js
         showView('app-view');
     });
 }