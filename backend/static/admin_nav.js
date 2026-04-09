/**
 * admin_nav.js — Общая навигация и auth-обёртка для админских страниц.
 *
 * Использование: подключить <script src="/static/admin_nav.js"></script>
 * Вызвать AdminNav.init({ currentPage: 'bot-config' }) после загрузки DOM.
 */
const AdminNav = (() => {
    const _nativeFetch = window.fetch.bind(window); // capture before any override
    let _token = null;
    let _user = null;

    /** Получить токен из sessionStorage */
    function getToken() {
        if (!_token) _token = sessionStorage.getItem('admin_token');
        return _token;
    }

    /** Получить данные пользователя */
    function getUser() {
        if (!_user) {
            const raw = sessionStorage.getItem('admin_user');
            if (raw) {
                try { _user = JSON.parse(raw); } catch { _user = null; }
            }
        }
        return _user;
    }

    /** Проверить авторизацию, при отсутствии — redirect */
    function requireAuth() {
        const token = getToken();
        if (!token) {
            const next = encodeURIComponent(window.location.pathname);
            window.location.href = '/admin-login?next=' + next;
            return false;
        }
        return true;
    }

    /** Выйти */
    function logout() {
        sessionStorage.removeItem('admin_token');
        sessionStorage.removeItem('admin_user');
        _token = null;
        _user = null;
        window.location.href = '/admin-login';
    }

    /** Обёртка fetch с авторизацией. При 401 — redirect на логин. */
    async function authFetch(url, options = {}) {
        const token = getToken();
        if (!token) {
            logout();
            throw new Error('Not authenticated');
        }
        const headers = { ...(options.headers || {}) };
        headers['Authorization'] = 'Bearer ' + token;
        if (!(options.body instanceof FormData) && !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }
        const resp = await _nativeFetch(url, { ...options, headers });
        if (resp.status === 401) {
            logout();
            throw new Error('Session expired');
        }
        if (resp.status === 403) {
            throw new Error('Недостаточно прав');
        }
        return resp;
    }

    /** Отрисовать навигацию */
    function renderNav(currentPage) {
        const user = getUser();
        const role = user?.role || '';
        const displayName = user?.display_name || user?.username || '';

        const nav = document.createElement('nav');
        nav.className = 'admin-nav';
        nav.innerHTML = `
            <div class="admin-nav__links">
                <a href="/bot-config" class="admin-nav__link ${currentPage === 'bot-config' ? 'active' : ''}">Настройки бота</a>
                <a href="/kb-admin" class="admin-nav__link ${currentPage === 'kb-admin' ? 'active' : ''}">База знаний</a>
                ${role === 'superadmin' ? `
                <a href="/admin-users" class="admin-nav__link ${currentPage === 'admin-users' ? 'active' : ''}">Пользователи</a>
                <a href="/admin-audit" class="admin-nav__link ${currentPage === 'admin-audit' ? 'active' : ''}">Аудит</a>
                ` : ''}
            </div>
            <div class="admin-nav__user">
                <span class="admin-nav__name">${_escapeHtml(displayName)}</span>
                <span class="admin-nav__role">${_escapeHtml(role)}</span>
                <button class="admin-nav__logout" onclick="AdminNav.logout()" title="Выйти">Выйти</button>
            </div>
        `;
        document.body.prepend(nav);
    }

    function _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /** Инициализация: проверить auth, отрисовать nav, вернуть user */
    function init({ currentPage = '' } = {}) {
        if (!requireAuth()) return null;
        renderNav(currentPage);

        // Добавить CSS-класс роли на body
        const user = getUser();
        if (user?.role) {
            document.body.classList.add('role-' + user.role);
        }
        return user;
    }

    /** Проверить, является ли пользователь viewer */
    function isViewer() {
        return getUser()?.role === 'viewer';
    }

    return { init, getToken, getUser, authFetch, logout, requireAuth, isViewer };
})();
