/**
 * frontend/js/ui/popup.js
 * =======================
 * Liquid-glass modal popups for Confirm, Prompt, and Alert dialogs.
 * Replaces native browser alert(), confirm(), and prompt() with
 * wave-themed, accessible, non-blocking promise-based dialogs.
 */

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function ensurePopupDOM() {
    let overlay = document.getElementById('sanadaActionPopup');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'sanadaActionPopup';
        overlay.className = 'auth-modal sanada-action-popup-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-hidden', 'true');
        overlay.innerHTML = `
            <div class="auth-modal-card logout-modal-card sanada-action-popup-card">
                <button type="button" class="auth-modal-close" id="popupCloseBtn" aria-label="Close dialog" title="Close (Esc)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
                <h2 class="modal-title" id="popupTitle">Confirm</h2>
                <p class="modal-subtitle" id="popupMessage"></p>
                <div class="popup-input-wrap" id="popupInputWrap" style="display: none; margin-top: 16px;">
                    <input type="text" class="text-input popup-input-field" id="popupInput" placeholder="Enter text...">
                </div>
                <div class="logout-modal-actions popup-actions" id="popupActions">
                    <button type="button" class="btn btn-modal-cancel" id="popupCancelBtn"><span>Cancel</span></button>
                    <button type="button" class="btn btn-modal-confirm-primary" id="popupConfirmBtn"><span>Confirm</span></button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    return overlay;
}

/**
 * Displays a liquid-glass confirmation popup dialog.
 * @returns {Promise<boolean>}
 */
export function showConfirmPopup({
    title = 'Confirm Action',
    message = 'Are you sure you want to proceed?',
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    isDanger = false
} = {}) {
    return new Promise((resolve) => {
        const overlay = ensurePopupDOM();
        const titleEl = document.getElementById('popupTitle');
        const msgEl = document.getElementById('popupMessage');
        const inputWrap = document.getElementById('popupInputWrap');
        const cancelBtn = document.getElementById('popupCancelBtn');
        const confirmBtn = document.getElementById('popupConfirmBtn');
        const closeBtn = document.getElementById('popupCloseBtn');

        if (titleEl) titleEl.textContent = title;
        if (msgEl) msgEl.textContent = message;
        if (inputWrap) inputWrap.style.display = 'none';

        const actionsWrap = document.getElementById('popupActions');
        if (actionsWrap) actionsWrap.classList.remove('single-action');

        if (cancelBtn) {
            cancelBtn.style.display = 'inline-flex';
            cancelBtn.innerHTML = `<span>${escapeHtml(cancelText)}</span>`;
        }

        if (confirmBtn) {
            confirmBtn.innerHTML = `<span>${escapeHtml(confirmText)}</span>`;
            if (isDanger) {
                confirmBtn.className = 'btn btn-modal-confirm-danger';
            } else {
                confirmBtn.className = 'btn btn-modal-confirm-primary';
            }
        }

        function cleanup() {
            overlay.classList.remove('open');
            overlay.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
            confirmBtn?.removeEventListener('click', onConfirm);
            cancelBtn?.removeEventListener('click', onCancel);
            closeBtn?.removeEventListener('click', onCancel);
            overlay.removeEventListener('click', onBackdrop);
            window.removeEventListener('keydown', onKey);
        }

        function onConfirm() {
            cleanup();
            resolve(true);
        }

        function onCancel() {
            cleanup();
            resolve(false);
        }

        function onBackdrop(e) {
            if (e.target === overlay) {
                onCancel();
            }
        }

        function onKey(e) {
            if (e.key === 'Escape') {
                onCancel();
            } else if (e.key === 'Enter') {
                onConfirm();
            }
        }

        confirmBtn?.addEventListener('click', onConfirm);
        cancelBtn?.addEventListener('click', onCancel);
        closeBtn?.addEventListener('click', onCancel);
        overlay.addEventListener('click', onBackdrop);
        window.addEventListener('keydown', onKey);

        overlay.classList.add('open');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        confirmBtn?.focus();
    });
}

/**
 * Displays a liquid-glass prompt popup dialog with text input.
 * @returns {Promise<string | null>}
 */
export function showPromptPopup({
    title = 'Input Required',
    message = 'Please enter a value:',
    defaultValue = '',
    placeholder = 'Enter label...',
    confirmText = 'Save',
    cancelText = 'Cancel'
} = {}) {
    return new Promise((resolve) => {
        const overlay = ensurePopupDOM();
        const titleEl = document.getElementById('popupTitle');
        const msgEl = document.getElementById('popupMessage');
        const inputWrap = document.getElementById('popupInputWrap');
        const inputEl = document.getElementById('popupInput');
        const cancelBtn = document.getElementById('popupCancelBtn');
        const confirmBtn = document.getElementById('popupConfirmBtn');
        const closeBtn = document.getElementById('popupCloseBtn');

        if (titleEl) titleEl.textContent = title;
        if (msgEl) msgEl.textContent = message;

        if (inputWrap) inputWrap.style.display = 'block';
        if (inputEl) {
            inputEl.value = defaultValue || '';
            inputEl.placeholder = placeholder;
        }

        const actionsWrap = document.getElementById('popupActions');
        if (actionsWrap) actionsWrap.classList.remove('single-action');

        if (cancelBtn) {
            cancelBtn.style.display = 'inline-flex';
            cancelBtn.innerHTML = `<span>${escapeHtml(cancelText)}</span>`;
        }

        if (confirmBtn) {
            confirmBtn.innerHTML = `<span>${escapeHtml(confirmText)}</span>`;
            confirmBtn.className = 'btn btn-modal-confirm-primary';
        }

        function cleanup() {
            overlay.classList.remove('open');
            overlay.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
            confirmBtn?.removeEventListener('click', onConfirm);
            cancelBtn?.removeEventListener('click', onCancel);
            closeBtn?.removeEventListener('click', onCancel);
            overlay.removeEventListener('click', onBackdrop);
            window.removeEventListener('keydown', onKey);
        }

        function onConfirm() {
            const val = inputEl ? inputEl.value : '';
            cleanup();
            resolve(val);
        }

        function onCancel() {
            cleanup();
            resolve(null);
        }

        function onBackdrop(e) {
            if (e.target === overlay) {
                onCancel();
            }
        }

        function onKey(e) {
            if (e.key === 'Escape') {
                onCancel();
            } else if (e.key === 'Enter') {
                onConfirm();
            }
        }

        confirmBtn?.addEventListener('click', onConfirm);
        cancelBtn?.addEventListener('click', onCancel);
        closeBtn?.addEventListener('click', onCancel);
        overlay.addEventListener('click', onBackdrop);
        window.addEventListener('keydown', onKey);

        overlay.classList.add('open');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        setTimeout(() => {
            inputEl?.focus();
            inputEl?.select();
        }, 50);
    });
}

/**
 * Displays a liquid-glass alert notification popup.
 * @returns {Promise<void>}
 */
export function showAlertPopup({
    title = 'Notice',
    message = '',
    confirmText = 'OK'
} = {}) {
    return new Promise((resolve) => {
        const overlay = ensurePopupDOM();
        const titleEl = document.getElementById('popupTitle');
        const msgEl = document.getElementById('popupMessage');
        const inputWrap = document.getElementById('popupInputWrap');
        const cancelBtn = document.getElementById('popupCancelBtn');
        const confirmBtn = document.getElementById('popupConfirmBtn');
        const closeBtn = document.getElementById('popupCloseBtn');

        if (titleEl) titleEl.textContent = title;
        if (msgEl) msgEl.textContent = message;
        if (inputWrap) inputWrap.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'none';

        const actionsWrap = document.getElementById('popupActions');
        if (actionsWrap) actionsWrap.classList.add('single-action');

        if (confirmBtn) {
            confirmBtn.innerHTML = `<span>${escapeHtml(confirmText)}</span>`;
            confirmBtn.className = 'btn btn-modal-confirm-primary';
        }

        function cleanup() {
            overlay.classList.remove('open');
            overlay.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
            confirmBtn?.removeEventListener('click', onConfirm);
            closeBtn?.removeEventListener('click', onConfirm);
            overlay.removeEventListener('click', onConfirm);
            window.removeEventListener('keydown', onKey);
        }

        function onConfirm() {
            cleanup();
            resolve();
        }

        function onKey(e) {
            if (e.key === 'Escape' || e.key === 'Enter') {
                onConfirm();
            }
        }

        confirmBtn?.addEventListener('click', onConfirm);
        closeBtn?.addEventListener('click', onConfirm);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) onConfirm(); });
        window.addEventListener('keydown', onKey);

        overlay.classList.add('open');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        confirmBtn?.focus();
    });
}

// Expose globally for convenience
if (typeof window !== 'undefined') {
    window.showConfirmPopup = showConfirmPopup;
    window.showPromptPopup = showPromptPopup;
    window.showAlertPopup = showAlertPopup;
}
