/**
 * frontend/js/pages/account.js
 * ============================
 * Redesigned Account Profile, Run History & Credential Controller.
 *
 * Provides:
 *  1. Profile header (Avatar, username, email, role, member since).
 *  2. Recent Run History (Shared with Studio page, max 5 runs, rename/pin/delete/clear).
 *  3. Default Inference Preferences (Task, mode, threshold, all_4_stems) loaded by Studio.
 *  4. Change Password with crypt() verification, validation, and demo account notice.
 *  5. API keys link and Sign out button.
 */

import { getAuth } from '../auth-guard.js';
import {
    getRunHistory,
    renderRunHistoryList,
    renderHistorySkeleton,
    clearAllRuns,
    changePassword,
    getCurrentUserMeta
} from '../services/runHistory.js';
import { showConfirmPopup } from '../ui/popup.js';

export function initAccountPage() {
    updateAccountProfile();
    initChangePasswordForm();
    initAccountRunHistory();

    // Re-render when role or session changes
    window.addEventListener('sanada:roleChanged', () => {
        updateAccountProfile();
        initChangePasswordForm();
        refreshAccountRuns();
    });

    // Re-render run history whenever a run is recorded, renamed, or deleted
    window.addEventListener('sanada:runHistoryChanged', () => {
        refreshAccountRuns();
    });

    // Re-render run history whenever Account page is navigated or routed to
    window.addEventListener('sanada:routeChanged', (e) => {
        if (e.detail?.route === 'account') {
            updateAccountProfile();
            refreshAccountRuns();
        }
    });

    window.addEventListener('hashchange', () => {
        const hash = (window.location.hash || '').replace(/^#\/?/, '').split('/')[0].trim();
        if (hash === 'account' || hash === 'profile' || hash === 'settings') {
            updateAccountProfile();
            refreshAccountRuns();
        }
    });

    // Wire Sign Out button to global logout flow
    const accountLogoutBtn = document.getElementById('accountLogoutBtn');
    if (accountLogoutBtn) {
        accountLogoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const logoutBtn = document.getElementById('logoutBtn') || document.querySelector('.btn-logout');
            if (logoutBtn) {
                logoutBtn.click();
            } else if (typeof window.openLogoutModal === 'function') {
                window.openLogoutModal();
            }
        });
    }
}

/**
 * Updates profile header information.
 */
export function updateAccountProfile() {
    const userMeta = getCurrentUserMeta();
    const auth = getAuth();
    const role = userMeta.role || 'unauthenticated';
    const username = userMeta.username || (role === 'developer' ? 'developer' : 'user');

    const nameEl = document.getElementById('accountDisplayName');
    const handleEl = document.getElementById('accountUsernameTag');
    const emailEl = document.getElementById('accountEmailTag');
    const roleBadgeEl = document.getElementById('accountRoleBadge');
    const avatarInitialEl = document.getElementById('accountAvatarInitial');
    const memberSinceEl = document.getElementById('accountMemberSince');
    const apiKeysLink = document.getElementById('accountApiKeysLink');

    if (nameEl) {
        nameEl.textContent = role === 'developer'
            ? 'Developer Account'
            : (role === 'user' ? 'User Account' : 'Guest Account');
    }

    if (handleEl) {
        handleEl.textContent = `@${username}`;
    }

    if (emailEl) {
        const resolvedEmail = userMeta.email || auth?.email || `${username}@sanada.ai`;
        emailEl.textContent = resolvedEmail;
    }

    if (avatarInitialEl) {
        avatarInitialEl.textContent = username.charAt(0).toUpperCase();
    }

    if (roleBadgeEl) {
        if (role === 'developer') {
            roleBadgeEl.textContent = 'Developer (Full Access)';
            roleBadgeEl.className = 'badge badge-dev';
        } else if (role === 'user') {
            roleBadgeEl.textContent = 'User (Inference Studio)';
            roleBadgeEl.className = 'badge badge-user';
        } else {
            roleBadgeEl.textContent = 'Unauthenticated';
            roleBadgeEl.className = 'badge';
        }
    }

    if (memberSinceEl) {
        if (auth?.created_at) {
            try {
                const d = new Date(auth.created_at);
                memberSinceEl.textContent = `Member since ${d.toLocaleDateString([], { month: 'short', year: 'numeric' })}`;
            } catch (e) {
                memberSinceEl.textContent = 'Member since Nov 2026';
            }
        } else {
            memberSinceEl.textContent = 'Member since Nov 2026';
        }
    }

    if (apiKeysLink) {
        apiKeysLink.style.display = (role === 'developer' || role === 'user') ? 'inline-flex' : 'none';
    }
}

/**
 * Initializes and wires the Change Password form with demo protection.
 */
function initChangePasswordForm() {
    const userMeta = getCurrentUserMeta();
    const form = document.getElementById('accountPasswordForm');
    const demoBanner = document.getElementById('accountDemoPasswordNotice');
    const currentPassInput = document.getElementById('cpCurrentPass');
    const newPassInput = document.getElementById('cpNewPass');
    const confirmPassInput = document.getElementById('cpConfirmPass');
    const submitBtn = document.getElementById('cpSubmitBtn');
    const errorEl = document.getElementById('cpErrorMsg');

    if (!form || !currentPassInput || !newPassInput || !confirmPassInput || !submitBtn) return;

    // If demo account: disable form and show notice
    if (userMeta.isDemo) {
        currentPassInput.disabled = true;
        newPassInput.disabled = true;
        confirmPassInput.disabled = true;
        submitBtn.disabled = true;
        if (demoBanner) demoBanner.style.display = 'flex';
        return;
    }

    // Regular authenticated user: enable form
    currentPassInput.disabled = false;
    newPassInput.disabled = false;
    confirmPassInput.disabled = false;
    submitBtn.disabled = false;
    if (demoBanner) demoBanner.style.display = 'none';

    form.onsubmit = async (e) => {
        e.preventDefault();
        if (errorEl) {
            errorEl.style.display = 'none';
            errorEl.textContent = '';
        }

        const oldPass = currentPassInput.value.trim();
        const newPass = newPassInput.value.trim();
        const confirmPass = confirmPassInput.value.trim();

        if (!oldPass) {
            showError('Please enter your current password.');
            currentPassInput.focus();
            return;
        }

        if (!newPass) {
            showError('Please enter a new password.');
            newPassInput.focus();
            return;
        }

        if (newPass.length < 8) {
            showError('New password must be at least 8 characters long.');
            newPassInput.focus();
            return;
        }

        if (oldPass === newPass) {
            showError('New password must be different from current password.');
            newPassInput.focus();
            return;
        }

        if (newPass !== confirmPass) {
            showError('New password and confirmation do not match.');
            confirmPassInput.focus();
            return;
        }

        submitBtn.disabled = true;
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span>Updating Password...</span>';

        try {
            const res = await changePassword(oldPass, newPass);
            if (!res.success) {
                showError(res.error || 'Password update failed.');
                return;
            }

            form.reset();
            if (typeof window.showToast === 'function') {
                window.showToast('Password changed successfully!', 'success');
            }
        } catch (err) {
            showError(err?.message || 'Network error updating password.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    };

    function showError(msg) {
        if (errorEl) {
            errorEl.textContent = msg;
            errorEl.style.display = 'block';
        } else if (typeof window.showToast === 'function') {
            window.showToast(msg, 'error');
        }
    }
}

/**
 * Initializes the Run History list on the Account page.
 */
async function initAccountRunHistory() {
    const clearBtn = document.querySelector('#accountRunHistoryCard [data-action="clear-runs"]');
    if (clearBtn) {
        clearBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const confirmed = await showConfirmPopup({
                title: 'Clear Run History',
                message: 'Are you sure you want to clear your inference run history? This action cannot be undone.',
                confirmText: 'Clear All',
                isDanger: true
            });
            if (confirmed) {
                await clearAllRuns();
                if (typeof window.showToast === 'function') {
                    window.showToast('All run history cleared.', 'info');
                }
            }
        });
    }

    await refreshAccountRuns();
}

async function refreshAccountRuns() {
    const listEl = document.getElementById('accountRunHistoryList');
    if (!listEl) return;
    renderHistorySkeleton(listEl, 3);
    const runs = await getRunHistory();
    renderRunHistoryList(listEl, runs);
}
