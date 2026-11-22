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

import { getAuth, logout } from '../auth-guard.js';
import {
    getRunHistory,
    renderRunHistoryList,
    renderHistorySkeleton,
    clearAllRuns,
    changePassword,
    getCurrentUserMeta
} from '../services/runHistory.js';
import { getSupabaseClient, isSupabaseConfigured, deleteUserAccountWithSupabase } from '../services/supabase.js';
import { showConfirmPopup } from '../ui/popup.js';

export function initAccountPage() {
    updateAccountProfile();
    initChangePasswordForm();
    initDeleteAccountModal();
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
        handleEl.textContent = `Username: ${username}`;
    }

    if (emailEl) {
        const resolvedEmail = userMeta.email || auth?.email || `${username}@sanada.ai`;
        emailEl.textContent = `Email: ${resolvedEmail}`;
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

    // Dynamic sync: Fetch real Supabase email and profile to ensure exact match
    if (isSupabaseConfigured() && !userMeta.isDemo) {
        const client = getSupabaseClient();
        if (client) {
            (async () => {
                try {
                    let dbEmail = null;
                    let dbUsername = null;
                    let dbCreatedAt = null;

                    // 1. Try get_user_profile RPC
                    try {
                        const { data: prof } = await client.rpc('get_user_profile', {
                            p_user_id: String(userMeta.id || username)
                        });
                        if (Array.isArray(prof) && prof.length > 0) {
                            if (prof[0].email) dbEmail = prof[0].email;
                            if (prof[0].username) dbUsername = prof[0].username;
                            if (prof[0].created_at) dbCreatedAt = prof[0].created_at;
                        }
                    } catch (pErr) {}

                    // 2. Query public.app_users table
                    if (!dbEmail) {
                        try {
                            let query = client.from('app_users').select('username, email, created_at');
                            if (userMeta.id) {
                                query = query.eq('id', userMeta.id);
                            } else if (username) {
                                query = query.ilike('username', username);
                            }
                            const { data: userData } = await query.maybeSingle();
                            if (userData) {
                                if (userData.email) dbEmail = userData.email;
                                if (userData.username) dbUsername = userData.username;
                                if (userData.created_at) dbCreatedAt = userData.created_at;
                            }
                        } catch (uErr) {}
                    }

                    // 3. Fallback: check Supabase Auth session / user
                    if (!dbEmail && client.auth) {
                        try {
                            const { data: suData } = await client.auth.getUser();
                            if (suData && suData.user && suData.user.email) {
                                dbEmail = suData.user.email;
                                if (suData.user.created_at) dbCreatedAt = suData.user.created_at;
                            }
                        } catch (aErr) {}
                    }

                    // 4. Update DOM with real Supabase data
                    if (dbEmail && emailEl) {
                        emailEl.textContent = `Email: ${dbEmail}`;
                    }
                    if (dbUsername && handleEl) {
                        handleEl.textContent = `Username: ${dbUsername}`;
                    }
                    if (dbCreatedAt && memberSinceEl) {
                        try {
                            const d = new Date(dbCreatedAt);
                            memberSinceEl.textContent = `Member since ${d.toLocaleDateString([], { month: 'short', year: 'numeric' })}`;
                        } catch (e) {}
                    }

                    // 5. Update session storage cache
                    if (dbEmail || dbUsername) {
                        try {
                            const currentRaw = sessionStorage.getItem('sanada_auth');
                            if (currentRaw) {
                                const currentParsed = JSON.parse(currentRaw);
                                if (dbEmail) currentParsed.email = dbEmail;
                                if (dbUsername) currentParsed.username = dbUsername;
                                if (dbCreatedAt) currentParsed.created_at = dbCreatedAt;
                                sessionStorage.setItem('sanada_auth', JSON.stringify(currentParsed));
                            }
                        } catch (sErr) {}
                    }
                } catch (syncErr) {
                    console.warn('[SANADA Account] Profile sync error:', syncErr);
                }
            })();
        }
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

    const toggleBtnIds = ['cpCurrentPassToggle', 'cpNewPassToggle', 'cpConfirmPassToggle'];

    function resetPasswordVisibility() {
        ['cpCurrentPass', 'cpNewPass', 'cpConfirmPass'].forEach(id => {
            const inp = document.getElementById(id);
            if (inp) inp.type = 'password';
        });
        toggleBtnIds.forEach(tid => {
            const btn = document.getElementById(tid);
            if (btn) {
                const showEye = btn.querySelector('.eye-show');
                const hideEye = btn.querySelector('.eye-hide');
                if (showEye) showEye.style.setProperty('display', 'block', 'important');
                if (hideEye) hideEye.style.setProperty('display', 'none', 'important');
                btn.setAttribute('aria-label', 'Show password');
                btn.setAttribute('title', 'Show password');
            }
        });
    }

    // Direct onclick assignment prevents multiple stacked event listeners
    function wirePasswordToggle(toggleId, inputId) {
        const btn = document.getElementById(toggleId);
        if (!btn) return;

        btn.onclick = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility(inputId, btn);
            }
        };
    }

    wirePasswordToggle('cpCurrentPassToggle', 'cpCurrentPass');
    wirePasswordToggle('cpNewPassToggle', 'cpNewPass');
    wirePasswordToggle('cpConfirmPassToggle', 'cpConfirmPass');

    // If demo account: disable form and show notice
    if (userMeta.isDemo) {
        currentPassInput.disabled = true;
        newPassInput.disabled = true;
        confirmPassInput.disabled = true;
        submitBtn.disabled = true;
        toggleBtnIds.forEach(tid => {
            const btn = document.getElementById(tid);
            if (btn) btn.disabled = true;
        });
        if (demoBanner) demoBanner.style.display = 'flex';
        return;
    }

    // Regular authenticated user: enable form
    currentPassInput.disabled = false;
    newPassInput.disabled = false;
    confirmPassInput.disabled = false;
    submitBtn.disabled = false;
    toggleBtnIds.forEach(tid => {
        const btn = document.getElementById(tid);
        if (btn) btn.disabled = false;
    });
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
            resetPasswordVisibility();
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

/**
 * Initializes and wires the Danger Zone: Delete Account modal with password confirmation.
 */
function initDeleteAccountModal() {
    const triggerBtn = document.getElementById('accountDeleteAccountBtn');
    const modal = document.getElementById('sanadaDeleteAccountModal');
    const closeBtn = document.getElementById('modalDeleteAccountCloseBtn');
    const cancelBtn = document.getElementById('modalDeleteAccountCancelBtn');
    const form = document.getElementById('deleteAccountForm');
    const passInput = document.getElementById('deleteAccountPasswordInput');
    const passToggle = document.getElementById('deleteAccountPassToggle');
    const submitBtn = document.getElementById('modalDeleteAccountConfirmBtn');
    const errorBox = document.getElementById('modalDeleteAccountError');
    const errorText = document.getElementById('modalDeleteAccountErrorText');

    if (!triggerBtn || !modal || !form || !passInput || !submitBtn) return;

    function openModal() {
        const userMeta = getCurrentUserMeta();
        if (userMeta.isDemo) {
            if (typeof window.notify === 'function') {
                window.notify('Built-in demo accounts cannot be deleted.', 'error');
            } else if (typeof window.showToast === 'function') {
                window.showToast('Built-in demo accounts cannot be deleted.', 'error');
            } else {
                alert('Built-in demo accounts cannot be deleted.');
            }
            return;
        }

        modal.classList.add('open');
        modal.setAttribute('aria-hidden', 'false');
        passInput.value = '';
        passInput.type = 'password';
        if (errorBox) errorBox.style.display = 'none';

        // Reset eye toggle to hidden password icon
        if (passToggle) {
            const showEye = passToggle.querySelector('.eye-show');
            const hideEye = passToggle.querySelector('.eye-hide');
            if (showEye) showEye.style.setProperty('display', 'block', 'important');
            if (hideEye) hideEye.style.setProperty('display', 'none', 'important');
        }

        setTimeout(() => passInput.focus(), 80);
    }

    function closeModal() {
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        passInput.value = '';
        if (errorBox) errorBox.style.display = 'none';
    }

    triggerBtn.onclick = (e) => {
        e.preventDefault();
        openModal();
    };

    closeBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal();
    });

    cancelBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal();
    });

    // Close on backdrop click outside card
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('open')) {
            closeModal();
        }
    });

    // Password visibility toggle
    if (passToggle) {
        passToggle.onclick = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility('deleteAccountPasswordInput', passToggle);
            }
        };
    }

    // Form submit: password verification and cascade deletion
    form.onsubmit = async (e) => {
        e.preventDefault();
        if (errorBox) errorBox.style.display = 'none';

        const enteredPass = passInput.value.trim();
        if (!enteredPass) {
            showDeleteError('Please enter your password to confirm deletion.');
            passInput.focus();
            return;
        }

        const userMeta = getCurrentUserMeta();
        const userIdent = userMeta.username || userMeta.id;

        submitBtn.disabled = true;
        const originalContent = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span>Deleting Account...</span>';

        try {
            const res = await deleteUserAccountWithSupabase(userIdent, enteredPass);
            if (!res.success) {
                showDeleteError(res.error || 'Failed to delete account. Please verify your password.');
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalContent;
                passInput.focus();
                return;
            }

            closeModal();

            if (typeof window.notify === 'function') {
                window.notify('Account permanently deleted.', 'info');
            } else if (typeof window.showToast === 'function') {
                window.showToast('Account permanently deleted.', 'info');
            }

            // Clear session and redirect to landing
            setTimeout(() => {
                if (typeof logout === 'function') {
                    logout();
                } else if (typeof window.logout === 'function') {
                    window.logout();
                } else if (typeof window.SANADA_AUTH?.logout === 'function') {
                    window.SANADA_AUTH.logout();
                } else {
                    sessionStorage.removeItem('sanada_auth');
                    window.location.href = 'index.html';
                }
            }, 500);
        } catch (err) {
            showDeleteError(err?.message || 'Network error while attempting account deletion.');
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalContent;
        }
    };

    function showDeleteError(msg) {
        if (errorBox && errorText) {
            errorText.textContent = msg;
            errorBox.style.display = 'flex';
        } else if (typeof window.notify === 'function') {
            window.notify(msg, 'error');
        } else if (typeof window.showToast === 'function') {
            window.showToast(msg, 'error');
        }
    }
}
