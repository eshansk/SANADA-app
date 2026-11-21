/**
 * frontend/js/pages/account.js
 * ============================
 * Account Profile & Session Management Controller.
 * Displays authenticated user profile, permissions, session security,
 * and workspace actions.
 */

import { getAuth } from '../auth-guard.js';

export function initAccountPage() {
    updateAccountProfile();

    // Listen for role switch or auth changes
    window.addEventListener('sanada:roleChanged', () => {
        updateAccountProfile();
    });

    // Wire Account sign out button to global logout flow
    const accountLogoutBtn = document.getElementById('accountLogoutBtn');
    if (accountLogoutBtn) {
        accountLogoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const logoutBtn = document.getElementById('logoutBtn') || document.querySelector('.btn-logout');
            if (logoutBtn) {
                logoutBtn.click();
            }
        });
    }

    // Wire Reset Workspace Session Cache button
    const accountClearSessionBtn = document.getElementById('accountClearSessionBtn');
    if (accountClearSessionBtn) {
        accountClearSessionBtn.addEventListener('click', (e) => {
            e.preventDefault();
            // Clear temporary prediction and stem cache but retain auth
            sessionStorage.removeItem('sanada_last_inference');
            sessionStorage.removeItem('sanada_active_audio');
            if (typeof window.showToast === 'function') {
                window.showToast('Workspace session cache refreshed.', 'success');
            }
            updateAccountProfile();
        });
    }
}

export function updateAccountProfile() {
    const auth = getAuth();
    const role = auth?.role || document.body?.dataset?.role || 'unauthenticated';
    const username = auth?.username || (role === 'developer' ? 'developer' : (role === 'user' ? 'user' : 'Guest'));

    // DOM Element targets
    const nameEl = document.getElementById('accountDisplayName');
    const handleEl = document.getElementById('accountUsernameTag');
    const roleBadgeEl = document.getElementById('accountRoleBadge');
    const accessTierEl = document.getElementById('accountAccessTier');
    const avatarInitialEl = document.getElementById('accountAvatarInitial');
    const sessionStartedEl = document.getElementById('accountSessionStarted');
    const statRoleEl = document.getElementById('accountStatRole');
    const statRoleSubEl = document.getElementById('accountStatRoleSub');
    const devOnlyEntitlement = document.getElementById('accountEntitlementDevOnly');
    const apiKeysLink = document.getElementById('accountApiKeysLink');

    if (nameEl) {
        nameEl.textContent = role === 'developer'
            ? 'Developer Account'
            : (role === 'user' ? 'User Account' : 'Guest Account');
    }

    if (handleEl) {
        handleEl.textContent = `@${username}`;
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

    if (accessTierEl) {
        accessTierEl.innerHTML = role === 'developer'
            ? 'Tier 1 &bull; Full Training &amp; DSP'
            : 'Tier 2 &bull; Studio &amp; API';
    }

    if (statRoleEl) {
        statRoleEl.textContent = role === 'developer' ? 'DEV' : (role === 'user' ? 'USER' : 'GUEST');
    }

    if (statRoleSubEl) {
        statRoleSubEl.textContent = role === 'developer'
            ? 'Full DSP & Checkpoints'
            : (role === 'user' ? 'Inference & Audio Player' : 'Public Overview');
    }

    if (devOnlyEntitlement) {
        if (role === 'developer') {
            devOnlyEntitlement.style.opacity = '1';
            const dot = devOnlyEntitlement.querySelector('.entitlement-status-dot');
            if (dot) dot.className = 'entitlement-status-dot online';
        } else {
            devOnlyEntitlement.style.opacity = '0.45';
            const dot = devOnlyEntitlement.querySelector('.entitlement-status-dot');
            if (dot) dot.className = 'entitlement-status-dot';
        }
    }

    if (apiKeysLink) {
        // API route is developer-only or authorized user
        apiKeysLink.style.display = (role === 'developer' || role === 'user') ? 'inline-flex' : 'none';
    }

    if (sessionStartedEl) {
        if (auth?.timestamp) {
            const date = new Date(auth.timestamp);
            sessionStartedEl.textContent = `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        } else {
            sessionStartedEl.textContent = 'Active Session';
        }
    }
}
