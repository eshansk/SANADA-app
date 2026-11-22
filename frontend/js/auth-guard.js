/**
 * frontend/js/auth-guard.js
 * =========================
 * SANADA Authentication & Role Display Controller.
 *
 * Provides:
 *  1. Early synchronous role detection (preventing layout shifts / flashing nav buttons).
 *  2. Unauthenticated landing mode (0 navigation buttons, hero shows Login button).
 *  3. Dual login selector & modal (Login as Developer vs Login as User).
 *  4. Supabase PostgreSQL authentication via public.authenticate_user(p_username, p_password) RPC.
 *  5. Role-based display switching (Developer suite vs User machine listening studio).
 *  6. Session persistence and clean logout handling.
 */

import {
    isSupabaseConfigured,
    loginWithSupabase,
    signUpWithSupabase,
    verifyOtpWithSupabase,
    resendOtpWithSupabase,
    getSupabaseUrl,
    getSupabaseKey
} from './services/supabase.js';

// Demo credentials configuration
export const DEMO_CONFIG = {
    developer: {
        username: 'developer',
        password: 'dev123',
        role: 'developer',
        label: 'Developer',
        badge: 'Full Suite',
        desc: 'Source counting, tagging, separation, datasets & checkpoints'
    },
    user: {
        username: 'user',
        password: 'user123',
        role: 'user',
        label: 'User',
        badge: 'Inference',
        desc: 'Neural source count estimation & speech separation studio'
    }
};

// Global Password Visibility Toggle (Debounced & Safe)
let _lastToggleTime = 0;
let _lastToggleTarget = null;
export function togglePasswordVisibility(inputId, btnEl) {
    const now = Date.now();
    const targetId = typeof inputId === 'string' ? inputId : (inputId && inputId.id ? inputId.id : '');
    if (targetId && _lastToggleTarget === targetId && (now - _lastToggleTime) < 250) {
        return;
    }
    _lastToggleTime = now;
    _lastToggleTarget = targetId;

    const input = typeof inputId === 'string' ? document.getElementById(inputId) : inputId;
    if (!input) return;

    let btn = btnEl ? (btnEl.closest ? btnEl.closest('button, .modal-pass-toggle, .password-toggle-btn') : btnEl) : null;
    if (!btn && input.parentElement) {
        btn = input.parentElement.querySelector('.modal-pass-toggle, .password-toggle-btn');
    }

    const isPass = (input.type === 'password');
    input.type = isPass ? 'text' : 'password';

    if (btn) {
        const showEye = btn.querySelector('.eye-show');
        const hideEye = btn.querySelector('.eye-hide');
        if (showEye) showEye.style.setProperty('display', isPass ? 'none' : 'block', 'important');
        if (hideEye) hideEye.style.setProperty('display', isPass ? 'block' : 'none', 'important');
        btn.setAttribute('aria-label', isPass ? 'Hide password' : 'Show password');
        btn.setAttribute('title', isPass ? 'Hide password' : 'Show password');
    }

    try {
        input.focus();
        const len = input.value.length;
        input.setSelectionRange(len, len);
    } catch (e) {}
}

if (typeof window !== 'undefined') {
    window.togglePasswordVisibility = togglePasswordVisibility;
}

// 1. Synchronous phase: run immediately to tag document root before first paint
(function setInitialRoleEarly() {
    if (typeof window === 'undefined') return;
    try {
        const raw = sessionStorage.getItem('sanada_auth');
        if (raw) {
            const auth = JSON.parse(raw);
            if (auth && (auth.role === 'developer' || auth.role === 'user')) {
                document.documentElement.dataset.role = auth.role;
                return;
            }
        }
    } catch (e) {}
    document.documentElement.dataset.role = 'unauthenticated';
})();

// Session Accessors
export function getAuth() {
    try {
        const raw = sessionStorage.getItem('sanada_auth');
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed && (parsed.role === 'developer' || parsed.role === 'user')) {
            return parsed;
        }
    } catch (e) {}
    return null;
}

export function setAuth(role, username, extra = {}) {
    try {
        sessionStorage.setItem('sanada_auth', JSON.stringify({
            role: role,
            username: username || (role === 'developer' ? 'developer' : 'user'),
            id: extra.id || null,
            email: extra.email || null,
            created_at: extra.created_at || null,
            is_demo: Boolean(extra.is_demo),
            timestamp: Date.now()
        }));
    } catch (e) {
        console.warn('Storage write error', e);
    }
}

export function clearAuth() {
    try {
        sessionStorage.removeItem('sanada_auth');
    } catch (e) {
        console.warn('Storage remove error', e);
    }
}

// Global Toast helper fallback
function notify(message, type = 'info') {
    if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
        window.showToast(message, type);
    }
}

// 2. Role Display Synchronization
export function applyRoleDisplay(role) {
    const activeRole = (role === 'developer' || role === 'user') ? role : 'unauthenticated';

    if (typeof document !== 'undefined') {
        document.documentElement.dataset.role = activeRole;
        if (document.body) {
            document.body.dataset.role = activeRole;
        }

        // Synchronize auth buttons across navbar and hero
        const heroSignupBtn = document.getElementById('heroSignupBtn');
        const navSignupBtn = document.getElementById('navSignupBtn');
        const heroLoginBtn = document.getElementById('heroLoginBtn');
        const navLoginBtn = document.getElementById('navLoginBtn');
        const navAccountBtn = document.getElementById('navAccountBtn');
        const mobileMenuAccountLink = document.getElementById('mobileMenuAccountLink');

        if (activeRole === 'developer' || activeRole === 'user') {
            if (heroSignupBtn) heroSignupBtn.style.display = 'none';
            if (navSignupBtn) navSignupBtn.style.display = 'none';
            if (heroLoginBtn) heroLoginBtn.style.display = 'none';
            if (navLoginBtn) navLoginBtn.style.display = 'none';
            if (navAccountBtn) navAccountBtn.style.display = 'inline-flex';
            if (mobileMenuAccountLink) mobileMenuAccountLink.style.display = '';
        } else {
            if (heroSignupBtn) heroSignupBtn.style.display = '';
            if (navSignupBtn) navSignupBtn.style.display = '';
            if (heroLoginBtn) heroLoginBtn.style.display = '';
            if (navLoginBtn) navLoginBtn.style.display = '';
            if (navAccountBtn) navAccountBtn.style.display = 'none';
            if (mobileMenuAccountLink) mobileMenuAccountLink.style.display = 'none';
        }

        // Trigger custom role event for SPA router
        window.dispatchEvent(new CustomEvent('sanada:roleChanged', { detail: { role: activeRole } }));
    }
}

// 3. Modal State & Controller (Sign In, Create Account, OTP Verification)
let currentModalRole = 'user';
let currentModalView = 'signin'; // 'signin' | 'signup' | 'otp'
let pendingSignup = { username: '', email: '', password: '' };
let resendCooldown = 0;
let resendTimer = null;

export function startResendCooldown(seconds = 60) {
    if (resendTimer) {
        clearInterval(resendTimer);
        resendTimer = null;
    }
    resendCooldown = seconds;
    const btn = document.getElementById('modalOtpResendBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = `Resend Code (${resendCooldown}s)`;
        btn.style.opacity = '0.6';
        btn.style.pointerEvents = 'none';
    }

    resendTimer = setInterval(() => {
        resendCooldown -= 1;
        if (resendCooldown <= 0) {
            clearInterval(resendTimer);
            resendTimer = null;
            resendCooldown = 0;
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'Resend Code';
                btn.style.opacity = '1';
                btn.style.pointerEvents = 'auto';
            }
        } else {
            if (btn) {
                btn.textContent = `Resend Code (${resendCooldown}s)`;
            }
        }
    }, 1000);
}

export function switchModalView(viewName) {
    currentModalView = (viewName === 'signup' || viewName === 'otp') ? viewName : 'signin';

    const signinView = document.getElementById('modalSigninView');
    const signupView = document.getElementById('modalSignupView');
    const otpView = document.getElementById('modalOtpView');

    hideModalAlerts();

    if (signinView) signinView.style.display = (currentModalView === 'signin') ? 'block' : 'none';
    if (signupView) signupView.style.display = (currentModalView === 'signup') ? 'block' : 'none';
    if (otpView) otpView.style.display = (currentModalView === 'otp') ? 'block' : 'none';

    if (currentModalView === 'signin') {
        setTimeout(() => {
            document.getElementById('modalUsernameInput')?.focus();
        }, 50);
    } else if (currentModalView === 'signup') {
        setTimeout(() => {
            document.getElementById('modalSignupUsernameInput')?.focus();
        }, 50);
    } else if (currentModalView === 'otp') {
        const targetLabel = document.getElementById('modalOtpTargetEmail');
        if (targetLabel && pendingSignup.email) {
            targetLabel.textContent = pendingSignup.email;
        }
        setTimeout(() => {
            document.getElementById('modalOtpInput')?.focus();
        }, 50);
    }
}

export function setModalActiveRole(role) {
    currentModalRole = (role === 'developer') ? 'developer' : 'user';

    const devCard = document.getElementById('modalRoleDevBtn');
    const userCard = document.getElementById('modalRoleUserBtn');
    const submitBtn = document.getElementById('modalSubmitBtn');
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    const usernameLabel = document.getElementById('modalUsernameLabel');
    hideModalAlerts();

    // Both roles support logging in with Username or Email
    if (usernameLabel) usernameLabel.textContent = 'Username or Email';
    if (usernameInput) usernameInput.placeholder = 'Enter username or email';
    if (passwordInput) passwordInput.placeholder = 'Enter password';

    if (currentModalRole === 'developer') {
        devCard?.classList.add('active');
        devCard?.setAttribute('aria-selected', 'true');
        userCard?.classList.remove('active');
        userCard?.setAttribute('aria-selected', 'false');

        if (submitBtn) submitBtn.innerHTML = '<span>Log In as Developer</span>';
    } else {
        userCard?.classList.add('active');
        userCard?.setAttribute('aria-selected', 'true');
        devCard?.classList.remove('active');
        devCard?.setAttribute('aria-selected', 'false');

        if (submitBtn) submitBtn.innerHTML = '<span>Log In as User</span>';
    }
}

export function openAuthModal(preferredRole = 'user') {
    const modal = document.getElementById('sanadaAuthModal');
    if (!modal) return;

    setModalActiveRole(preferredRole || 'user');

    // Default to sign-in view
    switchModalView('signin');

    // Keep fields empty when opening
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    if (usernameInput) usernameInput.value = '';
    if (passwordInput) passwordInput.value = '';

    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    document.body.style.overflow = 'hidden';

    // Focus input
    setTimeout(() => {
        usernameInput?.focus();
    }, 50);
}

export function openSignupModal() {
    const modal = document.getElementById('sanadaAuthModal');
    if (!modal) return;

    setModalActiveRole('user');
    switchModalView('signup');

    // Keep fields empty when opening
    const regUsernameInput = document.getElementById('modalSignupUsernameInput');
    const regEmailInput = document.getElementById('modalSignupEmailInput');
    const regPasswordInput = document.getElementById('modalSignupPasswordInput');
    if (regUsernameInput) regUsernameInput.value = '';
    if (regEmailInput) regEmailInput.value = '';
    if (regPasswordInput) regPasswordInput.value = '';

    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    document.body.style.overflow = 'hidden';

    setTimeout(() => {
        regUsernameInput?.focus();
    }, 50);
}

export function closeAuthModal() {
    if (window.location.pathname.endsWith('signup.html')) {
        window.location.href = 'index.html';
        return;
    }

    const modal = document.getElementById('sanadaAuthModal');
    if (!modal) return;

    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';

    // Clear fields on close
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    const regUsernameInput = document.getElementById('modalSignupUsernameInput');
    const regEmailInput = document.getElementById('modalSignupEmailInput');
    const regPasswordInput = document.getElementById('modalSignupPasswordInput');
    const otpInput = document.getElementById('modalOtpInput');

    if (usernameInput) usernameInput.value = '';
    if (passwordInput) passwordInput.value = '';
    if (regUsernameInput) regUsernameInput.value = '';
    if (regEmailInput) regEmailInput.value = '';
    if (regPasswordInput) regPasswordInput.value = '';
    if (otpInput) otpInput.value = '';

    if (resendTimer) {
        clearInterval(resendTimer);
        resendTimer = null;
        resendCooldown = 0;
    }

    hideModalAlerts();
    switchModalView('signin');
}

function showModalError(message) {
    const errorBox = document.getElementById('modalAuthError');
    const errorText = document.getElementById('modalAuthErrorText');
    const successBox = document.getElementById('modalAuthSuccess');
    if (successBox) successBox.style.display = 'none';
    if (errorBox && errorText) {
        errorText.textContent = message;
        errorBox.style.display = 'flex';
    }
}

function showModalSuccess(message) {
    const successBox = document.getElementById('modalAuthSuccess');
    const successText = document.getElementById('modalAuthSuccessText');
    const errorBox = document.getElementById('modalAuthError');
    if (errorBox) errorBox.style.display = 'none';
    if (successBox && successText) {
        successText.textContent = message;
        successBox.style.display = 'flex';
    }
}

function hideModalAlerts() {
    const errorBox = document.getElementById('modalAuthError');
    const successBox = document.getElementById('modalAuthSuccess');
    if (errorBox) errorBox.style.display = 'none';
    if (successBox) successBox.style.display = 'none';
}

// 4. Authenticate & Role Routing
export async function authenticate(username, password, explicitRole) {
    const rawUser = (username || '').trim();
    const rawPass = (password || '').trim();

    const submitBtn = document.getElementById('modalSubmitBtn');
    const originalBtnContent = submitBtn ? submitBtn.innerHTML : '';
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Verifying...</span>';
    }

    const errorBox = document.getElementById('modalAuthError');
    if (errorBox) errorBox.style.display = 'none';

    try {
        // --- PATH A: Supabase Authentication ---
        if (isSupabaseConfigured()) {
            const res = await loginWithSupabase(rawUser, rawPass);
            if (!res.success) {
                showModalError(res.error || 'Invalid username or password.');
                return false;
            }

            const dbUser = res.user;
            const activeRole = (dbUser.role === 'developer') ? 'developer' : 'user';

            setAuth(activeRole, dbUser.username, {
                id: dbUser.id,
                email: dbUser.email,
                created_at: dbUser.created_at,
                is_demo: false
            });
            applyRoleDisplay(activeRole);
            closeAuthModal();

            if (explicitRole && explicitRole !== activeRole) {
                notify(`Signed in as ${activeRole.toUpperCase()} (account is registered as ${activeRole})`, 'info');
            } else {
                notify(`Signed in as ${activeRole === 'developer' ? 'Developer (Full Suite)' : 'User (Inference Studio)'}`, 'success');
            }

            // If on login.html, route to index.html
            if (window.location.pathname.endsWith('login.html')) {
                window.location.href = 'index.html';
            }
            return true;
        }

        // --- PATH B: Local Demo Mode Fallback (when Supabase keys not set yet) ---
        const u = rawUser.toLowerCase();
        let matchedRole = explicitRole;
        if (!matchedRole) {
            if (u === DEMO_CONFIG.developer.username) matchedRole = 'developer';
            else if (u === DEMO_CONFIG.user.username) matchedRole = 'user';
            else matchedRole = currentModalRole;
        }

        const cfg = DEMO_CONFIG[matchedRole];
        if (cfg && u === cfg.username.toLowerCase() && rawPass === cfg.password) {
            setAuth(cfg.role, cfg.username, {
                is_demo: true,
                email: `${cfg.username}@sanada.ai`
            });
            applyRoleDisplay(cfg.role);
            closeAuthModal();

            notify(`Signed in as ${cfg.label} (Demo fallback — configure Supabase in frontend/js/services/supabase.js)`, 'info');

            if (window.location.pathname.endsWith('login.html')) {
                window.location.href = 'index.html';
            }
            return true;
        }

        showModalError('Supabase is not configured yet. Please paste your SUPABASE_URL and SUPABASE_ANON_KEY into frontend/js/services/supabase.js (or use developer / dev123 for demo mode).');
        return false;
    } catch (err) {
        console.error('[SANADA Auth] Unexpected error during authentication:', err);
        showModalError(err.message || 'Authentication error. Please try again.');
        return false;
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnContent;
        }
    }
}

// 5. Logout Confirmation & Action
export function openLogoutModal() {
    const modal = document.getElementById('sanadaLogoutModal');
    if (!modal) {
        if (confirm('Are you sure you want to log out of SANADA?')) {
            logout();
        }
        return;
    }
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
}

export function closeLogoutModal() {
    const modal = document.getElementById('sanadaLogoutModal');
    if (!modal) return;
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
    const authModal = document.getElementById('sanadaAuthModal');
    if (!authModal || !authModal.classList.contains('open')) {
        document.body.style.overflow = '';
    }
}

export function logout() {
    clearAuth();
    applyRoleDisplay('unauthenticated');
    closeAuthModal();
    closeLogoutModal();

    notify('Logged out. Returned to landing page.', 'info');

    // Route to home on SPA
    if (typeof window !== 'undefined') {
        if (window.location.pathname.endsWith('user.html') || window.location.pathname.endsWith('login.html')) {
            window.location.href = 'index.html#/home';
        } else {
            window.location.hash = '#/home';
            if (typeof window.switchRoute === 'function') {
                window.switchRoute('home');
            }
        }
    }
}

// 6. DOM Event Bindings
function initAuthGuard() {
    // Sync current session state with DOM
    const current = getAuth();
    applyRoleDisplay(current ? current.role : 'unauthenticated');

    // Hero Login button
    const heroLoginBtn = document.getElementById('heroLoginBtn');
    if (heroLoginBtn) {
        heroLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openAuthModal('user');
        });
    }

    // Hero Signup button (Opens Create Account popup)
    const heroSignupBtn = document.getElementById('heroSignupBtn');
    if (heroSignupBtn) {
        heroSignupBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openSignupModal();
        });
    }

    // Top Navbar Login button
    const navLoginBtn = document.getElementById('navLoginBtn');
    if (navLoginBtn) {
        navLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openAuthModal('user');
        });
    }

    // Top Navbar Signup button
    const navSignupBtn = document.getElementById('navSignupBtn');
    if (navSignupBtn) {
        navSignupBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openSignupModal();
        });
    }

    // Auto-open signup modal on standalone signup page or #/signup hash
    if (
        window.location.pathname.endsWith('signup.html') ||
        window.location.hash === '#/signup' ||
        window.location.search.includes('signup')
    ) {
        openSignupModal();
    }

    // Logout buttons: ask confirmation before logging out
    const logoutBtns = document.querySelectorAll('#logoutBtn, .btn-logout, .logout-link, .mobile-menu-logout, #mobileMenuLogoutBtn');
    logoutBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            // Close mobile menu if open
            const navTabs = document.querySelector('.navbar .nav-tabs');
            const navToggle = document.getElementById('navToggle');
            if (navTabs) navTabs.classList.remove('open');
            if (navToggle) {
                navToggle.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
            }
            openLogoutModal();
        });
    });

    // Logout Confirmation Modal controls
    const logoutCancelBtn = document.getElementById('modalLogoutCancelBtn');
    logoutCancelBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
    });

    const logoutCloseBtn = document.getElementById('modalLogoutCloseBtn');
    logoutCloseBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
    });

    const logoutConfirmBtn = document.getElementById('modalLogoutConfirmBtn');
    logoutConfirmBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
        logout();
    });

    const logoutModal = document.getElementById('sanadaLogoutModal');
    if (logoutModal) {
        logoutModal.addEventListener('click', (e) => {
            if (e.target === logoutModal) {
                closeLogoutModal();
            }
        });
    }

    // Modal Close button
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            closeAuthModal();
        });
    }

    // Close modal if user clicks Home tab or brand in navbar
    const homeTab = document.querySelector('.navbar .nav-tabs .tab-link[data-route="home"]');
    homeTab?.addEventListener('click', () => closeAuthModal());
    const brandLink = document.querySelector('.navbar .brand');
    brandLink?.addEventListener('click', () => closeAuthModal());

    // Modal Backdrop click
    const modal = document.getElementById('sanadaAuthModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeAuthModal();
            }
        });
    }

    // Escape key listener for both modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (logoutModal && logoutModal.classList.contains('open')) {
                closeLogoutModal();
            } else if (modal && modal.classList.contains('open')) {
                closeAuthModal();
            }
        }
    });

    // Dual Role selector cards in modal
    const devCard = document.getElementById('modalRoleDevBtn');
    const userCard = document.getElementById('modalRoleUserBtn');
    if (devCard) {
        devCard.addEventListener('click', () => setModalActiveRole('developer'));
    }
    if (userCard) {
        userCard.addEventListener('click', () => setModalActiveRole('user'));
    }

    // Modal Login Form submission
    const modalForm = document.getElementById('modalLoginForm');
    const modalUserInput = document.getElementById('modalUsernameInput');
    const modalPassInput = document.getElementById('modalPasswordInput');

    // Auto-clear error when user types
    modalUserInput?.addEventListener('input', () => {
        const errorBox = document.getElementById('modalAuthError');
        if (errorBox) errorBox.style.display = 'none';
    });
    modalPassInput?.addEventListener('input', () => {
        const errorBox = document.getElementById('modalAuthError');
        if (errorBox) errorBox.style.display = 'none';
    });

    if (modalForm) {
        modalForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const u = modalUserInput?.value;
            const p = modalPassInput?.value;
            await authenticate(u, p, currentModalRole);
        });
    }

    // Password reveal toggles (guaranteed single-execution via direct onclick)
    const passToggle = document.getElementById('modalPassToggle');
    if (passToggle) {
        passToggle.onclick = (e) => {
            if (e) { e.preventDefault(); e.stopPropagation(); }
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility('modalPasswordInput', passToggle);
            }
        };
    }

    // Switch between Sign In and Create Account
    const switchToSignupBtn = document.getElementById('modalSwitchToSignupBtn');
    switchToSignupBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        switchModalView('signup');
    });

    const switchToSigninBtn = document.getElementById('modalSwitchToSigninBtn');
    switchToSigninBtn?.addEventListener('click', (e) => {
        if (window.location.pathname.endsWith('signup.html')) {
            window.location.href = 'login.html';
        } else {
            e.preventDefault();
            switchModalView('signin');
        }
    });

    const otpBackBtn = document.getElementById('modalOtpBackBtn');
    otpBackBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        switchModalView('signup');
        const sBtn = document.getElementById('modalSignupSubmitBtn');
        if (sBtn) {
            sBtn.disabled = false;
            sBtn.innerHTML = '<span>Send Verification Code (User)</span>';
        }
    });

    // Signup password reveal toggle
    const regPassToggle = document.getElementById('modalSignupPassToggle');
    if (regPassToggle) {
        regPassToggle.onclick = (e) => {
            if (e) { e.preventDefault(); e.stopPropagation(); }
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility('modalSignupPasswordInput', regPassToggle);
            }
        };
    }

    // Auto-clear alerts when user types in signup & OTP fields
    ['modalSignupUsernameInput', 'modalSignupEmailInput', 'modalSignupPasswordInput', 'modalOtpInput'].forEach(id => {
        const el = document.getElementById(id);
        el?.addEventListener('input', () => {
            hideModalAlerts();
        });
    });

    // Create Account Form submission (Sign Up -> send OTP)
    const signupForm = document.getElementById('modalSignupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideModalAlerts();

            const uInput = document.getElementById('modalSignupUsernameInput');
            const eInput = document.getElementById('modalSignupEmailInput');
            const pInput = document.getElementById('modalSignupPasswordInput');
            const submitBtn = document.getElementById('modalSignupSubmitBtn');

            const defaultSignupBtnHtml = '<span>Send Verification Code (User)</span>';
            const resetSignupBtn = () => {
                const btn = document.getElementById('modalSignupSubmitBtn');
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = defaultSignupBtnHtml;
                }
            };

            const username = (uInput?.value || '').trim();
            const email = (eInput?.value || '').trim().toLowerCase();
            const password = (pInput?.value || '').trim();

            if (!username) {
                showModalError('Please choose a username.');
                resetSignupBtn();
                uInput?.focus();
                return;
            }
            if (username.length < 3) {
                showModalError('Username must be at least 3 characters.');
                resetSignupBtn();
                uInput?.focus();
                return;
            }
            if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
                showModalError('Username can only contain letters, numbers, underscores, and hyphens.');
                resetSignupBtn();
                uInput?.focus();
                return;
            }
            if (!email || !email.includes('@') || !email.includes('.')) {
                showModalError('Please enter a valid email address.');
                resetSignupBtn();
                eInput?.focus();
                return;
            }
            if (!password || password.length < 6) {
                showModalError('Password must be at least 6 characters long.');
                resetSignupBtn();
                pInput?.focus();
                return;
            }

            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span>Sending Code...</span>';
            }

            // Immediately record pending signup data so OTP form always has the target email
            pendingSignup = { username, email, password };
            try {
                sessionStorage.setItem('sanada_pending_signup', JSON.stringify(pendingSignup));
            } catch (sErr) {}

            try {
                if (!isSupabaseConfigured()) {
                    showModalError('Supabase is not configured yet. Paste your credentials into frontend/js/services/supabase.js');
                    resetSignupBtn();
                    return;
                }

                const res = await signUpWithSupabase(username, email, password);
                if (!res.success) {
                    showModalError(res.error || 'Failed to send verification code. Please try again.');
                    resetSignupBtn();
                    if (res.error && res.error.toLowerCase().includes('username')) {
                        uInput?.focus();
                    } else if (res.error && res.error.toLowerCase().includes('email')) {
                        eInput?.focus();
                    }
                    return;
                }

                pendingSignup = { username, email, password };
                try {
                    sessionStorage.setItem('sanada_pending_signup', JSON.stringify(pendingSignup));
                } catch (sErr) {}

                switchModalView('otp');
                showModalSuccess(`A 6-digit verification code was sent to ${email}.`);
                startResendCooldown(60);
            } catch (err) {
                console.error('[SANADA Auth] Signup error:', err);
                showModalError(err.message || 'An unexpected error occurred during account creation.');
                resetSignupBtn();
            } finally {
                resetSignupBtn();
            }
        });
    }

    // Resend OTP button
    const otpResendBtn = document.getElementById('modalOtpResendBtn');
    otpResendBtn?.addEventListener('click', async (e) => {
        e.preventDefault();
        if (resendCooldown > 0) return;
        hideModalAlerts();

        const btn = document.getElementById('modalOtpResendBtn');
        if (btn) btn.textContent = 'Sending...';

        let resendEmail = (pendingSignup.email || '').trim().toLowerCase();
        if (!resendEmail) {
            try {
                const stored = sessionStorage.getItem('sanada_pending_signup');
                if (stored) resendEmail = (JSON.parse(stored).email || '').trim().toLowerCase();
            } catch (e) {}
        }
        if (!resendEmail) {
            resendEmail = (document.getElementById('modalSignupEmailInput')?.value || '').trim().toLowerCase();
        }

        if (!resendEmail) {
            showModalError('Missing email address. Please click "← Edit Details" to enter your email.');
            if (btn) btn.textContent = 'Resend Code';
            return;
        }

        try {
            const res = await resendOtpWithSupabase(resendEmail);
            if (!res.success) {
                showModalError(res.error || 'Failed to resend code.');
                if (btn) btn.textContent = 'Resend Code';
            } else {
                showModalSuccess(`A new 6-digit code has been sent to ${resendEmail}.`);
                startResendCooldown(60);
            }
        } catch (err) {
            showModalError(err.message || 'Failed to resend code.');
            if (btn) btn.textContent = 'Resend Code';
        }
    });

    // OTP Form submission (Verify OTP -> Activate -> Sign In)
    const otpForm = document.getElementById('modalOtpForm');
    if (otpForm) {
        otpForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideModalAlerts();

            const otpInput = document.getElementById('modalOtpInput');
            const submitBtn = document.getElementById('modalOtpSubmitBtn');
            const token = (otpInput?.value || '').trim();

            const defaultOtpBtnHtml = '<span>Verify &amp; Create Account</span>';
            const resetOtpBtn = () => {
                const btn = document.getElementById('modalOtpSubmitBtn');
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = defaultOtpBtnHtml;
                }
            };

            if (!token) {
                showModalError('Please enter the 6-digit verification code.');
                resetOtpBtn();
                otpInput?.focus();
                return;
            }

            // Multi-layer fallback to guarantee targetEmail is never missing
            let targetEmail = (pendingSignup.email || '').trim().toLowerCase();
            let targetUser = (pendingSignup.username || '').trim();
            let targetPass = (pendingSignup.password || '').trim();

            if (!targetEmail) {
                try {
                    const stored = sessionStorage.getItem('sanada_pending_signup');
                    if (stored) {
                        const parsed = JSON.parse(stored);
                        targetEmail = (parsed.email || '').trim().toLowerCase();
                        targetUser = targetUser || (parsed.username || '').trim();
                        targetPass = targetPass || (parsed.password || '').trim();
                    }
                } catch (e) {}
            }

            if (!targetEmail) {
                targetEmail = (document.getElementById('modalSignupEmailInput')?.value || '').trim().toLowerCase();
                targetUser = targetUser || (document.getElementById('modalSignupUsernameInput')?.value || '').trim();
                targetPass = targetPass || (document.getElementById('modalSignupPasswordInput')?.value || '').trim();
            }

            pendingSignup = { username: targetUser, email: targetEmail, password: targetPass };

            if (!targetEmail) {
                showModalError('Missing email address. Please click "← Edit Details" to enter your email.');
                resetOtpBtn();
                return;
            }

            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span>Verifying Code...</span>';
            }

            try {
                const res = await verifyOtpWithSupabase(
                    targetEmail,
                    token,
                    targetUser,
                    targetPass
                );

                if (!res.success) {
                    showModalError(res.error || 'Invalid or expired verification code.');
                    resetOtpBtn();
                    return;
                }

                const user = res.user;
                const username = user?.username || targetUser || 'user';
                const role = 'user';

                setAuth(role, username, { id: user?.id, email: targetEmail });
                applyRoleDisplay(role);
                closeAuthModal();

                notify(`Account verified successfully! Welcome to SANADA, ${username}.`, 'success');

                // Clear pending signup data
                pendingSignup = { username: '', email: '', password: '' };
                try {
                    sessionStorage.removeItem('sanada_pending_signup');
                } catch (e) {}

                if (window.location.pathname.endsWith('login.html') || window.location.pathname.endsWith('signup.html')) {
                    window.location.href = 'index.html';
                }
            } catch (err) {
                console.error('[SANADA Auth] OTP verification error:', err);
                showModalError(err.message || 'Verification failed. Please try again.');
                resetOtpBtn();
            } finally {
                resetOtpBtn();
            }
        });
    }
}

// Make functions accessible globally for plain script tags or inline calls
if (typeof window !== 'undefined') {
    window.SANADA_AUTH = {
        getAuth,
        setAuth,
        clearAuth,
        applyRoleDisplay,
        openAuthModal,
        openSignupModal,
        closeAuthModal,
        switchModalView,
        openLogoutModal,
        closeLogoutModal,
        startResendCooldown,
        authenticate,
        logout,
        isSupabaseConfigured,
        DEMO_CONFIG
    };
    window.logout = logout;
    window.openAuthModal = openAuthModal;
    window.openSignupModal = openSignupModal;
    window.closeAuthModal = closeAuthModal;
    window.switchModalView = switchModalView;
    window.openLogoutModal = openLogoutModal;
    window.closeLogoutModal = closeLogoutModal;
    window.startResendCooldown = startResendCooldown;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAuthGuard);
    } else {
        initAuthGuard();
    }
}
