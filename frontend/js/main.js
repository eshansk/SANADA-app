/**
 * frontend/js/main.js
 * ====================
 * Master Application Entry Point.
 * Sole orchestrator and single boot execution for the SANADA web platform.
 * Wires router, UI sub-systems, page handlers, audio engines, and telemetry.
 */

import { API_BASE } from './config.js';
import { API } from './services/api.js';
import { showToast } from './ui/toast.js';

if (typeof window !== 'undefined') {
    window.showToast = showToast;
}
import { initRouter, switchRoute } from './ui/router.js';
import { initHomeVideo, updateWaveVisibility } from './ui/homeVideo.js?v=6';
import { initCardTilt } from './ui/tilt.js?v=18';
import { initHomePage } from './pages/home.js';
import { initStudioPage } from './pages/studio.js?v=3';
import { initResultsPage, loadAnalytics } from './pages/results.js';
import { initDeveloperPage } from './pages/developer.js';
import { initApiPage } from './pages/api.js';
import { initPaperPage } from './pages/paper.js';
import { initAccountPage, updateAccountProfile } from './pages/account.js';

async function checkHealth() {
    const backendDot = document.getElementById('backendDot');
    const backendText = document.getElementById('backendText');
    const studioOfflineNotice = document.getElementById('studioOfflineNotice');

    try {
        const data = await API.getHealth();
        if (backendDot) backendDot.className = 'status-dot online';
        if (backendText) backendText.textContent = `Engine online (${data.device.toUpperCase()})`;
        if (studioOfflineNotice) studioOfflineNotice.style.display = 'none';
        return true;
    } catch (e) {
        if (backendDot) backendDot.className = 'status-dot';
        if (backendText) backendText.textContent = 'Engine offline';
        if (studioOfflineNotice) studioOfflineNotice.style.display = 'flex';
        return false;
    }
}

export function syncNavTabsPosition() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    const navInner = navbar.querySelector('.nav-inner');
    const navTabs = document.querySelector('.nav-tabs');
    const navStatus = navInner ? navInner.querySelector('.nav-status') : null;
    if (!navInner || !navTabs) return;

    if (window.innerWidth <= 1100) {
        if (navTabs.parentElement !== navbar) {
            navbar.appendChild(navTabs);
        }
    } else {
        if (navTabs.parentElement !== navInner) {
            if (navStatus) {
                navInner.insertBefore(navTabs, navStatus);
            } else {
                navInner.appendChild(navTabs);
            }
        }
    }
}

function initMobileNav() {
    syncNavTabsPosition();
    window.addEventListener('resize', syncNavTabsPosition);
    window.addEventListener('orientationchange', syncNavTabsPosition);

    const navToggle = document.getElementById('navToggle');
    if (!navToggle) return;

    navToggle.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        syncNavTabsPosition();
        const navTabs = document.querySelector('.nav-tabs');
        if (!navTabs) return;
        const isOpen = navTabs.classList.toggle('open');
        navToggle.classList.toggle('active', isOpen);
        navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });

    document.addEventListener('click', (e) => {
        const navTabs = document.querySelector('.nav-tabs');
        if (!navTabs) return;

        if (e.target.closest('.nav-tabs .tab-link')) {
            navTabs.classList.remove('open');
            navToggle.classList.remove('active');
            navToggle.setAttribute('aria-expanded', 'false');
            return;
        }

        if (!e.target.closest('.navbar')) {
            navTabs.classList.remove('open');
            navToggle.classList.remove('active');
            navToggle.setAttribute('aria-expanded', 'false');
        }
    });
}

function initPingButton() {
    const pingBtn = document.getElementById('pingBtn');
    if (!pingBtn) return;

    pingBtn.addEventListener('click', async () => {
        const ok = await checkHealth();
        if (ok) {
            showToast('Backend API is healthy and reachable!', 'success');
        } else {
            showToast(`Cannot connect to ${API_BASE}. Start backend/app.py.`, 'warning');
        }
    });
}

function initVisibilityControls() {
    if (typeof document === 'undefined') return;
    document.addEventListener('visibilitychange', () => {
        document.body.classList.toggle('doc-hidden', document.hidden);
    });
}

function initMobileTouchFeedback() {
    if (typeof document === 'undefined') return;

    let touchTimers = new Map();

    // Enable CSS :active & .is-tapped smooth tap feedback on mobile devices
    document.addEventListener('touchstart', (e) => {
        const btn = e.target.closest('.btn, .btn-primary, .btn-run-inference, .tab-link, .nav-toggle, .btn-nav-login, .btn-nav-signup, #heroSignupBtn, #heroLoginBtn, #heroOpenStudioBtn, #heroReadPaperBtn, #pingBtn, .btn-modal-submit, .btn-modal-cancel, .btn-ctrl, .preset-chip-btn');
        if (btn) {
            if (touchTimers.has(btn)) {
                clearTimeout(touchTimers.get(btn));
                touchTimers.delete(btn);
            }
            btn.classList.add('is-tapped');
        }
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const btn = e.target.closest('.btn, .btn-primary, .btn-run-inference, .tab-link, .nav-toggle, .btn-nav-login, .btn-nav-signup, #heroSignupBtn, #heroLoginBtn, #heroOpenStudioBtn, #heroReadPaperBtn, #pingBtn, .btn-modal-submit, .btn-modal-cancel, .btn-ctrl, .preset-chip-btn');
        if (btn) {
            const timer = setTimeout(() => {
                btn.classList.remove('is-tapped');
                touchTimers.delete(btn);
            }, 450);
            touchTimers.set(btn, timer);
        }
    }, { passive: true });

    document.addEventListener('touchcancel', () => {
        touchTimers.forEach(t => clearTimeout(t));
        touchTimers.clear();
        document.querySelectorAll('.is-tapped').forEach(el => el.classList.remove('is-tapped'));
    }, { passive: true });
}

export function boot() {
    // 1. Initialize Single-Page Router & sync route changes
    initRouter((route) => {
        updateWaveVisibility();
        if (route === 'analytics') {
            loadAnalytics();
        } else if (route === 'account') {
            updateAccountProfile();
        }
    });

    // 2. Initialize Page Controllers
    initHomePage();
    initStudioPage();
    initResultsPage();
    initDeveloperPage();
    initApiPage();
    initPaperPage();
    initAccountPage();

    // 3. Initialize Shared Navigation & Controls
    initMobileNav();
    initPingButton();
    initVisibilityControls();
    initMobileTouchFeedback();
    initCardTilt();

    // 4. Initialize Background Visual Video
    initHomeVideo('ctaVideo');

    // 5. Initial Vitality Telemetry Check
    checkHealth();
}

// Single Entry Point Initialization
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
}
