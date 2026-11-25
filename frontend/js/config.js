/**
 * frontend/js/config.js
 * =====================
 * Global runtime configuration and audio processing constraints.
 * Supports same-origin Web/Nginx hosting and Capacitor Android APK native bridging.
 */

export const PRODUCTION_API_URL = '';
export const MOBILE_FALLBACK_API_URL = 'https://sanada-api.azurewebsites.net';

function resolveApiBase() {
    let base = '';
    if (typeof window !== 'undefined' && window.SANADA_API_BASE) {
        base = window.SANADA_API_BASE;
    } else {
        try {
            if (typeof window !== 'undefined' && window.localStorage) {
                const stored = localStorage.getItem('SANADA_API_BASE');
                if (stored) base = stored;
            }
        } catch (e) {
            // Ignore localStorage security/sandbox errors
        }
    }
    const isNativeMobile = typeof window !== 'undefined' && (
        window.location.protocol === 'capacitor:' ||
        window.location.protocol === 'file:' ||
        Boolean(window.Capacitor?.isNativePlatform?.())
    );
    if (!base && isNativeMobile) {
        base = window.SANADA_MOBILE_API_URL || MOBILE_FALLBACK_API_URL;
    }
    if (!base) {
        base = PRODUCTION_API_URL;
    }
    return (base || '').replace(/\/+$/, '');
}

export const API_BASE = resolveApiBase();

export const TARGET_SR = 16000;
export const EXPECTED_FRAMES = 128000;
export const EXPECTED_DURATION = 8.0;
