/**
 * frontend/js/config.js
 * =====================
 * Global runtime configuration and audio processing constraints.
 * Centralizes API endpoint discovery and DSP parameters.
 */

export const PRODUCTION_API_URL = '';

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
    if (!base) {
        base = PRODUCTION_API_URL;
    }
    return (base || '').replace(/\/+$/, '');
}

export const API_BASE = resolveApiBase();

export const TARGET_SR = 16000;
export const EXPECTED_FRAMES = 128000;
export const EXPECTED_DURATION = 8.0;
