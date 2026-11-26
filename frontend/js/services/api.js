/**
 * frontend/js/services/api.js
 * ===========================
 * REST communication service for SANADA FastAPI backend.
 */

import { API_BASE } from '../config.js';

async function _throwBackendError(res) {
    let detail = `HTTP ${res.status}`;
    try {
        const errJson = await res.json();
        if (errJson?.detail) detail = `${detail}: ${errJson.detail}`;
    } catch (_) {}
    throw new Error(`Inference request failed (${detail})`);
}

export const API = {
    async getHealth() {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
        return await res.json();
    },

    async validateAudio(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(`Upload validation failed: HTTP ${res.status}`);
        return await res.json();
    },

    async predictAudio(file, task = 'all', mode = 'auto', all_4_stems = false, threshold = null) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('task', task);
        formData.append('mode', mode);
        formData.append('all_4_stems', all_4_stems ? 'true' : 'false');
        if (typeof threshold === 'number' && !Number.isNaN(threshold)) {
            formData.append('threshold', threshold.toString());
        }
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) await _throwBackendError(res);
        return await res.json();
    },

    async predictLiveAudio(file, all_4_stems = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('all_4_stems', all_4_stems ? 'true' : 'false');
        const res = await fetch(`${API_BASE}/api/predict/live`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) await _throwBackendError(res);
        return await res.json();
    },

    async getStats() {
        const res = await fetch(`${API_BASE}/api/stats`);
        if (!res.ok) throw new Error(`Failed to load analytics: HTTP ${res.status}`);
        return await res.json();
    },

    async triggerTraining(config) {
        const res = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error(`Training trigger failed: HTTP ${res.status}`);
        return await res.json();
    },

    async pollTrainingJob(jobId) {
        const res = await fetch(`${API_BASE}/api/train/${jobId}`);
        if (!res.ok) throw new Error(`Job polling failed: HTTP ${res.status}`);
        return await res.json();
    },

    async saveModel(task, versionTag, notes) {
        const res = await fetch(`${API_BASE}/api/save-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task,
                version_tag: versionTag,
                notes
            })
        });
        if (!res.ok) throw new Error(`Save model failed: HTTP ${res.status}`);
        return await res.json();
    },

    getStemUrl(relativeUrl) {
        if (!relativeUrl) return '';
        if (relativeUrl.startsWith('data:') || relativeUrl.startsWith('blob:') || relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://')) return relativeUrl;
        const clean = relativeUrl.startsWith('/') ? relativeUrl.slice(1) : relativeUrl;
        if (!clean.includes('/')) {
            return `${API_BASE}/static/demos/${clean}`;
        }
        return `${API_BASE}/${clean}`;
    }
};
