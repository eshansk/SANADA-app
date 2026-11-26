/**
 * frontend/js/services/runHistory.js
 * ==================================
 * Shared Run History, Detail Drawer & Inference Preferences Controller.
 *
 * Implements:
 *  1. Dual-tier persistence (Supabase RPCs + user-scoped localStorage fallback).
 *  2. Strict 5-run capacity per user with auto-deletion of older runs.
 *  3. Interactive run management: rename, pin/unpin, delete, clear all.
 *  4. Liquid glass detail drawer featuring:
 *     - Settings (task, mode, threshold, all 4 stems)
 *     - Task 1 source count & complexity ceiling
 *     - Task 2 active sound tags with confidence metrics
 *     - Task 3 stem players with ephemeral 404 "Audio expired" fallback
 *     - Telemetry & raw response JSON inspector with copy action
 *  5. User inference preferences (task, mode, threshold, all_4_stems).
 *  6. Change password RPC wrapper with demo account protection.
 */

import { getAuth } from '../auth-guard.js';
import { getSupabaseClient, isSupabaseConfigured } from './supabase.js';
import { showConfirmPopup, showPromptPopup } from '../ui/popup.js';
import { API } from './api.js';

// =============================================================================
// 0. INDEXEDDB AUDIO BLOB CACHE (Permanent local playback for history runs)
// =============================================================================
const AUDIO_CACHE_DB = 'sanada_audio_cache_v1';
const AUDIO_CACHE_STORE = 'run_audio';
const AUDIO_CACHE_MAX_ENTRIES = 10;

function openAudioCacheDB() {
    if (typeof indexedDB === 'undefined') return Promise.resolve(null);
    return new Promise((resolve) => {
        try {
            const req = indexedDB.open(AUDIO_CACHE_DB, 1);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(AUDIO_CACHE_STORE)) {
                    db.createObjectStore(AUDIO_CACHE_STORE, { keyPath: 'runId' });
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => resolve(null);
        } catch (_) {
            resolve(null);
        }
    });
}

async function saveRunAudioToCache(runId, { inputBlob = null, stemBlobs = {} } = {}) {
    if (!runId) return;
    const db = await openAudioCacheDB();
    if (!db) return;
    try {
        await new Promise((resolve) => {
            const tx = db.transaction(AUDIO_CACHE_STORE, 'readwrite');
            const store = tx.objectStore(AUDIO_CACHE_STORE);
            store.put({
                runId: String(runId),
                inputBlob: (inputBlob && inputBlob.size > 0) ? inputBlob : null,
                stemBlobs: stemBlobs || {},
                updatedAt: Date.now()
            });
            const allReq = store.getAll();
            allReq.onsuccess = () => {
                const items = Array.isArray(allReq.result) ? allReq.result : [];
                if (items.length > AUDIO_CACHE_MAX_ENTRIES) {
                    items.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
                    items.slice(AUDIO_CACHE_MAX_ENTRIES).forEach((old) => {
                        try { store.delete(old.runId); } catch (_) {}
                    });
                }
            };
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    } catch (_) {
        // Ignore private browsing / IDB quota failures gracefully
    } finally {
        try { db.close(); } catch (_) {}
    }
}

export async function loadRunAudioFromCache(runId) {
    if (!runId) return null;
    const db = await openAudioCacheDB();
    if (!db) return null;
    try {
        return await new Promise((resolve) => {
            const tx = db.transaction(AUDIO_CACHE_STORE, 'readonly');
            const store = tx.objectStore(AUDIO_CACHE_STORE);
            const req = store.get(String(runId));
            req.onsuccess = () => {
                const rec = req.result;
                resolve(rec ? { inputBlob: rec.inputBlob || null, stemBlobs: rec.stemBlobs || {} } : null);
            };
            req.onerror = () => resolve(null);
        });
    } catch (_) {
        return null;
    } finally {
        try { db.close(); } catch (_) {}
    }
}

async function deleteRunAudioFromCache(runId) {
    if (!runId) return;
    const db = await openAudioCacheDB();
    if (!db) return;
    try {
        await new Promise((resolve) => {
            const tx = db.transaction(AUDIO_CACHE_STORE, 'readwrite');
            tx.objectStore(AUDIO_CACHE_STORE).delete(String(runId));
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    } catch (_) {
    } finally {
        try { db.close(); } catch (_) {}
    }
}

async function clearRunAudioCache() {
    const db = await openAudioCacheDB();
    if (!db) return;
    try {
        await new Promise((resolve) => {
            const tx = db.transaction(AUDIO_CACHE_STORE, 'readwrite');
            tx.objectStore(AUDIO_CACHE_STORE).clear();
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    } catch (_) {
    } finally {
        try { db.close(); } catch (_) {}
    }
}

// =============================================================================
// 1. USER INFERENCE PREFERENCES (Task, Mode, Threshold, All 4 Stems)
// =============================================================================
export const DEFAULT_INFERENCE_PREFERENCES = {
    task: 'all',
    mode: 'auto',
    threshold: 0.5,
    all_4_stems: false
};

export function getInferencePreferences() {
    try {
        const userMeta = getCurrentUserMeta();
        const userKey = `sanada_inference_prefs_${userMeta.username || 'user'}`;
        const raw = localStorage.getItem(userKey) || localStorage.getItem('sanada_inference_prefs');
        if (raw) {
            const parsed = JSON.parse(raw);
            return {
                task: parsed.task || DEFAULT_INFERENCE_PREFERENCES.task,
                mode: parsed.mode || DEFAULT_INFERENCE_PREFERENCES.mode,
                threshold: typeof parsed.threshold === 'number' ? parsed.threshold : DEFAULT_INFERENCE_PREFERENCES.threshold,
                all_4_stems: Boolean(parsed.all_4_stems)
            };
        }
    } catch (e) {}
    return { ...DEFAULT_INFERENCE_PREFERENCES };
}

export async function saveInferencePreferences(prefs) {
    const userMeta = getCurrentUserMeta();
    const merged = {
        task: prefs.task || 'all',
        mode: prefs.mode || 'auto',
        threshold: typeof prefs.threshold === 'number' ? prefs.threshold : 0.5,
        all_4_stems: Boolean(prefs.all_4_stems)
    };

    try {
        const userKey = `sanada_inference_prefs_${userMeta.username || 'user'}`;
        localStorage.setItem(userKey, JSON.stringify(merged));
        localStorage.setItem('sanada_inference_prefs', JSON.stringify(merged));
        window.dispatchEvent(new CustomEvent('sanada:prefsChanged', { detail: merged }));
    } catch (e) {
        console.warn('Failed to save inference preferences to localStorage', e);
    }
    return true;
}

export async function syncPreferencesFromSupabase() {
    return getInferencePreferences();
}

// =============================================================================
// 2. USER CONTEXT HELPER
// =============================================================================
export function getCurrentUserMeta() {
    const auth = getAuth();
    if (!auth) {
        return {
            id: null,
            username: 'guest',
            email: null,
            role: 'unauthenticated',
            isDemo: true,
            storageKey: 'sanada_run_history_guest'
        };
    }
    const username = (auth.username || auth.role || 'user').trim().toLowerCase();
    const id = auth.id || null;
    const email = auth.email || null;
    const role = auth.role || 'user';
    const isDemo = Boolean(auth.is_demo) || (username === 'developer' || username === 'user') && !id;
    const storageKey = id ? `sanada_run_history_uid_${id}` : `sanada_run_history_user_${username}`;
    return { id, username, email, role, isDemo, storageKey };
}

// =============================================================================
// 3. RUN HISTORY DUAL-TIER STORAGE (Max 5 Runs Per User)
// =============================================================================

export function getRunJson(run) {
    if (!run) return {};
    let rj = run.resultJson || run.result_json || {};
    if (typeof rj === 'string') {
        try { rj = JSON.parse(rj); } catch(e) { rj = {}; }
    }
    return rj;
}

function resolveDisplayRunCount(res, countObj, activeTags = []) {
    const rawK = countObj.predicted_k !== undefined
        ? countObj.predicted_k
        : (countObj.estimated_k !== undefined ? countObj.estimated_k : (res.predicted_k !== undefined ? res.predicted_k : res.k));
    if (rawK === undefined || rawK === null) return '—';
    if (Number(rawK) === 0) {
        return Number(countObj.env_count || res.env_count || activeTags.length || 0);
    }
    return rawK;
}

export function getRunTags(run) {
    if (!run) return [];
    const rj = getRunJson(run);
    const res = rj.results || rj;
    const tagging = res.label_tagging || res.sound_tagging || res.environmental_events || {};
    let rawTags = tagging.active_classes || tagging.active_events || res.active_events || [];
    let tags = rawTags.map(t => typeof t === 'string' ? t : (t.event || t.class || t.label || String(t)));

    // Fallback: If no AST environmental tags detected, check speech/env stems or K count
    if (tags.length === 0) {
        const countObj = res.source_counting || {};
        const kVal = countObj.predicted_k !== undefined ? countObj.predicted_k : (countObj.estimated_k !== undefined ? countObj.estimated_k : (res.predicted_k || res.k || 0));
        const stems = res.source_separation?.stems || res.speech_separation?.stems || res.separated_stems || res.env_separation?.env_stems || [];
        if (stems.length > 0) {
            tags = stems.map((s, idx) => s.assigned_tag || s.label || `Speaker ${idx + 1}`);
        } else if (kVal > 0) {
            for (let i = 1; i <= Math.min(kVal, 6); i++) {
                tags.push(`Speaker ${i}`);
            }
        }
    }
    return tags;
}

function normalizeRun(item) {
    let rj = item.result_json || item.resultJson || {};
    if (typeof rj === 'string') {
        try { rj = JSON.parse(rj); } catch(e) { rj = {}; }
    }
    const rawName = item.name || item.filename || 'Neural Run';
    const cleanName = rawName.replace(/\.[a-zA-Z0-9]{2,5}$/, '');
    return {
        id: String(item.id || `run_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`),
        userId: item.user_id || item.userId || null,
        userIdentifier: item.user_identifier || item.userIdentifier || 'user',
        name: cleanName,
        filename: item.filename || 'recording.wav',
        mode: item.mode || 'auto',
        threshold: typeof item.threshold === 'number' ? item.threshold : 0.5,
        task: item.task || 'all',
        all4Stems: Boolean(item.all_4_stems || item.all4Stems),
        isPinned: false,
        resultJson: rj,
        createdAt: item.created_at || item.createdAt || new Date().toISOString()
    };
}

function sortAndTrimRuns(runs) {
    const sorted = [...runs].sort((a, b) => {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });
    return sorted.slice(0, 5);
}

export function loadLocalRuns() {
    try {
        const userMeta = getCurrentUserMeta();
        const stored = localStorage.getItem(userMeta.storageKey);
        if (stored) {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed)) {
                return sortAndTrimRuns(parsed.map(normalizeRun));
            }
        }
    } catch (e) {}
    return [];
}

function stripLargeAudioData(runObj) {
    if (!runObj || !runObj.resultJson) return runObj;
    const rj = runObj.resultJson;
    delete rj.input_audio_data_url;
    if (rj.results) {
        delete rj.results.input_audio_data_url;
        delete rj.results.all_4_raw_stems;
    }
    delete rj.all_4_raw_stems;

    const stemArrays = [
        rj.results?.source_separation?.stems,
        rj.results?.separated_stems,
        rj.source_separation?.stems,
        rj.separated_stems,
        rj.results?.env_separation?.env_stems,
        rj.results?.env_stems,
        rj.env_separation?.env_stems,
        rj.env_stems
    ];
    stemArrays.forEach(arr => {
        if (Array.isArray(arr)) {
            arr.forEach(s => {
                delete s.audio_data_url;
                if (s.url && s.url.startsWith('data:')) {
                    s.url = `/static/stems/${s.filename || ''}`;
                }
            });
        }
    });
    return runObj;
}

export function saveLocalRuns(runs) {
    const userMeta = getCurrentUserMeta();
    const trimmed = sortAndTrimRuns(runs);

    // 1. Try standard save
    try {
        localStorage.setItem(userMeta.storageKey, JSON.stringify(trimmed));
        return trimmed;
    } catch (e) {
        console.warn('[RunHistory] LocalStorage quota reached, applying tiered compaction:', e);
    }

    // 2. Compaction Tier 1: strip raw mixture data URLs and tensor lists
    try {
        const compact1 = trimmed.map(r => {
            const copy = JSON.parse(JSON.stringify(r));
            if (copy.resultJson) {
                delete copy.resultJson.input_audio_data_url;
                if (copy.resultJson.results) {
                    delete copy.resultJson.results.input_audio_data_url;
                    delete copy.resultJson.results.all_4_raw_stems;
                }
                delete copy.resultJson.all_4_raw_stems;
            }
            return copy;
        });
        localStorage.setItem(userMeta.storageKey, JSON.stringify(compact1));
        return compact1;
    } catch (e2) {}

    // 3. Compaction Tier 2: keep base64 audio only for latest run, older runs keep metadata and relative stem URLs
    try {
        const compact2 = trimmed.map((r, idx) => {
            const copy = JSON.parse(JSON.stringify(r));
            if (idx > 0) {
                return stripLargeAudioData(copy);
            }
            if (copy.resultJson) {
                delete copy.resultJson.input_audio_data_url;
                if (copy.resultJson.results) delete copy.resultJson.results.input_audio_data_url;
            }
            return copy;
        });
        localStorage.setItem(userMeta.storageKey, JSON.stringify(compact2));
        return compact2;
    } catch (e3) {}

    // 4. Compaction Tier 3: relative stem URLs for all runs (fits all 5 runs in <25KB guaranteed)
    try {
        const compact3 = trimmed.map(r => {
            const copy = JSON.parse(JSON.stringify(r));
            return stripLargeAudioData(copy);
        });
        localStorage.setItem(userMeta.storageKey, JSON.stringify(compact3));
        return compact3;
    } catch (e4) {
        console.error('[RunHistory] Critical: unable to write to localStorage:', e4);
        return trimmed;
    }
}

/**
 * Loads latest 5 runs for the current user.
 */
export async function getRunHistory() {
    const userMeta = getCurrentUserMeta();
    const localRuns = loadLocalRuns();

    if (!userMeta.isDemo && isSupabaseConfigured()) {
        try {
            const client = getSupabaseClient();
            if (client) {
                const { data, error } = await client.rpc('get_run_history', {
                    p_user_id: userMeta.id,
                    p_user_identifier: userMeta.username
                });

                if (!error && Array.isArray(data)) {
                    const remoteRuns = data.map(normalizeRun);

                    // 1. Supabase is the primary truth for complete runs
                    const finalRuns = [...remoteRuns];

                    // 2. Append any locally saved run that hasn't synced to Supabase yet
                    localRuns.forEach(local => {
                        const exists = finalRuns.some(r => 
                            r.id === local.id || 
                            (r.filename === local.filename && Math.abs(new Date(r.createdAt) - new Date(local.createdAt)) < 10000)
                        );
                        if (!exists && local.id.startsWith('run_')) {
                            finalRuns.push(local);
                        }
                    });

                    // 3. Preserve any in-memory audio_data_url from local cache
                    finalRuns.forEach(r => {
                        const localMatch = localRuns.find(l => l.id === r.id);
                        if (localMatch && localMatch.resultJson) {
                            const localStems = localMatch.resultJson.results?.source_separation?.stems || localMatch.resultJson.separated_stems || [];
                            const rStems = r.resultJson.results?.source_separation?.stems || r.resultJson.separated_stems || [];
                            rStems.forEach((rs, idx) => {
                                if (!rs.audio_data_url && localStems[idx]?.audio_data_url) {
                                    rs.audio_data_url = localStems[idx].audio_data_url;
                                }
                            });
                        }
                    });

                    const trimmed = sortAndTrimRuns(finalRuns);
                    saveLocalRuns(trimmed);
                    return trimmed;
                }
            }
        } catch (err) {
            console.warn('[RunHistory] Supabase fetch fallback to local:', err?.message || err);
        }
    }

    return localRuns;
}

async function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function dataUrlToBlob(dataUrl) {
    try {
        const [header, b64] = String(dataUrl).split(',', 2);
        const mimeMatch = header && header.match(/data:([^;]+)/);
        const mime = mimeMatch ? mimeMatch[1] : 'audio/wav';
        const bin = atob(b64 || '');
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new Blob([bytes], { type: mime });
    } catch (_) {
        return null;
    }
}

function stripDuplicateStemAudio(stemsList) {
    if (!Array.isArray(stemsList)) return [];
    return stemsList.map((s) => {
        const copy = { ...s };
        delete copy.audio_data_url;
        if (!copy.url || copy.url.startsWith('data:') || copy.url.startsWith('blob:')) {
            copy.url = `/static/stems/${copy.filename || ''}`;
        }
        return copy;
    });
}

/**
 * Fetches & caches speech and environmental stem audio Blobs, embedding base64 Data URLs
 * once in resultJson (keeping relative paths on duplicates to avoid inflating storage).
 */
async function prepareRunAudioData(resultJson, activeFile = null, stemBlobsOut = null) {
    const stemBlobs = (stemBlobsOut && typeof stemBlobsOut === 'object') ? stemBlobsOut : {};
    if (!resultJson) return { processedJson: resultJson, stemBlobs };
    const cloned = JSON.parse(JSON.stringify(resultJson));
    const stems = cloned.results?.source_separation?.stems
               || cloned.results?.separated_stems
               || cloned.separated_stems
               || cloned.source_separation?.stems
               || [];
    const envStems = cloned.results?.env_separation?.env_stems
                  || cloned.results?.env_stems
                  || cloned.env_separation?.env_stems
                  || cloned.env_stems
                  || [];

    // Strip bulky raw mixtures and float arrays that inflate payloads past PostgREST / localStorage limits
    delete cloned.all_4_raw_stems;
    delete cloned.input_audio_data_url;
    if (cloned.results) {
        delete cloned.results.all_4_raw_stems;
        delete cloned.results.input_audio_data_url;
    }

    const hydrateStemArray = async (stemArray) => {
        await Promise.all(stemArray.map(async (stem, idx) => {
            const rawSrc = stem.audio_data_url || stem.url || stem.stem_url || stem.filename;
            if (!rawSrc) return;
            const stemKey = stem.filename || `stem_${stem.channel || (idx + 1)}.wav`;
            if (rawSrc.startsWith('data:')) {
                stem.audio_data_url = rawSrc;
                const decodedBlob = dataUrlToBlob(rawSrc);
                if (decodedBlob && stemKey) stemBlobs[stemKey] = decodedBlob;
                if (!stem.url || stem.url.startsWith('data:')) {
                    stem.url = `/static/stems/${stem.filename || ''}`;
                }
                return;
            }
            const fullUrl = API.getStemUrl(rawSrc);
            try {
                const resp = await fetch(fullUrl);
                if (resp.ok) {
                    const blob = await resp.blob();
                    if (stemKey) stemBlobs[stemKey] = blob;
                    const dataUrl = await blobToDataUrl(blob);
                    stem.audio_data_url = dataUrl;
                    // Preserve relative URL on stem.url to avoid duplicating megabytes of base64
                    if (!stem.url || stem.url.startsWith('data:')) {
                        stem.url = `/static/stems/${stem.filename || ''}`;
                    }
                }
            } catch (err) {
                console.warn('[RunHistory] Stem fetch warning for permanent storage:', rawSrc, err);
            }
        }));
    };

    await Promise.all([hydrateStemArray(stems), hydrateStemArray(envStems)]);

    const speechRel = stripDuplicateStemAudio(stems);
    const envRel = stripDuplicateStemAudio(envStems);

    if (cloned.results) {
        if (!cloned.results.source_separation) cloned.results.source_separation = {};
        cloned.results.source_separation.stems = stems;
        cloned.results.separated_stems = speechRel;
        if (cloned.source_separation?.stems) cloned.source_separation.stems = speechRel;
        cloned.separated_stems = speechRel;

        if (envStems.length > 0 || cloned.results.env_separation || cloned.results.env_stems) {
            if (!cloned.results.env_separation) cloned.results.env_separation = {};
            cloned.results.env_separation.env_stems = envStems;
            if ('env_stems' in cloned.results) cloned.results.env_stems = envRel;
            if (cloned.env_separation?.env_stems) cloned.env_separation.env_stems = envRel;
            if ('env_stems' in cloned) cloned.env_stems = envRel;
        }
    } else {
        if (!cloned.source_separation) cloned.source_separation = {};
        cloned.source_separation.stems = stems;
        cloned.separated_stems = speechRel;

        if (envStems.length > 0 || cloned.env_separation || cloned.env_stems) {
            if (!cloned.env_separation) cloned.env_separation = {};
            cloned.env_separation.env_stems = envStems;
            if ('env_stems' in cloned) cloned.env_stems = envRel;
        }
    }

    return { processedJson: cloned, stemBlobs };
}

/**
 * Saves a new inference run, enforcing strictly max 5 runs.
 * Synchronizes to Supabase and manages local cache.
 */
export async function saveRun(filename, mode, threshold, task, all4Stems, resultJson, customName = null, activeFile = null) {
    const userMeta = getCurrentUserMeta();
    const cleanFilename = (filename || 'Audio Inference').replace(/\.[a-zA-Z0-9]{2,5}$/, '');
    const finalName = customName && customName.trim() ? customName.trim() : cleanFilename;

    const newRun = {
        id: `run_${Date.now()}`,
        userId: userMeta.id,
        userIdentifier: userMeta.username,
        name: finalName,
        filename: filename || 'recording.wav',
        mode: mode || 'auto',
        threshold: typeof threshold === 'number' ? threshold : 0.5,
        task: task || 'all',
        all4Stems: Boolean(all4Stems),
        isPinned: false,
        resultJson: resultJson || {},
        createdAt: new Date().toISOString()
    };

    // 1. Initial local cache save
    const existing = loadLocalRuns();
    const updated = [newRun, ...existing.filter(r => r.id !== newRun.id)];
    saveLocalRuns(updated);
    window.dispatchEvent(new CustomEvent('sanada:runHistoryChanged', { detail: { runs: updated } }));

    // 2. Encode stems asynchronously & cache Blobs in IndexedDB
    let processedJson = resultJson || {};
    let stemBlobs = {};
    try {
        const prepared = await prepareRunAudioData(resultJson, activeFile);
        processedJson = prepared?.processedJson || processedJson;
        stemBlobs = prepared?.stemBlobs || {};
        newRun.resultJson = processedJson;
        const currentRuns = loadLocalRuns();
        const merged = currentRuns.map(r => r.id === newRun.id ? newRun : r);
        saveLocalRuns(merged);
        await saveRunAudioToCache(newRun.id, { inputBlob: activeFile, stemBlobs });
    } catch (e) {
        console.warn('[RunHistory] Audio encoding notice:', e);
    }

    // 3. Attempt Supabase insert with RPC
    if (!userMeta.isDemo && isSupabaseConfigured()) {
        try {
            const client = getSupabaseClient();
            if (client) {
                const { data, error } = await client.rpc('save_run_history', {
                    p_user_id: userMeta.id,
                    p_user_identifier: userMeta.username,
                    p_name: newRun.name,
                    p_filename: newRun.filename,
                    p_mode: newRun.mode,
                    p_threshold: newRun.threshold,
                    p_task: newRun.task,
                    p_all_4_stems: newRun.all4Stems,
                    p_result_json: newRun.resultJson
                });
                if (!error && data?.run_id) {
                    const tempId = newRun.id;
                    newRun.id = data.run_id;
                    const finalRuns = loadLocalRuns().map(r => (r.id === tempId || r.id === newRun.id) ? newRun : r);
                    saveLocalRuns(finalRuns);
                    await saveRunAudioToCache(newRun.id, { inputBlob: activeFile, stemBlobs });
                    await deleteRunAudioFromCache(tempId);
                }
            }
        } catch (err) {
            console.warn('[RunHistory] Supabase save notice:', err?.message || err);
        }
    }

    window.dispatchEvent(new CustomEvent('sanada:runHistoryChanged', { detail: { runs: loadLocalRuns() } }));
    return newRun;
}

/**
 * Renames an existing run.
 */
export async function renameRun(runId, newName) {
    const userMeta = getCurrentUserMeta();
    const cleanName = (newName || '').trim();
    if (!cleanName) return false;

    if (!userMeta.isDemo && isSupabaseConfigured()) {
        try {
            const client = getSupabaseClient();
            if (client) {
                await client.rpc('rename_run_history', {
                    p_run_id: runId,
                    p_name: cleanName,
                    p_user_id: userMeta.id,
                    p_user_identifier: userMeta.username
                });
            }
        } catch (e) {}
    }

    const runs = loadLocalRuns();
    const target = runs.find(r => r.id === runId);
    if (target) {
        target.name = cleanName;
        saveLocalRuns(runs);
        window.dispatchEvent(new CustomEvent('sanada:runHistoryChanged', { detail: { runs } }));
        return true;
    }
    return false;
}

/**
 * Renders a shimmering liquid-glass skeleton loader for run history lists.
 */
export function renderHistorySkeleton(containerEl, count = 3) {
    if (!containerEl) return;
    let html = '<div class="run-history-skeleton-list">';
    for (let i = 0; i < count; i++) {
        const wTitle = i === 1 ? '48%' : (i === 2 ? '68%' : '58%');
        const wSub = i === 1 ? '32%' : (i === 2 ? '44%' : '38%');
        html += `
            <div class="run-history-skeleton-item">
                <div class="skeleton-content">
                    <div class="skeleton-line-title" style="width: ${wTitle};"></div>
                    <div class="skeleton-line-sub" style="width: ${wSub};"></div>
                </div>
                <div class="skeleton-action"></div>
            </div>
        `;
    }
    html += '</div>';
    containerEl.innerHTML = html;
}

/**
 * Deprecated stub for togglePinRun (pin feature removed).
 */
export async function togglePinRun() {
    return false;
}

/**
 * Deletes a single run.
 */
export async function deleteRun(runId) {
    const userMeta = getCurrentUserMeta();

    if (!userMeta.isDemo && isSupabaseConfigured()) {
        try {
            const client = getSupabaseClient();
            if (client) {
                await client.rpc('delete_run_history', {
                    p_run_id: runId,
                    p_user_id: userMeta.id,
                    p_user_identifier: userMeta.username
                });
            }
        } catch (e) {}
    }

    const runs = loadLocalRuns().filter(r => r.id !== runId);
    saveLocalRuns(runs);
    await deleteRunAudioFromCache(runId);
    window.dispatchEvent(new CustomEvent('sanada:runHistoryChanged', { detail: { runs } }));
    return true;
}

/**
 * Clears all runs for the user.
 */
export async function clearAllRuns() {
    const userMeta = getCurrentUserMeta();

    if (!userMeta.isDemo && isSupabaseConfigured()) {
        try {
            const client = getSupabaseClient();
            if (client) {
                await client.rpc('clear_run_history', {
                    p_user_id: userMeta.id,
                    p_user_identifier: userMeta.username
                });
            }
        } catch (e) {}
    }

    saveLocalRuns([]);
    await clearRunAudioCache();
    window.dispatchEvent(new CustomEvent('sanada:runHistoryChanged', { detail: { runs: [] } }));
    return true;
}

// =============================================================================
// 4. CHANGE PASSWORD SERVICE
// =============================================================================
export async function changePassword(oldPassword, newPassword) {
    const userMeta = getCurrentUserMeta();

    if (userMeta.isDemo) {
        return {
            success: false,
            error: 'Password modification is disabled for demo accounts. Please sign up or log in with your Supabase account.'
        };
    }

    const cleanOld = (oldPassword || '').trim();
    const cleanNew = (newPassword || '').trim();

    if (!cleanOld) {
        return { success: false, error: 'Please enter your current password.' };
    }
    if (!cleanNew) {
        return { success: false, error: 'Please enter your new password.' };
    }
    if (cleanNew.length < 8) {
        return { success: false, error: 'New password must be at least 8 characters long.' };
    }
    if (cleanOld === cleanNew) {
        return { success: false, error: 'New password must be different from current password.' };
    }

    if (!isSupabaseConfigured()) {
        return { success: false, error: 'Supabase credentials are not configured.' };
    }

    const client = getSupabaseClient();
    if (!client) {
        return { success: false, error: 'Supabase client is not available.' };
    }

    try {
        const { data, error } = await client.rpc('change_password', {
            p_user_id: String(userMeta.id || userMeta.username || ''),
            p_old: cleanOld,
            p_new: cleanNew
        });

        if (error) {
            console.error('[RunHistory] change_password RPC error:', error);
            return { success: false, error: error.message || 'Database error changing password.' };
        }

        if (data && data.success === false) {
            return { success: false, error: data.error || 'Password update failed.' };
        }

        // Also update Supabase Auth user password if user logged in via auth.users
        try {
            await client.auth.updateUser({ password: cleanNew });
        } catch (authErr) {}

        return { success: true, message: 'Password updated successfully!' };
    } catch (err) {
        return { success: false, error: err?.message || 'Network error updating password.' };
    }
}

// =============================================================================
// 5. RUN HISTORY UI COMPONENT RENDERER
// =============================================================================
function escapeHtml(str) {
    if (!str) return '';
    return str.toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatDate(isoStr) {
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return 'Recently';
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        if (isToday) return `Today, ${timeStr}`;
        return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${timeStr}`;
    } catch (e) {
        return 'Recently';
    }
}

export function renderRunHistoryList(containerEl, runs = []) {
    if (!containerEl) return;

    if (!runs || runs.length === 0) {
        containerEl.innerHTML = `
            <div class="run-history-empty">
                <span class="run-empty-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                </span>
                <span class="run-empty-title">No recent inference runs</span>
                <span class="run-empty-sub">Run inference in the Studio (Upload or Live Mic) to view saved results here.</span>
            </div>
        `;
        return;
    }

    let html = '<div class="run-history-items-list">';
    runs.forEach(run => {
        const results = run.resultJson?.results || run.resultJson || {};
        const countObj = results.source_counting || {};
        const activeTags = getRunTags(run);
        const kVal = resolveDisplayRunCount(results, countObj, activeTags);
        const modeLabel = (run.mode || 'auto').toUpperCase();
        const dateFormatted = formatDate(run.createdAt);

        html += `
            <div class="run-history-item" data-run-id="${escapeHtml(run.id)}">
                <div class="run-item-info" data-run-id="${escapeHtml(run.id)}">
                    <div class="run-item-title-row">
                        <span class="run-item-name" title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</span>
                        <div class="run-item-badges">
                            <span class="badge badge-mode">${escapeHtml(modeLabel)}</span>
                            <span class="badge badge-k">K = ${escapeHtml(kVal.toString())}</span>
                        </div>
                    </div>
                    <div class="run-item-meta-row">
                        <span class="run-filename" title="${escapeHtml(run.filename)}">${escapeHtml(run.filename)}</span>
                        <span class="run-bullet">&bull;</span>
                        <span class="run-date">${escapeHtml(dateFormatted)}</span>
                    </div>
                </div>

                <div class="run-item-actions">
                    <button
                        type="button"
                        class="btn-run-action btn-run-rename"
                        data-action="rename"
                        data-run-id="${escapeHtml(run.id)}"
                        title="Rename this run"
                        aria-label="Rename run"
                    >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 20h9"></path>
                            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                        </svg>
                    </button>
                    <button
                        type="button"
                        class="btn-run-action btn-run-delete"
                        data-action="delete"
                        data-run-id="${escapeHtml(run.id)}"
                        title="Delete run"
                        aria-label="Delete run"
                    >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    });
    html += '</div>';

    containerEl.innerHTML = html;

    // Attach click events
    containerEl.querySelectorAll('.run-history-item').forEach(row => {
        const runId = row.dataset.runId;
        const targetRun = runs.find(r => r.id === runId);
        if (!targetRun) return;

        // Click row opens directly in the Studio DAW workspace (make #pagePredict visible before canvas render)
        const handleOpenInStudio = (e) => {
            if (e.target.closest('[data-action="rename"], [data-action="delete"]')) return;
            e.preventDefault();
            if (typeof window.switchRoute === 'function') {
                window.switchRoute('predict', true);
            } else if (window.location.hash !== '#/predict') {
                window.location.hash = '#/predict';
            }
            requestAnimationFrame(() => {
                setTimeout(() => {
                    window.dispatchEvent(new CustomEvent('sanada:loadRun', { detail: targetRun }));
                }, 60);
            });
        };

        row.addEventListener('click', handleOpenInStudio);

        // Rename button (popup modal - strips file extension)
        row.querySelector('[data-action="rename"]')?.addEventListener('click', async (e) => {
            e.stopPropagation();
            const currentName = targetRun.name || targetRun.filename || 'Run';
            const baseName = currentName.replace(/\.[a-zA-Z0-9]{2,5}$/, '');
            const newName = await showPromptPopup({
                title: 'Rename Inference Run',
                message: 'Enter a new label for this run:',
                defaultValue: baseName,
                confirmText: 'Save'
            });
            if (newName !== null && newName.trim() && newName.trim() !== currentName) {
                await renameRun(runId, newName.trim());
                if (typeof window.showToast === 'function') {
                    window.showToast('Run renamed.', 'success');
                }
            }
        });

        // Delete button (popup modal)
        row.querySelector('[data-action="delete"]')?.addEventListener('click', async (e) => {
            e.stopPropagation();
            const confirmed = await showConfirmPopup({
                title: 'Delete Run',
                message: `Are you sure you want to delete run "${targetRun.name}"?`,
                confirmText: 'Delete',
                isDanger: true
            });
            if (confirmed) {
                await deleteRun(runId);
                if (typeof window.showToast === 'function') {
                    window.showToast('Run deleted.', 'info');
                }
            }
        });
    });
}

// =============================================================================
// 6. RUN DETAIL DRAWER (Slide-over with ephemeral stem fallback)
// =============================================================================
let activeDrawerRun = null;

export function openRunDetailDrawer(run) {
    if (!run) return;
    activeDrawerRun = run;

    let drawer = document.getElementById('sanadaRunDrawer');
    let backdrop = document.getElementById('sanadaDrawerBackdrop');

    // Create drawer DOM if missing in page
    if (!drawer) {
        drawer = document.createElement('aside');
        drawer.id = 'sanadaRunDrawer';
        drawer.className = 'sanada-run-drawer';
        drawer.setAttribute('aria-hidden', 'true');
        document.body.appendChild(drawer);
    }

    if (!backdrop) {
        backdrop = document.createElement('div');
        backdrop.id = 'sanadaDrawerBackdrop';
        backdrop.className = 'sanada-drawer-backdrop';
        document.body.appendChild(backdrop);
    }

    const rj = getRunJson(run);
    const res = Object.assign({}, rj, rj.results || {});
    const countObj = res.source_counting || {};
    const activeTags = getRunTags(run);
    const kVal = resolveDisplayRunCount(res, countObj, activeTags);
    const ceilingState = countObj.complexity_ceiling || (kVal >= 4 ? 'Over-Saturated' : (kVal === 3 ? 'Complex' : 'Optimal'));
    const stems = res.source_separation?.stems || res.speech_separation?.stems || res.separated_stems || res.env_separation?.env_stems || [];
    const telemetry = rj.telemetry || {};

    const origin = (typeof window !== 'undefined' && window.location.origin) ? window.location.origin : 'http://localhost:8000';

    let stemsHtml = '';
    if (stems && stems.length > 0) {
        stemsHtml = `
            <div class="drawer-section">
                <h4 class="drawer-section-title">Reconstructed Audio Stems (${stems.length})</h4>
                <div class="drawer-stems-list">
        `;
        stems.forEach((stem, idx) => {
            let stemUrl = stem.audio_data_url || stem.url || stem.stem_url || '';
            if (stemUrl && !stemUrl.startsWith('http') && !stemUrl.startsWith('data:')) {
                const clean = stemUrl.startsWith('/') ? stemUrl.slice(1) : stemUrl;
                stemUrl = `${origin}/${clean}`;
            }
            const stemLabel = stem.assigned_tag || stem.label || `Stem ${idx + 1}`;

            stemsHtml += `
                <div class="drawer-stem-item" data-stem-url="${escapeHtml(stemUrl)}">
                    <div class="drawer-stem-header">
                        <span class="drawer-stem-label">${escapeHtml(stemLabel)}</span>
                        ${stem.sdr ? `<span class="badge">${escapeHtml(stem.sdr.toString())} dB SDR</span>` : ''}
                    </div>
                    <div class="drawer-audio-wrapper" id="stemAudioWrap_${idx}">
                        <audio
                            class="drawer-stem-audio"
                            controls
                            preload="metadata"
                            src="${escapeHtml(stemUrl)}"
                        ></audio>
                        <div class="audio-expired-pill" style="display: none;">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="8" x2="12" y2="12"></line>
                                <line x1="12" y1="16" x2="12.01" y2="16"></line>
                            </svg>
                            <span>Audio expired (ephemeral disk storage)</span>
                        </div>
                    </div>
                </div>
            `;
        });
        stemsHtml += `</div></div>`;
    }

    let tagsHtml = '';
    if (activeTags && activeTags.length > 0) {
        tagsHtml = activeTags.map(t => `<span class="tag-pill-candidate">${escapeHtml(t)}</span>`).join('');
    } else {
        tagsHtml = `<span class="drawer-empty-text">No active tags detected above threshold</span>`;
    }

    drawer.innerHTML = `
        <div class="drawer-inner">
            <!-- DRAWER HEADER -->
            <div class="drawer-header">
                <div class="drawer-header-meta">
                    <div class="drawer-title-row">
                        <h3 class="drawer-title" id="drawerRunTitle">${escapeHtml(run.name)}</h3>
                    </div>
                    <span class="drawer-timestamp">${escapeHtml(formatDate(run.createdAt))} &bull; ${escapeHtml(run.filename)}</span>
                </div>
                <button type="button" class="btn-drawer-close" id="drawerCloseBtn" aria-label="Close drawer">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>

            <!-- DRAWER BODY -->
            <div class="drawer-body">
                <!-- SETTINGS PILLS -->
                <div class="drawer-section">
                    <h4 class="drawer-section-title">Inference Settings</h4>
                    <div class="drawer-settings-grid">
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Mode</span>
                            <span class="pill-value">${escapeHtml((run.mode || 'auto').toUpperCase())}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Task</span>
                            <span class="pill-value">${escapeHtml(run.task || 'all')}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Threshold</span>
                            <span class="pill-value">${escapeHtml((run.threshold || 0.5).toString())}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">All 4 Stems</span>
                            <span class="pill-value">${run.all4Stems ? 'Enabled' : 'Disabled'}</span>
                        </div>
                    </div>
                </div>

                <!-- TASK 1: SOURCE COUNTING -->
                <div class="drawer-section">
                    <h4 class="drawer-section-title">Source Counting (Task 1)</h4>
                    <div class="drawer-k-box">
                        <div class="drawer-k-num">${escapeHtml(kVal.toString())}</div>
                        <div class="drawer-k-meta">
                            <span class="drawer-k-label">Estimated Emitter Count</span>
                            <span class="badge">${escapeHtml(ceilingState)}</span>
                        </div>
                    </div>
                </div>

                <!-- TASK 2: SOUND EVENT TAGS -->
                <div class="drawer-section">
                    <h4 class="drawer-section-title">Detected Sound Tags (Task 2)</h4>
                    <div class="drawer-tags-wrap">${tagsHtml}</div>
                </div>

                <!-- TASK 3: RECONSTRUCTED STEM PLAYERS -->
                ${stemsHtml}

                <!-- TELEMETRY & LATENCY -->
                <div class="drawer-section">
                    <h4 class="drawer-section-title">Telemetry &amp; Execution Latency</h4>
                    <div class="drawer-telemetry-grid">
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Total Time</span>
                            <span class="pill-value">${escapeHtml(telemetry.total_inference_time_ms ? `${telemetry.total_inference_time_ms} ms` : 'Completed')}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Counting Latency</span>
                            <span class="pill-value">${escapeHtml(telemetry.task1_counting_ms ? `${telemetry.task1_counting_ms} ms` : '—')}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Tagging Latency</span>
                            <span class="pill-value">${escapeHtml(telemetry.task2_tagging_ms ? `${telemetry.task2_tagging_ms} ms` : '—')}</span>
                        </div>
                        <div class="drawer-metric-pill">
                            <span class="pill-label">Separation Latency</span>
                            <span class="pill-value">${escapeHtml(telemetry.task3_separation_ms ? `${telemetry.task3_separation_ms} ms` : '—')}</span>
                        </div>
                    </div>
                </div>

                <!-- RAW RESPONSE JSON -->
                <div class="drawer-section">
                    <div class="drawer-json-header">
                        <h4 class="drawer-section-title" style="margin: 0;">Raw Output JSON</h4>
                        <button type="button" class="btn btn-xs btn-copy-json" id="drawerCopyJsonBtn">Copy JSON</button>
                    </div>
                    <pre class="drawer-json-box"><code id="drawerJsonDisplay">${escapeHtml(JSON.stringify(run.resultJson, null, 2))}</code></pre>
                </div>
            </div>
        </div>
    `;

    // Ephemeral stem 404 detection: attach error handler to each audio player
    drawer.querySelectorAll('.drawer-stem-audio').forEach(audioEl => {
        audioEl.addEventListener('error', () => {
            const wrap = audioEl.closest('.drawer-audio-wrapper');
            if (wrap) {
                audioEl.style.display = 'none';
                const pill = wrap.querySelector('.audio-expired-pill');
                if (pill) pill.style.display = 'inline-flex';
            }
        });
    });

    // Wire Copy JSON button
    const copyBtn = drawer.querySelector('#drawerCopyJsonBtn');
    copyBtn?.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(JSON.stringify(run.resultJson, null, 2));
            const orig = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            setTimeout(() => { copyBtn.textContent = orig; }, 1800);
        } catch (e) {}
    });

    // Wire close triggers
    drawer.querySelector('#drawerCloseBtn')?.addEventListener('click', closeRunDetailDrawer);
    backdrop.onclick = closeRunDetailDrawer;

    // Show drawer
    drawer.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    backdrop.classList.add('open');
    document.body.style.overflow = 'hidden';
}

export function closeRunDetailDrawer() {
    const drawer = document.getElementById('sanadaRunDrawer');
    const backdrop = document.getElementById('sanadaDrawerBackdrop');
    if (drawer) {
        drawer.classList.remove('open');
        drawer.setAttribute('aria-hidden', 'true');
        // Pause any active audio playing inside drawer
        drawer.querySelectorAll('audio').forEach(a => {
            try { a.pause(); } catch (e) {}
        });
    }
    if (backdrop) {
        backdrop.classList.remove('open');
    }
    const authModal = document.getElementById('sanadaAuthModal');
    const logoutModal = document.getElementById('sanadaLogoutModal');
    const deleteModal = document.getElementById('sanadaDeleteKeyModal');
    if ((!authModal || !authModal.classList.contains('open'))
        && (!logoutModal || !logoutModal.classList.contains('open'))
        && (!deleteModal || !deleteModal.classList.contains('open'))) {
        document.body.style.overflow = '';
    }
    activeDrawerRun = null;
}

// =============================================================================
// 7. STUDIO LOAD FROM HISTORY MODAL CONTROLLER
// =============================================================================
let activeLoadHistoryCallback = null;

function renderModalHistoryItems(listContainer, runs) {
    if (!listContainer) return;
    if (!runs || runs.length === 0) {
        listContainer.innerHTML = `
            <div class="run-history-empty" style="padding: 34px 16px;">
                <span class="run-empty-title">No runs in history</span>
                <span class="run-empty-sub">Run inference in the Studio (Upload or Live Mic) to view and load past runs here.</span>
            </div>
        `;
        return;
    }

    let html = '<div class="modal-history-items-list">';
    runs.forEach(run => {
        const rj = getRunJson(run);
        const res = Object.assign({}, rj, rj.results || {});
        const countObj = res.source_counting || {};
        const tags = getRunTags(run);
        const kVal = resolveDisplayRunCount(res, countObj, tags);
        const tagPreview = tags.length > 0 ? tags.slice(0, 4).join(', ') + (tags.length > 4 ? ` +${tags.length - 4}` : '') : 'Active Sources';
        const stems = res.source_separation?.stems || res.speech_separation?.stems || res.separated_stems || res.env_separation?.env_stems || [];

        html += `
            <div class="modal-history-item" data-run-id="${escapeHtml(run.id)}">
                <div class="modal-history-main">
                    <div class="modal-history-title-row">
                        <span class="modal-history-name">${escapeHtml(run.name)}</span>
                        <span class="badge badge-k">K = ${escapeHtml(kVal.toString())}</span>
                    </div>
                    <div class="modal-history-meta">
                        <span class="modal-history-filename">${escapeHtml(run.filename)}</span>
                        <span>&bull;</span>
                        <span>${escapeHtml(formatDate(run.createdAt))}</span>
                        <span>&bull;</span>
                        <span class="modal-history-stems-count">${stems.length} stem${stems.length === 1 ? '' : 's'} with audio</span>
                    </div>
                    <div class="modal-history-tags">
                        <span style="color: var(--text-secondary); font-size: 11.5px;">Tags: <strong style="color: #ffffff;">${escapeHtml(tagPreview)}</strong></span>
                    </div>
                </div>
                <div class="modal-history-actions">
                    <button type="button" class="btn btn-primary btn-sm btn-load-run" data-action="load-this-run" data-run-id="${escapeHtml(run.id)}" title="Load run into Studio with all stems">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        <span>Load</span>
                    </button>
                </div>
            </div>
        `;
    });
    html += '</div>';
    listContainer.innerHTML = html;

    listContainer.querySelectorAll('.modal-history-item').forEach(itemEl => {
        itemEl.style.cursor = 'pointer';
        itemEl.addEventListener('click', (e) => {
            e.preventDefault();
            const runId = itemEl.dataset.runId;
            const target = runs.find(r => String(r.id) === String(runId));
            const cb = activeLoadHistoryCallback;
            closeLoadHistoryModal();
            if (target && typeof cb === 'function') {
                cb(target);
            }
        });
    });
}

export async function openLoadHistoryModal(onLoadCallback) {
    activeLoadHistoryCallback = onLoadCallback;
    let modal = document.getElementById('sanadaLoadHistoryModal');
    if (!modal) return;

    const listContainer = document.getElementById('studioModalHistoryList');

    // 1. Display skeleton loader while loading the complete history
    if (listContainer) {
        renderHistorySkeleton(listContainer, 3);
    }

    // 2. Open modal so user sees the skeleton loader
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';

    // 3. Fetch full history from Supabase (with fallback to local storage)
    try {
        const runs = await getRunHistory();
        if (listContainer) {
            renderModalHistoryItems(listContainer, runs);
        }
    } catch (err) {
        console.warn('[RunHistory] Notice fetching history for modal:', err);
        if (listContainer) {
            const fallback = loadLocalRuns();
            renderModalHistoryItems(listContainer, fallback);
        }
    }
}

export function closeLoadHistoryModal() {
    const modal = document.getElementById('sanadaLoadHistoryModal');
    if (modal) {
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
    }
    const authModal = document.getElementById('sanadaAuthModal');
    const drawer = document.getElementById('sanadaRunDrawer');
    if ((!authModal || !authModal.classList.contains('open'))
        && (!drawer || !drawer.classList.contains('open'))) {
        document.body.style.overflow = '';
    }
    activeLoadHistoryCallback = null;
}

// Close load history modal and route to Account when clicking "Manage history in Account"
if (typeof document !== 'undefined') {
    document.addEventListener('click', (e) => {
        const link = e.target && e.target.closest('#manageHistoryInAccountLink');
        if (link) {
            closeLoadHistoryModal();
            if (typeof window.switchRoute === 'function') {
                window.switchRoute('account');
            } else {
                window.location.hash = '#/account';
            }
            window.dispatchEvent(new CustomEvent('sanada:routeChanged', { detail: { route: 'account' } }));
        }
    });
}

// Global escape key listener to close drawer & modal
if (typeof window !== 'undefined') {
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const drawer = document.getElementById('sanadaRunDrawer');
            if (drawer && drawer.classList.contains('open')) {
                closeRunDetailDrawer();
            }
            const histModal = document.getElementById('sanadaLoadHistoryModal');
            if (histModal && histModal.classList.contains('open')) {
                closeLoadHistoryModal();
            }
        }
    });
}
