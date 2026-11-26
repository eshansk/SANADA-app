/**
 * frontend/js/audio/audioEngine.js
 * ================================
 * Web Audio API Engine for client-side decoding, playback, synthetic mix generation,
 * linear 16-bit PCM WAV encoding, and microphone streaming.
 */

import { TARGET_SR, EXPECTED_FRAMES, EXPECTED_DURATION } from '../config.js';

export function roundTo(num, dec) {
    const factor = Math.pow(10, dec);
    return Math.round(num * factor) / factor;
}

export class AudioEngine {
    constructor() {
        this.ctx = null;
        this.currentBuffer = null;
        this.sourceNode = null;
        this.isPlaying = false;
        this.startTime = 0;
        this.pauseTime = 0;
        this.onPlaybackUpdate = null;
        this.onPlaybackEnded = null;

        // MediaRecorder variables
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordTimer = null;
        this.recElapsed = 0;
        this.isRecording = false;
    }

    initContext() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume().catch(() => {});
        }
        return this.ctx;
    }

    /**
     * Decodes binary ArrayBuffer into an AudioBuffer
     */
    async decodeAudio(arrayBuffer) {
        const ctx = this.initContext();
        // Clone slice before decoding because decodeAudioData detaches the underlying ArrayBuffer
        const copy1 = arrayBuffer.slice ? arrayBuffer.slice(0) : arrayBuffer;
        try {
            const decoded = await ctx.decodeAudioData(copy1);
            this.currentBuffer = decoded;
            return decoded;
        } catch (promiseErr) {
            return new Promise((resolve, reject) => {
                try {
                    const copy2 = arrayBuffer.slice ? arrayBuffer.slice(0) : arrayBuffer;
                    ctx.decodeAudioData(copy2, (decoded) => {
                        this.currentBuffer = decoded;
                        resolve(decoded);
                    }, (err) => {
                        reject(err || promiseErr);
                    });
                } catch (callbackErr) {
                    reject(callbackErr || promiseErr);
                }
            });
        }
    }

    /**
     * Validates decoded buffer against specifications
     */
    inspectBuffer(buffer) {
        const sr = buffer.sampleRate;
        const ch = buffer.numberOfChannels;
        const frames = buffer.length;
        const duration = buffer.duration;
        const channelData = buffer.getChannelData(0);

        let maxAmp = 0;
        for (let i = 0; i < channelData.length; i++) {
            const abs = Math.abs(channelData[i]);
            if (abs > maxAmp) maxAmp = abs;
        }

        const peakDbfs = maxAmp > 0 ? (20 * Math.log10(maxAmp)).toFixed(1) : '-inf';
        const isStandard = (sr === TARGET_SR && ch === 1 && Math.abs(duration - EXPECTED_DURATION) <= 0.05);

        return {
            sampleRate: sr,
            channels: ch,
            frames,
            duration: roundTo(duration, 2),
            peakAmplitude: roundTo(maxAmp, 4),
            peakDbfs,
            isStandard
        };
    }

    /**
     * Generates a downsampled peak envelope for canvas rendering
     */
    extractPeaks(buffer, numBins = 600) {
        const data = buffer.getChannelData(0);
        const step = Math.ceil(data.length / numBins);
        const peaks = [];

        for (let b = 0; b < numBins; b++) {
            let min = 1.0;
            let max = -1.0;
            const start = b * step;
            const end = Math.min(start + step, data.length);
            for (let i = start; i < end; i++) {
                const val = data[i];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            peaks.push({ min: min > 1.0 ? 0 : min, max: max < -1.0 ? 0 : max });
        }
        return peaks;
    }

    /**
     * Synthesizes an 8.0s 16 kHz test audio mix with overlapping speech harmonics
     */
    createSyntheticDemoBuffer(speakers = 3) {
        this.initContext();
        const buffer = this.ctx.createBuffer(1, EXPECTED_FRAMES, TARGET_SR);
        const data = buffer.getChannelData(0);

        for (let i = 0; i < EXPECTED_FRAMES; i++) {
            const t = i / TARGET_SR;
            let sum = 0;
            // Primary speaker
            sum += 0.3 * Math.sin(2 * Math.PI * 220 * t) * (0.6 + 0.4 * Math.cos(2 * Math.PI * 1.5 * t));
            // Secondary speaker (with -3.0 dB gain rule)
            if (speakers >= 2) {
                sum += (0.3 * 0.7079) * Math.sin(2 * Math.PI * 440 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 2.5 * t));
            }
            // Third speaker
            if (speakers >= 3) {
                sum += (0.25 * 0.7079) * Math.sin(2 * Math.PI * 660 * t) * (0.7 + 0.3 * Math.cos(2 * Math.PI * 3.0 * t));
            }
            // Fourth speaker
            if (speakers >= 4) {
                sum += (0.2 * 0.7079) * Math.sin(2 * Math.PI * 880 * t) * (0.4 + 0.6 * Math.sin(2 * Math.PI * 1.0 * t));
            }
            data[i] = sum;
        }
        this.currentBuffer = buffer;
        return buffer;
    }

    _extractMonoResampledFloat32(buffer, targetSr = TARGET_SR, targetFrames = null) {
        const numCh = buffer.numberOfChannels || 1;
        const inLen = buffer.length;
        const mono = new Float32Array(inLen);
        for (let ch = 0; ch < numCh; ch++) {
            const data = buffer.getChannelData(ch);
            for (let i = 0; i < inLen; i++) mono[i] += data[i] / numCh;
        }
        const srcSr = buffer.sampleRate || targetSr;
        let resampled;
        if (srcSr === targetSr) {
            resampled = mono;
        } else {
            const ratio = srcSr / targetSr;
            const outLen = Math.max(1, Math.round(inLen / ratio));
            resampled = new Float32Array(outLen);
            if (ratio > 1.0) {
                // Box-filter anti-aliased decimation + linear interpolation for clean downsampling (e.g. 48k -> 16k)
                for (let i = 0; i < outLen; i++) {
                    const start = Math.floor(i * ratio);
                    const end = Math.min(inLen, Math.max(start + 1, Math.floor((i + 1) * ratio)));
                    let acc = 0;
                    for (let j = start; j < end; j++) acc += mono[j];
                    resampled[i] = acc / (end - start);
                }
            } else {
                for (let i = 0; i < outLen; i++) {
                    const pos = i * ratio;
                    const i0 = Math.floor(pos);
                    const i1 = Math.min(i0 + 1, inLen - 1);
                    const frac = pos - i0;
                    resampled[i] = mono[i0] * (1 - frac) + mono[i1] * frac;
                }
            }
        }
        if (targetFrames && targetFrames > 0) {
            const fixed = new Float32Array(targetFrames);
            fixed.set(resampled.subarray(0, Math.min(resampled.length, targetFrames)));
            return fixed;
        }
        return resampled;
    }

    _padOrTrimNativeMonoBuffer(decoded, durationSec = EXPECTED_DURATION) {
        const ctx = this.initContext();
        const sr = decoded.sampleRate || ctx.sampleRate || 48000;
        const targetLen = Math.max(1, Math.round(sr * durationSec));
        const outBuf = ctx.createBuffer(1, targetLen, sr);
        const out = outBuf.getChannelData(0);
        const numCh = decoded.numberOfChannels || 1;
        const copyLen = Math.min(decoded.length, targetLen);
        for (let ch = 0; ch < numCh; ch++) {
            const src = decoded.getChannelData(ch);
            for (let i = 0; i < copyLen; i++) out[i] += src[i] / numCh;
        }
        return outBuf;
    }

    /**
     * Converts an AudioBuffer to a downloadable linear 16-bit PCM WAV Blob
     */
    bufferToWavBlob(buffer, targetSr = null, targetFrames = null) {
        const useResampledMono = Boolean(targetSr);
        const sampleRate = useResampledMono ? targetSr : buffer.sampleRate;
        const channels = useResampledMono
            ? [this._extractMonoResampledFloat32(buffer, targetSr, targetFrames)]
            : Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i));
        const numChan = channels.length;
        const numFrames = channels[0] ? channels[0].length : 0;
        const length = numFrames * numChan * 2 + 44;
        const out = new DataView(new ArrayBuffer(length));
        let offset = 0;
        let pos = 0;

        function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
        function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8);
        setUint32(0x45564157); // "WAVE"
        setUint32(0x20746d66); // "fmt "
        setUint32(16);
        setUint16(1); // PCM
        setUint16(numChan);
        setUint32(sampleRate);
        setUint32(sampleRate * 2 * numChan);
        setUint16(numChan * 2);
        setUint16(16);
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4);

        while (pos < length) {
            for (let i = 0; i < numChan; i++) {
                const raw = Math.max(-1, Math.min(1, channels[i][offset] || 0));
                const int16 = (raw < 0 ? raw * 32768 : raw * 32767) | 0;
                out.setInt16(pos, int16, true);
                pos += 2;
            }
            offset++;
        }
        return new Blob([out.buffer], { type: 'audio/wav' });
    }

    /**
     * Starts live microphone capture using MediaRecorder
     */
    async startRecording(onProgress, onComplete, onError) {
        this.initContext();
        this.stopPlayback();
        this.recordedChunks = [];
        this.recElapsed = 0;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const preferredTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
            const chosenType = (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported)
                ? (preferredTypes.find(t => MediaRecorder.isTypeSupported(t)) || '')
                : '';
            this.mediaRecorder = chosenType ? new MediaRecorder(stream, { mimeType: chosenType }) : new MediaRecorder(stream);
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.recordedChunks.push(e.data);
            };

            this.mediaRecorder.onstop = async () => {
                try {
                    const mimeType = this.mediaRecorder.mimeType || chosenType || 'audio/webm';
                    const blob = new Blob(this.recordedChunks, { type: mimeType });
                    const arrayBuffer = await blob.arrayBuffer();
                    const decoded = await this.decodeAudio(arrayBuffer);
                    const playbackBuffer = this._padOrTrimNativeMonoBuffer(decoded, EXPECTED_DURATION);
                    this.currentBuffer = playbackBuffer;
                    this.pauseTime = 0;
                    this.isPlaying = false;
                    if (onComplete) onComplete(playbackBuffer);
                } catch (decErr) {
                    if (onError) onError(decErr);
                }
            };

            this.mediaRecorder.start(100);
            this.isRecording = true;

            this.recordTimer = setInterval(() => {
                this.recElapsed += 0.1;
                if (onProgress) onProgress(this.recElapsed);
                if (this.recElapsed >= EXPECTED_DURATION) {
                    this.stopRecording();
                }
            }, 100);
        } catch (err) {
            if (onError) onError(err);
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        clearInterval(this.recordTimer);
        this.isRecording = false;
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(t => t.stop());
        }
    }

    // --- PLAYBACK ENGINE ---
    async play(onUpdate, onEnded) {
        if (!this.currentBuffer) return;
        this.initContext();
        if (this.sourceNode) {
            this.sourceNode.onended = null;
            try { this.sourceNode.stop(); this.sourceNode.disconnect(); } catch (e) {}
            this.sourceNode = null;
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            try { await this.ctx.resume(); } catch (_) {}
        }
        this.onPlaybackUpdate = onUpdate;
        this.onPlaybackEnded = onEnded;

        this.sourceNode = this.ctx.createBufferSource();
        this.sourceNode.buffer = this.currentBuffer;
        this.sourceNode.connect(this.ctx.destination);

        const dur = this.currentBuffer.duration || EXPECTED_DURATION;
        const offset = (this.pauseTime >= dur - 0.01 || this.pauseTime < 0) ? 0 : this.pauseTime;
        this.sourceNode.start(0, offset);
        this.startTime = this.ctx.currentTime - offset;
        this.isPlaying = true;

        this.sourceNode.onended = () => {
            if (this.isPlaying) {
                this.stopPlayback();
                if (this.onPlaybackEnded) this.onPlaybackEnded();
            }
        };

        this._runPlaybackLoop();
    }

    pausePlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); this.sourceNode.disconnect(); } catch (e) {}
        }
        this.pauseTime = this.ctx.currentTime - this.startTime;
        this.isPlaying = false;
    }

    stopPlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); this.sourceNode.disconnect(); } catch (e) {}
        }
        this.isPlaying = false;
        this.pauseTime = 0;
    }

    seek(fraction) {
        if (!this.currentBuffer) return;
        this.pauseTime = fraction * this.currentBuffer.duration;
        if (this.isPlaying) {
            this.pausePlayback();
            this.play(this.onPlaybackUpdate, this.onPlaybackEnded);
        }
    }

    _runPlaybackLoop() {
        if (!this.isPlaying) return;
        const cur = this.ctx.currentTime - this.startTime;
        const dur = this.currentBuffer.duration;
        if (cur <= dur && this.onPlaybackUpdate) {
            this.onPlaybackUpdate(cur, dur, cur / dur);
        }
        requestAnimationFrame(() => this._runPlaybackLoop());
    }
}

export const audioEngine = new AudioEngine();
