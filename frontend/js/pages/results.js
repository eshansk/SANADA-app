/**
 * frontend/js/pages/results.js
 * ============================
 * Analytics & Benchmark Comparison Page Logic (SOTA Wave 1-4 + Paper Baseline).
 */

import { API } from '../services/api.js';
import { PAPER_BENCHMARKS, PATH_B_FALLBACK_BENCHMARKS, SOTA_WAVE4_FALLBACK_BENCHMARKS } from '../data/paperBenchmarks.js';
import { ChartHelper } from '../ui/charts.js';

export async function loadAnalytics() {
    const DOM = {
        statTotalClips: document.getElementById('statTotalClips'),
        statTotalHours: document.getElementById('statTotalHours'),
        statSampleRate: document.getElementById('statSampleRate'),
        statClippingGain: document.getElementById('statClippingGain'),
        statPureSpeechTile: document.getElementById('statPureSpeechTile'),
        statPureSpeechRejection: document.getElementById('statPureSpeechRejection'),
        statPureSpeechSub: document.getElementById('statPureSpeechSub'),
        sisdrTableBody: document.getElementById('sisdrTableBody'),
        countingTableBody: document.getElementById('countingTableBody'),
        taggingTableBody: document.getElementById('taggingTableBody'),
        sisdrChartContainer: document.getElementById('sisdrChartContainer'),
        countingChartContainer: document.getElementById('countingChartContainer'),
        taggingChartContainer: document.getElementById('taggingChartContainer')
    };

    const defaultSisdr = SOTA_WAVE4_FALLBACK_BENCHMARKS.separationSisdr.results;
    const defaultCounting = SOTA_WAVE4_FALLBACK_BENCHMARKS.counting.results;
    const defaultTagging = SOTA_WAVE4_FALLBACK_BENCHMARKS.tagging.results;

    if (DOM.sisdrTableBody) DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(defaultSisdr);
    if (DOM.countingTableBody) DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(defaultCounting);
    if (DOM.taggingTableBody) DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(defaultTagging);

    if (DOM.sisdrChartContainer) ChartHelper.renderSisdrChart(DOM.sisdrChartContainer, defaultSisdr);
    if (DOM.countingChartContainer) ChartHelper.renderCountingChart(DOM.countingChartContainer, defaultCounting);
    if (DOM.taggingChartContainer) ChartHelper.renderTaggingChart(DOM.taggingChartContainer, defaultTagging);

    if (SOTA_WAVE4_FALLBACK_BENCHMARKS.pure_speech_rejection) {
        const psr = SOTA_WAVE4_FALLBACK_BENCHMARKS.pure_speech_rejection;
        if (DOM.statPureSpeechTile) DOM.statPureSpeechTile.style.display = 'block';
        if (DOM.statPureSpeechRejection) DOM.statPureSpeechRejection.textContent = `${psr.speech_detection_accuracy_pct}%`;
        if (DOM.statPureSpeechSub) DOM.statPureSpeechSub.textContent = `${psr.speech_false_positive_rate_pct}% False Pos Rate`;
    }

    try {
        const data = await API.getStats();

        if (DOM.statTotalClips && data.suite_overview) {
            DOM.statTotalClips.textContent = Number(data.suite_overview.total_synthesized_mixtures || 15559).toLocaleString();
        }
        if (DOM.statTotalHours && data.suite_overview) {
            DOM.statTotalHours.textContent = `${data.suite_overview.total_duration_hours}h`;
        }
        if (DOM.statSampleRate && data.suite_overview) {
            DOM.statSampleRate.textContent = `${data.suite_overview.sample_rate_hz} Hz`;
        }
        if (DOM.statClippingGain) {
            DOM.statClippingGain.textContent = '-3.0 dBFS';
        }

        const psr = data.path_b_benchmarks?.pure_speech_rejection
            || data.pathb_benchmarks?.pure_speech_rejection
            || SOTA_WAVE4_FALLBACK_BENCHMARKS.pure_speech_rejection;
        if (psr) {
            if (DOM.statPureSpeechTile) DOM.statPureSpeechTile.style.display = 'block';
            if (DOM.statPureSpeechRejection) DOM.statPureSpeechRejection.textContent = `${psr.speech_detection_accuracy_pct}%`;
            if (DOM.statPureSpeechSub) DOM.statPureSpeechSub.textContent = `${psr.speech_false_positive_rate_pct}% False Pos Rate`;
        }

        const b = data.sota_benchmarks || data.complexity_ceiling_benchmarks;
        if (b) {
            if (DOM.sisdrTableBody && b.source_separation_sisdr_db) {
                DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(b.source_separation_sisdr_db);
            }
            if (DOM.countingTableBody && b.source_counting_mae) {
                DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(b.source_counting_mae);
            }
            if (DOM.taggingTableBody && b.multi_label_tagging_asl) {
                DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(b.multi_label_tagging_asl);
            }
            if (DOM.sisdrChartContainer && b.source_separation_sisdr_db) {
                ChartHelper.renderSisdrChart(DOM.sisdrChartContainer, b.source_separation_sisdr_db);
            }
            if (DOM.countingChartContainer && b.source_counting_mae) {
                ChartHelper.renderCountingChart(DOM.countingChartContainer, b.source_counting_mae);
            }
            if (DOM.taggingChartContainer && b.multi_label_tagging_asl) {
                ChartHelper.renderTaggingChart(DOM.taggingChartContainer, b.multi_label_tagging_asl);
            }
        }
    } catch (err) {
        // Backend offline; already populated from verified SOTA_WAVE4_FALLBACK_BENCHMARKS
    }
}

export function initResultsPage() {
    loadAnalytics();
}
