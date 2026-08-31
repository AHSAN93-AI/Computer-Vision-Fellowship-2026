/**
 * static/js/dashboard.js — Live dashboard logic.
 * Handles: WebSocket feed, metric polling, video controls, alert list.
 */

import * as api from './api.js';
import {
  createEventsTimeChart, updateEventsTimeChart,
  createEventsByTypeChart, updateEventsByTypeChart,
  createSeverityChart, updateSeverityChart,
  createOccupancyChart, updateOccupancyChart,
  createFpsChart, pushFps,
} from './charts.js';
import { initZoneEditor, setEditorActive } from './zone-editor.js';

// ─── State ────────────────────────────────────────────────────────

const state = {
  ws: null,
  wsConnected: false,
  pollTimer: null,
  charts: {},
  alertQueue: [],
  maxAlerts: 20,
};

// ─── DOM shortcuts ────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function setText(id, val) {
  const el = $(id);
  if (el) el.textContent = val;
}

function setHtml(id, val) {
  const el = $(id);
  if (el) el.innerHTML = val;
}

// ─── Toast ────────────────────────────────────────────────────────

function toast(msg, type = 'info', duration = 4000) {
  const container = $('toast-container');
  if (!container) return;

  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
  container.appendChild(t);

  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(20px)';
    t.style.transition = 'all 0.3s ease';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ─── WebSocket ────────────────────────────────────────────────────

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/live`;

  state.ws = new WebSocket(url);

  state.ws.onopen = () => {
    state.wsConnected = true;
    updateConnectionStatus(true);
    // Keep-alive ping every 20s
    state._wsPing = setInterval(() => {
      if (state.ws?.readyState === WebSocket.OPEN) state.ws.send('ping');
    }, 20000);
  };

  state.ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg === 'pong') return;

      if (msg.type === 'frame') {
        handleFrameMessage(msg);
      } else if (msg.type === 'alert') {
        handleAlertMessage(msg.data);
      }
    } catch (e) {
      // ignore parse errors
    }
  };

  state.ws.onclose = () => {
    state.wsConnected = false;
    updateConnectionStatus(false);
    clearInterval(state._wsPing);
    // Reconnect after 3s
    setTimeout(connectWebSocket, 3000);
  };

  state.ws.onerror = () => {
    state.ws?.close();
  };
}

function handleFrameMessage(msg) {
  // Update live video feed
  const img = $('live-feed');
  if (img && msg.frame) {
    img.src = `data:image/jpeg;base64,${msg.frame}`;
    img.style.display = 'block';
    $('video-placeholder')?.style?.setProperty?.('display', 'none');
    const ph = $('video-placeholder');
    if (ph) ph.style.display = 'none';
  }

  const d = msg.data || {};

  // Update overlay tags
  setText('tag-fps', `FPS: ${d.fps || 0}`);
  setText('tag-tracks', `Tracks: ${d.active_tracks || 0}`);
  setText('tag-events', `Events: ${d.active_events || 0}`);

  // Update metric cards
  setText('metric-fps', (d.fps || 0).toFixed(1));
  setText('metric-tracks', d.active_tracks || 0);
  setText('metric-events', d.active_events || 0);

  // Zone occupancies
  updateZoneOccupancy(d.zone_occupancies || {});

  // Line counts
  updateLineCounts(d.line_counts || {});

  // Push FPS sparkline
  pushFps(state.charts.fps, d.fps || 0);

  // Performance metrics
  if (d.perf) updatePerfMeters(d.perf);
}

function handleAlertMessage(data) {
  addAlert({
    severity: data.severity || 'INFO',
    title: data.event_type?.replace(/_/g, ' ') || 'Alert',
    zone: data.zone_id || '',
    time: new Date().toLocaleTimeString(),
    description: data.description || '',
  });
  toast(`${data.severity}: ${data.event_type?.replace(/_/g, ' ')} in ${data.zone_id}`,
    data.severity === 'CRITICAL' || data.severity === 'HIGH' ? 'error' : data.severity === 'WARNING' ? 'warning' : 'info');
}

// ─── Connection Status ────────────────────────────────────────────

function updateConnectionStatus(connected) {
  const dot = $('ws-dot');
  const txt = $('ws-status');
  if (dot) dot.className = `live-dot ${connected ? 'active' : 'offline'}`;
  if (txt) txt.textContent = connected ? 'Live' : 'Offline';
}

// ─── Metric Polling (fallback/supplement) ────────────────────────

async function pollMetrics() {
  try {
    const live = await api.getLiveMetrics();
    if (!live) return;

    if (!state.wsConnected) {
      // Fallback: update metrics from REST when WS is down
      setText('metric-fps', (live.fps || 0).toFixed(1));
      setText('metric-tracks', live.active_tracks || 0);
      setText('metric-events', live.active_events || 0);
      updateZoneOccupancy(live.zone_occupancies || {});
      updateLineCounts(live.line_counts || {});
    }

    // Status dot
    const pipelineDot = $('pipeline-dot');
    const pipelineStatus = $('pipeline-status');
    if (pipelineDot) pipelineDot.className = `live-dot ${live.running ? 'active' : 'offline'}`;
    if (pipelineStatus) pipelineStatus.textContent = live.running ? 'Running' : 'Idle';
  } catch (e) {
    // silently ignore
  }

  // Analytics summary
  try {
    const summary = await api.getAnalyticsSummary();
    if (!summary) return;
    setText('metric-total-events', summary.total || 0);
    updateEventsTimeChart(state.charts.eventsTime, summary.hourly_last_24h);
    updateEventsByTypeChart(state.charts.byType, summary.by_type);
    updateSeverityChart(state.charts.severity, summary.by_severity);
  } catch (e) {
    // ignore
  }

  // Occupancy charts
  try {
    const allOcc = await api.getAllOccupancy();
    if (allOcc?.zones?.length > 0) {
      const zone = allOcc.zones[0];
      setText('metric-occupancy', zone.current_count || 0);
    }
  } catch (e) { /* ignore */ }
}

// ─── Zone Occupancy ───────────────────────────────────────────────

function updateZoneOccupancy(occupancies) {
  const container = $('zone-occupancy-list');
  if (!container) return;

  const entries = Object.entries(occupancies);
  if (entries.length === 0) {
    container.innerHTML = '<p class="text-muted" style="font-size:0.8rem;text-align:center;padding:8px;">No zones configured</p>';
    return;
  }

  container.innerHTML = entries.map(([zid, count]) => `
    <div class="perf-row">
      <span class="perf-label">${zid}</span>
      <div class="perf-bar-wrap">
        <div class="progress-bar" style="width:${Math.min(count * 10, 100)}%"></div>
      </div>
      <span class="perf-value">${count} veh.</span>
    </div>
  `).join('');
}

function updateLineCounts(lineCounts) {
  const container = $('line-counts-list');
  if (!container) return;

  const entries = Object.entries(lineCounts);
  if (entries.length === 0) {
    container.innerHTML = '<p class="text-muted" style="font-size:0.8rem;text-align:center;padding:8px;">No lines configured</p>';
    return;
  }

  container.innerHTML = entries.map(([zid, counts]) => `
    <div class="flex items-center justify-between" style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
      <span class="text-secondary" style="font-size:0.8rem">${zid}</span>
      <div class="flex gap-2">
        <span class="badge badge-active">↑ IN: ${counts.in || 0}</span>
        <span class="badge badge-info">↓ OUT: ${counts.out || 0}</span>
      </div>
    </div>
  `).join('');
}

// ─── Performance Meters ───────────────────────────────────────────

function updatePerfMeters(perf) {
  const meters = [
    { id: 'perf-det', key: 'avg_detection_ms', label: 'Detection' },
    { id: 'perf-track', key: 'avg_tracking_ms', label: 'Tracking' },
    { id: 'perf-rules', key: 'avg_rules_ms', label: 'Rules' },
    { id: 'perf-total', key: 'avg_total_ms', label: 'Total/frame' },
  ];

  const maxMs = 100; // scale reference

  meters.forEach(({ id, key }) => {
    const ms = perf[key] || 0;
    const bar = $(`${id}-bar`);
    const val = $(`${id}-val`);
    if (bar) {
      const pct = Math.min((ms / maxMs) * 100, 100);
      bar.style.width = `${pct}%`;
      bar.className = `progress-bar ${pct > 70 ? 'danger' : pct > 40 ? 'warning' : ''}`;
    }
    if (val) val.textContent = `${ms.toFixed(1)}ms`;
  });

  const cpuEl = $('perf-cpu-val');
  const memEl = $('perf-mem-val');
  if (cpuEl && perf.cpu_percent != null) cpuEl.textContent = `${perf.cpu_percent.toFixed(1)}%`;
  if (memEl && perf.memory_mb != null)  memEl.textContent = `${perf.memory_mb.toFixed(0)} MB`;
}

// ─── Alert List ───────────────────────────────────────────────────

function addAlert(alert) {
  state.alertQueue.unshift(alert);
  if (state.alertQueue.length > state.maxAlerts) state.alertQueue.pop();
  renderAlerts();
}

function renderAlerts() {
  const container = $('alert-list');
  if (!container) return;

  if (state.alertQueue.length === 0) {
    container.innerHTML = '<p class="text-muted" style="text-align:center;font-size:0.8rem;padding:16px;">No recent alerts</p>';
    return;
  }

  const icons = { CRITICAL: '🚨', HIGH: '🔶', WARNING: '⚠️', INFO: 'ℹ️' };
  const badgeClass = { CRITICAL: 'critical', HIGH: 'danger', WARNING: 'warning', INFO: 'info' };
  container.innerHTML = state.alertQueue.map(a => `
    <div class="alert-item ${(a.severity || '').toLowerCase()}">
      <span class="alert-icon">${icons[a.severity] || '⚡'}</span>
      <div class="alert-content">
        <div class="alert-title">${escHtml(a.title)}</div>
        <div class="alert-meta">${escHtml(a.zone)} · ${a.time}</div>
      </div>
      <span class="badge badge-${badgeClass[a.severity] || 'info'}">${a.severity}</span>
    </div>
  `).join('');
}

// ─── Video Controls ───────────────────────────────────────────────

function initVideoControls() {
  // Tab switching
  document.querySelectorAll('.control-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.control-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.control-panel-body > div').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const target = $(`panel-${tab.dataset.tab}`);
      if (target) target.classList.add('active');
    });
  });

  // File upload
  $('upload-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = $('video-file')?.files?.[0];
    if (!file) return toast('Please select a video file', 'warning');
    try {
      await api.uploadVideo(file);
      toast('Video uploaded and processing started', 'success');
      showProcessingState(true);
    } catch (err) {
      toast(err.message || 'Upload failed', 'error');
    }
  });

  // Webcam
  $('webcam-btn')?.addEventListener('click', async () => {
    const idx = parseInt($('webcam-index')?.value || '0');
    try {
      await api.startWebcam(idx);
      toast(`Webcam ${idx} started`, 'success');
      showProcessingState(true);
    } catch (err) {
      toast(err.message || 'Webcam start failed', 'error');
    }
  });

  // RTSP
  $('rtsp-btn')?.addEventListener('click', async () => {
    const url = $('rtsp-url')?.value?.trim();
    if (!url) return toast('Enter an RTSP URL', 'warning');
    try {
      await api.startRtsp(url);
      toast('RTSP stream started', 'success');
      showProcessingState(true);
    } catch (err) {
      toast(err.message || 'RTSP failed', 'error');
    }
  });

  // Stop
  $('stop-btn')?.addEventListener('click', async () => {
    try {
      await api.stopPipeline();
      toast('Pipeline stopped', 'info');
      showProcessingState(false);
    } catch (err) {
      toast(err.message || 'Stop failed', 'error');
    }
  });
}

function showProcessingState(running) {
  const stopBtn = $('stop-btn');
  if (stopBtn) stopBtn.disabled = !running;
}

// ─── Helpers ─────────────────────────────────────────────────────

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Init ─────────────────────────────────────────────────────────

function initCharts() {
  state.charts.eventsTime = createEventsTimeChart('chart-events-time');
  state.charts.byType     = createEventsByTypeChart('chart-by-type');
  state.charts.severity   = createSeverityChart('chart-severity');
  state.charts.occupancy  = createOccupancyChart('chart-occupancy', 'Parking Area');
  state.charts.fps        = createFpsChart('chart-fps');
}

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initVideoControls();
  initZoneEditorControls();
  initTestRunner();
  connectWebSocket();

  // Poll REST for analytics every 5s
  pollMetrics();
  state.pollTimer = setInterval(pollMetrics, 5000);
});

// ─── Run Tests Handler ──────────────────────────────────────────

function initTestRunner() {
  const btn = $('run-tests-btn');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = '⏳ Running...';
    const resultsEl = $('test-results');
    if (resultsEl) {
      resultsEl.innerHTML = '<p style="font-size:0.8rem;text-align:center;padding:16px;color:var(--accent)">Running test suite and experiments... This may take up to 30 seconds.</p>';
    }

    try {
      const resp = await fetch('/api/tests/run', { method: 'POST' });
      const data = await resp.json();

      // Show results
      const detailEl = $('test-results-detail');
      if (detailEl) detailEl.style.display = 'block';

      // Test suite summary
      const testDiv = $('test-suite-summary');
      if (testDiv && data.tests) {
        const t = data.tests;
        const icon = t.status === 'PASSED' ? '✅' : '❌';
        testDiv.innerHTML = `
          <div>${icon} <strong>${t.status}</strong></div>
          <div style="color:var(--accent)">✅ Passed: <strong>${t.passed}</strong></div>
          <div style="color:${t.failed > 0 ? 'var(--danger)' : 'var(--text-muted)'}">❌ Failed: <strong>${t.failed}</strong></div>
          <div style="color:var(--text-muted)">⚠️ Errors: ${t.errors}</div>
          <div style="margin-top:4px;font-size:0.75rem;color:var(--text-muted)">${escHtml(t.summary_line)}</div>
        `;
      }

      // Experiment summary
      const expDiv = $('experiment-summary');
      if (expDiv && data.experiments) {
        const e = data.experiments;
        const icon = e.status === 'COMPLETED' ? '✅' : '❌';
        let html = `
          <div>${icon} <strong>${e.status}</strong></div>
          <div>📊 Data points: <strong>${e.total_data_points}</strong></div>
        `;
        if (e.summary?.experiments) {
          const names = Object.keys(e.summary.experiments);
          html += '<div style="margin-top:4px">';
          names.forEach(name => {
            html += `<div style="color:var(--text-muted);font-size:0.75rem">• ${name.replace(/_/g, ' ')}</div>`;
          });
          html += '</div>';
        }
        expDiv.innerHTML = html;
      }

      // Show download button
      const dlBtn = $('download-csv-btn');
      if (dlBtn) dlBtn.style.display = 'inline-flex';

      // Update results area
      if (resultsEl) {
        resultsEl.innerHTML = `<p style="font-size:0.8rem;text-align:center;padding:4px;color:var(--accent)">
          Completed in ${data.elapsed_seconds || 0}s — ${data.timestamp || ''}</p>`;
      }

      toast('Tests and experiments completed!', data.tests?.status === 'PASSED' ? 'success' : 'warning');
    } catch (err) {
      if (resultsEl) {
        resultsEl.innerHTML = `<p style="font-size:0.8rem;text-align:center;padding:16px;color:var(--danger)">
          Error: ${escHtml(err.message)}</p>`;
      }
      toast('Test run failed: ' + err.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '▶ Run Tests';
    }
  });
}

// ─── Zone Editor Controls ────────────────────────────────────────────────

function initZoneEditorControls() {
  const feedImg   = $('live-feed');
  const container = $('video-container');
  const statusEl  = $('zone-status');
  const saveBtn   = $('zone-save-btn');
  const toggleBtn = $('zone-edit-toggle');
  const reloadBtn = $('zone-reload-btn');

  if (!container) return;

  // Init the canvas-based editor
  initZoneEditor(feedImg, container, statusEl, saveBtn);

  let editing = false;

  toggleBtn?.addEventListener('click', () => {
    editing = !editing;
    setEditorActive(editing);
    if (toggleBtn) {
      toggleBtn.textContent = editing ? '⏹️ Stop Editing' : '✏️ Start Editing';
      toggleBtn.className   = editing ? 'btn btn-danger' : 'btn btn-primary';
      toggleBtn.style.width = '100%';
    }
  });

  reloadBtn?.addEventListener('click', () => {
    if (editing) setEditorActive(true); // re-loads zones
  });
}
