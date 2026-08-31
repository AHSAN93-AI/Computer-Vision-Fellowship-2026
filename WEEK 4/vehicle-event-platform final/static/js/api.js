/**
 * static/js/api.js — Thin wrapper around all REST API calls.
 * All functions return Promises and handle errors gracefully.
 */

const BASE = '';  // Same origin

// ─── Video Control ────────────────────────────────────────────────

export async function uploadVideo(file, sourceId = 'uploaded_video') {
  const form = new FormData();
  form.append('file', file);
  form.append('source_id', sourceId);
  const res = await fetch(`${BASE}/api/video/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
  return res.json();
}

export async function startWebcam(index = 0) {
  const res = await fetch(`${BASE}/api/video/webcam/start?webcam_index=${index}`, { method: 'POST' });
  if (!res.ok) throw new Error((await res.json()).detail || 'Webcam start failed');
  return res.json();
}

export async function startRtsp(url) {
  const res = await fetch(`${BASE}/api/video/rtsp/start?rtsp_url=${encodeURIComponent(url)}`, { method: 'POST' });
  if (!res.ok) throw new Error((await res.json()).detail || 'RTSP start failed');
  return res.json();
}

export async function stopPipeline() {
  const res = await fetch(`${BASE}/api/video/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Stop failed');
  return res.json();
}

export async function getVideoStatus() {
  const res = await fetch(`${BASE}/api/video/status`);
  if (!res.ok) return null;
  return res.json();
}

// ─── Analytics ────────────────────────────────────────────────────

export async function getLiveMetrics() {
  const res = await fetch(`${BASE}/api/analytics/live`);
  if (!res.ok) return null;
  return res.json();
}

export async function getAnalyticsSummary() {
  const res = await fetch(`${BASE}/api/analytics/summary`);
  if (!res.ok) return null;
  return res.json();
}

export async function getOccupancy(zoneId, minutes = 10) {
  const res = await fetch(`${BASE}/api/analytics/occupancy/${zoneId}?minutes=${minutes}`);
  if (!res.ok) return null;
  return res.json();
}

export async function getAllOccupancy() {
  const res = await fetch(`${BASE}/api/analytics/occupancy`);
  if (!res.ok) return null;
  return res.json();
}

export async function getDwellStats(zoneId) {
  const res = await fetch(`${BASE}/api/analytics/dwell/${zoneId}`);
  if (!res.ok) return null;
  return res.json();
}

export async function getPerformance() {
  const res = await fetch(`${BASE}/api/analytics/performance`);
  if (!res.ok) return null;
  return res.json();
}

// ─── Events ───────────────────────────────────────────────────────

export async function getActiveEvents() {
  const res = await fetch(`${BASE}/api/events/active`);
  if (!res.ok) return { events: [] };
  return res.json();
}

export async function listEvents(params = {}) {
  const q = new URLSearchParams();
  if (params.event_type) q.set('event_type', params.event_type);
  if (params.severity)   q.set('severity', params.severity);
  if (params.zone_id)    q.set('zone_id', params.zone_id);
  if (params.status)     q.set('status', params.status);
  if (params.start_time) q.set('start_time', params.start_time);
  if (params.end_time)   q.set('end_time', params.end_time);
  if (params.page)       q.set('page', params.page);
  if (params.page_size)  q.set('page_size', params.page_size);

  const res = await fetch(`${BASE}/api/events?${q}`);
  if (!res.ok) return { total: 0, events: [] };
  return res.json();
}

export async function getEvent(eventId) {
  const res = await fetch(`${BASE}/api/events/${eventId}`);
  if (!res.ok) return null;
  return res.json();
}

export async function acknowledgeEvent(eventId) {
  const res = await fetch(`${BASE}/api/events/${eventId}/acknowledge`, { method: 'PATCH' });
  if (!res.ok) throw new Error('Acknowledge failed');
  return res.json();
}

export async function resolveEvent(eventId) {
  const res = await fetch(`${BASE}/api/events/${eventId}/resolve`, { method: 'PATCH' });
  if (!res.ok) throw new Error('Resolve failed');
  return res.json();
}

export function getEventsCsvUrl(params = {}) {
  const q = new URLSearchParams();
  if (params.event_type) q.set('event_type', params.event_type);
  if (params.severity)   q.set('severity', params.severity);
  if (params.zone_id)    q.set('zone_id', params.zone_id);
  if (params.status)     q.set('status', params.status);
  return `${BASE}/api/events/export/csv?${q}`;
}

// ─── Zone Management ──────────────────────────────────────────────

export async function getZones() {
  const res = await fetch(`${BASE}/api/zones`);
  if (!res.ok) return { zones: [] };
  return res.json();
}

export async function updateZone(zoneId, data) {
  const res = await fetch(`${BASE}/api/zones/${encodeURIComponent(zoneId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'Zone update failed');
  return res.json();
}
