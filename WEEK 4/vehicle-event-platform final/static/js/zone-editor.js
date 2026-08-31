/**
 * static/js/zone-editor.js — Interactive zone editor.
 *
 * Renders draggable / resizable zones over the live video feed on a canvas.
 * Polygon zones: drag inside to move, drag corner handles to resize.
 * Line-crossing zones: drag each endpoint independently.
 * Saves updated coordinates back to config.yaml via PUT /api/zones/{id}.
 *
 * Reference frame for config coordinates: 1280 × 720 px.
 */

import { getZones, updateZone } from './api.js';

// ─── Constants ────────────────────────────────────────────────────────────────

const REF_W = 1280;   // config coordinate space width
const REF_H = 720;    // config coordinate space height
const HANDLE_R = 7;   // hit-test radius for corner / endpoint handles (px)
const LINE_HIT = 8;   // distance threshold for clicking a line zone body

// Zone colors (BGR in config → RGB for canvas)
const ZONE_COLORS = {
  entrance_lane:   'rgba(0, 255, 0,  0.85)',
  parking_area:    'rgba(255, 200, 0, 0.85)',
  no_parking_zone: 'rgba(255, 60,  60, 0.85)',
};

const FILL_ALPHA   = 0.15;
const STROKE_WIDTH = 2;

// ─── State ───────────────────────────────────────────────────────────────────

const editorState = {
  active:    false,
  zones:     [],           // raw zone objects from config
  canvas:    null,
  ctx:       null,
  feed:      null,         // <img id="live-feed">
  container: null,         // video container div

  // Drag state
  dragging:     false,
  dragZoneIdx:  -1,
  dragHandleIdx: -1,       // corner/endpoint index, -1 means dragging whole zone
  dragStartX:   0,
  dragStartY:   0,
  dragOrigPoints: [],      // snapshot of points at drag start

  // Hover state
  hoverZoneIdx:  -1,
  hoverHandleIdx: -1,

  dirty: false,            // unsaved changes exist
  statusEl: null,          // status message element
  saveBtn:  null,
  reloadIntervalId: null,
};

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Initialize the zone editor (call once from dashboard.js).
 * @param {HTMLImageElement} feedImg   - the live-feed <img>
 * @param {HTMLElement}      container - video-container div
 * @param {HTMLElement}      statusEl  - element to show save status
 * @param {HTMLButtonElement} saveBtn  - save button
 */
export function initZoneEditor(feedImg, container, statusEl, saveBtn) {
  editorState.feed      = feedImg;
  editorState.container = container;
  editorState.statusEl  = statusEl;
  editorState.saveBtn   = saveBtn;

  // Create canvas
  const canvas = document.createElement('canvas');
  canvas.id = 'zone-canvas';
  canvas.style.cssText = `
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 10;
    cursor: default;
  `;
  container.style.position = 'relative';
  container.appendChild(canvas);

  editorState.canvas = canvas;
  editorState.ctx    = canvas.getContext('2d');

  // Bind events
  canvas.addEventListener('mousedown',  onMouseDown);
  canvas.addEventListener('mousemove',  onMouseMove);
  canvas.addEventListener('mouseup',    onMouseUp);
  canvas.addEventListener('mouseleave', onMouseLeave);

  if (saveBtn) saveBtn.addEventListener('click', saveAllZones);

  // Size canvas on resize
  new ResizeObserver(syncCanvasSize).observe(container);
  syncCanvasSize();
}

/** Toggle the editor on/off. */
export function setEditorActive(active) {
  editorState.active = active;
  const canvas    = editorState.canvas;
  const container = editorState.container;
  if (!canvas) return;

  if (active) {
    canvas.style.pointerEvents = 'all';
    container?.classList.add('zone-editing-active');
    loadZones();
  } else {
    canvas.style.pointerEvents = 'none';
    container?.classList.remove('zone-editing-active');
    editorState.dragging = false;
    editorState.dirty    = false;
    clearCanvas();
    setStatus('');
    if (editorState.saveBtn) editorState.saveBtn.disabled = true;
  }
}

// ─── Zone loading ─────────────────────────────────────────────────────────────

async function loadZones() {
  setStatus('Loading zones…');
  try {
    const data = await getZones();
    editorState.zones = data.zones || [];
    drawAll();
    setStatus(`${editorState.zones.length} zone(s) loaded. Drag to edit.`);
    if (editorState.saveBtn) editorState.saveBtn.disabled = true;
    editorState.dirty = false;
  } catch (e) {
    setStatus('⚠️ Failed to load zones');
  }
}

// ─── Canvas sizing ────────────────────────────────────────────────────────────

function syncCanvasSize() {
  const canvas = editorState.canvas;
  const container = editorState.container;
  if (!canvas || !container) return;
  canvas.width  = container.clientWidth;
  canvas.height = container.clientHeight;
  if (editorState.active) drawAll();
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────

/** Config [x,y] → canvas pixel [cx, cy]. */
function refToCanvas(rx, ry) {
  const c = editorState.canvas;
  return [
    (rx / REF_W) * c.width,
    (ry / REF_H) * c.height,
  ];
}

/** Canvas pixel [cx, cy] → config [x, y] (clamped to reference frame). */
function canvasToRef(cx, cy) {
  const c = editorState.canvas;
  return [
    Math.round(Math.max(0, Math.min(REF_W, (cx / c.width)  * REF_W))),
    Math.round(Math.max(0, Math.min(REF_H, (cy / c.height) * REF_H))),
  ];
}

/** Get canvas-space points for a zone (polygon or line). */
function getCanvasPoints(zone) {
  const pts = zone.type === 'polygon' ? zone.polygon : zone.line;
  if (!pts) return [];
  return pts.map(([rx, ry]) => refToCanvas(rx, ry));
}

// ─── Hit testing ──────────────────────────────────────────────────────────────

/** Returns {zoneIdx, handleIdx} for the topmost hit at (mx, my). */
function hitTest(mx, my) {
  const zones = editorState.zones;

  // Iterate zones in reverse so topmost-drawn is hit first
  for (let zi = zones.length - 1; zi >= 0; zi--) {
    const zone = zones[zi];
    const pts  = getCanvasPoints(zone);
    if (!pts.length) continue;

    // Check corner / endpoint handles first
    for (let hi = 0; hi < pts.length; hi++) {
      const [cx, cy] = pts[hi];
      if (Math.hypot(mx - cx, my - cy) <= HANDLE_R + 2) {
        return { zoneIdx: zi, handleIdx: hi };
      }
    }

    // Check zone body
    if (zone.type === 'polygon') {
      if (pointInPolygon(mx, my, pts)) {
        return { zoneIdx: zi, handleIdx: -1 };
      }
    } else if (zone.type === 'line_crossing') {
      if (pointNearLine(mx, my, pts[0], pts[1])) {
        return { zoneIdx: zi, handleIdx: -1 };
      }
    }
  }

  return { zoneIdx: -1, handleIdx: -1 };
}

function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersects =
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function pointNearLine(px, py, [x1, y1], [x2, y2]) {
  const dx = x2 - x1, dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - x1, py - y1) < LINE_HIT;
  let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const nx = x1 + t * dx, ny = y1 + t * dy;
  return Math.hypot(px - nx, py - ny) < LINE_HIT;
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

function clearCanvas() {
  const { ctx, canvas } = editorState;
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawAll() {
  clearCanvas();
  if (!editorState.active) return;

  const { ctx, zones, hoverZoneIdx } = editorState;

  zones.forEach((zone, zi) => {
    const pts     = getCanvasPoints(zone);
    if (!pts.length) return;

    const isHover    = zi === editorState.hoverZoneIdx;
    const isDragging = zi === editorState.dragZoneIdx && editorState.dragging;
    const color      = zoneColor(zone);

    if (zone.type === 'polygon') {
      drawPolygon(ctx, pts, color, isHover || isDragging);
    } else if (zone.type === 'line_crossing') {
      drawLine(ctx, pts, color, isHover || isDragging);
    }

    // Draw handles
    drawHandles(ctx, pts, color, isHover || isDragging);

    // Zone label
    drawLabel(ctx, zone, pts);
  });
}

function zoneColor(zone) {
  if (ZONE_COLORS[zone.id]) return ZONE_COLORS[zone.id];
  const [b, g, r] = zone.color || [200, 200, 200];
  return `rgba(${r}, ${g}, ${b}, 0.85)`;
}

function drawPolygon(ctx, pts, color, active) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...pts[0]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(...pts[i]);
  ctx.closePath();

  // Fill
  const fillColor = color.replace(/[\d.]+\)$/, `${FILL_ALPHA})`);
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Stroke
  ctx.strokeStyle = color;
  ctx.lineWidth = active ? STROKE_WIDTH + 1.5 : STROKE_WIDTH;
  ctx.setLineDash(active ? [6, 3] : []);
  ctx.stroke();
  ctx.restore();
}

function drawLine(ctx, pts, color, active) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...pts[0]);
  ctx.lineTo(...pts[1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = active ? 4 : 2.5;
  ctx.setLineDash(active ? [8, 4] : []);
  ctx.shadowColor = color;
  ctx.shadowBlur  = active ? 8 : 3;
  ctx.stroke();
  ctx.restore();
}

function drawHandles(ctx, pts, color, active) {
  pts.forEach(([cx, cy], hi) => {
    const isHoverHandle =
      editorState.hoverZoneIdx === editorState.hoverZoneIdx &&
      editorState.hoverHandleIdx === hi;

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, isHoverHandle ? HANDLE_R + 2 : HANDLE_R, 0, Math.PI * 2);
    ctx.fillStyle = active ? '#ffffff' : 'rgba(255,255,255,0.7)';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  });
}

function drawLabel(ctx, zone, pts) {
  if (!pts.length) return;

  // Centroid
  const cx = pts.reduce((s, [x]) => s + x, 0) / pts.length;
  const cy = pts.reduce((s, [, y]) => s + y, 0) / pts.length;
  const label = zone.name || zone.id;

  ctx.save();
  ctx.font = 'bold 12px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  // Background pill
  const tw = ctx.measureText(label).width + 12;
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.roundRect(cx - tw / 2, cy - 10, tw, 20, 5);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, cx, cy);
  ctx.restore();
}

// ─── Mouse Event Handlers ─────────────────────────────────────────────────────

function getRelativePos(e) {
  const rect = editorState.canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

function onMouseDown(e) {
  if (!editorState.active) return;
  const [mx, my] = getRelativePos(e);
  const { zoneIdx, handleIdx } = hitTest(mx, my);

  if (zoneIdx < 0) return;

  editorState.dragging      = true;
  editorState.dragZoneIdx   = zoneIdx;
  editorState.dragHandleIdx = handleIdx;
  editorState.dragStartX    = mx;
  editorState.dragStartY    = my;

  // Snapshot current ref points
  const zone = editorState.zones[zoneIdx];
  const pts  = zone.type === 'polygon' ? zone.polygon : zone.line;
  editorState.dragOrigPoints = pts.map(([x, y]) => [x, y]);

  editorState.canvas.style.cursor = 'grabbing';
  e.preventDefault();
}

function onMouseMove(e) {
  if (!editorState.active) return;
  const [mx, my] = getRelativePos(e);

  if (!editorState.dragging) {
    // Hover feedback
    const { zoneIdx, handleIdx } = hitTest(mx, my);
    editorState.hoverZoneIdx   = zoneIdx;
    editorState.hoverHandleIdx = handleIdx;

    if (zoneIdx < 0) {
      editorState.canvas.style.cursor = 'default';
    } else if (handleIdx >= 0) {
      editorState.canvas.style.cursor = 'crosshair';
    } else {
      editorState.canvas.style.cursor = 'grab';
    }

    drawAll();
    return;
  }

  // Drag
  const dx = mx - editorState.dragStartX;
  const dy = my - editorState.dragStartY;

  const c      = editorState.canvas;
  const zone   = editorState.zones[editorState.dragZoneIdx];
  const isLine = zone.type === 'line_crossing';
  const orig   = editorState.dragOrigPoints;

  // Convert delta from canvas px to ref px
  const refDx = (dx / c.width)  * REF_W;
  const refDy = (dy / c.height) * REF_H;

  const pts = isLine ? zone.line : zone.polygon;

  if (editorState.dragHandleIdx >= 0) {
    // Move single handle
    const hi  = editorState.dragHandleIdx;
    const [ox, oy] = orig[hi];
    pts[hi] = [
      Math.round(Math.max(0, Math.min(REF_W, ox + refDx))),
      Math.round(Math.max(0, Math.min(REF_H, oy + refDy))),
    ];
  } else {
    // Move whole zone
    orig.forEach(([ox, oy], i) => {
      pts[i] = [
        Math.round(Math.max(0, Math.min(REF_W, ox + refDx))),
        Math.round(Math.max(0, Math.min(REF_H, oy + refDy))),
      ];
    });
  }

  editorState.dirty = true;
  if (editorState.saveBtn) editorState.saveBtn.disabled = false;
  setStatus('Unsaved changes — click 💾 Save to apply');
  drawAll();
}

function onMouseUp(e) {
  if (!editorState.dragging) return;
  editorState.dragging = false;
  editorState.canvas.style.cursor = 'grab';
}

function onMouseLeave() {
  editorState.hoverZoneIdx   = -1;
  editorState.hoverHandleIdx = -1;
  if (!editorState.dragging) drawAll();
}

// ─── Save ─────────────────────────────────────────────────────────────────────

async function saveAllZones() {
  if (!editorState.dirty) return;

  setStatus('Saving…');
  if (editorState.saveBtn) editorState.saveBtn.disabled = true;

  const errors = [];

  for (const zone of editorState.zones) {
    const payload =
      zone.type === 'polygon'
        ? { polygon: zone.polygon }
        : { line: zone.line };
    try {
      await updateZone(zone.id, payload);
    } catch (err) {
      errors.push(`${zone.id}: ${err.message}`);
    }
  }

  if (errors.length) {
    setStatus(`⚠️ Some zones failed to save: ${errors.join('; ')}`);
    if (editorState.saveBtn) editorState.saveBtn.disabled = false;
  } else {
    editorState.dirty = false;
    setStatus('✅ All zones saved! Pipeline will use new coordinates.');
  }
}

// ─── Status ───────────────────────────────────────────────────────────────────

function setStatus(msg) {
  if (editorState.statusEl) editorState.statusEl.textContent = msg;
}
