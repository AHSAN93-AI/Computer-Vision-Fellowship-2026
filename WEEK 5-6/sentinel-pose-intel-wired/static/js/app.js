// Sentinel Pose Intel — vanilla JS dashboard
// Polls the Flask REST API in place of the old React/WebSocket frontend.

const POLL_MS = 1000;
let soundEnabled = true;
let lastAlertIds = new Set();

// ── Tabs ──────────────────────────────────────────────
const tabs = document.querySelectorAll(".tab");
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.getElementById("view-" + tab.dataset.tab).classList.add("active");

    if (tab.dataset.tab === "analytics") loadAnalytics();
    if (tab.dataset.tab === "alerts") loadAlertHistory();
    if (tab.dataset.tab === "config") loadConfig();
  });
});

// ── Clock ─────────────────────────────────────────────
function tickClock() {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString();
}
setInterval(tickClock, 1000);
tickClock();

// ── Alert sound (Web Audio synth beep) ───────────────
document.getElementById("soundToggle").addEventListener("click", (e) => {
  soundEnabled = !soundEnabled;
  e.target.classList.toggle("muted", !soundEnabled);
});

function playAlertSound() {
  if (!soundEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (err) {
    // Audio context may require a user gesture first — ignore.
  }
}

// ── Pipeline controls ─────────────────────────────────
document.getElementById("startBtn").addEventListener("click", async () => {
  const source = document.getElementById("sourceInput").value.trim();
  const url = source ? `/api/pipeline/start?source=${encodeURIComponent(source)}` : "/api/pipeline/start";
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  if (!data.success) alert("Failed to start pipeline: " + (data.error || "unknown error"));
});

document.getElementById("stopBtn").addEventListener("click", async () => {
  await fetch("/api/pipeline/stop", { method: "POST" });
});

// ── Live polling loop ─────────────────────────────────
async function pollLive() {
  try {
    const res = await fetch("/api/live");
    const data = await res.json();
    document.getElementById("connDot").classList.toggle("live", data.status.isRunning);
    renderStatus(data.status);
    renderCamera(data.cameras[0]);
    renderEntities(data.trackedEntities);
    renderAlerts(data.alerts);
  } catch (err) {
    document.getElementById("connDot").classList.remove("live");
  }
}
setInterval(pollLive, POLL_MS);
pollLive();

function renderStatus(status) {
  document.getElementById("ovFps").textContent = status.fps.toFixed(1);
  document.getElementById("ovInfer").textContent = status.inferenceTimeMs.toFixed(1) + "ms";
  document.getElementById("ovPeople").textContent = status.activePeople;
  document.getElementById("statFalls").textContent = status.totalFalls;
  document.getElementById("statPosture").textContent = status.totalPostureRisks;
  document.getElementById("statActivities").textContent = status.totalActivities;
  document.getElementById("statFrame").textContent = status.frameNumber;
}

function renderCamera(cam) {
  const badge = document.getElementById("camStatusBadge");
  if (!cam) {
    badge.textContent = "IDLE";
    badge.className = "badge idle";
    return;
  }
  badge.textContent = cam.statusText;
  badge.className = "badge " + cam.status.toLowerCase();
}

function renderEntities(entities) {
  const tbody = document.querySelector("#entitiesTable tbody");
  tbody.innerHTML = "";
  if (!entities.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-msg">No tracked entities</td></tr>';
    return;
  }
  for (const e of entities) {
    const tr = document.createElement("tr");
    if (e.isAlarm) tr.classList.add("alarm-row");
    if (e.fallCnnLabel === "Fall") tr.classList.add("fall-flag");
    tr.innerHTML = `
      <td>${e.id}</td>
      <td>${e.posture}</td>
      <td>${e.duration}</td>
      <td>${(e.confidence * 100).toFixed(0)}%</td>
      <td>${e.fallCnnLabel} (${(e.fallCnnConfidence * 100).toFixed(0)}%)</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderAlerts(alerts) {
  const list = document.getElementById("alertsList");
  list.innerHTML = "";
  if (!alerts.length) {
    list.innerHTML = '<div class="empty-msg">No active alerts</div>';
  }

  const currentIds = new Set();
  for (const a of alerts) {
    currentIds.add(a.id);
    if (!lastAlertIds.has(a.id) && a.severity === "CRITICAL") {
      playAlertSound();
    }
    const div = document.createElement("div");
    div.className = "alert-item" + (a.severity === "CRITICAL" ? " critical" : "") + (a.acknowledged ? " acked" : "");
    div.innerHTML = `
      <div class="alert-title">${a.title}</div>
      <div class="alert-meta"><span>${a.targetId} @ ${a.location}</span><span>${a.timestamp}</span></div>
      ${a.acknowledged ? "" : `<button class="btn btn-small" data-ack="${a.id}">Acknowledge</button>`}
    `;
    list.appendChild(div);
  }
  lastAlertIds = currentIds;

  list.querySelectorAll("[data-ack]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/alerts/${btn.dataset.ack}/acknowledge`, { method: "POST" });
      pollLive();
    });
  });
}

// ── Analytics view ────────────────────────────────────
async function loadAnalytics() {
  const [statsRes, eventsRes] = await Promise.all([
    fetch("/api/events/stats"),
    fetch("/api/events?limit=200"),
  ]);
  const stats = await statsRes.json();
  const events = await eventsRes.json();

  const typeBody = document.querySelector("#typeCountsTable tbody");
  typeBody.innerHTML = "";
  const types = Object.keys(stats.typeCounts || {});
  if (!types.length) {
    typeBody.innerHTML = '<tr><td colspan="3" class="empty-msg">No activity data yet</td></tr>';
  }
  for (const t of types) {
    const tr = document.createElement("tr");
    const avg = stats.avgDurations[t] !== undefined ? stats.avgDurations[t].toFixed(1) : "-";
    tr.innerHTML = `<td>${t}</td><td>${stats.typeCounts[t]}</td><td>${avg}</td>`;
    typeBody.appendChild(tr);
  }

  const evBody = document.querySelector("#eventsTable tbody");
  evBody.innerHTML = "";
  if (!events.length) {
    evBody.innerHTML = '<tr><td colspan="5" class="empty-msg">No events yet</td></tr>';
  }
  for (const ev of events) {
    const tr = document.createElement("tr");
    const start = ev.start_time ? new Date(ev.start_time * 1000).toLocaleTimeString() : "-";
    const dur = ev.duration !== null && ev.duration !== undefined ? Number(ev.duration).toFixed(1) + "s" : "-";
    tr.innerHTML = `
      <td>#${String(ev.person_id).padStart(3, "0")}</td>
      <td>${ev.activity_type}</td>
      <td>${start}</td>
      <td>${dur}</td>
      <td>${((ev.confidence || 0) * 100).toFixed(0)}%</td>
    `;
    evBody.appendChild(tr);
  }
}

// ── Alert history view ────────────────────────────────
async function loadAlertHistory() {
  const res = await fetch("/api/alerts?limit=200");
  const alerts = await res.json();
  const tbody = document.querySelector("#alertHistoryTable tbody");
  tbody.innerHTML = "";
  if (!alerts.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-msg">No alerts recorded</td></tr>';
    return;
  }
  for (const a of alerts) {
    const tr = document.createElement("tr");
    const ts = a.created_at || a.timestamp || "-";
    tr.innerHTML = `
      <td>${a.alert_id || "-"}</td>
      <td>${a.alert_type || "-"}</td>
      <td>${a.severity}</td>
      <td>#${String(a.person_id ?? "-").padStart(3, "0")}</td>
      <td>${a.status}</td>
      <td>${ts}</td>
      <td>${a.status !== "acknowledged" ? `<button class="btn btn-small" data-ack-h="${a.alert_id}">Ack</button>` : ""}</td>
    `;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll("[data-ack-h]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/alerts/${btn.dataset.ackH}/acknowledge`, { method: "POST" });
      loadAlertHistory();
    });
  });
}

// ── Config view ────────────────────────────────────────
async function loadConfig() {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  const form = document.getElementById("configForm");
  for (const [key, value] of Object.entries(cfg)) {
    const field = form.elements[key];
    if (!field) continue;
    if (field.type === "checkbox") field.checked = !!value;
    else field.value = value;
  }
}

document.getElementById("configForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const payload = {
    videoSource: form.videoSource.value,
    poseModel: form.poseModel.value,
    detectionThreshold: parseFloat(form.detectionThreshold.value),
    keypointThreshold: parseFloat(form.keypointThreshold.value),
    sequenceLength: parseInt(form.sequenceLength.value, 10),
    fallConfirmFrames: parseInt(form.fallConfirmFrames.value, 10),
    alertCooldownSeconds: parseFloat(form.alertCooldownSeconds.value),
    inactivityWarnSeconds: parseFloat(form.inactivityWarnSeconds.value),
    useFallClassifier: form.useFallClassifier.checked,
    fallClassifierConfidenceThreshold: parseFloat(form.fallClassifierConfidenceThreshold.value),
  };
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  const msg = document.getElementById("configSaveMsg");
  msg.textContent = data.success ? "Saved." : "Failed to save.";
  setTimeout(() => (msg.textContent = ""), 2000);
});
