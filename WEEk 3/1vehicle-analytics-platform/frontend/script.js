const API = "";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const feedHint = document.getElementById("feed-hint");
const sourceLabel = document.getElementById("source-label");
const drawInstructions = document.getElementById("draw-instructions");

const engineDot = document.getElementById("engine-dot");
const engineStatusText = document.getElementById("engine-status-text");
const recDot = document.getElementById("rec-dot");
const recStatusText = document.getElementById("rec-status-text");
const recordBtn = document.getElementById("btn-record");

let mode = "idle";          // idle | line | roi
let linePoints = [];
let roiPoints = [];
let savedLine = null;
let savedRoi = null;
let isRunning = false;
let isRecording = false;

// ------------------------------------------------------------- helpers --
function resizeCanvas() {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  redraw();
}
window.addEventListener("resize", resizeCanvas);

function scaleFromDisplay(x, y) {
  // maps a click on the displayed canvas to the underlying frame resolution
  const scaleX = (video.naturalWidth || overlay.width) / overlay.width;
  const scaleY = (video.naturalHeight || overlay.height) / overlay.height;
  return [Math.round(x * scaleX), Math.round(y * scaleY)];
}

function displayFromFrame(x, y) {
  const scaleX = overlay.width / (video.naturalWidth || overlay.width);
  const scaleY = overlay.height / (video.naturalHeight || overlay.height);
  return [x * scaleX, y * scaleY];
}

function redraw() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (savedLine) {
    const [x1, y1] = displayFromFrame(savedLine[0][0], savedLine[0][1]);
    const [x2, y2] = displayFromFrame(savedLine[1][0], savedLine[1][1]);
    drawLine(x1, y1, x2, y2, "#ffb020");
  }
  if (savedRoi) {
    drawPolygon(savedRoi.map(p => displayFromFrame(p[0], p[1])), "#35f2a0");
  }

  if (mode === "line" && linePoints.length === 1) {
    const [x, y] = linePoints[0];
    drawDot(x, y, "#ffb020");
  }
  if (mode === "roi" && roiPoints.length > 0) {
    drawPolygon(roiPoints, "#35f2a0", false);
    roiPoints.forEach(([x, y]) => drawDot(x, y, "#35f2a0"));
  }
}

function drawLine(x1, y1, x2, y2, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function drawDot(x, y, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
}

function drawPolygon(points, color, close = true) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  if (close) ctx.closePath();
  ctx.stroke();
}

// ------------------------------------------------------------- sources --
async function startWebcam() {
  await fetch(`${API}/api/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: "webcam" }),
  });
  attachStream();
  sourceLabel.textContent = "webcam";
}

async function startFile(filename) {
  await fetch(`${API}/api/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: "file", path: `../uploads/${filename}` }),
  });
  attachStream();
  sourceLabel.textContent = filename;
}

function attachStream() {
  video.src = `${API}/video_feed?t=${Date.now()}`;
  feedHint.style.display = "none";
  isRunning = true;
  engineDot.className = "dot dot-live";
  engineStatusText.textContent = "LIVE";
  video.onload = resizeCanvas;
  setTimeout(resizeCanvas, 400);
}

document.getElementById("btn-webcam").addEventListener("click", startWebcam);

document.getElementById("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("video", file);
  drawInstructions.textContent = "Uploading video...";
  const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
  const data = await res.json();
  drawInstructions.textContent = "";
  if (data.filename) startFile(data.filename);
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  await fetch(`${API}/api/stop`, { method: "POST" });
  isRunning = false;
  engineDot.className = "dot dot-idle";
  engineStatusText.textContent = "STANDBY";
});

document.getElementById("btn-remove-video").addEventListener("click", async () => {
  await fetch(`${API}/api/remove_source`, { method: "POST" });
  isRunning = false;
  video.src = "";
  feedHint.style.display = "flex";
  feedHint.textContent = "Select a source below to begin analysis";
  sourceLabel.textContent = "no source";
  engineDot.className = "dot dot-idle";
  engineStatusText.textContent = "STANDBY";
  savedLine = null;
  savedRoi = null;
  mode = "idle";
  drawInstructions.textContent = "";
  document.getElementById("file-input").value = "";
  redraw();
});

// -------------------------------------------------------------- drawing -
document.getElementById("btn-draw-line").addEventListener("click", () => {
  mode = "line";
  linePoints = [];
  drawInstructions.textContent = "Click two points on the feed to place the counting line.";
});

document.getElementById("btn-draw-roi").addEventListener("click", () => {
  mode = "roi";
  roiPoints = [];
  drawInstructions.textContent = "Click to add zone points, then press Enter to finish (min 3 points).";
});

document.getElementById("btn-clear").addEventListener("click", async () => {
  savedLine = null;
  savedRoi = null;
  linePoints = [];
  roiPoints = [];
  mode = "idle";
  drawInstructions.textContent = "";
  await fetch(`${API}/api/clear_line`, { method: "POST" });
  await fetch(`${API}/api/clear_roi`, { method: "POST" });
  redraw();
});

overlay.addEventListener("click", (e) => {
  const rect = overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (mode === "line") {
    linePoints.push([x, y]);
    redraw();
    if (linePoints.length === 2) {
      const [fx1, fy1] = scaleFromDisplay(linePoints[0][0], linePoints[0][1]);
      const [fx2, fy2] = scaleFromDisplay(linePoints[1][0], linePoints[1][1]);
      savedLine = [[fx1, fy1], [fx2, fy2]];
      fetch(`${API}/api/set_line`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x1: fx1, y1: fy1, x2: fx2, y2: fy2 }),
      });
      mode = "idle";
      drawInstructions.textContent = "Counting line set.";
      linePoints = [];
      redraw();
    }
  } else if (mode === "roi") {
    roiPoints.push([x, y]);
    redraw();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && mode === "roi" && roiPoints.length >= 3) {
    const framePts = roiPoints.map(([x, y]) => scaleFromDisplay(x, y));
    savedRoi = framePts;
    fetch(`${API}/api/set_roi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: framePts }),
    });
    mode = "idle";
    drawInstructions.textContent = "ROI zone set.";
    roiPoints = [];
    redraw();
  }
});

// ------------------------------------------------------------ threshold -
const confSlider = document.getElementById("conf-slider");
const confValue = document.getElementById("conf-value");
confSlider.addEventListener("input", () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
confSlider.addEventListener("change", () => {
  fetch(`${API}/api/set_threshold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value: parseFloat(confSlider.value) }),
  });
});

// --------------------------------------------------------------- image --
const imageInput = document.getElementById("image-input");
const imagePanel = document.getElementById("image-panel");
const imageResultImg = document.getElementById("image-result-img");
const imageStats = document.getElementById("image-stats");

imageInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("image", file);

  drawInstructions.textContent = "Analyzing image...";
  const res = await fetch(`${API}/api/detect_image`, { method: "POST", body: formData });
  const data = await res.json();
  drawInstructions.textContent = "";

  if (data.error) {
    alert(data.error);
    return;
  }

  imageResultImg.src = data.image;
  const s = data.stats;
  const perClassRows = Object.entries(s.per_class_count || {})
    .map(([cls, count]) => `<div class="row"><span>${cls}</span><span>${count}</span></div>`)
    .join("");
  imageStats.innerHTML = `
    <div class="row"><span>Total Objects</span><span>${s.total_objects}</span></div>
    ${perClassRows}
    <div class="row"><span>Avg Confidence</span><span>${s.avg_confidence.toFixed(2)}</span></div>
  `;
  imagePanel.style.display = "block";
  imagePanel.scrollIntoView({ behavior: "smooth" });

  // reset the input so the same file can be re-selected later
  imageInput.value = "";
});

document.getElementById("close-image-panel").addEventListener("click", () => {
  imagePanel.style.display = "none";
});

// -------------------------------------------------------------- heatmap -
let heatmapOn = false;
const heatmapBtn = document.getElementById("btn-heatmap");
heatmapBtn.addEventListener("click", async () => {
  heatmapOn = !heatmapOn;
  await fetch(`${API}/api/heatmap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: heatmapOn }),
  });
  heatmapBtn.textContent = heatmapOn ? "▦ Heatmap: On" : "▦ Heatmap: Off";
  heatmapBtn.classList.toggle("active", heatmapOn);
});

// ------------------------------------------------------------- record ---
recordBtn.addEventListener("click", async () => {
  if (!isRecording) {
    await fetch(`${API}/api/record/start`, { method: "POST" });
    isRecording = true;
    recordBtn.classList.add("active");
    recordBtn.textContent = "■ Stop Recording";
    recDot.className = "dot dot-rec";
    recStatusText.textContent = "RECORDING";
  } else {
    await fetch(`${API}/api/record/stop`, { method: "POST" });
    isRecording = false;
    recordBtn.classList.remove("active");
    recordBtn.textContent = "● Record";
    recDot.className = "dot dot-off";
    recStatusText.textContent = "NOT RECORDING";
    loadRecordings();
  }
});

// ------------------------------------------------------------ recordings-
async function loadRecordings() {
  const res = await fetch(`${API}/api/recordings`);
  const data = await res.json();
  const list = document.getElementById("recordings-list");
  if (!data.recordings || data.recordings.length === 0) {
    list.innerHTML = `<div class="empty-note">No recordings yet</div>`;
    return;
  }
  list.innerHTML = data.recordings
    .map(
      (f) => `<div class="recording-item"><span>${f}</span><a href="/api/recordings/${f}" download>↓ download</a></div>`
    )
    .join("");
}

// ------------------------------------------------------------ stats poll-
async function pollStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    const s = await res.json();
    document.getElementById("stat-active").textContent = s.active_objects;
    document.getElementById("stat-total").textContent = s.total_unique_objects;
    document.getElementById("stat-in").textContent = s.entered;
    document.getElementById("stat-out").textContent = s.exited;
    document.getElementById("stat-line-in").textContent = s.line_in;
    document.getElementById("stat-line-out").textContent = s.line_out;
    document.getElementById("stat-roi").textContent = s.roi_count;
    document.getElementById("stat-conf").textContent = s.avg_confidence.toFixed(2);
    document.getElementById("stat-fps").textContent = s.fps.toFixed(1);
    document.getElementById("stat-latency").textContent = `${s.processing_time_ms.toFixed(0)} ms`;
  } catch (err) {
    // backend not reachable yet
  }
}
setInterval(pollStats, 1000);
loadRecordings();
resizeCanvas();
