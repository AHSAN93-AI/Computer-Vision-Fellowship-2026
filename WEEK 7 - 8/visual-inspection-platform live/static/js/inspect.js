// ---------------------------------------------------------------------------
// Shared elements
// ---------------------------------------------------------------------------
const resultPanel = document.getElementById('result-panel');
const emptyHint = document.getElementById('empty-hint');
const productTypeSelect = document.getElementById('product-type');
const captureCanvas = document.getElementById('capture-canvas');

function setClock() {
  const el = document.getElementById('clock');
  if (el) el.textContent = new Date().toLocaleString();
}
setClock();
setInterval(setClock, 1000);

// Grabs the current frame of a <video> element onto the shared canvas and
// resolves with both a Blob (to upload) and a data URL (to preview).
function grabFrame(videoEl) {
  return new Promise((resolve, reject) => {
    if (!videoEl || !videoEl.videoWidth) {
      reject(new Error('Video is not ready yet.'));
      return;
    }
    captureCanvas.width = videoEl.videoWidth;
    captureCanvas.height = videoEl.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob((blob) => {
      if (!blob) { reject(new Error('Could not capture a frame.')); return; }
      resolve({ blob, dataUrl: captureCanvas.toDataURL('image/jpeg', 0.9) });
    }, 'image/jpeg', 0.9);
  });
}

// Posts a single frame/image through the existing single-image pipeline and
// renders the result. Every source (upload, webcam, conveyor, video) funnels
// through here so the backend never needs to know where the pixels came from.
async function runInspection(blob, previewUrl) {
  const fd = new FormData();
  fd.append('image', blob, 'frame.jpg');
  fd.append('product_type', productTypeSelect.value);

  const res = await fetch('/api/inspect', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Inspection failed');

  renderResult(data, previewUrl);
  refreshLineStrip();
  return data;
}

function renderResult(data, imgUrl) {
  emptyHint.style.display = 'none';
  resultPanel.style.display = 'block';

  if (imgUrl) document.getElementById('result-img').src = imgUrl;

  const stamp = document.getElementById('stamp');
  stamp.textContent = data.status;
  stamp.className = 'stamp is-' + data.status.toLowerCase();

  document.getElementById('result-subtitle').textContent =
    data.status === 'INVALID' ? ('Rejected before inference: ' + data.invalid_reason) : ('Inspection ' + data.inspection_id);

  document.getElementById('r-id').textContent = data.inspection_id;

  const statusBadge = document.getElementById('r-status');
  statusBadge.innerHTML = `<span class="badge is-${data.status.toLowerCase()}">${data.status}</span>`;

  document.getElementById('r-defects').textContent = (data.defects_detected && data.defects_detected.length)
    ? data.defects_detected.join(', ') : 'none';

  const sev = document.getElementById('r-severity');
  if (data.max_severity && data.max_severity !== 'None') {
    sev.innerHTML = `<span class="badge is-${data.max_severity.toLowerCase()}">${data.max_severity}</span>`;
  } else {
    sev.textContent = '—';
  }

  document.getElementById('r-conf').textContent = data.classifier_confidence != null ? data.classifier_confidence.toFixed(3) : '—';
  document.getElementById('r-anomaly').textContent = data.anomaly_score != null ? data.anomaly_score.toFixed(4) : '—';
  document.getElementById('r-area').textContent = data.defect_area_ratio != null ? data.defect_area_ratio.toFixed(4) : '—';
  document.getElementById('r-time').textContent = data.processing_time_ms != null ? data.processing_time_ms.toFixed(1) + ' ms' : '—';

  const reasons = document.getElementById('result-reasons');
  if (data.reasons && data.reasons.length) {
    reasons.textContent = 'Reasons: ' + data.reasons.join(' · ');
  } else {
    reasons.textContent = '';
  }
}

async function refreshLineStrip() {
  try {
    const res = await fetch('/api/analytics');
    const a = await res.json();
    document.getElementById('strip-total').textContent = a.total_inspections;
    document.getElementById('strip-pass').textContent = a.passed;
    document.getElementById('strip-fail').textContent = a.failed;
    document.getElementById('strip-invalid').textContent = a.invalid;
    document.getElementById('strip-rate').textContent = (a.pass_rate * 100).toFixed(1) + '%';
  } catch (e) { /* silent -- line strip is supplementary */ }
}
refreshLineStrip();

// ---------------------------------------------------------------------------
// Source tabs
// ---------------------------------------------------------------------------
const sourceChips = document.querySelectorAll('#source-tabs .chip[data-source]');
const sourcePanes = {
  image: document.getElementById('source-image'),
  webcam: document.getElementById('source-webcam'),
  conveyor: document.getElementById('source-conveyor'),
  video: document.getElementById('source-video'),
};

sourceChips.forEach((chip) => {
  chip.addEventListener('click', () => {
    const target = chip.dataset.source;
    sourceChips.forEach((c) => c.classList.toggle('is-active', c === chip));
    Object.entries(sourcePanes).forEach(([key, pane]) => {
      pane.style.display = key === target ? 'block' : 'none';
    });
    if (target !== 'webcam') stopWebcam();
    if (target !== 'conveyor') stopConveyorAuto();
  });
});

// ---------------------------------------------------------------------------
// Source: single image (dropzone)
// ---------------------------------------------------------------------------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const form = document.getElementById('inspect-form');
const runBtn = document.getElementById('run-btn');

let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('is-dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('is-dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('is-dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = 'block';
  dropzoneEmpty.style.display = 'none';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFile) {
    alert('Select or drop an image first.');
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = 'Running inspection…';
  try {
    await runInspection(selectedFile, previewImg.src);
  } catch (err) {
    alert('Inspection failed: ' + err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run inspection';
  }
});

// ---------------------------------------------------------------------------
// Source: webcam / USB industrial camera
// ---------------------------------------------------------------------------
const webcamVideo = document.getElementById('webcam-video');
const webcamEmpty = document.getElementById('webcam-empty');
const webcamStartBtn = document.getElementById('webcam-start-btn');
const webcamStopBtn = document.getElementById('webcam-stop-btn');
const webcamCaptureBtn = document.getElementById('webcam-capture-btn');
const webcamDeviceSelect = document.getElementById('webcam-device-select');
const webcamDeviceField = document.getElementById('webcam-device-field');
const webcamStatus = document.getElementById('webcam-status');

let webcamStream = null;

async function populateCameraList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === 'videoinput');
    webcamDeviceSelect.innerHTML = '';
    cams.forEach((cam, i) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      webcamDeviceSelect.appendChild(opt);
    });
    webcamDeviceField.style.display = cams.length > 1 ? 'block' : 'none';
  } catch (e) { /* best-effort */ }
}
populateCameraList();
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', populateCameraList);
}

async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    webcamStatus.textContent = 'This browser does not support camera access.';
    webcamStatus.className = 'pane-status is-error';
    return;
  }
  webcamStatus.textContent = 'Requesting camera access…';
  webcamStatus.className = 'pane-status';
  try {
    const constraints = {
      video: webcamDeviceSelect.value
        ? { deviceId: { exact: webcamDeviceSelect.value } }
        : true,
    };
    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamVideo.srcObject = webcamStream;
    webcamVideo.style.display = 'block';
    webcamEmpty.style.display = 'none';
    webcamStartBtn.disabled = true;
    webcamStopBtn.disabled = false;
    webcamCaptureBtn.disabled = false;
    webcamStatus.textContent = 'Live — camera streaming.';
    webcamStatus.className = 'pane-status is-live';
    await populateCameraList();
  } catch (err) {
    webcamStatus.textContent = 'Camera error: ' + (err.message || err.name || 'permission denied');
    webcamStatus.className = 'pane-status is-error';
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  webcamVideo.srcObject = null;
  webcamVideo.style.display = 'none';
  webcamEmpty.style.display = 'block';
  webcamStartBtn.disabled = false;
  webcamStopBtn.disabled = true;
  webcamCaptureBtn.disabled = true;
  webcamStatus.textContent = '';
  webcamStatus.className = 'pane-status';
}

webcamStartBtn.addEventListener('click', startWebcam);
webcamStopBtn.addEventListener('click', stopWebcam);
webcamCaptureBtn.addEventListener('click', async () => {
  webcamCaptureBtn.disabled = true;
  webcamCaptureBtn.textContent = 'Inspecting…';
  try {
    const { blob, dataUrl } = await grabFrame(webcamVideo);
    await runInspection(blob, dataUrl);
  } catch (err) {
    alert('Inspection failed: ' + err.message);
  } finally {
    webcamCaptureBtn.disabled = !webcamStream;
    webcamCaptureBtn.textContent = 'Capture frame & inspect';
  }
});

// ---------------------------------------------------------------------------
// Source: recorded conveyor video (auto-inspects on a timer while it plays)
// ---------------------------------------------------------------------------
const conveyorZone = document.getElementById('conveyor-input-zone');
const conveyorFileInput = document.getElementById('conveyor-file-input');
const conveyorVideo = document.getElementById('conveyor-video');
const conveyorChangeRow = document.getElementById('conveyor-change-row');
const conveyorChangeBtn = document.getElementById('conveyor-change-btn');
const conveyorIntervalInput = document.getElementById('conveyor-interval');
const conveyorToggleBtn = document.getElementById('conveyor-toggle-btn');
const conveyorStatus = document.getElementById('conveyor-status');
const conveyorLog = document.getElementById('conveyor-log');

let conveyorTimer = null;
let conveyorBusy = false;

conveyorZone.addEventListener('click', () => conveyorFileInput.click());
conveyorFileInput.addEventListener('change', () => {
  if (conveyorFileInput.files.length) loadConveyorVideo(conveyorFileInput.files[0]);
});

function loadConveyorVideo(file) {
  stopConveyorAuto();
  conveyorVideo.src = URL.createObjectURL(file);
  conveyorVideo.style.display = 'block';
  conveyorZone.style.display = 'none';
  conveyorChangeRow.style.display = 'flex';
  conveyorToggleBtn.disabled = false;
  conveyorVideo.play().catch(() => { /* user can hit play manually */ });
}

conveyorChangeBtn.addEventListener('click', () => {
  stopConveyorAuto();
  conveyorVideo.pause();
  conveyorVideo.removeAttribute('src');
  conveyorVideo.load();
  conveyorVideo.style.display = 'none';
  conveyorZone.style.display = 'block';
  conveyorChangeRow.style.display = 'none';
  conveyorToggleBtn.disabled = true;
});

function addConveyorLogRow(data, errorMessage) {
  const empty = conveyorLog.querySelector('.conveyor-log__empty');
  if (empty) empty.remove();
  const row = document.createElement('div');
  row.className = 'conveyor-log__row';
  const time = new Date().toLocaleTimeString();
  if (errorMessage) {
    row.innerHTML = `<span>${errorMessage}</span><span class="conveyor-log__meta">${time}</span>`;
  } else {
    row.innerHTML = `<span>${data.inspection_id}</span><span class="badge is-${data.status.toLowerCase()}">${data.status}</span><span class="conveyor-log__meta">${time}</span>`;
  }
  conveyorLog.prepend(row);
  while (conveyorLog.children.length > 8) conveyorLog.removeChild(conveyorLog.lastChild);
}

async function conveyorTick() {
  if (conveyorBusy || conveyorVideo.paused || conveyorVideo.ended) return;
  conveyorBusy = true;
  try {
    const { blob, dataUrl } = await grabFrame(conveyorVideo);
    const data = await runInspection(blob, dataUrl);
    addConveyorLogRow(data, null);
  } catch (err) {
    addConveyorLogRow(null, err.message);
  } finally {
    conveyorBusy = false;
  }
}

function startConveyorAuto() {
  const secs = Math.max(1, parseFloat(conveyorIntervalInput.value) || 3);
  conveyorVideo.play().catch(() => {});
  conveyorTick();
  conveyorTimer = setInterval(conveyorTick, secs * 1000);
  conveyorToggleBtn.textContent = 'Stop auto-inspection';
  conveyorStatus.textContent = `Auto-inspecting every ${secs}s while the footage plays.`;
  conveyorStatus.className = 'pane-status is-live';
}

function stopConveyorAuto() {
  if (conveyorTimer) {
    clearInterval(conveyorTimer);
    conveyorTimer = null;
  }
  conveyorToggleBtn.textContent = 'Start auto-inspection';
  if (conveyorStatus) {
    conveyorStatus.textContent = '';
    conveyorStatus.className = 'pane-status';
  }
}

conveyorToggleBtn.addEventListener('click', () => {
  if (conveyorTimer) stopConveyorAuto();
  else startConveyorAuto();
});

// ---------------------------------------------------------------------------
// Source: uploaded video (scrub to a frame, capture it manually)
// ---------------------------------------------------------------------------
const videoZone = document.getElementById('video-input-zone');
const videoFileInput = document.getElementById('video-file-input');
const uploadVideo = document.getElementById('upload-video');
const videoChangeRow = document.getElementById('video-change-row');
const videoChangeBtn = document.getElementById('video-change-btn');
const videoCaptureBtn = document.getElementById('video-capture-btn');

videoZone.addEventListener('click', () => videoFileInput.click());
videoFileInput.addEventListener('change', () => {
  if (videoFileInput.files.length) loadUploadVideo(videoFileInput.files[0]);
});

function loadUploadVideo(file) {
  uploadVideo.src = URL.createObjectURL(file);
  uploadVideo.style.display = 'block';
  videoZone.style.display = 'none';
  videoChangeRow.style.display = 'flex';
  videoCaptureBtn.disabled = false;
}

videoChangeBtn.addEventListener('click', () => {
  uploadVideo.pause();
  uploadVideo.removeAttribute('src');
  uploadVideo.load();
  uploadVideo.style.display = 'none';
  videoZone.style.display = 'block';
  videoChangeRow.style.display = 'none';
  videoCaptureBtn.disabled = true;
});

videoCaptureBtn.addEventListener('click', async () => {
  videoCaptureBtn.disabled = true;
  videoCaptureBtn.textContent = 'Inspecting…';
  try {
    const { blob, dataUrl } = await grabFrame(uploadVideo);
    await runInspection(blob, dataUrl);
  } catch (err) {
    alert('Inspection failed: ' + err.message);
  } finally {
    videoCaptureBtn.disabled = false;
    videoCaptureBtn.textContent = 'Capture current frame & inspect';
  }
});
