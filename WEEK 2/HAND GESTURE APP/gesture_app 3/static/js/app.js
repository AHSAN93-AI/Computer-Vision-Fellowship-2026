/* GestureVision — webcam + image + video detection, shared stats & model switch */

(() => {
  // ---------------------------------------------------------------------
  // Shared state
  // ---------------------------------------------------------------------
  const CLASSES = window.GESTURE_CLASSES || [];
  let currentModel = window.DEFAULT_MODEL || "augmented";
  let threshold = 0.5;

  const legendList = document.getElementById("legendList");
  const detectionsList = document.getElementById("detectionsList");
  const detCount = document.getElementById("detCount");
  const footerStatus = document.getElementById("footerStatus");

  const statObjects = document.getElementById("statObjects");
  const statTime = document.getElementById("statTime");
  const statAvgConf = document.getElementById("statAvgConf");
  const statMinMax = document.getElementById("statMinMax");
  const confScores = document.getElementById("confScores");

  // -----------------------------------------------------------------
  // Tabs
  // -----------------------------------------------------------------
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      tabButtons.forEach((b) => b.classList.toggle("is-active", b === btn));
      tabPanels.forEach((p) => p.classList.toggle("is-active", p.dataset.panel === mode));

      // Stop the webcam if the user navigates away from that tab
      if (mode !== "webcam") {
        stopCamera();
      }
    });
  });

  // -----------------------------------------------------------------
  // Model switch (best.pt "augmented" vs bestnoaug.pt "baseline")
  // -----------------------------------------------------------------
  const modelSwitch = document.getElementById("modelSwitch");
  modelSwitch.querySelectorAll(".model-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      currentModel = pill.dataset.model;
      modelSwitch.querySelectorAll(".model-pill").forEach((p) => p.classList.toggle("is-active", p === pill));
      footerStatus.textContent = `model: ${currentModel}`;
    });
  });

  // -----------------------------------------------------------------
  // Confidence threshold (shared across all three modes)
  // -----------------------------------------------------------------
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  threshold = parseFloat(thresholdSlider.value);
  thresholdValue.textContent = threshold.toFixed(2);

  thresholdSlider.addEventListener("input", () => {
    threshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = threshold.toFixed(2);
  });

  // -----------------------------------------------------------------
  // Shared rendering: stats panel, detections list, legend highlight
  // -----------------------------------------------------------------
  function renderStats(stats, timeMs, timeLabel) {
    if (!stats) {
      statObjects.textContent = "0";
      statTime.textContent = "-- ms";
      statAvgConf.textContent = "--";
      statMinMax.textContent = "-- / --";
      confScores.innerHTML = `<p class="empty-hint">No detections yet.</p>`;
      return;
    }
    statObjects.textContent = stats.num_objects;
    statTime.textContent = timeLabel || `${timeMs} ms`;
    statAvgConf.textContent = stats.num_objects ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "--";
    statMinMax.textContent = stats.num_objects
      ? `${(stats.min_confidence * 100).toFixed(0)}% / ${(stats.max_confidence * 100).toFixed(0)}%`
      : "-- / --";
  }

  function renderConfidenceChips(detections) {
    if (!detections || detections.length === 0) {
      confScores.innerHTML = `<p class="empty-hint">No detections yet.</p>`;
      return;
    }
    confScores.innerHTML = detections
      .map((d) => `<span class="conf-chip">${escapeHtml(d.label)} · ${(d.confidence * 100).toFixed(1)}%</span>`)
      .join("");
  }

  function renderDetections(detections) {
    if (!detections || detections.length === 0) {
      detectionsList.innerHTML = `<p class="empty-hint">No gestures detected yet.</p>`;
      detCount.textContent = "0 active";
    } else {
      detCount.textContent = `${detections.length} active`;
      detectionsList.innerHTML = detections
        .map(
          (d) => `
        <div class="detection-card">
          <div class="detection-top">
            <span class="detection-name">${escapeHtml(d.label)}</span>
            <span class="detection-conf">${(d.confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="detection-bar">
            <div class="detection-bar-fill" style="width:${(d.confidence * 100).toFixed(1)}%"></div>
          </div>
        </div>`
        )
        .join("");
    }

    const activeLabels = new Set(detections.map((d) => d.label));
    legendList.querySelectorAll(".legend-item").forEach((li) => {
      li.classList.toggle("is-active", activeLabels.has(li.dataset.class));
    });

    renderConfidenceChips(detections);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // =====================================================================
  // MODE 1: WEBCAM
  // =====================================================================
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const stage = document.getElementById("stage");
  const placeholder = document.getElementById("stagePlaceholder");

  const toggleCamBtn = document.getElementById("toggleCam");
  const toggleCamLabel = document.getElementById("toggleCamLabel");
  const camStatus = document.getElementById("camStatus");
  const fpsTag = document.getElementById("fpsTag");

  let stream = null;
  let running = false;
  let inFlight = false;
  let loopHandle = null;

  const CAPTURE_INTERVAL_MS = 250; // ~4 predictions/sec, gentle on CPU-only inference

  const grabCanvas = document.createElement("canvas");
  const grabCtx = grabCanvas.getContext("2d");

  function setCamStatus(on) {
    camStatus.innerHTML = on
      ? '<span class="dot dot-on"></span> CAMERA LIVE'
      : '<span class="dot dot-off"></span> CAMERA OFF';
  }

  function resizeOverlay() {
    const rect = stage.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
  }
  window.addEventListener("resize", resizeOverlay);

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = stream;
      video.style.display = "block";
      placeholder.style.display = "none";
      await video.play();
      resizeOverlay();

      running = true;
      setCamStatus(true);
      toggleCamBtn.classList.add("is-active");
      toggleCamLabel.textContent = "Stop Camera";
      footerStatus.textContent = "streaming";
      loop();
    } catch (err) {
      console.error("Camera error:", err);
      footerStatus.textContent = "camera access denied";
      alert("Could not access the webcam. Please allow camera permissions and try again.");
    }
  }

  function stopCamera() {
    if (!running && !stream) return;
    running = false;
    if (loopHandle) clearTimeout(loopHandle);
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.style.display = "none";
    placeholder.style.display = "flex";
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    setCamStatus(false);
    toggleCamBtn.classList.remove("is-active");
    toggleCamLabel.textContent = "Start Camera";
    footerStatus.textContent = "idle";
    fpsTag.textContent = "-- ms / inference";
  }

  toggleCamBtn.addEventListener("click", () => {
    running ? stopCamera() : startCamera();
  });

  async function loop() {
    if (!running) return;

    if (!inFlight && video.videoWidth > 0) {
      inFlight = true;
      try {
        await captureAndPredict();
      } catch (err) {
        console.error("Prediction error:", err);
      } finally {
        inFlight = false;
      }
    }
    loopHandle = setTimeout(loop, CAPTURE_INTERVAL_MS);
  }

  async function captureAndPredict() {
    grabCanvas.width = video.videoWidth;
    grabCanvas.height = video.videoHeight;
    // Mirror the frame so it matches what the user sees (video is CSS-flipped)
    grabCtx.translate(grabCanvas.width, 0);
    grabCtx.scale(-1, 1);
    grabCtx.drawImage(video, 0, 0, grabCanvas.width, grabCanvas.height);
    grabCtx.setTransform(1, 0, 0, 1, 0, 0);

    const dataUrl = grabCanvas.toDataURL("image/jpeg", 0.75);

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl, threshold, model: currentModel }),
    });

    if (!res.ok) throw new Error(`Server responded ${res.status}`);

    const data = await res.json();
    fpsTag.textContent = `${data.inference_ms} ms / inference`;
    drawWebcamDetections(data.detections);
    renderDetections(data.detections);
    renderStats(data.stats, data.inference_ms);
  }

  function drawWebcamDetections(detections) {
    resizeOverlay();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!detections || detections.length === 0) return;

    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();

    detections.forEach((d) => {
      const { x1, y1, x2, y2 } = d.box;
      const px1 = x1 * overlay.width;
      const py1 = y1 * overlay.height;
      const pw = (x2 - x1) * overlay.width;
      const ph = (y2 - y1) * overlay.height;

      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.shadowColor = accent;
      ctx.shadowBlur = 6;
      ctx.strokeRect(px1, py1, pw, ph);
      ctx.shadowBlur = 0;

      const label = `${d.label}  ${(d.confidence * 100).toFixed(0)}%`;
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      const textW = ctx.measureText(label).width + 12;

      ctx.fillStyle = accent;
      ctx.fillRect(px1, Math.max(0, py1 - 22), textW, 22);

      ctx.fillStyle = "#0a0b0d";
      ctx.fillText(label, px1 + 6, Math.max(14, py1 - 6));
    });
  }

  resizeOverlay();

  // =====================================================================
  // MODE 2: IMAGE UPLOAD
  // =====================================================================
  const imageInput = document.getElementById("imageInput");
  const imagePreview = document.getElementById("imagePreview");
  const imagePlaceholder = document.getElementById("imagePlaceholder");
  const runImageDetectionBtn = document.getElementById("runImageDetection");
  const downloadImageBtn = document.getElementById("downloadImage");
  const imgFpsTag = document.getElementById("imgFpsTag");

  let selectedImageFile = null;
  let lastImageExportUrl = null;

  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;
    selectedImageFile = file;

    const url = URL.createObjectURL(file);
    imagePreview.src = url;
    imagePreview.style.display = "block";
    imagePlaceholder.style.display = "none";
    runImageDetectionBtn.disabled = false;
    downloadImageBtn.disabled = true;
    lastImageExportUrl = null;
    imgFpsTag.textContent = "-- ms / inference";
  });

  runImageDetectionBtn.addEventListener("click", async () => {
    if (!selectedImageFile) return;
    runImageDetectionBtn.disabled = true;
    runImageDetectionBtn.textContent = "Detecting…";
    footerStatus.textContent = "running image detection";

    try {
      const form = new FormData();
      form.append("image", selectedImageFile);
      form.append("threshold", threshold);
      form.append("model", currentModel);

      const res = await fetch("/predict_image", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();

      imagePreview.src = data.annotated_image;
      imgFpsTag.textContent = `${data.inference_ms} ms / inference`;
      lastImageExportUrl = data.export_url;
      downloadImageBtn.disabled = false;

      renderDetections(data.detections.map((d) => ({ label: d.label, confidence: d.confidence })));
      renderStats(data.stats, data.inference_ms);
      footerStatus.textContent = "image detection complete";
    } catch (err) {
      console.error("Image detection error:", err);
      footerStatus.textContent = "image detection failed";
      alert("Something went wrong running detection on this image.");
    } finally {
      runImageDetectionBtn.disabled = false;
      runImageDetectionBtn.textContent = "Run Detection";
    }
  });

  downloadImageBtn.addEventListener("click", () => {
    if (!lastImageExportUrl) return;
    const a = document.createElement("a");
    a.href = lastImageExportUrl;
    a.download = "gesturevision-annotated.jpg";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // =====================================================================
  // MODE 3: VIDEO UPLOAD
  // =====================================================================
  const videoInput = document.getElementById("videoInput");
  const videoPreview = document.getElementById("videoPreview");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const videoLoading = document.getElementById("videoLoading");
  const runVideoDetectionBtn = document.getElementById("runVideoDetection");
  const downloadVideoBtn = document.getElementById("downloadVideo");
  const videoFpsTag = document.getElementById("videoFpsTag");

  let selectedVideoFile = null;
  let lastVideoExportUrl = null;

  videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];
    if (!file) return;
    selectedVideoFile = file;

    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    videoPreview.style.display = "block";
    videoPlaceholder.style.display = "none";
    runVideoDetectionBtn.disabled = false;
    downloadVideoBtn.disabled = true;
    lastVideoExportUrl = null;
    videoFpsTag.textContent = "-- ms / video";
  });

  runVideoDetectionBtn.addEventListener("click", async () => {
    if (!selectedVideoFile) return;
    runVideoDetectionBtn.disabled = true;
    runVideoDetectionBtn.textContent = "Detecting…";
    videoLoading.classList.add("is-active");
    footerStatus.textContent = "running video detection";

    try {
      const form = new FormData();
      form.append("video", selectedVideoFile);
      form.append("threshold", threshold);
      form.append("model", currentModel);

      const res = await fetch("/predict_video", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();

      videoPreview.src = data.export_url;
      videoFpsTag.textContent = `${data.processing_ms} ms / video (${data.frame_count} frames)`;
      lastVideoExportUrl = data.export_url;
      downloadVideoBtn.disabled = false;

      renderDetections([]); // per-object list doesn't map cleanly to a whole video
      renderStats(data.stats, data.processing_ms, `${(data.processing_ms / 1000).toFixed(1)} s`);
      footerStatus.textContent = "video detection complete";
    } catch (err) {
      console.error("Video detection error:", err);
      footerStatus.textContent = "video detection failed";
      alert("Something went wrong running detection on this video.");
    } finally {
      runVideoDetectionBtn.disabled = false;
      runVideoDetectionBtn.textContent = "Run Detection";
      videoLoading.classList.remove("is-active");
    }
  });

  downloadVideoBtn.addEventListener("click", () => {
    if (!lastVideoExportUrl) return;
    const a = document.createElement("a");
    a.href = lastVideoExportUrl;
    a.download = "gesturevision-annotated.mp4";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // Initial paint
  renderStats(null);
})();
