const sliders = {
  conf: document.getElementById('s-conf'),
  anomaly: document.getElementById('s-anomaly'),
  count: document.getElementById('s-count'),
  area: document.getElementById('s-area'),
  minor: document.getElementById('s-minor'),
  major: document.getElementById('s-major'),
};
const values = {
  conf: document.getElementById('v-conf'),
  anomaly: document.getElementById('v-anomaly'),
  count: document.getElementById('v-count'),
  area: document.getElementById('v-area'),
  minor: document.getElementById('v-minor'),
  major: document.getElementById('v-major'),
};

for (const key in sliders) {
  sliders[key].addEventListener('input', () => {
    values[key].textContent = sliders[key].value;
  });
}

function wireChips(containerId) {
  const container = document.getElementById(containerId);
  container.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => chip.classList.toggle('is-active'));
  });
}
wireChips('critical-chips');
wireChips('floor-chips');

function activeChips(containerId) {
  return Array.from(document.querySelectorAll(`#${containerId} .chip.is-active`)).map(c => c.dataset.class);
}

function setActiveChips(containerId, activeList) {
  document.querySelectorAll(`#${containerId} .chip`).forEach(chip => {
    chip.classList.toggle('is-active', activeList.includes(chip.dataset.class));
  });
}

async function loadSettings() {
  const res = await fetch('/api/settings');
  const s = await res.json();

  sliders.conf.value = s.decision.classifier_confidence_threshold;
  sliders.anomaly.value = s.decision.anomaly_score_threshold;
  sliders.count.value = s.decision.max_allowed_defect_count;
  sliders.area.value = s.decision.max_allowed_area_ratio;
  sliders.minor.value = s.severity.minor_max_area_ratio;
  sliders.major.value = s.severity.major_max_area_ratio;

  for (const key in sliders) values[key].textContent = sliders[key].value;

  setActiveChips('critical-chips', s.decision.critical_classes || []);
  setActiveChips('floor-chips', Object.keys(s.severity.minimum_severity_by_class || {}));
}

async function saveSettings() {
  const floorClasses = activeChips('floor-chips');
  const minimum_severity_by_class = {};
  floorClasses.forEach(c => { minimum_severity_by_class[c] = 'Major'; });

  const payload = {
    decision: {
      classifier_confidence_threshold: parseFloat(sliders.conf.value),
      anomaly_score_threshold: parseFloat(sliders.anomaly.value),
      max_allowed_defect_count: parseInt(sliders.count.value, 10),
      max_allowed_area_ratio: parseFloat(sliders.area.value),
      critical_classes: activeChips('critical-chips'),
    },
    severity: {
      minor_max_area_ratio: parseFloat(sliders.minor.value),
      major_max_area_ratio: parseFloat(sliders.major.value),
      minimum_severity_by_class,
    },
  };

  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = 'Settings saved — applied immediately';
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2200);
}

document.getElementById('save-btn').addEventListener('click', saveSettings);
document.getElementById('reset-btn').addEventListener('click', async () => {
  const res = await fetch('/api/settings');
  const s = await res.json();
  sliders.conf.value = s.defaults.decision.classifier_confidence_threshold;
  sliders.anomaly.value = s.defaults.decision.anomaly_score_threshold;
  sliders.count.value = s.defaults.decision.max_allowed_defect_count;
  sliders.area.value = s.defaults.decision.max_allowed_area_ratio;
  sliders.minor.value = s.defaults.severity.minor_max_area_ratio;
  sliders.major.value = s.defaults.severity.major_max_area_ratio;
  for (const key in sliders) values[key].textContent = sliders[key].value;
  setActiveChips('critical-chips', s.defaults.decision.critical_classes || []);
  setActiveChips('floor-chips', Object.keys(s.defaults.severity.minimum_severity_by_class || {}));
});

loadSettings();
