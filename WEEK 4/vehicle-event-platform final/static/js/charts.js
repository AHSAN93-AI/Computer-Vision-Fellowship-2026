/**
 * static/js/charts.js — Chart.js chart instances for the dashboard.
 * Exports createXxxChart() functions and updateXxxChart() helpers.
 */

// ─── Color Palette ────────────────────────────────────────────────

const COLORS = {
  accent:   'rgba(99, 179, 237, 0.85)',
  success:  'rgba(104, 211, 145, 0.85)',
  warning:  'rgba(246, 173, 85, 0.85)',
  danger:   'rgba(252, 129, 129, 0.85)',
  critical: 'rgba(252, 72, 72, 0.85)',
  purple:   'rgba(159, 122, 234, 0.85)',
  teal:     'rgba(118, 228, 247, 0.85)',
};

const BORDER_COLORS = {
  accent:   'rgba(99, 179, 237, 1)',
  success:  'rgba(104, 211, 145, 1)',
  warning:  'rgba(246, 173, 85, 1)',
  danger:   'rgba(252, 129, 129, 1)',
  critical: 'rgba(252, 72, 72, 1)',
  purple:   'rgba(159, 122, 234, 1)',
  teal:     'rgba(118, 228, 247, 1)',
};

const SEVERITY_COLORS = {
  CRITICAL: COLORS.critical,
  HIGH:     COLORS.danger,
  WARNING:  COLORS.warning,
  INFO:     COLORS.accent,
};

const TYPE_COLOR_POOL = [
  COLORS.accent, COLORS.success, COLORS.warning, COLORS.danger,
  COLORS.purple, COLORS.teal, COLORS.critical,
];

// ─── Defaults ────────────────────────────────────────────────────

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400 },
  plugins: {
    legend: {
      labels: {
        color: '#94a3b8',
        boxWidth: 12,
        font: { family: 'Inter', size: 11 },
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: 'rgba(15,21,38,0.95)',
      titleColor: '#e2e8f0',
      bodyColor: '#94a3b8',
      borderColor: 'rgba(255,255,255,0.08)',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
    },
  },
  scales: {
    x: {
      grid:  { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#475569', font: { size: 10 } },
    },
    y: {
      grid:  { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#475569', font: { size: 10 } },
      beginAtZero: true,
    },
  },
};

// ─── Events Over Time (Line) ──────────────────────────────────────

export function createEventsTimeChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Events',
        data: [],
        borderColor: BORDER_COLORS.accent,
        backgroundColor: 'rgba(99,179,237,0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: BORDER_COLORS.accent,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: false },
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, suggestedMax: 5 },
      },
    },
  });
}

export function updateEventsTimeChart(chart, hourlyData) {
  if (!chart || !hourlyData) return;

  const now = new Date();
  const labels = [];
  const counts = new Array(24).fill(0);

  if (Array.isArray(hourlyData)) {
    hourlyData.forEach(d => {
      if (d.hour >= 0 && d.hour < 24) {
        counts[d.hour] = d.count;
      }
    });
  }

  for (let i = 23; i >= 0; i--) {
    const h = new Date(now - i * 3600000);
    labels.push(`${h.getHours()}:00`);
  }

  chart.data.labels = labels;
  chart.data.datasets[0].data = counts;
  chart.update();
}

// ─── Events By Type (Doughnut) ────────────────────────────────────

export function createEventsByTypeChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'doughnut',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0, hoverOffset: 6 }] },
    options: {
      ...CHART_DEFAULTS,
      cutout: '65%',
      scales: {},  // no scales for doughnut
    },
  });
}

export function updateEventsByTypeChart(chart, byType) {
  if (!chart || !byType) return;
  const labels = Object.keys(byType);
  const data   = Object.values(byType);
  const colors = labels.map((_, i) => TYPE_COLOR_POOL[i % TYPE_COLOR_POOL.length]);

  chart.data.labels = labels.map(l => l.replace(/_/g, ' '));
  chart.data.datasets[0].data = data;
  chart.data.datasets[0].backgroundColor = colors;
  chart.update();
}

// ─── Events By Severity (Bar) ─────────────────────────────────────

export function createSeverityChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['CRITICAL', 'HIGH', 'WARNING', 'INFO'],
      datasets: [{
        label: 'Events',
        data: [0, 0, 0, 0],
        backgroundColor: [COLORS.critical, COLORS.danger, COLORS.warning, COLORS.accent],
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, ticks: { ...CHART_DEFAULTS.scales.y.ticks, precision: 0 } },
      },
    },
  });
}

export function updateSeverityChart(chart, bySeverity) {
  if (!chart || !bySeverity) return;
  chart.data.datasets[0].data = [
    bySeverity['CRITICAL'] || 0,
    bySeverity['HIGH']     || 0,
    bySeverity['WARNING']  || 0,
    bySeverity['INFO']     || 0,
  ];
  chart.update();
}

// ─── Occupancy Trend (Line) ───────────────────────────────────────

export function createOccupancyChart(canvasId, zoneName = 'Occupancy') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: zoneName,
          data: [],
          borderColor: BORDER_COLORS.success,
          backgroundColor: 'rgba(104,211,145,0.08)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
        },
        {
          label: 'Capacity',
          data: [],
          borderColor: 'rgba(252,129,129,0.6)',
          borderDash: [6, 3],
          fill: false,
          tension: 0,
          pointRadius: 0,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins },
      scales: { ...CHART_DEFAULTS.scales },
    },
  });
}

export function updateOccupancyChart(chart, timeSeries, maxCapacity) {
  if (!chart || !timeSeries) return;
  const labels = timeSeries.map(d => {
    const t = new Date(d.timestamp * 1000);
    return `${t.getHours()}:${String(t.getMinutes()).padStart(2, '0')}`;
  });
  const counts = timeSeries.map(d => d.count);
  const cap = timeSeries.map(() => maxCapacity || null);

  chart.data.labels = labels;
  chart.data.datasets[0].data = counts;
  chart.data.datasets[1].data = cap;
  chart.update();
}

// ─── FPS / Performance Sparkline ─────────────────────────────────

export function createFpsChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const history = Array(60).fill(0);

  return {
    _history: history,
    _chart: new Chart(ctx, {
      type: 'line',
      data: {
        labels: history.map((_, i) => i),
        datasets: [{
          data: history,
          borderColor: BORDER_COLORS.teal,
          backgroundColor: 'rgba(118,228,247,0.06)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 1.5,
        }],
      },
      options: {
        ...CHART_DEFAULTS,
        animation: false,
        plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: { display: false, beginAtZero: true, suggestedMax: 35 },
        },
      },
    }),
  };
}

export function pushFps(fpsChartObj, fps) {
  if (!fpsChartObj) return;
  fpsChartObj._history.push(fps);
  fpsChartObj._history.shift();
  fpsChartObj._chart.data.datasets[0].data = fpsChartObj._history;
  fpsChartObj._chart.update();
}
