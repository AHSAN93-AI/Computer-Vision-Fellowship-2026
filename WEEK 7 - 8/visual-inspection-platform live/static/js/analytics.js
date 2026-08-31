const CHART_COLORS = {
  amber: '#f2a93b',
  pass: '#3ddc84',
  fail: '#e5484d',
  invalid: '#9098a3',
  grid: '#2e3338',
  text: '#8a929c',
};

Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

async function loadAnalytics() {
  const res = await fetch('/api/analytics');
  const a = await res.json();

  document.getElementById('a-total').textContent = a.total_inspections;
  document.getElementById('a-pass-rate').textContent = (a.pass_rate * 100).toFixed(1) + '%';
  document.getElementById('a-defect-rate').textContent = (a.defect_rate * 100).toFixed(1) + '%';
  document.getElementById('a-invalid-rate').textContent = (a.invalid_rate * 100).toFixed(1) + '%';
  document.getElementById('a-latency').textContent = a.avg_processing_time_ms.toFixed(1) + ' ms';
  const throughput = a.avg_processing_time_ms > 0 ? (1000 / a.avg_processing_time_ms).toFixed(2) : '0.00';
  document.getElementById('a-throughput').textContent = throughput + ' img/s';
  document.getElementById('a-passed').textContent = a.passed;
  document.getElementById('a-failed').textContent = a.failed;
  document.getElementById('a-invalid').textContent = a.invalid;

  renderDefectsChart(a.defects_by_category);
  renderTimeseriesChart(a.defect_rate_over_time);
  renderSeverityChart(a.severity_distribution);
}

function renderDefectsChart(byCategory) {
  const ctx = document.getElementById('chart-defects');
  const labels = Object.keys(byCategory);
  const values = Object.values(byCategory);
  if (!labels.length) { ctx.replaceWith(noDataMsg()); return; }
  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: CHART_COLORS.amber, borderRadius: 2 }] },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_COLORS.grid } },
        y: { grid: { color: CHART_COLORS.grid }, beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}

function renderTimeseriesChart(rows) {
  const ctx = document.getElementById('chart-timeseries');
  if (!rows.length) { ctx.replaceWith(noDataMsg()); return; }
  const labels = rows.map(r => r.day);
  const rates = rows.map(r => r.total ? (r.fails / r.total) * 100 : 0);
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: rates,
        borderColor: CHART_COLORS.fail,
        backgroundColor: 'rgba(229,72,77,0.12)',
        tension: 0.25,
        fill: true,
        pointRadius: 2,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_COLORS.grid } },
        y: { grid: { color: CHART_COLORS.grid }, beginAtZero: true, ticks: { callback: (v) => v + '%' } },
      },
    },
  });
}

function renderSeverityChart(dist) {
  const ctx = document.getElementById('chart-severity');
  const labels = Object.keys(dist);
  const values = Object.values(dist);
  if (!labels.length) { ctx.replaceWith(noDataMsg()); return; }
  const colorMap = { Minor: CHART_COLORS.amber, Major: '#f2a93b', Critical: CHART_COLORS.fail };
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: labels.map(l => colorMap[l] || CHART_COLORS.invalid) }],
    },
    options: { plugins: { legend: { position: 'right', labels: { boxWidth: 10 } } } },
  });
}

function noDataMsg() {
  const div = document.createElement('div');
  div.className = 'empty-state';
  div.innerHTML = '<div class="empty-state__title">No data yet</div>Run some inspections to populate this chart.';
  return div;
}

loadAnalytics();
