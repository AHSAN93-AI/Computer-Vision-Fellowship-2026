const tableBody = document.getElementById('table-body');
const emptyState = document.getElementById('empty-state');
const tableWrap = document.getElementById('table-wrap');

const filterStatus = document.getElementById('filter-status');
const filterDefect = document.getElementById('filter-defect');
const filterDate = document.getElementById('filter-date');
const filterClear = document.getElementById('filter-clear');
const exportBtn = document.getElementById('export-btn');

function buildQuery() {
  const params = new URLSearchParams();
  if (filterStatus.value) params.set('status', filterStatus.value);
  if (filterDefect.value) params.set('defect', filterDefect.value);
  if (filterDate.value) params.set('date', filterDate.value);
  return params.toString();
}

async function loadHistory() {
  const qs = buildQuery();
  const res = await fetch('/api/inspections' + (qs ? '?' + qs : ''));
  const rows = await res.json();
  renderTable(rows);
  exportBtn.href = '/api/inspections/export' + (qs ? '?' + qs : '');
}

function renderTable(rows) {
  tableBody.innerHTML = '';
  if (!rows.length) {
    tableWrap.style.display = 'none';
    emptyState.style.display = 'block';
    return;
  }
  tableWrap.style.display = 'block';
  emptyState.style.display = 'none';

  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.inspection_id}</td>
      <td>${new Date(r.timestamp).toLocaleString()}</td>
      <td><span class="badge is-${r.status.toLowerCase()}">${r.status}</span></td>
      <td>${r.max_severity && r.max_severity !== 'None' ? `<span class="badge is-${r.max_severity.toLowerCase()}">${r.max_severity}</span>` : '—'}</td>
      <td>${r.defect_count}</td>
      <td>${r.classifier_confidence != null ? r.classifier_confidence.toFixed(3) : '—'}</td>
      <td>${r.anomaly_score != null ? r.anomaly_score.toFixed(4) : '—'}</td>
      <td>${r.defect_area_ratio != null ? r.defect_area_ratio.toFixed(4) : '—'}</td>
      <td>${r.processing_time_ms != null ? r.processing_time_ms.toFixed(1) + ' ms' : '—'}</td>
    `;
    tr.addEventListener('click', () => openEvidence(r));
    tableBody.appendChild(tr);
  }
}

function openEvidence(row) {
  if (row.status === 'INVALID') return; // no evidence images for invalid rows
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `
    <div class="modal">
      <div class="modal__header">
        <div>
          <div style="font-family:var(--font-display); font-weight:600; font-size:15px;">${row.inspection_id}</div>
          <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-secondary);">${row.status} &middot; ${row.max_severity}</div>
        </div>
        <button class="modal__close" id="modal-close">&times;</button>
      </div>
      <div class="evidence-imgs">
        <figure><img src="/api/inspections/${row.inspection_id}/evidence/original"><figcaption>Original</figcaption></figure>
        <figure><img src="/api/inspections/${row.inspection_id}/evidence/heatmap"><figcaption>Anomaly heatmap</figcaption></figure>
        <figure><img src="/api/inspections/${row.inspection_id}/evidence/annotated"><figcaption>Annotated</figcaption></figure>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) backdrop.remove(); });
  backdrop.querySelector('#modal-close').addEventListener('click', () => backdrop.remove());
}

filterStatus.addEventListener('change', loadHistory);
filterDefect.addEventListener('change', loadHistory);
filterDate.addEventListener('change', loadHistory);
filterClear.addEventListener('click', () => {
  filterStatus.value = '';
  filterDefect.value = '';
  filterDate.value = '';
  loadHistory();
});

loadHistory();
