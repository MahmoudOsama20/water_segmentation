/* ── AquaVision — app.js ────────────────────────────────────────── */

// ── Health check ────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();

    setDot('dot-flask',   'ok');
    setDot('dot-unet',    data.unet    === 'ok' ? 'ok' : 'down');
    setDot('dot-deeplab', data.deeplab === 'ok' ? 'ok' : 'down');
  } catch {
    ['dot-flask', 'dot-unet', 'dot-deeplab'].forEach(id => setDot(id, 'down'));
  }
}

function setDot(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'dot ' + state;
}

// Run on load and every 15s
checkHealth();
setInterval(checkHealth, 15000);

// ── File upload helpers ─────────────────────────────────────────
function triggerUpload(inputId) {
  document.getElementById(inputId).click();
}

function onFileChosen(input, badgeId, cardId) {
  const badge = document.getElementById(badgeId);
  const card  = document.getElementById(cardId);

  if (input.files[0]) {
    const name = input.files[0].name;
    // Truncate long filenames
    const display = name.length > 22 ? name.slice(0, 19) + '…' : name;
    badge.textContent = '✓ ' + display;
    card.classList.add('has-file');
  } else {
    badge.textContent = 'Click to upload';
    card.classList.remove('has-file');
  }
}

// ── UI state helpers ────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('runBtn').disabled         = on;
  document.getElementById('spinner').style.display   = on ? 'flex' : 'none';
  document.getElementById('errorBanner').style.display = 'none';
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  document.getElementById('errorMsg').textContent = msg;
  banner.style.display   = 'flex';
  setLoading(false);
}

function hideResults() {
  document.getElementById('results').style.display    = 'none';
  document.getElementById('metricsWrap').style.display = 'none';
}

// ── Animate water bar ───────────────────────────────────────────
function animateBar(id, pct) {
  const el = document.getElementById(id);
  if (!el) return;
  // Small delay so CSS transition fires
  setTimeout(() => { el.style.width = Math.min(pct, 100) + '%'; }, 80);
}

// ── Main prediction ─────────────────────────────────────────────
async function runPrediction() {
  const imageFile = document.getElementById('imageFile').files[0];
  if (!imageFile) {
    showError('Please upload a GeoTIFF image file first.');
    return;
  }

  setLoading(true);
  hideResults();

  // Reset water bars
  ['unet-bar', 'deeplab-bar', 'ensemble-bar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = '0%';
  });

  const formData = new FormData();
  formData.append('image', imageFile);

  const maskFile = document.getElementById('maskFile').files[0];
  if (maskFile) formData.append('mask', maskFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    // ── Show masks ──
    document.getElementById('unet-img').src     = 'data:image/png;base64,' + data.unet_mask;
    document.getElementById('deeplab-img').src  = 'data:image/png;base64,' + data.deeplab_mask;
    document.getElementById('ensemble-img').src = 'data:image/png;base64,' + data.ensemble_mask;

    // ── Water percentages ──
    const u = data.water_pct.unet;
    const d = data.water_pct.deeplab;
    const e = data.water_pct.ensemble;

    document.getElementById('unet-pct').textContent     = u + '%';
    document.getElementById('deeplab-pct').textContent  = d + '%';
    document.getElementById('ensemble-pct').textContent = e + '%';

    // ── Animate bars ──
    animateBar('unet-bar',     u);
    animateBar('deeplab-bar',  d);
    animateBar('ensemble-bar', e);

    // ── Show results ──
    document.getElementById('results').style.display = 'block';

    // ── Metrics (optional) ──
    if (data.metrics) {
      renderMetrics(data.metrics);
      document.getElementById('metricsWrap').style.display = 'block';
    }

    // Smooth scroll to results
    setTimeout(() => {
      document.getElementById('results').scrollIntoView({
        behavior: 'smooth', block: 'start'
      });
    }, 100);

  } catch (e) {
    showError('Request failed: ' + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Render metrics table ────────────────────────────────────────
function renderMetrics(metrics) {
  const tbody = document.getElementById('metricsBody');
  tbody.innerHTML = '';

  // Find best IoU
  const ious   = Object.values(metrics).map(m => m.IoU);
  const bestIou = Math.max(...ious);

  const modelIcons = {
    'U-Net'     : '◈',
    'DeepLabV3+': '◆',
    'Ensemble'  : '✦',
  };

  for (const [model, m] of Object.entries(metrics)) {
    const isBest = m.IoU === bestIou;
    const icon   = modelIcons[model] || '·';

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>
        <span style="color:var(--text-muted);margin-right:8px;font-size:12px">${icon}</span>
        ${model}
        ${isBest ? '<span class="badge-best">best</span>' : ''}
      </td>
      <td>${renderMetricCell(m.IoU)}</td>
      <td>${renderMetricCell(m.F1)}</td>
      <td>${renderMetricCell(m.Precision)}</td>
      <td>${renderMetricCell(m.Recall)}</td>
    `;
    tbody.appendChild(row);
  }
}

function renderMetricCell(value) {
  const barWidth = Math.round(value * 60);
  const color    = value >= 0.8 ? '#34d399' : value >= 0.6 ? '#60a5fa' : '#f59e0b';
  return `
    <span class="metric-value">
      <span style="color:${color};font-weight:500">${value.toFixed(4)}</span>
      <span class="metric-bar-mini" style="width:${barWidth}px;background:${color}44;
            box-shadow: 0 0 6px ${color}44"></span>
    </span>`;
}
