"""
EV Battery Decision Tree — Flask Inference Server
===================================================
Loads the trained .pkl model bundle and exposes a REST API
for real-time and batch predictions.

Endpoints:
    GET  /health           → liveness + model metadata
    POST /predict          → single-row prediction
    POST /predict/batch    → batch prediction (up to 10 000 rows)
    GET  /model/info       → detailed model metadata

Usage:
    python app.py                          # development
    gunicorn -w 4 -b 0.0.0.0:5000 app:app # production

Author : <your-name>
Version: 2.1.0
"""

import logging
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, render_template_string

# ──────────────────────────────────────────────
# App & logging
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH   = Path(os.getenv("MODEL_PATH", "ev_battery_model.pkl"))
FEATURES     = ["temperature", "voltage", "current", "power"]
BATCH_LIMIT  = 10_000

# ── Physical bounds — must mirror the dataset generator exactly ──
BOUNDS = {
    "temperature": (0.0,   80.0),   # °C  — [0, 80)
    "voltage"    : (8.0,   13.0),   # V   — [8, 13)
    "current"    : (0.0,   18.0),   # A   — [0, 18]   ← updated from (6.0, 18.0)
    "power"      : (0.0,  234.0),   # W   — derived: [0×8, 13×18)  ← updated from (48.0, 234.0)
}

# ── Label thresholds (for display/docs only — model handles actual logic) ──
THRESHOLDS = {
    "NORMAL"  : {"temp": "<40°C",  "volt": "≥11.5V", "curr": "≤10A"},
    "ALERT"   : {"temp": "≥70°C",  "volt": "<9.5V",  "curr": ">15A"},
    "CRITICAL": {"temp": "≥75°C",  "volt": "<8.5V",  "curr": "≥18A"},
}

# Colour palette for states (used in the HTML dashboard)
STATE_STYLE = {
    "NORMAL"  : {"color": "#22c55e", "bg": "#052e16", "icon": "✅"},
    "ALERT"   : {"color": "#f59e0b", "bg": "#1c1202", "icon": "⚠️"},
    "CRITICAL": {"color": "#ef4444", "bg": "#1f0606", "icon": "🚨"},
}


# ──────────────────────────────────────────────
# Model loading  (cached singleton)
# ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    """Load and cache the model bundle from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.resolve()}\n"
            "Run train_model.py first to generate the .pkl file."
        )
    log.info("Loading model from %s …", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    log.info(
        "✅  Model loaded  |  accuracy=%.4f%%  |  depth=%d  |  leaves=%d",
        bundle["accuracy"] * 100,
        bundle["tree_depth"],
        bundle["n_leaves"],
    )
    return bundle


def _model():
    return _load_bundle()["model"]

def _encoder():
    return _load_bundle()["label_encoder"]


# ──────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────
def _validate_row(data: dict) -> tuple[list[float], list[str]]:
    """
    Validate and extract feature values from a dict.
    Returns (feature_values, errors).

    Validation rules:
      temperature : [0.0,  80.0)  — strictly below 80
      voltage     : [8.0,  13.0)  — strictly below 13
      current     : [0.0,  18.0]  — inclusive 18   ← updated from 6.0
      power       : [0.0, 234.0)  — derived; validated loosely  ← updated from 48.0
    """
    errors = []
    values = []

    for feat in FEATURES:
        if feat not in data:
            errors.append(f"Missing field: '{feat}'")
            continue
        try:
            val = float(data[feat])
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be a number, got: {data[feat]!r}")
            continue

        lo, hi = BOUNDS[feat]

        if feat == "current":
            # current is inclusive on both ends [0, 18]
            if not (lo <= val <= hi):
                errors.append(
                    f"'{feat}' out of range [{lo}, {hi}]: got {val}"
                )
        elif feat == "power":
            # power is derived — apply a loose sanity check only
            if not (lo <= val < hi + 10):   # small buffer for float rounding
                errors.append(
                    f"'{feat}' out of expected range [{lo}, {hi}): got {val}"
                )
        else:
            # temperature and voltage — strictly below upper bound
            if not (lo <= val < hi):
                errors.append(
                    f"'{feat}' out of range [{lo}, {hi}): got {val}"
                )

        values.append(val)

    return values, errors


# ──────────────────────────────────────────────
# HTML dashboard (served at /)
# ──────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EV Battery Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

  :root {
    --bg       : #0a0f0d;
    --surface  : #111a15;
    --border   : #1e3329;
    --text     : #c8ffe0;
    --muted    : #4d7a61;
    --green    : #22c55e;
    --amber    : #f59e0b;
    --red      : #ef4444;
    --radius   : 10px;
    --mono     : 'Share Tech Mono', monospace;
    --sans     : 'Exo 2', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem 4rem;
  }

  body::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(34,197,94,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34,197,94,.03) 1px, transparent 1px);
    background-size: 28px 28px;
    z-index: 0;
  }

  .wrapper { position: relative; z-index: 1; width: 100%; max-width: 820px; }

  header {
    display: flex; flex-direction: column; align-items: center;
    gap: .4rem; margin-bottom: 2.5rem; text-align: center;
  }
  .badge {
    font-family: var(--mono); font-size: .7rem; letter-spacing: .15em;
    color: var(--green); border: 1px solid var(--green);
    padding: .2rem .7rem; border-radius: 99px; text-transform: uppercase;
  }
  h1 {
    font-size: clamp(1.6rem, 4vw, 2.6rem); font-weight: 800;
    letter-spacing: -.02em; line-height: 1.1;
    background: linear-gradient(135deg, #c8ffe0 30%, #22c55e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { font-size: .85rem; color: var(--muted); font-weight: 300; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.25rem;
  }
  .card h2 {
    font-size: .78rem; font-family: var(--mono); letter-spacing: .12em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 1.1rem;
    border-bottom: 1px solid var(--border); padding-bottom: .6rem;
  }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media(max-width:500px){ .grid-2 { grid-template-columns:1fr; } }

  .field label {
    display: block; font-size: .72rem; font-family: var(--mono);
    color: var(--muted); letter-spacing: .1em; margin-bottom: .35rem;
    text-transform: uppercase;
  }
  .field input {
    width: 100%; padding: .6rem .85rem;
    background: #0d1711; border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-family: var(--mono);
    font-size: .92rem; outline: none; transition: border .2s;
  }
  .field input:focus { border-color: var(--green); }
  .hint { font-size: .65rem; color: var(--muted); margin-top: .25rem; }

  .derive-row {
    display: flex; align-items: center; gap: .5rem;
    font-size: .75rem; color: var(--muted); margin-top: .4rem;
  }
  .derive-row input[type=checkbox] { accent-color: var(--green); }

  .btn {
    width: 100%; padding: .75rem; margin-top: 1.2rem;
    background: var(--green); color: #021a0a;
    font-family: var(--sans); font-weight: 600; font-size: .95rem;
    border: none; border-radius: 8px; cursor: pointer;
    letter-spacing: .04em; transition: opacity .2s, transform .1s;
  }
  .btn:hover { opacity: .88; }
  .btn:active { transform: scale(.98); }
  .btn:disabled { opacity: .4; cursor: not-allowed; }

  #result { display: none; }
  .state-banner {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.4rem; border-radius: 8px;
    border-left: 4px solid; margin-bottom: 1rem;
  }
  .state-icon { font-size: 1.8rem; }
  .state-label { font-size: 1.6rem; font-weight: 800; letter-spacing: .05em; }
  .state-sub   { font-size: .78rem; font-family: var(--mono); color: var(--muted); }

  .prob-bar-wrap { margin-top: .8rem; }
  .prob-row { display: flex; align-items: center; gap: .75rem; margin-bottom: .45rem; }
  .prob-name { font-family: var(--mono); font-size: .72rem; width: 70px; flex-shrink: 0; }
  .prob-track {
    flex: 1; height: 6px; background: #1a2e22; border-radius: 99px; overflow: hidden;
  }
  .prob-fill  { height: 100%; border-radius: 99px; transition: width .5s ease; }
  .prob-pct   { font-family: var(--mono); font-size: .72rem; width: 40px; text-align: right; }

  .error-box {
    background: #1f0606; border: 1px solid #7f1d1d;
    border-radius: 8px; padding: .9rem 1.1rem;
    font-family: var(--mono); font-size: .8rem; color: #fca5a5;
  }
  .error-box ul { padding-left: 1rem; }

  .meta-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
    gap: .75rem; margin-top: .2rem;
  }
  .meta-item { display: flex; flex-direction: column; gap: .2rem; }
  .meta-item .mk { font-family: var(--mono); font-size: .65rem; color: var(--muted); text-transform:uppercase; }
  .meta-item .mv { font-family: var(--mono); font-size: .82rem; color: var(--green); }

  details summary {
    cursor: pointer; font-family: var(--mono); font-size: .72rem;
    color: var(--muted); letter-spacing: .1em; text-transform: uppercase;
    user-select: none; padding: .4rem 0;
  }
  pre {
    background: #0a100c; border: 1px solid var(--border);
    border-radius: 6px; padding: .8rem 1rem; overflow-x: auto;
    font-family: var(--mono); font-size: .75rem; color: var(--muted);
    margin-top: .5rem;
  }

  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #021a0a; border-top-color: transparent;
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-right: .4rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* threshold reference table */
  .thresh-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: .75rem; }
  .thresh-table th { color: var(--muted); text-align: left; padding: .3rem .5rem;
                     border-bottom: 1px solid var(--border); font-weight: 400; }
  .thresh-table td { padding: .35rem .5rem; border-bottom: 1px solid #12221a; }
  .thresh-table tr:last-child td { border-bottom: none; }
  .tag-normal   { color: #22c55e; } .tag-alert { color: #f59e0b; } .tag-critical { color: #ef4444; }
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <span class="badge">Decision Tree Classifier v2.1</span>
    <h1>EV Battery Health Monitor</h1>
    <p class="subtitle">Real-time battery state prediction · NORMAL · ALERT · CRITICAL</p>
  </header>

  <!-- Model Metadata -->
  <div class="card" id="meta-card">
    <h2>Model Metadata</h2>
    <div class="meta-grid" id="meta-grid">
      <div class="meta-item"><span class="mk">Status</span><span class="mv" id="m-status">Loading…</span></div>
      <div class="meta-item"><span class="mk">Accuracy</span><span class="mv" id="m-acc">—</span></div>
      <div class="meta-item"><span class="mk">Macro F1</span><span class="mv" id="m-f1">—</span></div>
      <div class="meta-item"><span class="mk">Tree Depth</span><span class="mv" id="m-depth">—</span></div>
      <div class="meta-item"><span class="mk">Leaves</span><span class="mv" id="m-leaves">—</span></div>
      <div class="meta-item"><span class="mk">Trained On</span><span class="mv" id="m-rows">—</span></div>
    </div>
  </div>

  <!-- Threshold Reference -->
  <div class="card">
    <h2>Label Thresholds (Reference)</h2>
    <table class="thresh-table">
      <thead><tr><th>State</th><th>Temperature</th><th>Voltage</th><th>Current</th></tr></thead>
      <tbody>
        <tr><td class="tag-normal">NORMAL</td><td>&lt; 40 °C</td><td>≥ 11.5 V</td><td>≤ 10 A</td></tr>
        <tr><td class="tag-alert">ALERT</td><td>≥ 70 °C</td><td>&lt; 9.5 V</td><td>&gt; 15 A</td></tr>
        <tr><td class="tag-critical">CRITICAL</td><td>≥ 75 °C</td><td>&lt; 8.5 V</td><td>≥ 18 A</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Input Form -->
  <div class="card">
    <h2>Battery Sensor Input</h2>
    <div class="grid-2">
      <div class="field">
        <label for="temperature">Temperature (°C)</label>
        <input id="temperature" type="number" step="0.01" placeholder="e.g. 35.0" min="0" max="79.99"/>
        <p class="hint">Range: 0 – 79.99 °C</p>
      </div>
      <div class="field">
        <label for="voltage">Voltage (V)</label>
        <input id="voltage" type="number" step="0.001" placeholder="e.g. 11.5" min="8.0" max="12.999"/>
        <p class="hint">Range: 8.0 – 12.99 V</p>
      </div>
      <div class="field">
        <label for="current">Current (A)</label>
        <input id="current" type="number" step="0.01" placeholder="e.g. 9.0" min="0" max="18"/>
        <p class="hint">Range: 0 – 18 A</p>
      </div>
      <div class="field">
        <label for="power">Power (W)</label>
        <input id="power" type="number" step="0.01" placeholder="Auto from V × I" min="0" max="234"/>
        <p class="hint">Derived: voltage × current  [0 – 234 W]</p>
        <div class="derive-row">
          <input type="checkbox" id="auto-power" checked/>
          <label for="auto-power">Auto-compute from voltage × current</label>
        </div>
      </div>
    </div>
    <button class="btn" id="predict-btn" onclick="runPredict()">Predict Battery State</button>
  </div>

  <!-- Result -->
  <div class="card" id="result">
    <h2>Prediction Result</h2>
    <div id="result-inner"></div>
    <details style="margin-top:1rem">
      <summary>Raw JSON Response</summary>
      <pre id="raw-json"></pre>
    </details>
  </div>

</div>

<script>
(async () => {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('m-status').textContent  = d.status === 'ok' ? '✅ Online' : '❌ ' + d.status;
    document.getElementById('m-acc').textContent     = d.model ? (d.model.accuracy * 100).toFixed(3) + '%' : '—';
    document.getElementById('m-f1').textContent      = d.model ? (d.model.macro_f1  * 100).toFixed(3) + '%' : '—';
    document.getElementById('m-depth').textContent   = d.model ? d.model.tree_depth  : '—';
    document.getElementById('m-leaves').textContent  = d.model ? d.model.n_leaves.toLocaleString() : '—';
    document.getElementById('m-rows').textContent    = d.model ? d.model.trained_on_rows.toLocaleString() + ' rows' : '—';
  } catch(e) {
    document.getElementById('m-status').textContent = '❌ Unreachable';
  }
})();

['voltage','current'].forEach(id => {
  document.getElementById(id).addEventListener('input', autoFillPower);
});
document.getElementById('auto-power').addEventListener('change', autoFillPower);
function autoFillPower() {
  if (!document.getElementById('auto-power').checked) return;
  const v = parseFloat(document.getElementById('voltage').value);
  const i = parseFloat(document.getElementById('current').value);
  if (!isNaN(v) && !isNaN(i)) {
    document.getElementById('power').value = (v * i).toFixed(4);
  }
}

const STATE_COLOR = { NORMAL:'#22c55e', ALERT:'#f59e0b', CRITICAL:'#ef4444' };
const STATE_BG    = { NORMAL:'#052e16', ALERT:'#1c1202', CRITICAL:'#1f0606' };
const STATE_ICON  = { NORMAL:'✅', ALERT:'⚠️', CRITICAL:'🚨' };

async function runPredict() {
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  const payload = {
    temperature: document.getElementById('temperature').value,
    voltage    : document.getElementById('voltage').value,
    current    : document.getElementById('current').value,
    power      : document.getElementById('power').value,
  };

  const resultCard = document.getElementById('result');
  const inner      = document.getElementById('result-inner');
  const rawJson    = document.getElementById('raw-json');

  try {
    const resp = await fetch('/predict', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(payload),
    });
    const data = await resp.json();
    rawJson.textContent = JSON.stringify(data, null, 2);
    resultCard.style.display = 'block';

    if (!resp.ok || data.errors) {
      inner.innerHTML = `
        <div class="error-box">
          <strong>Validation errors:</strong>
          <ul>${(data.errors || [data.error || 'Unknown error']).map(e => '<li>' + e + '</li>').join('')}</ul>
        </div>`;
    } else {
      const s   = data.prediction;
      const col = STATE_COLOR[s] || '#aaa';
      const bg  = STATE_BG[s]    || '#111';
      const ic  = STATE_ICON[s]  || '?';
      const probs = data.probabilities || {};

      const probBars = ['NORMAL','ALERT','CRITICAL'].map(cls => {
        const pct = ((probs[cls] || 0) * 100).toFixed(1);
        return `
          <div class="prob-row">
            <span class="prob-name">${cls}</span>
            <div class="prob-track">
              <div class="prob-fill" style="width:${pct}%;background:${STATE_COLOR[cls]}"></div>
            </div>
            <span class="prob-pct" style="color:${STATE_COLOR[cls]}">${pct}%</span>
          </div>`;
      }).join('');

      inner.innerHTML = `
        <div class="state-banner" style="background:${bg};border-color:${col}">
          <span class="state-icon">${ic}</span>
          <div>
            <div class="state-label" style="color:${col}">${s}</div>
            <div class="state-sub">Inference: ${data.inference_ms?.toFixed(2) ?? '—'} ms</div>
          </div>
        </div>
        <div class="prob-bar-wrap">${probBars}</div>`;
    }
  } catch(e) {
    resultCard.style.display = 'block';
    inner.innerHTML = `<div class="error-box">Network / server error: ${e.message}</div>`;
    rawJson.textContent = '';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict Battery State';
  }
}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/health", methods=["GET"])
def health():
    try:
        bundle = _load_bundle()
        return jsonify({
            "status": "ok",
            "model" : {
                "accuracy"       : bundle["accuracy"],
                "macro_f1"       : bundle["macro_f1"],
                "tree_depth"     : bundle["tree_depth"],
                "n_leaves"       : bundle["n_leaves"],
                "trained_on_rows": bundle["trained_on_rows"],
                "sklearn_version": bundle.get("sklearn_version", "unknown"),
                "features"       : bundle["features"],
                "classes"        : bundle["classes"],
            },
        }), 200
    except FileNotFoundError as e:
        return jsonify({"status": "model_not_found", "detail": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        bundle = _load_bundle()
        return jsonify({
            "features"       : bundle["features"],
            "classes"        : bundle["classes"],
            "dt_params"      : bundle["dt_params"],
            "accuracy"       : bundle["accuracy"],
            "macro_f1"       : bundle["macro_f1"],
            "tree_depth"     : bundle["tree_depth"],
            "n_leaves"       : bundle["n_leaves"],
            "trained_on_rows": bundle["trained_on_rows"],
            "sklearn_version": bundle.get("sklearn_version", "unknown"),
            "bounds"         : BOUNDS,
            "thresholds"     : THRESHOLDS,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    values, errors = _validate_row(data)
    if errors:
        return jsonify({"errors": errors}), 422

    bundle  = _load_bundle()
    clf     = bundle["model"]
    le      = bundle["label_encoder"]
    features = bundle["features"]

    X = np.array([values], dtype=np.float32)

    t0    = time.perf_counter()
    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label       = le.inverse_transform([pred])[0]
    proba_dict  = {le.inverse_transform([i])[0]: float(p)
                   for i, p in enumerate(proba)}

    log.info("Prediction: %-10s  (%.3f ms)  input=%s",
             label, elapsed_ms,
             {f: round(v, 4) for f, v in zip(features, values)})

    return jsonify({
        "prediction"   : label,
        "probabilities": proba_dict,
        "inference_ms" : round(elapsed_ms, 4),
        "input"        : {f: v for f, v in zip(features, values)},
    }), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True, silent=True)
    if data is None or "rows" not in data:
        return jsonify({"error": "JSON body must have a 'rows' key"}), 400

    rows = data["rows"]
    if not isinstance(rows, list) or len(rows) == 0:
        return jsonify({"error": "'rows' must be a non-empty list"}), 400
    if len(rows) > BATCH_LIMIT:
        return jsonify({"error": f"Batch exceeds limit of {BATCH_LIMIT} rows"}), 400

    all_values = []
    all_errors = []
    for i, row in enumerate(rows):
        vals, errs = _validate_row(row)
        if errs:
            all_errors.append({"row": i, "errors": errs})
        else:
            all_values.append(vals)

    if all_errors:
        return jsonify({"errors": all_errors}), 422

    bundle = _load_bundle()
    clf    = bundle["model"]
    le     = bundle["label_encoder"]

    X = np.array(all_values, dtype=np.float32)

    t0    = time.perf_counter()
    preds = clf.predict(X)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    labels = le.inverse_transform(preds).tolist()

    log.info("Batch prediction: %d rows  |  %.3f ms  |  %.0f rows/s",
             len(rows), elapsed_ms, len(rows) / (elapsed_ms / 1000))

    return jsonify({
        "predictions" : labels,
        "count"       : len(labels),
        "inference_ms": round(elapsed_ms, 4),
    }), 200


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        _load_bundle()
    except FileNotFoundError as exc:
        log.error("❌  %s", exc)
        log.error("Run  python train_model.py  first, then restart the server.")
        raise SystemExit(1)

    port = int(os.getenv("PORT", 5000))
    log.info("🚀  Starting EV Battery Monitor on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)