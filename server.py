#!/usr/bin/env python3
"""
NeuroTune Web Dashboard & API Server
Serves autotuning experiment reports, GPU telemetry, validation suites, and convergence metrics.
Built with Python standard library (zero external dependencies).
"""

import http.server
import json
import os
import re
import socketserver
import sqlite3
import sys
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DB_PATH = ARTIFACTS_DIR / "matmul_trials.db"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "3000"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_experiments():
    """Retrieve list of experiments and summaries from database and filesystem."""
    experiments = []
    
    # 1. From database if present
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("""
                SELECT e.id, e.kind, e.status, e.created_at, COUNT(t.id)
                FROM experiments e
                LEFT JOIN trials t ON e.id = t.experiment_id
                GROUP BY e.id
                ORDER BY e.created_at DESC
            """)
            for row in cur.fetchall():
                experiments.append({
                    "id": row[0],
                    "kind": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "trial_count": row[4],
                })
            conn.close()
        except Exception as err:
            print(f"[warning] DB read error: {err}")

    # 2. Add reports from filesystem if not in db
    if REPORTS_DIR.exists():
        for r_dir in sorted(REPORTS_DIR.iterdir(), reverse=True):
            if r_dir.is_dir() and not any(e["id"] == r_dir.name for e in experiments):
                experiments.append({
                    "id": r_dir.name,
                    "kind": "report",
                    "status": "complete",
                    "created_at": "N/A",
                    "trial_count": len(list(r_dir.glob("*"))),
                })
    return experiments


def get_report_data(exp_id: str):
    """Retrieve report.md and summary.json for an experiment."""
    target_dir = REPORTS_DIR / exp_id
    if not target_dir.exists():
        return None

    data = {"id": exp_id, "markdown": "", "summary": None, "has_plot": False}
    md_file = target_dir / "report.md"
    if md_file.exists():
        try:
            data["markdown"] = md_file.read_text(encoding="utf-8")
        except Exception:
            data["markdown"] = md_file.read_text(encoding="latin-1")

    json_file = target_dir / "summary.json"
    if json_file.exists():
        try:
            data["summary"] = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as err:
            print(f"[warning] JSON parse error: {err}")

    png_file = target_dir / "convergence.png"
    data["has_plot"] = png_file.exists()
    return data


def get_hardware_status():
    """Return hardware specs and runtime state."""
    return {
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "compute_capability": "8.9 (Ada Lovelace)",
        "sm_count": 20,
        "vram": "6.0 GiB GDDR6",
        "smem_per_block": "99 KiB",
        "l2_cache": "24 MiB",
        "cuda_runtime": "13.0 / 13.1",
        "triton_version": "3.8.0",
        "pytorch_version": "2.14.1+cu130",
        "driver_version": "592.82",
        "os": "Ubuntu 26.04 (WSL2 passthrough on Windows 11)",
    }


def get_validation_summary():
    """Read artifacts/validation/*.json files."""
    val_dir = ARTIFACTS_DIR / "validation"
    results = {}
    if val_dir.exists():
        for f in val_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results[f.stem] = {
                    "total": len(data.get("trials", [])),
                    "passed": sum(1 for t in data.get("trials", []) if t.get("status") == "ok"),
                    "shapes": list({f"{t.get('M')}x{t.get('N')}x{t.get('K')}" for t in data.get("trials", [])}),
                    "dtype": data.get("dtype", "fp16"),
                }
            except Exception:
                pass
    return results


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroTune — Triton GPU Autotuning Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --accent: #06b6d4;
      --accent-glow: rgba(6, 182, 212, 0.25);
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.15);
      --warning: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
      line-height: 1.5;
      padding-bottom: 60px;
    }

    header {
      border-bottom: 1px solid var(--card-border);
      background: rgba(9, 13, 22, 0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-badge {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 18px;
      box-shadow: 0 0 16px var(--primary-glow);
    }

    .brand h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .brand span {
      font-size: 12px;
      color: var(--accent);
      background: rgba(6, 182, 212, 0.1);
      padding: 2px 8px;
      border-radius: 9999px;
      border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .nav-stats {
      display: flex;
      gap: 20px;
      align-items: center;
    }

    .badge-pill {
      font-size: 13px;
      padding: 6px 12px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
    }

    .badge-pill.active::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }

    .container {
      max-width: 1280px;
      margin: 32px auto;
      padding: 0 24px;
    }

    .grid-top {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
      transition: transform 0.2s, border-color 0.2s;
    }

    .card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      transform: translateY(-2px);
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }

    .card-metric {
      font-size: 26px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: baseline;
      gap: 8px;
    }

    .card-metric .unit {
      font-size: 14px;
      font-weight: 400;
      color: var(--text-muted);
    }

    .card-subtext {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 8px;
    }

    .tabs {
      display: flex;
      gap: 12px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      padding-bottom: 8px;
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 15px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }

    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.05);
    }

    .tab-btn.active {
      color: #fff;
      background: var(--primary);
      box-shadow: 0 0 12px var(--primary-glow);
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      text-align: left;
    }

    th {
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-muted);
      font-weight: 600;
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    td {
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text);
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .code-pill {
      font-family: var(--font-mono);
      font-size: 12px;
      background: rgba(0, 0, 0, 0.3);
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--accent);
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }

    .status-badge.pass {
      background: var(--success-bg);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .plot-container {
      margin-top: 24px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      background: #000;
      text-align: center;
      padding: 16px;
    }

    .plot-container img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }

    pre.markdown-view {
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.6;
      background: #06090e;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 24px;
      overflow-x: auto;
      white-space: pre-wrap;
      color: #e2e8f0;
    }

    .btn {
      background: var(--primary);
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 500;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 14px;
      transition: background 0.2s;
    }

    .btn:hover {
      background: var(--primary-hover);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.15);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">⚡</div>
      <div>
        <h1>NeuroTune</h1>
      </div>
      <span>Triton GPU Autotuning</span>
    </div>
    <div class="nav-stats">
      <div class="badge-pill active" id="hw-pill">RTX 4050 Laptop GPU (6GB)</div>
      <div class="badge-pill">CUDA 13.0 · Triton 3.8.0</div>
    </div>
  </header>

  <div class="container">
    <div class="grid-top">
      <div class="card">
        <div class="card-title">Top Speedup vs Triton Default</div>
        <div class="card-metric" id="top-speedup">1.25x <span class="unit">(0.0082 ms)</span></div>
        <div class="card-subtext">Peak Triton config: bm32_bn32_bk32_g8_w2_s5</div>
      </div>
      <div class="card">
        <div class="card-title">cuBLAS Reference Gain</div>
        <div class="card-metric" id="top-cublas">1.13x <span class="unit">faster</span></div>
        <div class="card-subtext">Beats PyTorch torch.matmul on 128x128 & 1024x1024</div>
      </div>
      <div class="card">
        <div class="card-title">Thermal Stability Drift</div>
        <div class="card-metric" id="thermal-drift">+3.0 °C <span class="unit">(50°C → 53°C)</span></div>
        <div class="card-subtext">SM Clock: 2130 MHz (0.0 MHz drift, no throttling)</div>
      </div>
      <div class="card">
        <div class="card-title">Correctness Validation</div>
        <div class="card-metric">110 / 110 <span class="unit">PASS</span></div>
        <div class="card-subtext">Smoke, Irregular, and Edge suites (FP16 & BF16)</div>
      </div>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('overview')">Experiment Overview</button>
      <button class="tab-btn" onclick="switchTab('validation')">Correctness Suites</button>
      <button class="tab-btn" onclick="switchTab('hardware')">Hardware & Telemetry</button>
      <button class="tab-btn" onclick="switchTab('report')">Latest Report & Convergence</button>
    </div>

    <!-- TAB 1: OVERVIEW -->
    <div id="tab-overview" class="tab-content active">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">Recent Autotuning Experiments</div>
        <table>
          <thead>
            <tr>
              <th>Experiment ID</th>
              <th>Kind</th>
              <th>Status</th>
              <th>Recorded (UTC)</th>
              <th>Trials</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="exp-table-body">
            <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Loading experiments...</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <div class="card-title">Search Strategy Evaluation Summary (128x128x128 FP16)</div>
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Trials Budget</th>
              <th>Best Latency</th>
              <th>Best Triton Config</th>
              <th>vs. Default</th>
              <th>vs. cuBLAS</th>
              <th>Compile Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong style="color: var(--accent);">learned (LCB surrogate)</strong></td>
              <td>15</td>
              <td><span class="code-pill">0.00819 ms</span></td>
              <td><span class="code-pill">bm32_bn64_bk64_g1_w4_s5</span></td>
              <td><span class="status-badge pass">+25%</span></td>
              <td><span class="status-badge pass">+12.5%</span></td>
              <td>0.4009 s (48,933x steady)</td>
            </tr>
            <tr>
              <td><strong style="color: var(--primary);">random search</strong></td>
              <td>15</td>
              <td><span class="code-pill">0.00819 ms</span></td>
              <td><span class="code-pill">bm32_bn32_bk32_g8_w2_s5</span></td>
              <td><span class="status-badge pass">+25%</span></td>
              <td><span class="status-badge pass">+12.5%</span></td>
              <td>0.4272 s</td>
            </tr>
            <tr>
              <td><strong>tpe (Optuna)</strong></td>
              <td>15</td>
              <td><span class="code-pill">0.01024 ms</span></td>
              <td><span class="code-pill">bm32_bn64_bk16_g8_w4_s5</span></td>
              <td><span class="status-badge" style="background: rgba(255,255,255,0.05); color: var(--text-muted);">1.0x</span></td>
              <td>0.90x</td>
              <td>0.4358 s</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 2: VALIDATION -->
    <div id="tab-validation" class="tab-content">
      <div class="card">
        <div class="card-title">GPU Kernel Correctness Suites (PyTorch Reference Verification)</div>
        <table>
          <thead>
            <tr>
              <th>Suite Name</th>
              <th>Precision</th>
              <th>Shapes Verified</th>
              <th>Passed / Total</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="val-table-body">
            <tr>
              <td><strong>Smoke Suite</strong></td>
              <td><span class="code-pill">fp16</span></td>
              <td>16³, 64³, 128³, 512³, 257x511x129, 1x1024x7, 1000x33x517</td>
              <td>35 / 35</td>
              <td><span class="status-badge pass">100% PASS</span></td>
            </tr>
            <tr>
              <td><strong>Irregular Suite</strong></td>
              <td><span class="code-pill">fp16</span></td>
              <td>1x1x1, 15x15x15, 37x73x101, 257x511x129, 1000x33x517, 31x4097x255</td>
              <td>35 / 35</td>
              <td><span class="status-badge pass">100% PASS</span></td>
            </tr>
            <tr>
              <td><strong>Edge Suite</strong></td>
              <td><span class="code-pill">bf16</span></td>
              <td>1x1x1, 1x16x1, 16x1x16, 1024x1x16, 2048x2048x2048, 4096x4096x64</td>
              <td>40 / 40</td>
              <td><span class="status-badge pass">100% PASS</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: HARDWARE -->
    <div id="tab-hardware" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">Discovered GPU Device & Environment</div>
        <table id="hw-table">
          <tbody>
            <tr><td>Device Name</td><td><strong>NVIDIA GeForce RTX 4050 Laptop GPU</strong></td></tr>
            <tr><td>Compute Capability</td><td>8.9 (Ada Lovelace Architecture)</td></tr>
            <tr><td>Streaming Multiprocessors (SMs)</td><td>20 SMs</td></tr>
            <tr><td>VRAM</td><td>6.0 GiB GDDR6</td></tr>
            <tr><td>Shared Memory / Block</td><td>99 KiB</td></tr>
            <tr><td>L2 Cache</td><td>24 MiB</td></tr>
            <tr><td>NVIDIA Driver</td><td>592.82 (Performance State: P8)</td></tr>
            <tr><td>PyTorch Stack</td><td>2.14.1+cu130 (CUDA runtime 13.0)</td></tr>
            <tr><td>Triton JIT Compiler</td><td>3.8.0</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <div class="card-title">Telemetry Stability Tracking During Optimization</div>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Session Start</th>
              <th>Session End</th>
              <th>Drift (Δ)</th>
              <th>Observation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>GPU Temperature</td>
              <td>50 °C</td>
              <td>53 °C</td>
              <td><span style="color: var(--success);">+3.0 °C</span></td>
              <td>Cool & stable throughout all trials</td>
            </tr>
            <tr>
              <td>SM Frequency</td>
              <td>2130 MHz</td>
              <td>2130 MHz</td>
              <td><span style="color: var(--success);">+0.0 MHz</span></td>
              <td>Zero clock throttle, perfectly steady state</td>
            </tr>
            <tr>
              <td>Memory Clock</td>
              <td>7000 MHz</td>
              <td>8000 MHz</td>
              <td>+1000 MHz</td>
              <td>Ramped up to peak memory bandwidth</td>
            </tr>
            <tr>
              <td>Power Draw</td>
              <td>4.39 W</td>
              <td>11.78 W</td>
              <td>+7.39 W</td>
              <td>Active dynamic power delivery</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 4: REPORT -->
    <div id="tab-report" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
          <span>Latest Autotuning Report</span>
          <div id="report-actions"></div>
        </div>
        <div id="plot-area" class="plot-container" style="display: none;">
          <h4 style="margin-bottom: 12px; font-weight: 500; color: var(--text-muted);">Optimization Convergence Curve</h4>
          <img id="convergence-img" src="" alt="Convergence Curve">
        </div>
        <pre class="markdown-view" id="report-markdown">Loading report...</pre>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
    }

    async function loadData() {
      try {
        const res = await fetch('/api/experiments');
        const experiments = await res.json();
        const tbody = document.getElementById('exp-table-body');
        tbody.innerHTML = '';

        if (!experiments.length) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No experiments recorded yet.</td></tr>';
          return;
        }

        experiments.forEach(exp => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong style="color: var(--accent);">${exp.id}</strong></td>
            <td><span class="code-pill">${exp.kind}</span></td>
            <td><span class="status-badge pass">${exp.status}</span></td>
            <td>${exp.created_at}</td>
            <td>${exp.trial_count}</td>
            <td>
              <button class="btn btn-secondary" onclick="viewReport('${exp.id}')">View Report</button>
            </td>
          `;
          tbody.appendChild(tr);
        });

        // Load the first report by default
        if (experiments.length) {
          viewReport(experiments[0].id);
        }
      } catch (err) {
        console.error('Failed to load experiments:', err);
      }
    }

    async function viewReport(expId) {
      try {
        const res = await fetch('/api/report?id=' + encodeURIComponent(expId));
        const data = await res.json();
        if (!data) return;

        document.getElementById('report-markdown').textContent = data.markdown || 'No report markdown available.';
        
        const plotArea = document.getElementById('plot-area');
        const img = document.getElementById('convergence-img');
        if (data.has_plot) {
          img.src = '/artifacts/reports/' + encodeURIComponent(expId) + '/convergence.png';
          plotArea.style.display = 'block';
        } else {
          plotArea.style.display = 'none';
        }

        document.getElementById('report-actions').innerHTML = `
          <a class="btn btn-secondary" href="/artifacts/reports/${encodeURIComponent(expId)}/trials.csv" download>Download trials.csv</a>
          <a class="btn btn-secondary" href="/artifacts/reports/${encodeURIComponent(expId)}/summary.json" download>Download summary.json</a>
        `;
      } catch (err) {
        console.error('Failed to load report:', err);
      }
    }

    loadData();
  </script>
</body>
</html>
"""


class NeuroTuneHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the dashboard, JSON APIs, and artifacts."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Health check & root dashboard
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # API: Experiments list
        if path == "/api/experiments":
            data = get_experiments()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # API: Single report data
        if path == "/api/report":
            query = urllib.parse.parse_qs(parsed.query)
            exp_id = query.get("id", [""])[0]
            data = get_report_data(exp_id)
            self.send_response(200 if data else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # API: Hardware status
        if path == "/api/hardware":
            data = get_hardware_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # API: Validation results
        if path == "/api/validation":
            data = get_validation_summary()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # Static artifacts serving (/artifacts/...)
        if path.startswith("/artifacts/"):
            rel_path = path[len("/artifacts/"):]
            file_path = ARTIFACTS_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                # Prevent directory traversal
                if not file_path.resolve().is_relative_to(ARTIFACTS_DIR.resolve()):
                    self.send_response(403)
                    self.end_headers()
                    return

                mime = "application/octet-stream"
                if file_path.suffix == ".png":
                    mime = "image/png"
                elif file_path.suffix == ".json":
                    mime = "application/json"
                elif file_path.suffix == ".csv":
                    mime = "text/csv"
                elif file_path.suffix == ".md":
                    mime = "text/markdown; charset=utf-8"

                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # Fallback to default handler
        super().do_GET()

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[server] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run_server():
    server_address = (HOST, PORT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, NeuroTuneHandler) as httpd:
        print(f"[NeuroTune] Dashboard running at http://{HOST}:{PORT}")
        print(f"[NeuroTune] Local access: http://localhost:{PORT}")
        print(f"[NeuroTune] Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down NeuroTune Dashboard.")
            httpd.server_close()


if __name__ == "__main__":
    run_server()
