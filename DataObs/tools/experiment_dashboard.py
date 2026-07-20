#!/usr/bin/env python3
"""Local dashboard for DataObs experiment pipeline runs."""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from flask import Flask, Response, jsonify, request, send_file
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Flask is required. Install flask or run inside the verl-cot env.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/data/hrh/COT/experiments")
STAGES = ["distill", "metrics", "sft", "sft_eval", "grpo", "grpo_eval"]
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(r"Training Progress:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)")
SFT_STEP_RE = re.compile(r"step:(\d+)\s+-\s+train/loss:([0-9.]+)")
GRPO_STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
FINAL_VALIDATION_RE = re.compile(r"Final validation metrics:\s*(\{.*?\})")
VAL_REWARD_RE = re.compile(
    r"(?P<key>val-[A-Za-z0-9_./@+-]+/reward/[A-Za-z0-9_./@+-]+):"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class DashboardConfig:
    experiments_root: Path


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def read_text(path: Path, limit_bytes: int | None = None) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if limit_bytes is None:
        return path.read_text(encoding="utf-8", errors="replace")
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > limit_bytes:
            f.seek(size - limit_bytes)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def safe_exp_dir(root: Path, experiment_id: str) -> Path:
    root = root.expanduser().resolve()
    exp_dir = (root / experiment_id).resolve()
    if root not in exp_dir.parents and exp_dir != root:
        raise ValueError("experiment-id escapes experiments root")
    return exp_dir


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def latest_global_step(path: Path) -> int | None:
    steps: list[int] = []
    if not path.exists():
        return None
    for item in path.glob("global_step_*"):
        if item.is_dir():
            suffix = item.name.removeprefix("global_step_")
            if suffix.isdigit():
                steps.append(int(suffix))
    return max(steps) if steps else None


def extract_final_validation_metrics(log_text: str) -> dict[str, float]:
    matches = FINAL_VALIDATION_RE.findall(log_text)
    if not matches:
        return {}
    try:
        parsed = ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def extract_latest_validation_rewards(log_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in VAL_REWARD_RE.finditer(log_text):
        try:
            out[match.group("key")] = float(match.group("value"))
        except (TypeError, ValueError):
            continue
    return out


def stage_artifacts(exp_dir: Path) -> dict[str, list[Path]]:
    return {
        "distill": [
            exp_dir / "distill" / "filtered_sft.summary.json",
            exp_dir / "distill" / "filtered_sft.parquet",
            exp_dir / "distill" / "filtered_sft.candidates.parquet",
        ],
        "metrics": [
            exp_dir / "dataobs" / exp_dir.name / "data_metrics_summary.csv",
        ],
        "sft": [
            exp_dir / "sft",
        ],
        "sft_eval": [
            exp_dir / "eval" / "sft" / "generated" / "responses_labeled.metrics.json",
            exp_dir / "eval" / "sft" / "generated" / "responses_labeled.json",
        ],
        "grpo": [
            exp_dir / "grpo",
            exp_dir / "grpo_metrics" / "grpo_metrics_derived.csv",
            exp_dir / "grpo_metrics" / "summary.json",
        ],
        "grpo_eval": [
            exp_dir / "eval" / "grpo" / "generated" / "responses_labeled.metrics.json",
            exp_dir / "eval" / "grpo" / "generated" / "responses_labeled.json",
        ],
    }


def command_stage_status(exp_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(exp_dir / "commands.jsonl")
    status = {stage: {"stage": stage, "status": "pending"} for stage in STAGES}
    for item in rows:
        stage = item.get("stage")
        if stage not in status:
            continue
        if "cmd" in item:
            status[stage].update(
                {
                    "status": "running_or_interrupted",
                    "started_at": item.get("time"),
                    "cmd": item.get("cmd_str") or " ".join(item.get("cmd", [])),
                    "env": item.get("env") or {},
                    "log_path": item.get("log_path"),
                }
            )
        if "returncode" in item:
            returncode = item.get("returncode")
            status[stage].update(
                {
                    "status": "success" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "elapsed_sec": item.get("elapsed_sec"),
                    "log_path": item.get("log_path"),
                }
            )
    return status


def infer_running_or_success(exp_dir: Path, status: dict[str, dict[str, Any]]) -> None:
    artifacts = stage_artifacts(exp_dir)
    for stage, paths in artifacts.items():
        if status[stage]["status"] in {"success", "failed", "running_or_interrupted"}:
            continue
        if any(path.exists() for path in paths):
            status[stage]["status"] = "artifact_exists"


def extract_progress(exp_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sft_log = clean_text(read_text(exp_dir / "logs" / "sft.log", 200_000))
    grpo_log = clean_text(read_text(exp_dir / "logs" / "grpo.log", 400_000))
    distill_summary = read_json(exp_dir / "distill" / "filtered_sft.summary.json") or {}

    if distill_summary:
        out["distill"] = {
            "kept": distill_summary.get("num_kept"),
            "candidates": distill_summary.get("num_candidates"),
            "pass_rate": distill_summary.get("teacher_pass_rate"),
            "elapsed_sec": distill_summary.get("elapsed_sec"),
        }

    sft_steps = SFT_STEP_RE.findall(sft_log)
    if sft_steps:
        last_step, last_loss = sft_steps[-1]
        out["sft"] = {"last_step": int(last_step), "last_train_loss": float(last_loss)}
    if "Total training steps:" in sft_log:
        match = re.search(r"Total training steps:\s*(\d+)", sft_log)
        if match:
            out.setdefault("sft", {})["total_steps"] = int(match.group(1))

    progress_matches = PROGRESS_RE.findall(grpo_log)
    grpo_steps = [int(x) for x in GRPO_STEP_RE.findall(grpo_log)]
    if progress_matches:
        pct, step, total = progress_matches[-1]
        out["grpo"] = {"progress_pct": int(pct), "step": int(step), "total_steps": int(total)}
    elif grpo_steps:
        out["grpo"] = {"step": max(grpo_steps)}
    latest_ckpt = latest_global_step(exp_dir / "grpo")
    if latest_ckpt is not None:
        out.setdefault("grpo", {})["latest_checkpoint"] = latest_ckpt
    final_val = extract_final_validation_metrics(grpo_log) or extract_latest_validation_rewards(grpo_log)
    if final_val:
        reward_items = {
            key: value
            for key, value in final_val.items()
            if "/reward/" in key or key.endswith("reward/mean@1")
        }
        primary_key, primary_value = next(iter(reward_items.items() or final_val.items()))
        out.setdefault("grpo", {})["final_validation"] = final_val
        out.setdefault("grpo", {})["final_validation_metric"] = primary_key
        out.setdefault("grpo", {})["final_validation_reward"] = primary_value
    return out


def eval_metrics(exp_dir: Path, split: str, eval_dir_name: str = "eval") -> dict[str, Any] | None:
    generated_dir = exp_dir / eval_dir_name / split / "generated"
    path = generated_dir / "responses_labeled.metrics.json"
    data = read_json(path)
    if not isinstance(data, dict):
        results_path = generated_dir / "results.json"
        results = read_json(results_path)
        if isinstance(results, dict) and "accuracy" in results:
            accuracy = results.get("accuracy")
            total = results.get("num_samples") or results.get("total")
            correct = results.get("num_correct") or results.get("correct")
            try:
                accuracy_float = float(accuracy)
            except (TypeError, ValueError):
                return None
            return {
                "path": str(results_path),
                "accuracy": accuracy_float * 100 if accuracy_float <= 1 else accuracy_float,
                "correct": correct,
                "total": total,
                "mean@1": accuracy_float if accuracy_float <= 1 else accuracy_float / 100,
                "pass@1/mean": accuracy_float if accuracy_float <= 1 else accuracy_float / 100,
            }
        labeled_path = generated_dir / "responses_labeled.json"
        labeled = read_json(labeled_path)
        if isinstance(labeled, list):
            scores = []
            for item in labeled:
                if not isinstance(item, dict) or "score" not in item:
                    continue
                try:
                    scores.append(float(item["score"]))
                except (TypeError, ValueError):
                    continue
            if scores:
                mean_score = sum(scores) / len(scores)
                return {
                    "path": str(labeled_path),
                    "accuracy": mean_score * 100,
                    "correct": int(sum(scores)),
                    "total": len(scores),
                    "mean@1": mean_score,
                    "pass@1/mean": mean_score,
                }
        return None
    first = next(iter(data.values()), None)
    if isinstance(first, dict):
        return {"path": str(path), **first}
    return {"path": str(path), "raw": data}


def ensure_grpo_metrics(exp_dir: Path) -> None:
    log_path = exp_dir / "logs" / "grpo.log"
    if not log_path.exists():
        return
    output_dir = exp_dir / "grpo_metrics"
    derived_csv = output_dir / "grpo_metrics_derived.csv"
    if derived_csv.exists() and derived_csv.stat().st_mtime >= log_path.stat().st_mtime:
        return
    script = REPO_ROOT / "DataObs" / "tools" / "grpo_log_metrics.py"
    subprocess.run(
        [sys.executable, str(script), "--log", str(log_path), "--output-dir", str(output_dir), "--plot"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def dashboard_data(root: Path, experiment_id: str, refresh_grpo: bool = False) -> dict[str, Any]:
    exp_dir = safe_exp_dir(root, experiment_id)
    if refresh_grpo:
        ensure_grpo_metrics(exp_dir)
    status = command_stage_status(exp_dir)
    infer_running_or_success(exp_dir, status)
    manifest = read_json(exp_dir / "manifest.json") or {}
    results = read_json(exp_dir / "results.json") or {}
    artifacts = stage_artifacts(exp_dir)
    grpo_summary = read_json(exp_dir / "grpo_metrics" / "summary.json")
    return {
        "experiment_id": experiment_id,
        "exp_dir": str(exp_dir),
        "exists": exp_dir.exists(),
        "manifest": manifest,
        "results": results,
        "stages": [status[stage] for stage in STAGES],
        "progress": extract_progress(exp_dir),
        "eval": {
            "sft": eval_metrics(exp_dir, "sft"),
            "grpo": eval_metrics(exp_dir, "grpo"),
        },
        "eval_old": {
            "sft": eval_metrics(exp_dir, "sft", "eval_old"),
            "grpo": eval_metrics(exp_dir, "grpo", "eval_old"),
        },
        "grpo_summary": grpo_summary,
        "artifacts": {
            stage: [file_info(path) for path in paths]
            for stage, paths in artifacts.items()
        },
        "plots": {
            name: file_info(exp_dir / "grpo_metrics" / name)
            for name in ["reward.png", "entropy.png", "grad_norm.png", "kl.png"]
        },
        "logs": {
            stage: file_info(exp_dir / "logs" / f"{stage}.log")
            for stage in STAGES
        },
    }


def create_app(config: DashboardConfig) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/experiments")
    def list_experiments() -> Response:
        root = Path(request.args.get("root") or config.experiments_root).expanduser()
        items = []
        if root.exists():
            for path in sorted(root.iterdir()):
                if path.is_dir() and ((path / "manifest.json").exists() or (path / "logs").exists()):
                    items.append(path.name)
        return jsonify({"root": str(root), "experiments": items})

    @app.get("/api/experiment")
    def experiment() -> Response:
        root = Path(request.args.get("root") or config.experiments_root).expanduser()
        experiment_id = request.args.get("id", "").strip()
        if not experiment_id:
            return jsonify({"error": "missing experiment id"}), 400
        try:
            data = dashboard_data(root, experiment_id, refresh_grpo=request.args.get("refresh_grpo") == "1")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(data)

    @app.get("/api/log")
    def log_tail() -> Response:
        root = Path(request.args.get("root") or config.experiments_root).expanduser()
        experiment_id = request.args.get("id", "").strip()
        stage = request.args.get("stage", "").strip()
        if stage not in STAGES:
            return jsonify({"error": "bad stage"}), 400
        try:
            exp_dir = safe_exp_dir(root, experiment_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        text = clean_text(read_text(exp_dir / "logs" / f"{stage}.log", 120_000))
        return jsonify({"stage": stage, "text": text})

    @app.get("/artifact")
    def artifact() -> Response:
        root = Path(request.args.get("root") or config.experiments_root).expanduser()
        experiment_id = request.args.get("id", "").strip()
        rel = request.args.get("path", "").strip()
        try:
            exp_dir = safe_exp_dir(root, experiment_id)
            path = (exp_dir / rel).resolve()
            if exp_dir not in path.parents and path != exp_dir:
                return jsonify({"error": "path escapes experiment"}), 400
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.exists() or not path.is_file():
            return jsonify({"error": "not found"}), 404
        return send_file(path)

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataObs Experiment Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #19212b;
      --muted: #667085;
      --good: #1f8a5b;
      --bad: #c43d3d;
      --warn: #b7791f;
      --info: #2f6fb0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 20px;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 18px; margin: 0 0 10px; }
    h2 { font-size: 15px; margin: 0 0 10px; }
    .controls {
      display: grid;
      grid-template-columns: minmax(280px, 1.3fr) minmax(240px, 1fr) minmax(260px, 1fr) auto auto auto;
      gap: 8px;
      align-items: center;
    }
    input, select, button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    button {
      cursor: pointer;
      background: #24384f;
      color: #fff;
      border-color: #24384f;
      white-space: nowrap;
    }
    main { padding: 16px 20px 32px; }
    .grid {
      display: grid;
      gap: 12px;
    }
    .top-grid {
      grid-template-columns: minmax(240px, 0.8fr) minmax(520px, 1.8fr) minmax(280px, 0.9fr);
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
      overflow-x: auto;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .kpi {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      min-height: 70px;
    }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 20px; font-weight: 700; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; font-size: 12px; }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 700;
    }
    .success { background: #e6f4ee; color: var(--good); }
    .failed { background: #fdeaea; color: var(--bad); }
    .running_or_interrupted { background: #e9f1fb; color: var(--info); }
    .pending { background: #eef0f3; color: var(--muted); }
    .artifact_exists { background: #fff4db; color: var(--warn); }
    .plots {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .plot img {
      display: block;
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    pre {
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      max-height: 460px;
      font-size: 12px;
      line-height: 1.45;
    }
    .paths {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }
    .muted { color: var(--muted); }
    .mt { margin-top: 12px; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .delta-pos { color: var(--good); font-weight: 700; }
    .delta-neg { color: var(--bad); font-weight: 700; }
    @media (max-width: 1100px) {
      .controls, .top-grid, .plots { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>DataObs Experiment Dashboard</h1>
    <div class="controls">
      <input id="root" value="/data/hrh/COT/experiments" aria-label="Experiments root">
      <select id="expSelect" aria-label="Choose experiment"></select>
      <input id="exp" placeholder="experiment-id or custom id" aria-label="Experiment id">
      <button id="load">Load</button>
      <button id="refresh">Refresh GRPO</button>
      <select id="stageLog" aria-label="Stage log">
        <option value="grpo">grpo.log</option>
        <option value="sft_eval">sft_eval.log</option>
        <option value="grpo_eval">grpo_eval.log</option>
        <option value="sft">sft.log</option>
        <option value="distill">distill.log</option>
        <option value="metrics">metrics.log</option>
      </select>
    </div>
  </header>
  <main class="grid">
    <section class="grid top-grid">
      <div class="card">
        <h2>Experiment</h2>
        <div id="expInfo" class="paths muted">No experiment loaded.</div>
      </div>
      <div class="card">
        <h2>Evaluation</h2>
        <div id="evalKpis"></div>
      </div>
      <div class="card">
        <h2>Progress</h2>
        <div class="kpis" id="progressKpis"></div>
      </div>
    </section>
    <section class="card">
      <h2>Stages</h2>
      <div id="stages"></div>
    </section>
    <section class="card">
      <h2>GRPO Curves</h2>
      <div id="plots" class="plots"></div>
    </section>
    <section class="card">
      <h2>Artifacts</h2>
      <div id="artifacts"></div>
    </section>
    <section class="card">
      <h2>Log Tail</h2>
      <pre id="logTail">Select an experiment.</pre>
    </section>
  </main>
  <script>
    const rootEl = document.getElementById('root');
    const expEl = document.getElementById('exp');
    const expSelectEl = document.getElementById('expSelect');
    const logEl = document.getElementById('logTail');
    let currentData = null;

    function fmt(v, digits=4) {
      if (v === null || v === undefined || v === '') return '-';
      if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(digits);
      return String(v);
    }
    function relPath(abs) {
      if (!currentData || !abs) return abs || '';
      const prefix = currentData.exp_dir + '/';
      return abs.startsWith(prefix) ? abs.slice(prefix.length) : abs;
    }
    function artifactLink(abs, text) {
      const rel = relPath(abs);
      const url = `/artifact?root=${encodeURIComponent(rootEl.value)}&id=${encodeURIComponent(expEl.value)}&path=${encodeURIComponent(rel)}`;
      return `<a href="${url}" target="_blank" rel="noreferrer">${text || rel}</a>`;
    }
    async function loadExperiments() {
      const res = await fetch(`/api/experiments?root=${encodeURIComponent(rootEl.value)}`);
      const data = await res.json();
      const options = ['<option value="">Select experiment...</option>'];
      options.push(...data.experiments.map(x => `<option value="${htmlEscape(x)}">${htmlEscape(x)}</option>`));
      expSelectEl.innerHTML = options.join('');
      if (!expEl.value && data.experiments.length) {
        expEl.value = data.experiments[data.experiments.length - 1];
        expSelectEl.value = expEl.value;
      } else if (expEl.value && data.experiments.includes(expEl.value)) {
        expSelectEl.value = expEl.value;
      }
    }
    async function loadData(refresh=false) {
      if (!expEl.value.trim()) return;
      const url = `/api/experiment?root=${encodeURIComponent(rootEl.value)}&id=${encodeURIComponent(expEl.value)}&refresh_grpo=${refresh ? '1' : '0'}`;
      const res = await fetch(url);
      currentData = await res.json();
      render();
      await loadLog();
    }
    async function loadLog() {
      if (!expEl.value.trim()) return;
      const stage = document.getElementById('stageLog').value;
      const res = await fetch(`/api/log?root=${encodeURIComponent(rootEl.value)}&id=${encodeURIComponent(expEl.value)}&stage=${encodeURIComponent(stage)}`);
      const data = await res.json();
      logEl.textContent = data.text || '(empty)';
      logEl.scrollTop = logEl.scrollHeight;
    }
    function renderKpis(container, items) {
      container.innerHTML = items.map(item => `
        <div class="kpi">
          <div class="label">${item.label}</div>
          <div class="value">${item.value}</div>
        </div>
      `).join('');
    }
    function metricValue(metrics, key) {
      if (!metrics) return null;
      const value = metrics[key];
      return value === undefined ? null : value;
    }
    function metricPct(metrics) {
      const accuracy = metricValue(metrics, 'accuracy');
      if (accuracy !== null) return accuracy;
      const pass = metricValue(metrics, 'pass@1/mean');
      return pass === null ? null : pass * 100;
    }
    function metricPass(metrics) {
      return metricValue(metrics, 'pass@1/mean');
    }
    function deltaFmt(current, old) {
      if (current === null || old === null) return '-';
      const delta = current - old;
      const cls = delta > 0 ? 'delta-pos' : (delta < 0 ? 'delta-neg' : '');
      const sign = delta > 0 ? '+' : '';
      return `<span class="${cls}">${sign}${delta.toFixed(2)}</span>`;
    }
    function metricSource(metrics) {
      if (!metrics || !metrics.path) return '-';
      return artifactLink(metrics.path, relPath(metrics.path));
    }
    function renderEvalTable(currentEval, oldEval, progress) {
      const rows = [
        {stage: 'SFT', old: oldEval.sft || null, current: currentEval.sft || null},
        {stage: 'GRPO', old: oldEval.grpo || null, current: currentEval.grpo || null},
      ];
      const finalVal = progress.grpo && progress.grpo.final_validation_reward;
      const finalValMetric = progress.grpo && progress.grpo.final_validation_metric;
      const finalValSource = currentData.logs && currentData.logs.grpo && currentData.logs.grpo.exists
        ? artifactLink(currentData.logs.grpo.path, 'logs/grpo.log')
        : '-';
      document.getElementById('evalKpis').innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th class="num">Old acc</th>
              <th class="num">Current acc</th>
              <th class="num">Delta pp</th>
              <th class="num">Old pass@1</th>
              <th class="num">Current pass@1</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>${rows.map(row => {
            const oldAcc = metricPct(row.old);
            const currentAcc = metricPct(row.current);
            return `
              <tr>
                <td>${row.stage}</td>
                <td class="num">${fmt(oldAcc, 2)}</td>
                <td class="num">${fmt(currentAcc, 2)}</td>
                <td class="num">${deltaFmt(currentAcc, oldAcc)}</td>
                <td class="num">${fmt(metricPass(row.old))}</td>
                <td class="num">${fmt(metricPass(row.current))}</td>
                <td class="paths">${metricSource(row.current)}</td>
              </tr>`;
          }).join('')}
            <tr>
              <td>GRPO final val</td>
              <td class="num">-</td>
              <td class="num">-</td>
              <td class="num">-</td>
              <td class="num">-</td>
              <td class="num">${fmt(finalVal)}</td>
              <td class="paths">${finalValMetric ? htmlEscape(finalValMetric) + '<br>' : ''}${finalValSource}</td>
            </tr>
          </tbody>
        </table>`;
    }
    function render() {
      const d = currentData;
      if (!d || d.error) {
        document.getElementById('expInfo').textContent = d ? d.error : 'No experiment loaded.';
        return;
      }
      document.getElementById('expInfo').innerHTML = `
        <div><b>${d.experiment_id}</b></div>
        <div>${d.exp_dir}</div>
        <div class="mt">dataset: ${fmt(d.manifest.dataset)} | base: ${fmt(d.manifest.base_model)}</div>
      `;
      const evalCurrent = d.eval || {};
      const evalOld = d.eval_old || {};
      const p = d.progress || {};
      renderEvalTable(evalCurrent, evalOld, p);
      renderKpis(document.getElementById('progressKpis'), [
        {label: 'Distill kept', value: fmt(p.distill && p.distill.kept)},
        {label: 'SFT step', value: `${fmt(p.sft && p.sft.last_step)}/${fmt(p.sft && p.sft.total_steps)}`},
        {label: 'GRPO step', value: p.grpo && p.grpo.total_steps ? `${fmt(p.grpo.step)}/${fmt(p.grpo.total_steps)}` : fmt(p.grpo && p.grpo.step)},
        {label: 'Latest ckpt', value: fmt(p.grpo && p.grpo.latest_checkpoint)},
      ]);
      document.getElementById('stages').innerHTML = `
        <table>
          <thead><tr><th>Stage</th><th>Status</th><th>Elapsed</th><th>Log</th><th>Env</th></tr></thead>
          <tbody>${d.stages.map(x => `
            <tr>
              <td>${x.stage}</td>
              <td><span class="badge ${x.status}">${x.status}</span></td>
              <td>${x.elapsed_sec ? fmt(x.elapsed_sec, 1) + 's' : '-'}</td>
              <td class="paths">${x.log_path || '-'}</td>
              <td class="paths">${x.env ? htmlEscape(JSON.stringify(x.env)) : '-'}</td>
            </tr>`).join('')}
          </tbody>
        </table>`;
      const plotNames = ['reward.png', 'entropy.png', 'grad_norm.png', 'kl.png'];
      document.getElementById('plots').innerHTML = plotNames.map(name => {
        const info = d.plots[name];
        if (!info || !info.exists) return `<div class="plot muted">${name} not found</div>`;
        const rel = `grpo_metrics/${name}`;
        const url = `/artifact?root=${encodeURIComponent(rootEl.value)}&id=${encodeURIComponent(expEl.value)}&path=${encodeURIComponent(rel)}&t=${Date.now()}`;
        return `<div class="plot"><h2>${name}</h2><img src="${url}" alt="${name}"></div>`;
      }).join('');
      const rows = [];
      for (const [stage, files] of Object.entries(d.artifacts)) {
        for (const f of files) {
          rows.push(`<tr><td>${stage}</td><td>${f.exists ? 'yes' : 'no'}</td><td class="paths">${f.exists ? artifactLink(f.path, relPath(f.path)) : relPath(f.path)}</td></tr>`);
        }
      }
      document.getElementById('artifacts').innerHTML = `<table><thead><tr><th>Stage</th><th>Exists</th><th>Path</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
    }
    function htmlEscape(s) {
      return s.replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    document.getElementById('load').addEventListener('click', () => loadData(false));
    document.getElementById('refresh').addEventListener('click', () => loadData(true));
    document.getElementById('stageLog').addEventListener('change', loadLog);
    rootEl.addEventListener('change', loadExperiments);
    expSelectEl.addEventListener('change', () => {
      if (expSelectEl.value) {
        expEl.value = expSelectEl.value;
        loadData(false);
      }
    });
    expEl.addEventListener('change', () => {
      const exists = Array.from(expSelectEl.options).some(option => option.value === expEl.value);
      expSelectEl.value = exists ? expEl.value : '';
    });
    loadExperiments().then(() => loadData(false));
    setInterval(() => loadData(false), 30000);
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local dashboard for DataObs experiment outputs.")
    parser.add_argument("--experiments-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_app(DashboardConfig(experiments_root=Path(args.experiments_root)))
    print(f"Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
