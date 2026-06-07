"""Small file-based experiment runtime helpers."""

from __future__ import annotations

import csv
import datetime as dt
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
  with Path(path).open("r") as f:
    data = yaml.safe_load(f)
  return data or {}


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  with target.open("w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def expand_config(value: Any) -> Any:
  if isinstance(value, dict):
    return {k: expand_config(v) for k, v in value.items()}
  if isinstance(value, list):
    return [expand_config(v) for v in value]
  if isinstance(value, str):
    return os.path.expanduser(os.path.expandvars(value))
  return value


def resolve_project_path(path: str | Path) -> Path:
  candidate = Path(path)
  if candidate.is_absolute():
    return candidate
  return PROJECT_ROOT / candidate


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
  result = dict(base)
  for key, value in updates.items():
    if isinstance(value, dict) and isinstance(result.get(key), dict):
      result[key] = deep_update(result[key], value)
    else:
      result[key] = value
  return result


def git_commit() -> str:
  try:
    proc = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=PROJECT_ROOT,
      check=True,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()
  except Exception:
    return "unknown"


def git_branch() -> str:
  try:
    proc = subprocess.run(
      ["git", "branch", "--show-current"],
      cwd=PROJECT_ROOT,
      check=True,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()
  except Exception:
    return "unknown"


def git_dirty() -> bool | str:
  try:
    proc = subprocess.run(
      ["git", "status", "--porcelain"],
      cwd=PROJECT_ROOT,
      check=True,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    return bool(proc.stdout.strip())
  except Exception:
    return "unknown"


def make_run_dir(config: dict[str, Any], mode: str) -> Path:
  experiment_cfg = config["experiment"]
  timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  root = resolve_project_path(experiment_cfg["output_root"])
  run_dir = root / experiment_cfg["id"] / f"{timestamp}_{mode}"
  run_dir.mkdir(parents=True, exist_ok=False)
  return run_dir


def run_metadata(config: dict[str, Any], mode: str) -> dict[str, Any]:
  return {
    "experiment": config.get("experiment", {}),
    "mode": mode,
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "git": {"branch": git_branch(), "commit": git_commit(), "dirty": git_dirty()},
    "host": {
      "platform": platform.platform(),
      "python": platform.python_version(),
    },
  }


def append_summary(row: dict[str, Any], config: dict[str, Any]) -> None:
  root = resolve_project_path(config["experiment"]["output_root"])
  path = root / "summary.csv"
  path.parent.mkdir(parents=True, exist_ok=True)
  existing_fields: list[str] = []
  if path.is_file():
    with path.open("r", newline="") as f:
      reader = csv.reader(f)
      existing_fields = next(reader, [])
  fields = list(dict.fromkeys(existing_fields + list(row.keys())))
  rows: list[dict[str, Any]] = []
  if path.is_file():
    with path.open("r", newline="") as f:
      reader = csv.DictReader(f)
      rows = list(reader)
  rows.append(row)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
