"""Generate a local HTML report from experiment summary logs."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any

from experiments import runtime


DEFAULT_SUMMARY = "./output/experiments/summary.csv"
DEFAULT_REPORT = "./output/experiments/report.html"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Input summary CSV.")
  parser.add_argument("--output", default=DEFAULT_REPORT, help="Output HTML report.")
  return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
  if not path.is_file():
    return [], []
  with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    return list(reader.fieldnames or []), list(reader)


def display_columns(fields: list[str]) -> list[str]:
  preferred = [
    "timestamp",
    "experiment_id",
    "mode",
    "branch",
    "run_dir",
    "submission.accuracy_inference",
    "submission.accuracy_tflite",
    "submission.roc_auc_inference",
    "submission.roc_auc_tflite",
    "test_Baseline_pytorch.top1_accuracy",
    "test_Baseline_tflite.top1_accuracy",
    "test_Baseline_pytorch.roc_auc_macro_ovr",
    "test_Baseline_tflite.roc_auc_macro_ovr",
    "test_Baseline_pytorch.size_bytes",
    "test_Baseline_tflite.size_bytes",
    "mss2d.feature_shape",
    "mss2d.num_params",
    "mss2d.final_validation_accuracy",
    "mss2d.best_validation_accuracy",
    "mss2d.best_accuracy_epoch",
    "mss2d.early_stop_epoch",
    "mss2d.test_accuracy",
    "mss2d.evaluated_checkpoint",
    "mss2d.best_checkpoint_path",
    "mss2d.model_path",
  ]
  return [field for field in preferred if field in fields]


def render_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
  if not rows:
    return "<p>No experiment rows found.</p>"
  header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
  body_rows = []
  for row in rows:
    cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
    body_rows.append(f"<tr>{cells}</tr>")
  return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html(columns: list[str], rows: list[dict[str, Any]], summary_path: Path) -> str:
  table = render_table(columns, rows)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BioDCASE TinyML Experiments</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202a;
      --muted: #5a6675;
      --line: #d7dde5;
      --panel: #f5f7fa;
      --accent: #0b6b6f;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      font-weight: 720;
    }}
    p {{
      margin: 0 0 24px;
      color: var(--muted);
    }}
    .meta {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 18px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef3f4;
      color: var(--accent);
      font-weight: 680;
      white-space: nowrap;
    }}
    td {{
      max-width: 360px;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main>
    <h1>BioDCASE TinyML Experiments</h1>
    <p>Local report generated from structured experiment logs.</p>
    <div class="meta">Source: {html.escape(str(summary_path))}</div>
    {table}
  </main>
</body>
</html>
"""


def main() -> None:
  summary_path = runtime.resolve_project_path(parse_args().summary)
  output_path = runtime.resolve_project_path(parse_args().output)
  fields, rows = read_rows(summary_path)
  columns = display_columns(fields) or fields
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(render_html(columns, rows, summary_path))
  print(f"Wrote report: {output_path}")


if __name__ == "__main__":
  main()
