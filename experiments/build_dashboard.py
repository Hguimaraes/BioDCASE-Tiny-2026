# --
# experiment dashboard builder
#
# renders docs/EXPERIMENT_PLAN.md (the canonical plan / discussion document)
# together with all run records found in experiments/runs/*/run.yaml into a
# single self-contained html page: docs/experiment_plan.html
#
# usage: python experiments/build_dashboard.py

import re
import html
import yaml

from pathlib import Path

ROOT = Path(__file__).parent.parent
PLAN_MD_PATH = ROOT / 'docs' / 'EXPERIMENT_PLAN.md'
RUNS_DIR = ROOT / 'experiments' / 'runs'
OUT_PATH = ROOT / 'docs' / 'experiment_plan.html'


# --
# minimal markdown -> html (covers what EXPERIMENT_PLAN.md uses)

def _inline(text):
  text = html.escape(text, quote=False)
  text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
  text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
  text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
  text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
  return text


def md_to_html(md):
  lines = md.splitlines()
  out, i, in_list, list_tag = [], 0, False, None

  def close_list():
    nonlocal in_list, list_tag
    if in_list: out.append('</{}>'.format(list_tag)); in_list = False; list_tag = None

  while i < len(lines):
    line = lines[i]

    # table block
    if line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-|:]+\|$', lines[i + 1]):
      close_list()
      headers = [c.strip() for c in line.strip('|').split('|')]
      out.append('<table><thead><tr>' + ''.join('<th>{}</th>'.format(_inline(h)) for h in headers) + '</tr></thead><tbody>')
      i += 2
      while i < len(lines) and lines[i].startswith('|'):
        cells = [c.strip() for c in lines[i].strip('|').split('|')]
        out.append('<tr>' + ''.join('<td>{}</td>'.format(_inline(c)) for c in cells) + '</tr>')
        i += 1
      out.append('</tbody></table>')
      continue

    # headings
    m = re.match(r'^(#{1,4})\s+(.*)$', line)
    if m:
      close_list()
      level = len(m.group(1)) + 1  # page title is h1, md '#' becomes h2
      sid = re.sub(r'[^a-z0-9]+', '-', m.group(2).lower()).strip('-')
      out.append('<h{l} id="{sid}">{t}</h{l}>'.format(l=min(level, 5), sid=sid, t=_inline(m.group(2))))
      i += 1
      continue

    # list items (unordered / ordered)
    m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', line)
    if m:
      tag = 'ol' if m.group(2).rstrip('.').isdigit() else 'ul'
      if not in_list: out.append('<{}>'.format(tag)); in_list, list_tag = True, tag
      # absorb hanging indented continuation lines
      item = m.group(3)
      while i + 1 < len(lines) and re.match(r'^\s{2,}\S', lines[i + 1]) and not re.match(r'^(\s*)([-*]|\d+\.)\s+', lines[i + 1]):
        i += 1; item += ' ' + lines[i].strip()
      out.append('<li>{}</li>'.format(_inline(item)))
      i += 1
      continue

    # blank
    if not line.strip():
      close_list()
      i += 1
      continue

    # paragraph (merge consecutive plain lines)
    close_list()
    para = [line.strip()]
    while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r'^(#{1,4}\s|\||(\s*)([-*]|\d+\.)\s)', lines[i + 1]):
      i += 1; para.append(lines[i].strip())
    out.append('<p>{}</p>'.format(_inline(' '.join(para))))
    i += 1

  close_list()
  return '\n'.join(out)


# --
# results table from run records

RESULT_COLUMNS = [
  ('run_id', 'Run'),
  ('track', 'Track'),
  ('branch', 'Branch'),
  ('seed', 'Seed'),
  ('val_acc', 'Val ACC'),
  ('val_auc', 'Val AUC'),
  ('tflite_int8_acc', 'int8 ACC'),
  ('tflite_int8_auc', 'int8 AUC'),
  ('tflite_bytes', 'tflite [B]'),
  ('params', 'Params'),
  ('macs', 'MACs'),
  ('status', 'Status'),
  ('notes', 'Notes'),
]


def collect_runs():
  rows = []
  for run_yaml in sorted(RUNS_DIR.glob('*/run.yaml')):
    r = yaml.safe_load(open(run_yaml))
    m, model = r.get('metrics', {}) or {}, r.get('model', {}) or {}
    rows.append({
      'run_id': r.get('run_id', run_yaml.parent.name),
      'track': r.get('track', '-'),
      'branch': (r.get('git') or {}).get('branch', '-'),
      'seed': r.get('seed', '-'),
      'val_acc': m.get('val_acc'),
      'val_auc': m.get('val_auc'),
      'tflite_int8_acc': m.get('tflite_int8_acc'),
      'tflite_int8_auc': m.get('tflite_int8_auc'),
      'tflite_bytes': model.get('tflite_bytes'),
      'params': model.get('params'),
      'macs': model.get('macs'),
      'status': r.get('status', '-'),
      'notes': r.get('notes', ''),
    })
  return rows


def _fmt(v):
  if v is None: return '—'
  if isinstance(v, float): return '{:.4f}'.format(v)
  if isinstance(v, int) and v >= 10000: return '{:,}'.format(v)
  return str(v)


def results_table_html(rows):
  if not rows: return '<p class="muted">No run records found yet in <code>experiments/runs/</code>.</p>'
  best_auc = max((r['tflite_int8_auc'] or r['val_auc'] or 0) for r in rows)
  head = ''.join('<th class="sortable" onclick="sortTable(this)">{}</th>'.format(h) for _, h in RESULT_COLUMNS)
  body = []
  for r in rows:
    is_best = (r['tflite_int8_auc'] or r['val_auc'] or -1) == best_auc
    tds = ''.join('<td>{}</td>'.format(_fmt(r[k])) for k, _ in RESULT_COLUMNS)
    body.append('<tr{}>{}</tr>'.format(' class="best"' if is_best else '', tds))
  return '<table id="results"><thead><tr>{}</tr></thead><tbody>{}</tbody></table>'.format(head, '\n'.join(body))


# --
# page template

CSS = """
:root { --bg:#fcfcfa; --fg:#1d2329; --accent:#0b6e4f; --line:#e3e3dd; --muted:#6b7280; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif; }
main { max-width:980px; margin:0 auto; padding:2rem 1.5rem 6rem; }
h1 { font-size:1.9rem; border-bottom:3px solid var(--accent); padding-bottom:.4rem; }
h2 { font-size:1.4rem; margin-top:2.4rem; color:var(--accent); }
h3 { font-size:1.15rem; margin-top:1.8rem; }
code { background:#eef0ec; padding:.1em .35em; border-radius:4px; font-size:.88em; }
table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.85rem; }
th, td { border:1px solid var(--line); padding:.4rem .55rem; text-align:left; vertical-align:top; }
th { background:#f0f2ee; position:sticky; top:0; }
th.sortable { cursor:pointer; }
th.sortable:hover { background:#e4e8e0; }
tr.best { background:#e8f5ee; font-weight:600; }
tr:nth-child(even):not(.best) { background:#f7f7f4; }
.muted { color:var(--muted); }
.tag { display:inline-block; background:var(--accent); color:#fff; border-radius:4px; padding:0 .5em; font-size:.78rem; margin-left:.5em; vertical-align:middle; }
#results-wrap { overflow-x:auto; }
footer { margin-top:3rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; }
"""

JS = """
function sortTable(th) {
  const table = th.closest('table'), tbody = table.tBodies[0];
  const idx = Array.from(th.parentNode.children).indexOf(th);
  const asc = !(th.dataset.asc === 'true');
  th.parentNode.querySelectorAll('th').forEach(h => delete h.dataset.asc);
  th.dataset.asc = asc;
  const parse = s => { const n = parseFloat(s.replace(/,/g, '')); return isNaN(n) ? s.toLowerCase() : n; };
  Array.from(tbody.rows)
    .sort((a, b) => { const x = parse(a.cells[idx].innerText), y = parse(b.cells[idx].innerText);
      return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); })
    .forEach(r => tbody.appendChild(r));
}
"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BioDCASE-Tiny 2026 — Experiment Plan &amp; Results</title>
<style>{css}</style>
</head>
<body>
<main>
<h1>BioDCASE-Tiny 2026 — Experiment Plan &amp; Results<span class="tag">{n_runs} runs</span></h1>

<h2 id="results-section">Results</h2>
<p class="muted">Click a column header to sort. Best row (by int8 AUC, falling back to val AUC) is highlighted.
Records are read from <code>experiments/runs/&lt;run_id&gt;/run.yaml</code>; rebuild this page with
<code>python experiments/build_dashboard.py</code>.</p>
<div id="results-wrap">
{results_table}
</div>

{plan_html}

<footer>Generated by <code>experiments/build_dashboard.py</code> from <code>docs/EXPERIMENT_PLAN.md</code> and <code>experiments/runs/</code>.</footer>
</main>
<script>{js}</script>
</body>
</html>
"""


if __name__ == '__main__':

  rows = collect_runs()
  page = PAGE.format(
    css=CSS,
    js=JS,
    n_runs=len(rows),
    results_table=results_table_html(rows),
    plan_html=md_to_html(PLAN_MD_PATH.read_text()),
  )
  OUT_PATH.write_text(page)
  print('Dashboard written to: {} ({} runs)'.format(OUT_PATH, len(rows)))
