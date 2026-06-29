#!/usr/bin/env python3
"""Build a self-contained HTML review app from a prepared review session directory.

Embeds all tile PNGs as base64 data URIs and the session manifest as inline JSON
so the resulting file can be opened directly in any modern browser — no Python,
pip, or network access required on the collaborator's machine.

Usage::

    python app/build_review_html.py --session review_sessions/<session_id>

The file ``review.html`` is written into the session directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Coral Oocyte Review — {session_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f0f2f5;
      color: #1a1a1a;
      min-height: 100vh;
    }}
    .layout {{
      display: flex;
      gap: 2rem;
      max-width: 960px;
      margin: 0 auto;
      padding: 2rem 1rem 4rem;
      align-items: flex-start;
    }}
    .col-image {{ flex: 3; min-width: 0; }}
    .col-controls {{ flex: 1; min-width: 200px; }}
    h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }}
    h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; }}
    .caption {{ color: #666; font-size: 0.85rem; margin-bottom: 1rem; }}
    .progress-wrap {{
      height: 6px;
      background: #d8dde4;
      border-radius: 3px;
      margin-bottom: 1.5rem;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      background: #1a73e8;
      border-radius: 3px;
      transition: width 0.25s ease;
    }}
    .tile-wrap {{
      background: #fff;
      border: 1px solid #d8dde4;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 0.5rem;
      line-height: 0;
    }}
    .tile-wrap img {{
      width: 100%;
      display: block;
      image-rendering: pixelated;
    }}
    .scale-caption {{ text-align: center; color: #888; font-size: 0.8rem; }}
    .status {{
      border-radius: 6px;
      padding: 0.6rem 0.75rem;
      font-size: 0.85rem;
      margin-bottom: 1rem;
    }}
    .status-yes      {{ background: #e6f4ea; color: #1e6e35; border: 1px solid #a8d5b5; }}
    .status-no       {{ background: #e8f0fe; color: #1a56cc; border: 1px solid #aac4f5; }}
    .status-unlabelled {{ background: #fef9e7; color: #7a5c00; border: 1px solid #f0d980; }}
    .btn {{
      display: block;
      width: 100%;
      padding: 0.8rem 1rem;
      font-size: 0.95rem;
      font-weight: 500;
      border-radius: 6px;
      border: 1px solid transparent;
      cursor: pointer;
      margin-bottom: 0.6rem;
      transition: background 0.12s;
    }}
    .btn:focus-visible {{ outline: 2px solid #1a73e8; outline-offset: 2px; }}
    .btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .btn-yes {{ background: #1a73e8; color: #fff; border-color: #1a73e8; }}
    .btn-yes:hover:not(:disabled) {{ background: #1558c0; border-color: #1558c0; }}
    .btn-no {{ background: #fff; color: #1a1a1a; border-color: #bbc0c7; }}
    .btn-no:hover:not(:disabled) {{ background: #f5f7fa; }}
    .nav-row {{ display: flex; gap: 0.5rem; margin-bottom: 0.6rem; }}
    .nav-row .btn {{ margin-bottom: 0; }}
    .btn-nav {{ background: #fff; color: #1a1a1a; border-color: #bbc0c7; }}
    .btn-nav:hover:not(:disabled) {{ background: #f5f7fa; }}
    hr.divider {{ border: none; border-top: 1px solid #d8dde4; margin: 0.75rem 0; }}
    .metric {{ margin-bottom: 0.75rem; }}
    .metric-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.15rem; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 700; }}
    .completion-box {{
      background: #e6f4ea;
      border: 1px solid #a8d5b5;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
    }}
    .completion-box h2 {{ color: #1e6e35; margin-bottom: 0.5rem; }}
    .completion-box p {{ font-size: 0.85rem; color: #2d5a3d; line-height: 1.5; }}
    .btn-download {{ background: #188038; color: #fff; border-color: #188038; }}
    .btn-download:hover:not(:disabled) {{ background: #146c2e; border-color: #146c2e; }}
    .error {{ color: #c62828; background: #fce8e6; border-radius: 6px; padding: 1rem; }}
    @media (max-width: 600px) {{
      .layout {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="col-image"    id="col-image"></div>
    <div class="col-controls" id="col-controls"></div>
  </div>
  <script>
    // ---- embedded session data ----
    const SESSION_DATA = {session_json};
    const TILE_IMAGES  = {tile_images_json};

    // ---- runtime ----
    const SESSION_ID  = SESSION_DATA.session_id;
    const STORAGE_KEY = "review_labels_" + SESSION_ID;
    const tiles = SESSION_DATA.tiles.slice().sort((a, b) => a.display_index - b.display_index);
    const total = tiles.length;
    let currentIndex = 0;

    function loadLabels() {{
      try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
      catch {{ return {{}}; }}
    }}

    function saveLabels(labels) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    }}

    function labelledCount(labels) {{
      return Object.keys(labels).length;
    }}

    function firstUnlabelledIndex(labels) {{
      for (let i = 0; i < tiles.length; i++) {{
        if (!(tiles[i].tile_id in labels)) return i;
      }}
      return null;
    }}

    function nextUnlabelledIndex(labels, fromIndex) {{
      for (let offset = 1; offset < tiles.length; offset++) {{
        const i = (fromIndex + offset) % tiles.length;
        if (!(tiles[i].tile_id in labels)) return i;
      }}
      return null;
    }}

    function clampIndex(idx) {{
      return Math.min(Math.max(idx, 0), total - 1);
    }}

    function renderImage(idx) {{
      const tile    = tiles[idx];
      const labels  = loadLabels();
      const labelled = labelledCount(labels);
      const pct     = (labelled / total * 100).toFixed(1);
      const imgSrc  = TILE_IMAGES[tile.png_filename];
      const col     = document.getElementById("col-image");

      if (!imgSrc) {{
        col.innerHTML = `<p class="error">Image not found: ${{tile.png_filename}}</p>`;
        return;
      }}

      col.innerHTML = `
        <h1>Coral Oocyte Review</h1>
        <p class="caption">Tile ${{idx + 1}} of ${{total}}</p>
        <div class="progress-wrap">
          <div class="progress-fill" style="width:${{pct}}%"></div>
        </div>
        <div class="tile-wrap">
          <img src="${{imgSrc}}" alt="Tile ${{idx + 1}}">
        </div>
        <p class="scale-caption">Scale: ${{tile.tile_size}}&thinsp;px</p>
      `;
    }}

    function renderControls(idx) {{
      const tile    = tiles[idx];
      const labels  = loadLabels();
      const labelled = labelledCount(labels);
      const remaining = total - labelled;
      const entry   = labels[tile.tile_id];
      const col     = document.getElementById("col-controls");

      let statusHtml;
      if (entry === undefined) {{
        statusHtml = `<div class="status status-unlabelled">Current answer: Not labelled</div>`;
      }} else if (entry.label === true) {{
        statusHtml = `<div class="status status-yes">Current answer: Yes</div>`;
      }} else {{
        statusHtml = `<div class="status status-no">Current answer: No</div>`;
      }}

      const prevDisabled = idx === 0         ? "disabled" : "";
      const nextDisabled = idx === total - 1 ? "disabled" : "";

      let completionHtml = "";
      if (remaining === 0) {{
        completionHtml = `
          <div class="completion-box">
            <h2>Review complete</h2>
            <p>${{total}} / ${{total}} tiles labelled.<br>
               Click below to download your results, then send the
               <strong>session.json</strong> file back to the study coordinator.</p>
          </div>
          <button class="btn btn-download" id="btn-download">Download session.json</button>
        `;
      }}

      col.innerHTML = `
        <h2>Answer</h2>
        ${{statusHtml}}
        <button class="btn btn-yes" id="btn-yes">Yes — I see an oocyte</button>
        <button class="btn btn-no"  id="btn-no">No — no oocyte here</button>
        <div class="nav-row">
          <button class="btn btn-nav" id="btn-prev" ${{prevDisabled}}>Prev</button>
          <button class="btn btn-nav" id="btn-next" ${{nextDisabled}}>Next</button>
        </div>
        <hr class="divider">
        <div class="metric">
          <div class="metric-label">Labelled</div>
          <div class="metric-value">${{labelled}} / ${{total}}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Remaining</div>
          <div class="metric-value">${{remaining}}</div>
        </div>
        ${{completionHtml}}
      `;

      document.getElementById("btn-yes").addEventListener("click", () => recordLabel(tile.tile_id, true));
      document.getElementById("btn-no").addEventListener("click",  () => recordLabel(tile.tile_id, false));
      document.getElementById("btn-prev").addEventListener("click", () => moveTo(idx - 1));
      document.getElementById("btn-next").addEventListener("click", () => moveTo(idx + 1));
      if (remaining === 0) {{
        document.getElementById("btn-download").addEventListener("click", downloadResults);
      }}
    }}

    function render() {{
      renderImage(currentIndex);
      renderControls(currentIndex);
    }}

    function moveTo(idx) {{
      currentIndex = clampIndex(idx);
      render();
    }}

    function recordLabel(tileId, label) {{
      const labels = loadLabels();
      const now    = new Date().toISOString().slice(0, 19);
      labels[tileId] = {{ label, labelled_at: now }};
      saveLabels(labels);
      const next = nextUnlabelledIndex(labels, currentIndex);
      currentIndex = next !== null ? next : clampIndex(currentIndex);
      render();
    }}

    function downloadResults() {{
      const labels  = loadLabels();
      const updated = JSON.parse(JSON.stringify(SESSION_DATA));
      for (const tile of updated.tiles) {{
        const entry = labels[tile.tile_id];
        if (entry) {{
          tile.collaborator_label = entry.label;
          tile.labelled_at        = entry.labelled_at;
        }}
      }}
      const text = JSON.stringify(updated, null, 2) + "\\n";
      const blob = new Blob([text], {{ type: "application/json" }});
      const url  = URL.createObjectURL(blob);
      const a    = Object.assign(document.createElement("a"), {{
        href: url, download: "session.json"
      }});
      a.click();
      URL.revokeObjectURL(url);
    }}

    // start at the first unlabelled tile
    const _init = firstUnlabelledIndex(loadLabels());
    currentIndex = _init !== null ? _init : 0;
    render();
  </script>
</body>
</html>
"""


def build_review_html(session_dir: Path) -> Path:
    """Embed tiles and session data into a single self-contained HTML file.

    Parameters
    ----------
    session_dir:
        Path to the prepared review session directory containing either
        ``session_meta.json`` + ``labels.json`` (current format) or the
        legacy ``session.json``, plus a ``tiles/`` sub-folder.

    Returns
    -------
    Path
        Path to the written ``review.html`` file (inside *session_dir*).
    """
    meta_path   = session_dir / "session_meta.json"
    labels_path = session_dir / "labels.json"
    legacy_path = session_dir / "session.json"

    if meta_path.exists() and labels_path.exists():
        with meta_path.open("r", encoding="utf-8") as fp:
            session_data = json.load(fp)
        with labels_path.open("r", encoding="utf-8") as fp:
            labels_data = json.load(fp)
        labels_by_id = {row["tile_id"]: row for row in labels_data.get("labels", [])}
        for tile in session_data.get("tiles", []):
            row = labels_by_id.get(tile["tile_id"], {})
            tile["collaborator_label"] = row.get("collaborator_label")
            tile["labelled_at"] = row.get("labelled_at")
    elif legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as fp:
            session_data = json.load(fp)
    else:
        raise FileNotFoundError(
            f"Session files not found in {session_dir}: expected session_meta.json + labels.json"
        )

    tile_images: dict[str, str] = {}
    for tile in session_data["tiles"]:
        img_path = session_dir / tile["png_filename"]
        if not img_path.exists():
            raise FileNotFoundError(f"Tile image not found: {img_path}")
        encoded = base64.b64encode(img_path.read_bytes()).decode("ascii")
        tile_images[tile["png_filename"]] = "data:image/png;base64," + encoded

    html = _HTML_TEMPLATE.format(
        session_id=session_data["session_id"],
        session_json=json.dumps(session_data, ensure_ascii=False),
        tile_images_json=json.dumps(tile_images, ensure_ascii=False),
    )

    out_path = session_dir / "review.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML review app from a prepared session directory."
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to the prepared review session directory (contains session.json and tiles/).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    session_dir = args.session if args.session.is_absolute() else REPO_ROOT / args.session
    try:
        out_path = build_review_html(session_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Written: {out_path}  ({size_mb:.1f} MB)")
    print("Share this single file with the collaborator — no installation needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
