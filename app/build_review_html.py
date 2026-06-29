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
    #app {{
      max-width: 580px;
      margin: 0 auto;
      padding: 2rem 1rem 4rem;
    }}
    h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }}
    .caption {{
      color: #666;
      font-size: 0.85rem;
      margin-bottom: 1rem;
    }}
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
    .scale-caption {{
      text-align: center;
      color: #888;
      font-size: 0.8rem;
      margin-bottom: 1.25rem;
    }}
    .btn {{
      display: block;
      width: 100%;
      padding: 0.8rem 1rem;
      font-size: 1rem;
      font-weight: 500;
      border-radius: 6px;
      border: 1px solid transparent;
      cursor: pointer;
      margin-bottom: 0.6rem;
      transition: background 0.12s, box-shadow 0.12s;
    }}
    .btn:focus-visible {{
      outline: 2px solid #1a73e8;
      outline-offset: 2px;
    }}
    .btn-yes {{
      background: #1a73e8;
      color: #fff;
      border-color: #1a73e8;
    }}
    .btn-yes:hover {{ background: #1558c0; border-color: #1558c0; }}
    .btn-no {{
      background: #fff;
      color: #1a1a1a;
      border-color: #bbc0c7;
    }}
    .btn-no:hover {{ background: #f5f7fa; }}
    hr.divider {{
      border: none;
      border-top: 1px solid #d8dde4;
      margin: 0.75rem 0;
    }}
    .remaining {{ color: #888; font-size: 0.85rem; }}
    .completion h1 {{ margin-bottom: 1rem; }}
    .completion p {{ line-height: 1.5; margin-bottom: 0.8rem; color: #444; }}
    .btn-download {{
      background: #188038;
      color: #fff;
      border-color: #188038;
      margin-top: 0.5rem;
    }}
    .btn-download:hover {{ background: #146c2e; border-color: #146c2e; }}
    .error {{ color: #c62828; background: #fce8e6; border-radius: 6px; padding: 1rem; }}
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    // ---- embedded session data ----
    const SESSION_DATA = {session_json};
    const TILE_IMAGES  = {tile_images_json};

    // ---- runtime ----
    const SESSION_ID  = SESSION_DATA.session_id;
    const STORAGE_KEY = "review_labels_" + SESSION_ID;
    const tiles = SESSION_DATA.tiles.slice().sort((a, b) => a.display_index - b.display_index);

    function loadLabels() {{
      try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
      catch {{ return {{}}; }}
    }}

    function saveLabels(labels) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    }}

    function firstUnlabelledIndex(labels) {{
      for (let i = 0; i < tiles.length; i++) {{
        if (!(tiles[i].tile_id in labels)) return i;
      }}
      return null;
    }}

    function render() {{
      const labels = loadLabels();
      const idx    = firstUnlabelledIndex(labels);
      const app    = document.getElementById("app");
      if (idx === null) {{ renderCompletion(app, labels); }}
      else              {{ renderTile(app, tiles[idx], idx, labels); }}
    }}

    function renderTile(app, tile, idx, labels) {{
      const total    = tiles.length;
      const labelled = Object.keys(labels).length;
      const pct      = (idx / total * 100).toFixed(1);
      const imgSrc   = TILE_IMAGES[tile.png_filename];

      if (!imgSrc) {{
        app.innerHTML = `<p class="error">Image not found: ${{tile.png_filename}}</p>`;
        return;
      }}

      app.innerHTML = `
        <h1>Coral Oocyte Review</h1>
        <p class="caption">Tile ${{idx + 1}} of ${{total}}</p>
        <div class="progress-wrap">
          <div class="progress-fill" style="width:${{pct}}%"></div>
        </div>
        <div class="tile-wrap">
          <img src="${{imgSrc}}" alt="Tile ${{idx + 1}}">
        </div>
        <p class="scale-caption">Scale: ${{tile.tile_size}}&thinsp;px</p>
        <button class="btn btn-yes" id="btn-yes">Yes — I see an oocyte</button>
        <button class="btn btn-no"  id="btn-no">No — no oocyte here</button>
        <hr class="divider">
        <p class="remaining">Remaining: ${{total - labelled}}</p>
      `;

      document.getElementById("btn-yes").addEventListener("click", () => recordLabel(tile.tile_id, true));
      document.getElementById("btn-no").addEventListener("click",  () => recordLabel(tile.tile_id, false));
    }}

    function recordLabel(tileId, label) {{
      const labels = loadLabels();
      const now    = new Date().toISOString().slice(0, 19);
      labels[tileId] = {{ label, labelled_at: now }};
      saveLabels(labels);
      render();
    }}

    function renderCompletion(app, labels) {{
      const total = tiles.length;
      app.innerHTML = `
        <div class="completion">
          <h1>Review complete</h1>
          <p>${{total}} / ${{total}} tiles labelled.</p>
          <p>Click the button below to download your results, then send the
             <strong>session.json</strong> file back to the study coordinator.</p>
          <button class="btn btn-download" id="btn-download">
            Download session.json
          </button>
        </div>
      `;
      document.getElementById("btn-download").addEventListener("click", () => downloadResults(labels));
    }}

    function downloadResults(labels) {{
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
        Path to the prepared review session directory containing
        ``session.json`` and a ``tiles/`` sub-folder.

    Returns
    -------
    Path
        Path to the written ``review.html`` file (inside *session_dir*).
    """
    session_path = session_dir / "session.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Session manifest not found: {session_path}")

    with session_path.open("r", encoding="utf-8") as fp:
        session_data = json.load(fp)

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
