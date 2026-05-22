"""Streamlit app for blinded oocyte-presence review sessions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

DISPLAY_SIZE_PX = 512
SESSION_STATE_KEY = "review_session_data"
CURRENT_INDEX_KEY = "review_current_index"
SESSION_DIR_KEY = "review_session_dir"


def main() -> None:
    """Run the blinded tile-review app."""
    args = _parse_args(sys.argv[1:])
    session_dir = args.session.resolve()
    session_path = session_dir / "session.json"

    st.set_page_config(page_title="Coral Oocyte Review", layout="centered")

    if not session_dir.exists():
        st.error(f"Session directory not found: {session_dir}")
        st.stop()
    if not session_path.exists():
        st.error(f"Session manifest not found: {session_path}")
        st.stop()

    if st.session_state.get(SESSION_DIR_KEY) != str(session_dir):
        session_data = _read_session(session_path)
        st.session_state[SESSION_STATE_KEY] = session_data
        st.session_state[CURRENT_INDEX_KEY] = _first_unlabelled_index(session_data)
        st.session_state[SESSION_DIR_KEY] = str(session_dir)

    session_data = st.session_state[SESSION_STATE_KEY]
    ordered_tiles = _ordered_tiles(session_data)
    current_index = st.session_state[CURRENT_INDEX_KEY]

    if current_index is None:
        _render_completion(session_path, session_data)
        return

    tile = ordered_tiles[current_index]
    total_tiles = len(ordered_tiles)
    labelled_count = sum(1 for item in ordered_tiles if item["collaborator_label"] is not None)
    current_position = current_index + 1

    st.title("Coral Oocyte Review")
    st.caption(f"Tile {current_position} of {total_tiles}")
    st.progress(current_position / total_tiles)
    st.image(str(session_dir / tile["png_filename"]), width=DISPLAY_SIZE_PX)
    st.caption(f"Scale: {tile['tile_size']} px")

    if st.button("Yes - I see an oocyte", use_container_width=True, type="primary"):
        _record_label(session_path, session_data, tile_display_index=tile["display_index"], label=True)
    if st.button("No - no oocyte here", use_container_width=True):
        _record_label(session_path, session_data, tile_display_index=tile["display_index"], label=False)

    st.divider()
    st.caption(f"Remaining: {total_tiles - labelled_count}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse Streamlit passthrough arguments."""
    parser = argparse.ArgumentParser(description="Review oocyte-presence tiles from a session.")
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to the prepared review session directory.",
    )
    return parser.parse_args(argv)


def _read_session(session_path: Path) -> dict[str, Any]:
    """Load one review session manifest from disk."""
    with session_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_session(session_path: Path, session_data: dict[str, Any]) -> None:
    """Write the in-progress review session manifest back to disk."""
    with session_path.open("w", encoding="utf-8") as fp:
        json.dump(session_data, fp, indent=2)
        fp.write("\n")


def _ordered_tiles(session_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tiles sorted by display order."""
    return sorted(session_data.get("tiles", []), key=lambda tile: tile["display_index"])


def _first_unlabelled_index(session_data: dict[str, Any]) -> int | None:
    """Return the position of the first unlabelled tile or ``None`` when complete."""
    ordered_tiles = _ordered_tiles(session_data)
    for index, tile in enumerate(ordered_tiles):
        if tile["collaborator_label"] is None:
            return index
    return None


def _record_label(
    session_path: Path,
    session_data: dict[str, Any],
    *,
    tile_display_index: int,
    label: bool,
) -> None:
    """Persist one collaborator label and advance to the next pending tile."""
    ordered_tiles = _ordered_tiles(session_data)
    tile = next(item for item in ordered_tiles if item["display_index"] == tile_display_index)
    tile["collaborator_label"] = label
    tile["labelled_at"] = datetime.now().isoformat(timespec="seconds")
    _write_session(session_path, session_data)
    st.session_state[CURRENT_INDEX_KEY] = _first_unlabelled_index(session_data)
    st.rerun()


def _render_completion(session_path: Path, session_data: dict[str, Any]) -> None:
    """Render the completion state once all tiles have been labelled."""
    total_tiles = len(session_data.get("tiles", []))
    st.title("Review complete")
    st.write("40 / 40 tiles labelled." if total_tiles == 40 else f"{total_tiles} / {total_tiles} tiles labelled.")
    st.write("Please send the following file back to the study coordinator:")
    st.code(str(session_path))


if __name__ == "__main__":
    main()