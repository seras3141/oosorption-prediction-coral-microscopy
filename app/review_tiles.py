"""Streamlit app for blinded oocyte-presence review sessions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

DISPLAY_SIZE_PX = 640
SESSION_STATE_KEY = "review_session_data"
CURRENT_INDEX_KEY = "review_current_index"
SESSION_DIR_KEY = "review_session_dir"
SESSION_META_FILENAME = "session_meta.json"
LABELS_FILENAME = "labels.json"


def main() -> None:
    """Run the blinded tile-review app."""
    args = _parse_args(sys.argv[1:])
    session_dir = args.session.resolve()
    meta_path = session_dir / SESSION_META_FILENAME
    labels_path = session_dir / LABELS_FILENAME

    st.set_page_config(page_title="Coral Oocyte Review", layout="wide")

    if not session_dir.exists():
        st.error(f"Session directory not found: {session_dir}")
        st.stop()
    if not meta_path.exists():
        st.error(f"Session metadata not found: {meta_path}")
        st.stop()
    if not labels_path.exists():
        st.error(f"Session labels file not found: {labels_path}")
        st.stop()

    if st.session_state.get(SESSION_DIR_KEY) != str(session_dir):
        try:
            session_data = _load_review_session(meta_path, labels_path)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state[SESSION_STATE_KEY] = session_data
        st.session_state[CURRENT_INDEX_KEY] = _first_unlabelled_index(session_data) or 0
        st.session_state[SESSION_DIR_KEY] = str(session_dir)

    session_data = st.session_state[SESSION_STATE_KEY]
    ordered_tiles = _ordered_tiles(session_data)
    if not ordered_tiles:
        st.error("This review session contains no tiles.")
        st.stop()

    current_index = _clamp_index(st.session_state[CURRENT_INDEX_KEY], len(ordered_tiles))
    st.session_state[CURRENT_INDEX_KEY] = current_index
    tile = ordered_tiles[current_index]
    total_tiles = len(ordered_tiles)
    labelled_count = _labelled_count(session_data)
    current_position = current_index + 1

    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        st.title("Coral Oocyte Review")
        st.caption(f"Tile {current_position} of {total_tiles}")
        st.progress(labelled_count / total_tiles)
        st.image(str(session_dir / tile["png_filename"]), width=DISPLAY_SIZE_PX)
        st.caption(f"Scale: {tile['tile_size']} px")

    with right_col:
        _render_controls(
            labels_path,
            session_data,
            tile=tile,
            current_index=current_index,
            total_tiles=total_tiles,
        )


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


def _read_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON object using a sibling temporary file."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")
        fp.flush()
    tmp_path.replace(path)


def _load_review_session(meta_path: Path, labels_path: Path) -> dict[str, Any]:
    """Load and validate split review-session files."""
    meta = _read_json(meta_path)
    labels = _read_json(labels_path)
    _validate_review_session(meta, labels)
    return _join_tiles_and_labels(meta, labels)


def _validate_review_session(meta: dict[str, Any], labels: dict[str, Any]) -> None:
    """Raise ``ValueError`` when session metadata and labels do not match."""
    if meta.get("session_id") != labels.get("session_id"):
        raise ValueError("session_meta.json and labels.json have different session_id values")

    tiles = meta.get("tiles", [])
    label_rows = labels.get("labels", [])
    if not isinstance(tiles, list) or not isinstance(label_rows, list):
        raise ValueError("Session metadata and labels must contain list fields")
    if len(tiles) != len(label_rows):
        raise ValueError("session_meta.json and labels.json contain different tile counts")

    labels_by_tile_id: dict[str, dict[str, Any]] = {}
    for label in label_rows:
        tile_id = label.get("tile_id")
        if not tile_id:
            raise ValueError("labels.json contains a label row without tile_id")
        if tile_id in labels_by_tile_id:
            raise ValueError(f"labels.json contains duplicate tile_id: {tile_id}")
        labels_by_tile_id[tile_id] = label

    for tile in tiles:
        tile_id = tile.get("tile_id")
        if not tile_id:
            raise ValueError("session_meta.json contains a tile without tile_id")
        label = labels_by_tile_id.get(tile_id)
        if label is None:
            raise ValueError(f"labels.json is missing tile_id: {tile_id}")
        if label.get("display_index") != tile.get("display_index"):
            raise ValueError(f"display_index mismatch for tile_id: {tile_id}")


def _join_tiles_and_labels(meta: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Return session metadata with collaborator labels joined onto each tile."""
    labels_by_tile_id = {label["tile_id"]: label for label in labels["labels"]}
    joined_tiles: list[dict[str, Any]] = []
    for tile in meta["tiles"]:
        label = labels_by_tile_id[tile["tile_id"]]
        joined_tile = dict(tile)
        joined_tile["collaborator_label"] = label["collaborator_label"]
        joined_tile["labelled_at"] = label["labelled_at"]
        joined_tiles.append(joined_tile)

    session_data = dict(meta)
    session_data["tiles"] = joined_tiles
    return session_data


def _labels_manifest_from_session(session_data: dict[str, Any]) -> dict[str, Any]:
    """Return the writable labels manifest from joined session data."""
    return {
        "session_id": session_data["session_id"],
        "labels": [
            {
                "display_index": tile["display_index"],
                "tile_id": tile["tile_id"],
                "collaborator_label": tile["collaborator_label"],
                "labelled_at": tile["labelled_at"],
            }
            for tile in _ordered_tiles(session_data)
        ],
    }


def _ordered_tiles(session_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tiles sorted by display order."""
    return sorted(session_data.get("tiles", []), key=lambda tile: tile["display_index"])


def _first_unlabelled_index(session_data: dict[str, Any]) -> int | None:
    """Return the position of the first unlabelled tile or ``None`` when complete."""
    for index, tile in enumerate(_ordered_tiles(session_data)):
        if tile["collaborator_label"] is None:
            return index
    return None


def _next_unlabelled_index(session_data: dict[str, Any], current_index: int) -> int | None:
    """Return the next unlabelled tile after the current position, wrapping once."""
    ordered_tiles = _ordered_tiles(session_data)
    search_order = list(range(current_index + 1, len(ordered_tiles)))
    search_order += list(range(0, current_index))
    for index in search_order:
        if ordered_tiles[index]["collaborator_label"] is None:
            return index
    return None


def _labelled_count(session_data: dict[str, Any]) -> int:
    """Return the number of labelled tiles in a joined session."""
    return sum(1 for item in _ordered_tiles(session_data) if item["collaborator_label"] is not None)


def _clamp_index(index: int, total_tiles: int) -> int:
    """Return an in-range tile index."""
    return min(max(int(index), 0), total_tiles - 1)


def _record_label(
    labels_path: Path,
    session_data: dict[str, Any],
    *,
    tile_id: str,
    label: bool,
    current_index: int,
) -> None:
    """Persist one collaborator label and advance the current index."""
    tile = next(item for item in session_data["tiles"] if item["tile_id"] == tile_id)
    tile["collaborator_label"] = label
    tile["labelled_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(labels_path, _labels_manifest_from_session(session_data))

    next_unlabelled = _next_unlabelled_index(session_data, current_index)
    if next_unlabelled is not None:
        st.session_state[CURRENT_INDEX_KEY] = next_unlabelled
    else:
        st.session_state[CURRENT_INDEX_KEY] = min(current_index + 1, len(session_data["tiles"]) - 1)
    st.rerun()


def _move_to_index(index: int, total_tiles: int) -> None:
    """Move the current tile index and rerun the app."""
    st.session_state[CURRENT_INDEX_KEY] = _clamp_index(index, total_tiles)
    st.rerun()


def _render_controls(
    labels_path: Path,
    session_data: dict[str, Any],
    *,
    tile: dict[str, Any],
    current_index: int,
    total_tiles: int,
) -> None:
    """Render review controls for the current tile."""
    labelled_count = _labelled_count(session_data)
    remaining = total_tiles - labelled_count

    st.subheader("Answer")
    current_label = tile["collaborator_label"]
    if current_label is True:
        st.success("Current answer: Yes")
    elif current_label is False:
        st.info("Current answer: No")
    else:
        st.warning("Current answer: Not labelled")

    if st.button("Yes - I see an oocyte", use_container_width=True, type="primary"):
        _record_label(
            labels_path,
            session_data,
            tile_id=tile["tile_id"],
            label=True,
            current_index=current_index,
        )
    if st.button("No - no oocyte here", use_container_width=True):
        _record_label(
            labels_path,
            session_data,
            tile_id=tile["tile_id"],
            label=False,
            current_index=current_index,
        )

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("Prev", use_container_width=True, disabled=current_index == 0):
            _move_to_index(current_index - 1, total_tiles)
    with nav_cols[1]:
        if st.button("Next", use_container_width=True, disabled=current_index == total_tiles - 1):
            _move_to_index(current_index + 1, total_tiles)

    st.divider()
    st.metric("Labelled", f"{labelled_count} / {total_tiles}")
    st.metric("Remaining", remaining)
    if remaining == 0:
        _render_completion(labels_path, total_tiles)


def _render_completion(labels_path: Path, total_tiles: int) -> None:
    """Render the completion state once all tiles have been labelled."""
    st.success("Review complete")
    st.write(f"{total_tiles} / {total_tiles} tiles labelled.")
    st.write("Please send this file back to the study coordinator:")
    st.code(str(labels_path))


if __name__ == "__main__":
    main()
