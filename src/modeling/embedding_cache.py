"""Cache frozen-encoder embeddings per tile, and serve them to the head trainer."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.modeling.tile_classification_dataset import TileClassificationDataset
from src.modeling.tile_index import LABEL_RULE_CENTROID, _resolve_repo_path, load_tile_index

LOG = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = "data/embedding_cache"
CACHE_FORMAT_VERSION = 2
POOLINGS = ("mean", "max")
EMBEDDINGS_FILE = "embeddings.npy"
TILES_FILE = "tiles.csv"
META_FILE = "meta.json"
MANIFEST_HASHES_FILE = "tile_manifests.sha256.json"
CONTENT_BINDINGS = ("tile_manifest_hashes_sha256", "embeddings_sha256")
# Format-1 meta.json is run-id-bound; never rewritten.
LEGACY_CACHE_BINDINGS = {
    "54909588b8d0c03b52d1fd2bc48e30d377295f1ec15089b5050661c62d622dae": {
        "tile_manifest_hashes_sha256": "d50efadb6d354dff9addccc803927036b1346aa50977d2c0198c6823c256cf1b",
        "embeddings_sha256": "e1933129b5fc06d4df797d6b04d564be1dfe17a795a16b888d27b0650be51396",
    },
    "32365347daa254eff5405e62ba4524781aa8d4d021ee4b8ee0231748d58b4a27": {
        "tile_manifest_hashes_sha256": "d50efadb6d354dff9addccc803927036b1346aa50977d2c0198c6823c256cf1b",
        "embeddings_sha256": "de8a38f5bc61e3179a536b5318aca8ec9cdb828794481db16a0b34672836212d",
    },
    "a8210ccd0345b961b2b19816fe2fc16cc0c774110c8d629c64d3c2e0c60eecaf": {
        "tile_manifest_hashes_sha256": "d50efadb6d354dff9addccc803927036b1346aa50977d2c0198c6823c256cf1b",
        "embeddings_sha256": "4b1d3f12e0ada49cc89650f5d6e24823d3639015b047e4943bbb6498ab33fc30",
    },
}


def cache_dir(encoder_name: str, tile_size: int, root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    """Return the cache directory for one encoder and tile size."""
    slug = encoder_name.replace("/", "-")
    return _resolve_repo_path(root) / slug / f"{tile_size:04d}"


def tile_list_sha256(frame: pd.DataFrame) -> str:
    """Hash the tile ids and image paths a cache covers, in cache order."""
    digest = hashlib.sha256()
    for tile_id, png_path in zip(frame["tile_id"], frame["png_path"]):
        digest.update(f"{tile_id}\t{png_path}\n".encode("utf-8"))
    return digest.hexdigest()


def tile_manifest_stats(frame: pd.DataFrame) -> dict[str, list[int]]:
    """Size and mtime of each tile manifest behind ``frame``, kept as provenance."""
    if "cut_name" not in frame.columns:
        return {}
    stats = {}
    for stem, cut_name in sorted(set(zip(frame["stem"], frame["cut_name"]))):
        relative = f"data/tiles/{stem}/{cut_name}/{cut_name}_tile_manifest.json"
        info = _resolve_repo_path(relative).stat()
        stats[relative] = [info.st_size, info.st_mtime_ns]
    return stats


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's contents, read in chunks."""
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def tile_manifest_sha256(relative_paths: list[str]) -> dict[str, str]:
    """Content hash of each tile manifest; re-tiling rewrites them."""
    return {
        relative: file_sha256(_resolve_repo_path(relative)) for relative in sorted(relative_paths)
    }


def _swap_into_place(staging: Path, out_dir: Path) -> None:
    """Replace ``out_dir`` with ``staging``, restoring the old cache if the move fails."""
    previous = out_dir.with_name(out_dir.name + ".previous")
    if out_dir.exists():
        if previous.exists():
            shutil.rmtree(previous)
        os.replace(out_dir, previous)
    try:
        os.replace(staging, out_dir)
    except OSError:
        if previous.exists():
            os.replace(previous, out_dir)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def local_encoder_revision(encoder_name: str) -> str | None:
    """Return the commit hash of the encoder weights in the local HuggingFace cache."""
    from huggingface_hub import try_to_load_from_cache

    config_path = try_to_load_from_cache(encoder_name, "config.json")
    if not isinstance(config_path, str):
        return None
    return Path(config_path).parent.name


def cache_tiles(tile_size: int) -> pd.DataFrame:
    """Every indexed tile at ``tile_size``, sorted by tile id: the cache's row order."""
    # Labels join later; centroid skips the geometry pass.
    index = load_tile_index(tile_sizes=(tile_size,), label_rule=LABEL_RULE_CENTROID)
    return index.sort_values("tile_id").reset_index(drop=True)


def build_cache(
    encoder_name: str,
    tile_size: int,
    root: str | Path = DEFAULT_CACHE_ROOT,
    batch_size: int = 64,
    num_workers: int = 4,
    tiles: pd.DataFrame | None = None,
    device: str | None = None,
) -> Path:
    """Encode every tile, or ``tiles`` if given, and write the cache atomically."""
    from torch.utils.data import DataLoader

    from src.modeling.encoders import encoder_input_px, encoder_spec, load_encoder
    from src.modeling.train_tile_classifier import build_transforms

    frame = cache_tiles(tile_size) if tiles is None else tiles.reset_index(drop=True)
    manifests = tile_manifest_stats(frame)
    manifest_hashes = tile_manifest_sha256(list(manifests))
    model = load_encoder(encoder_name=encoder_name, tile_size=tile_size)
    spec = encoder_spec(encoder_name)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target).eval()

    input_px = encoder_input_px(tile_size)
    dataset = TileClassificationDataset(
        frame.assign(label="negative", is_ambiguous=False),
        transform=build_transforms(train=False, input_px=input_px, mean=spec.mean, std=spec.std),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out_dir = cache_dir(encoder_name, tile_size, root)
    staging = out_dir.with_name(out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_sub = 4 if model.pools_sub_tiles else 1
    embeddings = np.lib.format.open_memmap(
        staging / EMBEDDINGS_FILE, mode="w+", dtype=np.float32,
        shape=(len(frame), n_sub, model.embedding_dim),
    )
    written = 0
    with torch.no_grad():
        for images, _ in loader:
            batch = model.sub_tile_embeddings(images.to(target)).float().cpu().numpy()
            embeddings[written:written + len(batch)] = batch
            written += len(batch)
            if written % (batch_size * 100) < batch_size:
                LOG.info("Encoded %d / %d tiles", written, len(frame))
    embeddings.flush()
    del embeddings

    frame[["tile_id", "png_path"]].to_csv(staging / TILES_FILE, index=False)
    hashes_bytes = (json.dumps(manifest_hashes, indent=2) + "\n").encode("utf-8")
    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "created": date.today().isoformat(),
        "encoder_name": encoder_name,
        "encoder_revision": local_encoder_revision(encoder_name),
        "embedding_dim": int(model.embedding_dim),
        "tile_size": tile_size,
        "n_tiles": int(len(frame)),
        "n_sub_tiles": n_sub,
        "input_px": input_px,
        "normalisation": {"mean": list(spec.mean), "std": list(spec.std)},
        "dtype": "float32",
        "augmentation": "none",
        "tile_list_sha256": tile_list_sha256(frame),
        "tile_manifests": manifests,
        "tile_manifest_hashes_sha256": hashlib.sha256(hashes_bytes).hexdigest(),
        "embeddings_sha256": file_sha256(staging / EMBEDDINGS_FILE),
    }
    (staging / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (staging / MANIFEST_HASHES_FILE).write_bytes(hashes_bytes)

    try:
        unchanged = tile_manifest_sha256(list(manifests)) == manifest_hashes
    except OSError:
        unchanged = False
    if not unchanged:
        shutil.rmtree(staging)
        raise ValueError(
            f"tile manifests changed while the cache was built; {out_dir} was left as it was"
        )
    _swap_into_place(staging, out_dir)
    LOG.info("Wrote %s (%d tiles x %d x %d)", out_dir, len(frame), n_sub, model.embedding_dim)
    return out_dir


@dataclass
class EmbeddingCache:
    """A read-only cache: per-tile sub-tile embeddings plus their metadata."""

    path: Path
    meta: dict[str, Any]
    meta_sha256: str
    embeddings: np.ndarray
    rows: dict[str, int]
    png_paths: list[str]

    @property
    def embedding_dim(self) -> int:
        return int(self.meta["embedding_dim"])

    def pooled(self, tile_ids: list[str], pooling: str = "mean") -> np.ndarray:
        """Return ``(len(tile_ids), embedding_dim)``, pooling sub-tiles as asked."""
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        missing = [t for t in tile_ids if t not in self.rows]
        if missing:
            raise ValueError(
                f"{len(missing)} tile(s) are not in the cache at {self.path}, e.g. "
                f"{missing[0]}; rebuild it for the current tile index"
            )
        block = np.asarray(self.embeddings[[self.rows[t] for t in tile_ids]])
        return block.mean(axis=1) if pooling == "mean" else block.max(axis=1)


def load_cache(
    path: str | Path,
    tile_size: int | None = None,
    encoder_name: str | None = None,
    check_encoder_revision: bool = True,
) -> EmbeddingCache:
    """Open a cache, refusing one built for another size, encoder or weight revision."""
    root = _resolve_repo_path(path)
    if not (root / META_FILE).exists():
        raise ValueError(f"{root} is not an embedding cache (no {META_FILE})")
    # Rebuilds swap directories, changing the inode.
    identity = root.stat().st_ino
    meta_bytes = (root / META_FILE).read_bytes()
    meta = json.loads(meta_bytes)
    meta_sha256 = hashlib.sha256(meta_bytes).hexdigest()
    version = meta.get("format_version")
    if version == CACHE_FORMAT_VERSION:
        for key in CONTENT_BINDINGS:
            if key not in meta:
                raise ValueError(f"{root}: {META_FILE} lacks {key}; rebuild it")
        bindings = {key: meta[key] for key in CONTENT_BINDINGS}
    elif version == 1:
        bindings = LEGACY_CACHE_BINDINGS.get(meta_sha256)
        if bindings is None:
            raise ValueError(f"{root} predates content-bound caches; rebuild it")
    else:
        raise ValueError(f"{root} is cache format {version}, which this code cannot read")
    if tile_size is not None and meta["tile_size"] != tile_size:
        raise ValueError(f"{root} caches {meta['tile_size']} px tiles, not {tile_size} px")
    if encoder_name is not None and meta["encoder_name"] != encoder_name:
        raise ValueError(f"{root} caches {meta['encoder_name']}, not {encoder_name}")
    if check_encoder_revision and meta.get("encoder_revision"):
        current = local_encoder_revision(meta["encoder_name"])
        if current is not None and current != meta["encoder_revision"]:
            raise ValueError(
                f"{root} was built from {meta['encoder_name']} revision "
                f"{meta['encoder_revision'][:12]}, but the local weights are {current[:12]}"
            )

    with (root / TILES_FILE).open(newline="", encoding="utf-8") as fp:
        listed = list(csv.DictReader(fp))
    frame = pd.DataFrame(listed, columns=["tile_id", "png_path"])
    if tile_list_sha256(frame) != meta["tile_list_sha256"]:
        raise ValueError(f"{root}: {TILES_FILE} does not match the hash in {META_FILE}")
    if not (root / MANIFEST_HASHES_FILE).exists():
        raise ValueError(f"{root} has no {MANIFEST_HASHES_FILE}; rebuild it")
    for name, key in ((MANIFEST_HASHES_FILE, "tile_manifest_hashes_sha256"),
                      (EMBEDDINGS_FILE, "embeddings_sha256")):
        if file_sha256(root / name) != bindings[key]:
            raise ValueError(f"{root}: {name} does not match its recorded hash; rebuild it")
    hashes = json.loads((root / MANIFEST_HASHES_FILE).read_bytes())
    if set(hashes) != set(meta.get("tile_manifests", {})):
        raise ValueError(f"{root}: {MANIFEST_HASHES_FILE} lists other manifests than {META_FILE}")
    for relative, recorded in hashes.items():
        manifest = _resolve_repo_path(relative)
        if not manifest.exists():
            raise ValueError(f"{root}: {relative} is gone; the tiles were regenerated")
        if file_sha256(manifest) != recorded:
            raise ValueError(f"{root}: {relative} changed since the cache was built; rebuild it")
    embeddings = np.load(root / EMBEDDINGS_FILE, mmap_mode="r")
    if embeddings.shape != (meta["n_tiles"], meta["n_sub_tiles"], meta["embedding_dim"]):
        raise ValueError(f"{root}: embeddings shape {embeddings.shape} disagrees with {META_FILE}")
    if root.stat().st_ino != identity:
        raise ValueError(f"{root} was replaced while it was loaded; load it again")
    return EmbeddingCache(
        path=root,
        meta=meta,
        meta_sha256=meta_sha256,
        embeddings=embeddings,
        rows={tile_id: row for row, tile_id in enumerate(frame["tile_id"])},
        png_paths=list(frame["png_path"]),
    )


def check_covers(cache: EmbeddingCache, frame: pd.DataFrame) -> None:
    """Refuse a cache that lacks a tile, or holds it under a different image path."""
    for tile_id, png_path in zip(frame["tile_id"], frame["png_path"]):
        row = cache.rows.get(tile_id)
        if row is None:
            raise ValueError(f"{tile_id} is not in the cache at {cache.path}; rebuild it")
        if cache.png_paths[row] != png_path:
            raise ValueError(
                f"{tile_id}: the cache at {cache.path} was built from {cache.png_paths[row]}, "
                f"the index now points at {png_path}; rebuild it"
            )


class CachedEmbeddingDataset(TileClassificationDataset):
    """Serve pooled cached embeddings in place of images, with the same targets."""

    def __init__(self, index: pd.DataFrame, cache: EmbeddingCache, pooling: str = "mean") -> None:
        super().__init__(index)
        check_covers(cache, self.index)
        self._embeddings = torch.from_numpy(
            cache.pooled(list(self.index["tile_id"]), pooling)
        )

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(float(self._targets[position]), dtype=torch.float32)
        return self._embeddings[position], target


def compare_with_live(
    cache: EmbeddingCache,
    n_tiles: int = 256,
    seed: int = 0,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, float]:
    """Re-encode a random sample live and report its largest deviation from the cache."""
    from torch.utils.data import DataLoader

    from src.modeling.encoders import encoder_spec, load_encoder
    from src.modeling.train_tile_classifier import build_transforms

    meta = cache.meta
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(meta["n_tiles"], size=min(n_tiles, meta["n_tiles"]), replace=False))
    tile_ids = list(cache.rows)
    frame = pd.DataFrame({
        "tile_id": [tile_ids[r] for r in rows],
        "png_path": [cache.png_paths[r] for r in rows],
        "tile_size": meta["tile_size"], "label": "negative", "is_ambiguous": False,
    })
    spec = encoder_spec(meta["encoder_name"])
    model = load_encoder(encoder_name=meta["encoder_name"], tile_size=meta["tile_size"])
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target).eval()
    loader = DataLoader(
        TileClassificationDataset(frame, transform=build_transforms(
            train=False, input_px=meta["input_px"], mean=spec.mean, std=spec.std)),
        batch_size=batch_size, shuffle=False,
    )
    live = []
    with torch.no_grad():
        for images, _ in loader:
            live.append(model.sub_tile_embeddings(images.to(target)).float().cpu().numpy())
    live_array = np.concatenate(live)
    cached = np.asarray(cache.embeddings[rows])
    abs_diff = np.abs(live_array - cached)
    scale = float(np.abs(cached).mean())
    return {
        "n_tiles": int(len(rows)),
        "max_abs_diff": float(abs_diff.max()),
        "max_abs_diff_over_mean_magnitude": float(abs_diff.max()) / scale,
        "mean_abs_embedding": scale,
        "pooled_mean_max_abs_diff": float(
            np.abs(live_array.mean(axis=1) - cached.mean(axis=1)).max()
        ),
    }
