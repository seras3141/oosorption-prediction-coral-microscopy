"""Publish an output directory whole: written to a staging directory, then moved into place."""

from __future__ import annotations

import errno
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path


def refuse_existing(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists; pass overwrite=True (--overwrite) to replace it"
        )


def swap_into_place(staging: Path, out_dir: Path) -> None:
    """Replace ``out_dir`` with ``staging``, restoring the old one if the move fails."""
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


def publish_directory(out_dir: Path, write: Callable[[Path], None], overwrite: bool) -> Path:
    """Run ``write`` on a fresh staging directory, then move it to ``out_dir``."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.staging-{uuid.uuid4().hex}")
    # mkdir, unlike mkdtemp, keeps the group-readable umask.
    staging.mkdir()
    try:
        write(staging)
        if overwrite:
            swap_into_place(staging, out_dir)
        else:
            refuse_existing(out_dir, overwrite)
            # rename refuses a non-empty target directory.
            try:
                os.rename(staging, out_dir)
            except OSError as exc:
                if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                    raise FileExistsError(f"{out_dir} came to exist while this run was writing") from exc
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return out_dir
