#!/usr/bin/env python3
"""Step 1 of 2 — Download GeoJSON annotation files from a shared folder.

Pipeline
--------
    Step 1 (this script):
        python scripts/download_geojson.py --out-dir data/dataset_YYYY_MM
        Downloads every *.geojson file from the shared source folder.

    Step 2:
        python scripts/download_ndpi.py --geojson-dir data/dataset_YYYY_MM
        Reads the downloaded GeoJSON filenames and fetches the corresponding
        NDPI slide files from PANGAEA.

The current source is a SharePoint/OneDrive shared folder, but this can change.
To point the script at a different location, pass a new --sharing-url at runtime
or update DEFAULT_SHARING_URL in this file.

Authentication
--------------
Authentication uses MSAL's device-code flow — no passwords or secret keys are
stored anywhere in this script.  On first run you will see a prompt like::

    To sign in, use a web browser to open https://microsoft.com/devicelogin
    and enter the code XXXXX to authenticate.

Open the URL in any browser (laptop, phone), sign in with your Helmholtz
account, and enter the code.  No browser is needed on the machine running the
script.  The resulting token is cached at
~/.cache/coral_microscopy_tokens.json; subsequent runs are fully silent until
the token expires (~90 days for organisational accounts).

Running on a SLURM cluster
--------------------------
SLURM jobs are non-interactive: the device-code prompt would block forever.
Follow these two steps instead:

  1. Authenticate once on the login node (interactive SSH session):

         python scripts/download_geojson.py --dry-run

     --dry-run triggers authentication and caches the token, but downloads
     nothing.  Open the printed URL on your laptop and sign in as above.

  2. Submit the actual download as a normal job — the cached token is used
     silently.  Pass --no-prompt so the job fails immediately with a clear
     error instead of hanging if the token has expired:

         sbatch --wrap="python scripts/download_geojson.py \\
                            --no-prompt \\
                            --out-dir data/dataset_YYYY_MM"

     If the job fails with "No cached token", repeat step 1 on the login node.
     Tokens last up to 90 days, so this should rarely be needed.

Requirements
------------
    uv add msal           # only new dependency; everything else is stdlib

Troubleshooting
---------------
If sign-in fails with an admin-consent or AADSTS error, your organisation may
restrict which apps can authenticate.  In that case, register a free Azure AD
app (takes ~3 minutes):

1. https://portal.azure.com → Azure AD → App registrations → New registration.
2. Choose "Public client / native" and add redirect URI:
       https://login.microsoftonline.com/common/oauth2/nativeclient
3. API permissions → Add → Microsoft Graph → Delegated → Files.Read.All.
4. Copy the Application (client) ID and pass it via --app-id.

No client secret is needed; device-code flow is designed for public clients.
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Current source: SharePoint shared folder containing the GeoJSON annotations.
# Change this (or pass --sharing-url) when the storage location moves.
DEFAULT_SHARING_URL = (
    "https://hmgu-my.sharepoint.com/:f:/g/personal/serena_sritharan_helmholtz-munich_de"
    "/IgAMw8mqI6erQr2MGiPFYw1_AYuTL7hYb_s6DFv5_F8O_Gg?e=Ywfx9l"
)

# Azure CLI public-client app ID — no secret required.
# Replace via --app-id if blocked by your organisation (see Troubleshooting above).
DEFAULT_APP_ID = "04b07795-8542-4523-8734-1b68cf4af4f2"

# 'organizations' accepts any Azure AD (work/school) account.
DEFAULT_TENANT = "organizations"

# Microsoft Graph delegated scope for read access to files.
GRAPH_SCOPES = ["https://graph.microsoft.com/Files.Read.All"]

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Default output: a new dated dataset folder alongside existing ones.
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "data" / "dataset_new"
TOKEN_CACHE_PATH = Path.home() / ".cache" / "coral_microscopy_tokens.json"

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Only files with these extensions are downloaded.  Everything else is skipped.
GEOJSON_SUFFIXES = {".geojson"}


# ---------------------------------------------------------------------------
# Authentication (MSAL device-code flow)
# ---------------------------------------------------------------------------

def _load_token_cache() -> "msal.SerializableTokenCache":  # type: ignore[name-defined]
    """Load the persisted MSAL token cache, or return an empty one."""
    import msal  # noqa: PLC0415

    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH.exists():
        cache.deserialize(TOKEN_CACHE_PATH.read_text())
    return cache


def _save_token_cache(cache: "msal.SerializableTokenCache") -> None:  # type: ignore[name-defined]
    """Write the token cache to disk if it changed."""
    if cache.has_state_changed:
        TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE_PATH.write_text(cache.serialize())
        logger.debug("Token cache saved to %s", TOKEN_CACHE_PATH)


def get_access_token(
    app_id: str = DEFAULT_APP_ID,
    tenant: str = DEFAULT_TENANT,
    scopes: list[str] | None = None,
    *,
    no_prompt: bool = False,
) -> str:
    """Return a Microsoft Graph bearer token, prompting for device-code login if needed.

    Tries a cached refresh token silently first.  Falls back to an interactive
    device-code flow unless *no_prompt* is set.

    Parameters
    ----------
    app_id : str
        Azure AD application (client) ID.  The default is the public Azure CLI
        app — no secret required.
    tenant : str
        Tenant ID, domain, or ``'organizations'`` for any Azure AD account.
    scopes : list[str] | None
        Delegated OAuth2 scopes.  Defaults to ``Files.Read.All``.
    no_prompt : bool
        If ``True``, raise ``RuntimeError`` immediately when no valid cached
        token exists instead of starting an interactive device-code flow.
        Use this for non-interactive environments such as SLURM jobs.

    Returns
    -------
    str
        Bearer access token string.

    Raises
    ------
    RuntimeError
        If authentication fails, or if *no_prompt* is set and no cached token
        is available.
    """
    import msal  # noqa: PLC0415

    if scopes is None:
        scopes = GRAPH_SCOPES

    cache = _load_token_cache()
    authority = f"https://login.microsoftonline.com/{tenant}"
    app = msal.PublicClientApplication(app_id, authority=authority, token_cache=cache)

    # Try a silent refresh using a cached token.
    accounts = app.get_accounts()
    result: dict[str, Any] | None = None
    if accounts:
        logger.info("Cached account found — attempting silent token refresh.")
        result = app.acquire_token_silent(scopes, account=accounts[0])

    if result is None:
        if no_prompt:
            raise RuntimeError(
                "No cached token found and --no-prompt is set.\n"
                "Authenticate first by running interactively on the login node:\n"
                "    python scripts/download_geojson.py --dry-run"
            )
        # Interactive device-code flow.
        flow = app.initiate_device_flow(scopes=scopes)
        if "user_code" not in flow:
            raise RuntimeError(
                "Failed to start device-code flow: "
                + flow.get("error_description", str(flow))
            )
        print("\n" + flow["message"])
        print()
        result = app.acquire_token_by_device_flow(flow)

    _save_token_cache(cache)

    if "access_token" not in result:
        raise RuntimeError(
            "Authentication failed: "
            + result.get("error_description", str(result))
        )

    logger.info("Token acquired (expires in %s s).", result.get("expires_in", "?"))
    return result["access_token"]


# ---------------------------------------------------------------------------
# Microsoft Graph helpers
# ---------------------------------------------------------------------------

def _graph_get(token: str, url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    """Authenticated GET against the Graph API; returns parsed JSON.

    Parameters
    ----------
    token : str
        Bearer access token.
    url : str
        Full Graph API endpoint URL.
    params : dict | None
        Optional URL query parameters.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    urllib.error.HTTPError
        On non-2xx responses.
    """
    import json  # noqa: PLC0415
    from urllib.parse import urlencode  # noqa: PLC0415

    if params:
        url = url + ("&" if "?" in url else "?") + urlencode(params)

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def encode_sharing_url(sharing_url: str) -> str:
    """Encode a SharePoint/OneDrive sharing URL for the Graph ``/shares`` endpoint.

    The Graph API requires the sharing URL encoded as ``u!`` + base64url with no ``=``
    padding.  See https://learn.microsoft.com/en-us/graph/api/shares-get.

    Parameters
    ----------
    sharing_url : str
        Raw sharing URL copied from SharePoint.

    Returns
    -------
    str
        Encoded ``{sharingToken}`` path segment.

    Examples
    --------
    >>> tok = encode_sharing_url("https://example.sharepoint.com/folder?e=abc")
    >>> tok.startswith("u!")
    True
    >>> "=" not in tok
    True
    """
    encoded = base64.urlsafe_b64encode(sharing_url.encode("utf-8")).decode("ascii")
    return "u!" + encoded.rstrip("=")


def resolve_sharing_link(token: str, sharing_url: str) -> dict[str, Any]:
    """Resolve a sharing URL to its Graph API ``driveItem`` representation.

    Parameters
    ----------
    token : str
        Bearer access token.
    sharing_url : str
        Raw SharePoint sharing URL.

    Returns
    -------
    dict
        Graph API ``driveItem`` for the shared folder.
    """
    encoded = encode_sharing_url(sharing_url)
    return _graph_get(token, f"{GRAPH_BASE}/shares/{encoded}/driveItem")


def list_drive_children(token: str, drive_id: str, item_id: str) -> list[dict[str, Any]]:
    """List all children of a folder, following ``@odata.nextLink`` pages.

    Parameters
    ----------
    token : str
        Bearer access token.
    drive_id : str
        OneDrive / SharePoint drive ID.
    item_id : str
        Drive item ID of the folder.

    Returns
    -------
    list[dict]
        All child ``driveItem`` objects (files and sub-folders).
    """
    url: str | None = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/children"
    items: list[dict[str, Any]] = []
    while url:
        page = _graph_get(token, url, params={"$top": "200"})
        items.extend(page.get("value", []))
        url = page.get("@odata.nextLink")
    return items


def collect_download_tasks(
    token: str,
    drive_id: str,
    item_id: str,
    out_dir: Path,
    relative: Path = Path(),
    *,
    suffixes: set[str] = GEOJSON_SUFFIXES,
) -> list[tuple[str, Path, int]]:
    """Recursively collect download tasks for files matching *suffixes*.

    Parameters
    ----------
    token : str
        Bearer access token.
    drive_id : str
        OneDrive drive ID.
    item_id : str
        Drive item ID of the root folder to traverse.
    out_dir : Path
        Local base output directory.
    relative : Path
        Accumulated relative sub-path (used during recursion).
    suffixes : set[str]
        File extensions to include (e.g. ``{".geojson"}``).  Pass
        ``set()`` to download everything.

    Returns
    -------
    list[tuple[str, Path, int]]
        Each element is ``(download_url, local_dest_path, file_size_bytes)``.
        ``download_url`` is a pre-authenticated direct URL that does not
        require a bearer token but typically expires within ~1 hour.
    """
    children = list_drive_children(token, drive_id, item_id)
    tasks: list[tuple[str, Path, int]] = []

    for child in children:
        name: str = child["name"]
        if "folder" in child:
            tasks.extend(
                collect_download_tasks(
                    token, drive_id, child["id"], out_dir, relative / name,
                    suffixes=suffixes,
                )
            )
        elif "file" in child:
            if suffixes and Path(name).suffix.lower() not in suffixes:
                logger.debug("Skipping %s (extension not in %s).", name, suffixes)
                continue
            dl_url: str = child.get("@microsoft.graph.downloadUrl", "")
            if not dl_url:
                logger.warning("No downloadUrl for %r — skipping.", name)
                continue
            dest = out_dir / relative / name
            size: int = child.get("size", 0)
            tasks.append((dl_url, dest, size))

    return tasks


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest: Path,
    file_size: int,
    *,
    verbose: bool = True,
) -> bool:
    """Download a pre-authenticated URL to *dest* with an in-place progress bar.

    Uses a ``.part`` staging file so an interrupted download never leaves a
    corrupt file at the destination.

    Parameters
    ----------
    url : str
        Pre-authenticated direct download URL (no bearer token required).
    dest : Path
        Final destination path.
    file_size : int
        Expected byte count (0 if unknown).
    verbose : bool
        Whether to print a progress bar to stdout.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.part")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as response, tmp_dest.open("wb") as fh:
            downloaded = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, file_size, verbose=verbose)

        tmp_dest.replace(dest)
        if verbose:
            print()  # newline after progress bar
        return True

    except urllib.error.HTTPError as exc:
        logger.error("HTTP %s for %s: %s", exc.code, dest.name, exc.reason)
        return False
    except urllib.error.URLError as exc:
        logger.error("URL error for %s: %s", dest.name, exc.reason)
        return False
    except OSError as exc:
        logger.error("I/O error writing %s: %s", dest.name, exc)
        return False
    finally:
        if tmp_dest.exists():
            tmp_dest.unlink()


def _print_progress(downloaded: int, total_size: int, *, verbose: bool) -> None:
    """Print an in-place download progress indicator to stdout."""
    if not verbose:
        return
    if total_size <= 0:
        kb = downloaded / 1_024
        print(f"\r    {kb:.0f} KB downloaded", end="", flush=True)
        return
    pct = min(100.0, downloaded / total_size * 100)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    size_kb = total_size / 1_024
    dl_kb = min(downloaded, total_size) / 1_024
    print(
        f"\r    [{bar}] {pct:5.1f}%  {dl_kb:.0f}/{size_kb:.0f} KB",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point for the GeoJSON download CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 of 2: download GeoJSON annotation files from a shared folder.\n"
            "Run download_ndpi.py afterwards with --geojson-dir pointing at the "
            "output directory to fetch the corresponding NDPI slides."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sharing-url",
        default=DEFAULT_SHARING_URL,
        metavar="URL",
        help="SharePoint 'Copy link' URL for the shared folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Local directory where GeoJSON files will be saved.  "
            "Rename after downloading to match the date, e.g. data/dataset_22_05."
        ),
    )
    parser.add_argument(
        "--app-id",
        default=DEFAULT_APP_ID,
        metavar="UUID",
        help="Azure AD application (client) ID.  See module docstring if authentication fails.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help="Azure AD tenant ID, domain, or 'organizations'.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all files, not just *.geojson.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help=(
            "Exit immediately with an error if no cached token exists, instead of "
            "starting an interactive device-code flow.  Use this for SLURM jobs. "
            "Authenticate first by running interactively with --dry-run on the login node."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Authenticate (if needed) and list files that would be downloaded, "
            "but do not fetch them.  Use this on the login node before submitting "
            "a SLURM job to ensure the token is cached."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        import msal  # noqa: F401, PLC0415
    except ImportError:
        print(
            "ERROR: 'msal' is not installed.\n"
            "Install it with:  uv add msal",
            file=sys.stderr,
        )
        return 1

    suffixes: set[str] = set() if args.all_files else GEOJSON_SUFFIXES
    suffix_label = "all files" if not suffixes else "/".join(sorted(suffixes))

    out_dir: Path = args.out_dir
    print(f"Output dir  : {out_dir.resolve()}")
    print(f"File types  : {suffix_label}")
    print(f"Sharing URL : {args.sharing_url[:80]}…")
    print()

    # --- Authenticate -------------------------------------------------------
    print("Authenticating with Microsoft Graph…")
    try:
        token = get_access_token(
            app_id=args.app_id, tenant=args.tenant, no_prompt=args.no_prompt
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Authenticated.\n")

    # --- Resolve sharing link -----------------------------------------------
    print("Resolving sharing link…")
    try:
        root_item = resolve_sharing_link(token, args.sharing_url)
    except urllib.error.HTTPError as exc:
        print(
            f"ERROR {exc.code} resolving sharing link: {exc.reason}\n"
            "Check that the URL is correct and that your account has access.",
            file=sys.stderr,
        )
        return 1

    drive_id: str = root_item["parentReference"]["driveId"]
    item_id: str = root_item["id"]
    folder_name: str = root_item.get("name", "unknown")
    print(f"Folder      : {folder_name}  (drive {drive_id[:8]}…)\n")

    # --- Enumerate files ----------------------------------------------------
    print(f"Enumerating {suffix_label}…")
    try:
        tasks = collect_download_tasks(
            token, drive_id, item_id, out_dir, suffixes=suffixes
        )
    except urllib.error.HTTPError as exc:
        print(f"ERROR {exc.code} enumerating folder: {exc.reason}", file=sys.stderr)
        return 1

    if not tasks:
        print(f"No {suffix_label} found in the shared folder.")
        return 0

    # Partition into already-present and pending.
    to_download: list[tuple[str, Path, int]] = []
    for dl_url, dest, size in tasks:
        if dest.exists():
            size_kb = dest.stat().st_size / 1_024
            print(f"  [OK]   {dest.name:<50s}  ({size_kb:.0f} KB — already present)")
        else:
            to_download.append((dl_url, dest, size))

    if not to_download:
        print("\nAll files already present — nothing to download.")
        _print_next_step(out_dir)
        return 0

    total_kb = sum(s for _, _, s in to_download) / 1_024
    print(f"\n{len(to_download)} file(s) to download  ({total_kb:.0f} KB total):")
    for _, dest, size in to_download:
        size_kb = size / 1_024
        print(f"  {dest.name:<50s}  {size_kb:.0f} KB")

    if args.dry_run:
        print("\n[dry-run] Exiting without downloading.")
        return 0

    # --- Download -----------------------------------------------------------
    print()
    failed: list[str] = []
    for i, (dl_url, dest, size) in enumerate(to_download, 1):
        print(f"[{i}/{len(to_download)}] {dest.name}")
        ok = download_file(dl_url, dest, size)
        if not ok:
            failed.append(dest.name)

    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(to_download)}):")
        for name in failed:
            print(f"  {name}")
        return 1

    print(f"Done. {len(to_download)} file(s) saved to {out_dir.resolve()}")
    _print_next_step(out_dir)
    return 0


def _print_next_step(out_dir: Path) -> None:
    """Print a reminder to run download_ndpi.py next."""
    print(
        f"\nNext step — download the corresponding NDPI slides:\n"
        f"    python scripts/download_ndpi.py --geojson-dir {out_dir}"
    )


if __name__ == "__main__":
    sys.exit(main())
