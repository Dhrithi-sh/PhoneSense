"""Stage 1: download the raw Amazon Reviews 2023 files into data/raw/.

Sources come from ``config.yaml`` (``dataset.urls``, then ``dataset.fallback_urls``).
A file that already exists with size > 0 is never downloaded again. Downloads go to a
``.part`` file that is renamed only when complete, so an interrupted download is retried.
If the source serves plain ``.jsonl`` but the target is ``.jsonl.gz``, the stream is gzipped
on the fly (saves ~75% disk).

Run: ``python -m phonesense.data.download``
"""

from __future__ import annotations

import gzip
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from phonesense.data.common import Settings

CHUNK = 1 << 20  # 1 MiB
SLOW_WINDOW = 30.0  # seconds
MIN_RATE = 512 * 1024  # bytes/s; below this over a window -> reconnect
MAX_RECONNECTS = 200
USER_AGENT = "phonesense-data-pipeline/1.0 (university IR project)"


def _exists(path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    plain = path.with_suffix("") if path.suffix == ".gz" else None
    return bool(plain and plain.exists() and plain.stat().st_size > 0)


def _open(url: str, offset: int):
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    if offset and resp.status != 206:
        resp.close()
        raise OSError("server does not support resuming (no HTTP 206)")
    return resp


def _fetch(url: str, target: Path, progress: bool = True) -> None:
    """Stream ``url`` into ``target`` (gzip on the fly if needed). Raises on failure.

    CDNs sometimes throttle a long-lived connection to a crawl. If throughput over a
    ``SLOW_WINDOW`` drops below ``MIN_RATE`` (or the connection breaks), we reconnect with an
    HTTP Range request at the byte offset reached and keep writing into the same output, so
    nothing is lost or duplicated.
    """
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    compress = target.suffix == ".gz" and not url.split("?")[0].endswith(".gz")
    resp = _open(url, 0)
    total = int(resp.headers.get("Content-Length") or 0) or None
    offset, reconnects = 0, 0
    sink = gzip.open(part, "wb", compresslevel=4) if compress else part.open("wb")
    with sink, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"download {target.name}", disable=not progress) as bar:
        win_start, win_bytes = time.monotonic(), 0
        while True:
            try:
                chunk = resp.read(CHUNK)
            except (OSError, TimeoutError) as e:
                chunk, err = None, e
            else:
                err = None
            if chunk == b"" and (total is None or offset >= total):
                break
            if chunk:
                sink.write(chunk)
                offset += len(chunk)
                win_bytes += len(chunk)
                bar.update(len(chunk))
            elapsed = time.monotonic() - win_start
            slow = elapsed >= SLOW_WINDOW and win_bytes / elapsed < MIN_RATE
            if err is not None or not chunk or slow:
                resp.close()
                reconnects += 1
                if reconnects > MAX_RECONNECTS:
                    raise OSError(f"giving up after {MAX_RECONNECTS} reconnects ({err})")
                why = err or ("slow connection" if slow else "connection closed early")
                bar.write(f"[download] reconnecting at {offset / 2**20:.0f} MiB ({why})")
                resp = _open(url, offset)
                win_start, win_bytes = time.monotonic(), 0
            elif elapsed >= SLOW_WINDOW:
                win_start, win_bytes = time.monotonic(), 0
    resp.close()
    if part.stat().st_size == 0:
        part.unlink()
        raise OSError("empty download")
    os.replace(part, target)


def download_file(name: str, target: Path, urls: list[str], progress: bool = True) -> bool:
    """Download ``target`` from the first working URL. Returns True if the file is present."""
    if _exists(target):
        print(f"[download] {target.name}: already present, skipping")
        return True
    for url in urls:
        print(f"[download] {name}: {url}")
        try:
            _fetch(url, target, progress)
            print(f"[download] saved {target} ({target.stat().st_size / 2**30:.2f} GiB)")
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[download] failed: {e}", file=sys.stderr)
    print(
        f"\n[download] Could not download the {name} file automatically.\n"
        f"  Please download it manually and save it as:\n    {target}\n"
        f"  (or the uncompressed version as {target.with_suffix('')}).\n"
        "  Sources tried:\n" + "".join(f"    {u}\n" for u in urls)
        + "  Hugging Face dataset: McAuley-Lab/Amazon-Reviews-2023 "
        "(raw/meta_categories/, raw/review_categories/)\n",
        file=sys.stderr,
    )
    return False


def run(s: Settings) -> bool:
    ok = True
    for name, target in (("meta", s.raw_meta), ("reviews", s.raw_reviews)):
        urls = [u for u in (s.urls.get(name), s.fallback_urls.get(name)) if u]
        ok &= download_file(name, target, urls, progress=s.progress)
    return ok


def main() -> None:
    if not run(Settings.from_config()):
        sys.exit(1)


if __name__ == "__main__":
    main()
