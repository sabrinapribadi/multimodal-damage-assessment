"""
Selective download of BRIGHT dataset events via HTTP range requests.

Reads only the zip central directory, then downloads only the requested
event tiles — avoids pulling full zips (pre-event.zip is 9.9 GB).

Usage:
    # Turkey only (PoC)
    python scripts/download_events.py turkey-earthquake

    # Turkey + Beirut
    python scripts/download_events.py turkey-earthquake beirut-explosion

    # All available events
    python scripts/download_events.py --all

Available event names (from BRIGHT standard_ML split):
    bata-explosion, beirut-explosion, congo-volcano, haiti-earthquake,
    hawaii-wildfire, la, libya-flood, marshall-wildfire, mexico-hurricane,
    morocco-earthquake, myanmar-hurricane, noto-earthquake,
    turkey-earthquake, ukraine-conflict

Output per event:
    data/processed/{event}/
        pre-event/   {event}_XXXXX_pre_disaster.tif
        post-event/  {event}_XXXXX_post_disaster.tif
        target/      {event}_XXXXX_building_damage.tif

Requires: pip install huggingface_hub httpx
"""
import json
import struct
import sys
import time
import zlib
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_url
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID   = "Kullervo/BRIGHT"
BASE_DIR  = Path(__file__).parent.parent / "data" / "processed"
CD_CACHE  = Path(__file__).parent.parent / "data" / ".cd_cache"   # central directory cache

ALL_EVENTS = [
    "bata-explosion", "beirut-explosion", "congo-volcano", "haiti-earthquake",
    "hawaii-wildfire", "la", "libya-flood", "marshall-wildfire",
    "mexico-hurricane", "morocco-earthquake", "myanmar-hurricane",
    "noto-earthquake", "turkey-earthquake", "ukraine-conflict",
]


class CDNExpired(RuntimeError):
    """Raised when a HuggingFace CDN pre-signed URL has expired (HTTP 403)."""


# ── Zip parsing helpers ───────────────────────────────────────────────────────

def _resolve_cdn_url(hf_url: str) -> tuple[str, int]:
    r = httpx.head(hf_url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return str(r.url), int(r.headers["content-length"])


def _range_get(cdn_url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Range-fetch bytes with exponential-backoff retry for transient network drops."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            r = httpx.get(cdn_url, headers={"Range": f"bytes={start}-{end}"},
                          follow_redirects=True, timeout=120)
            if r.status_code == 403:
                raise CDNExpired("HuggingFace CDN URL expired (403)")
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Range request failed: {r.status_code}")
            return r.content
        except CDNExpired:
            raise  # 403 needs URL refresh, not a bare retry
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt   # 1s → 2s → 4s → 8s → 16s
                time.sleep(wait)
    raise last_exc


def _find_eocd(cdn_url: str, file_size: int) -> tuple[int, int]:
    # 8 KB is enough for any zip footer — smaller than 64 KB, less likely to be rate-limited
    tail_size = min(8192, file_size)
    tail = _range_get(cdn_url, file_size - tail_size, file_size - 1)

    loc_idx = tail.rfind(b"PK\x06\x07")
    if loc_idx != -1:
        eocd64_offset = struct.unpack_from("<Q", tail, loc_idx + 8)[0]
        eocd64 = _range_get(cdn_url, eocd64_offset, eocd64_offset + 55)
        if eocd64[:4] != b"PK\x06\x06":
            raise ValueError("Bad zip64 EOCD signature")
        return int(struct.unpack_from("<Q", eocd64, 48)[0]), int(struct.unpack_from("<Q", eocd64, 40)[0])

    idx = tail.rfind(b"PK\x05\x06")
    if idx == -1:
        raise ValueError("EOCD not found")
    return int(struct.unpack_from("<I", tail, idx + 16)[0]), int(struct.unpack_from("<I", tail, idx + 12)[0])


def _parse_central_directory(data: bytes, filter_str: str) -> list[dict]:
    entries, pos = [], 0
    sig = b"PK\x01\x02"
    while pos < len(data):
        idx = data.find(sig, pos)
        if idx == -1 or idx + 46 > len(data):
            break

        comp_method       = struct.unpack_from("<H", data, idx + 10)[0]
        compressed_size   = struct.unpack_from("<I", data, idx + 20)[0]
        uncompressed_size = struct.unpack_from("<I", data, idx + 24)[0]
        fname_len         = struct.unpack_from("<H", data, idx + 28)[0]
        extra_len         = struct.unpack_from("<H", data, idx + 30)[0]
        comment_len       = struct.unpack_from("<H", data, idx + 32)[0]
        local_hdr_offset  = struct.unpack_from("<I", data, idx + 42)[0]

        fname = data[idx + 46 : idx + 46 + fname_len].decode("utf-8", errors="replace")

        if compressed_size == 0xFFFFFFFF or local_hdr_offset == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF:
            extra = data[idx + 46 + fname_len : idx + 46 + fname_len + extra_len]
            ep = 0
            while ep + 4 <= len(extra):
                tag  = struct.unpack_from("<H", extra, ep)[0]
                size = struct.unpack_from("<H", extra, ep + 2)[0]
                if tag == 0x0001:
                    foff = ep + 4
                    if uncompressed_size == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        uncompressed_size = struct.unpack_from("<Q", extra, foff)[0]; foff += 8
                    if compressed_size == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        compressed_size = struct.unpack_from("<Q", extra, foff)[0]; foff += 8
                    if local_hdr_offset == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        local_hdr_offset = struct.unpack_from("<Q", extra, foff)[0]
                    break
                ep += 4 + size

        if filter_str in fname and fname.endswith(".tif"):
            entries.append({
                "filename":         fname,
                "comp_method":      comp_method,
                "compressed_size":  int(compressed_size),
                "local_hdr_offset": int(local_hdr_offset),
            })

        pos = idx + 46 + fname_len + extra_len + comment_len
    return entries


def _load_or_fetch_entries(zip_name: str, cdn_url: str, file_size: int,
                           filter_str: str) -> list[dict]:
    """Return parsed central directory entries, using disk cache when available."""
    CD_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CD_CACHE / f"{zip_name}.{filter_str}.json"

    if cache_file.exists():
        print(f"  [{zip_name}] Loading central directory from cache ...", end=" ", flush=True)
        entries = json.loads(cache_file.read_text())
        print(f"OK ({len(entries)} entries)")
        return entries

    print(f"  [{zip_name}] Reading central directory ...", end=" ", flush=True)
    cd_offset, cd_size = _find_eocd(cdn_url, file_size)
    cd_data  = _range_get(cdn_url, cd_offset, cd_offset + cd_size - 1)
    entries  = _parse_central_directory(cd_data, filter_str)
    print(f"OK ({len(entries)} entries) — cached to disk")

    cache_file.write_text(json.dumps(entries))
    return entries


def _extract_entry(cdn_url: str, entry: dict, dest_dir: Path) -> Path:
    off  = entry["local_hdr_offset"]
    lhdr = _range_get(cdn_url, off, off + 29)
    if lhdr[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header for {entry['filename']}")
    fname_len  = struct.unpack_from("<H", lhdr, 26)[0]
    extra_len  = struct.unpack_from("<H", lhdr, 28)[0]
    data_start = off + 30 + fname_len + extra_len
    payload    = _range_get(cdn_url, data_start, data_start + entry["compressed_size"] - 1)

    method = entry["comp_method"]
    if method == 0:
        content = payload
    elif method == 8:
        content = zlib.decompress(payload, wbits=-15)
    else:
        raise ValueError(f"Unsupported compression method {method}")

    dest = dest_dir / Path(entry["filename"]).name
    dest.write_bytes(content)
    return dest


# ── Per-zip download ──────────────────────────────────────────────────────────

_cdn_cache: dict[str, tuple[str, int]] = {}

def _get_cdn(zip_name: str, force_refresh: bool = False) -> tuple[str, int]:
    if force_refresh or zip_name not in _cdn_cache:
        hf_url = hf_hub_url(REPO_ID, zip_name, repo_type="dataset")
        label  = "Refreshing" if force_refresh else "Resolving"
        print(f"  {label} {zip_name} CDN ...", end=" ", flush=True)
        cdn_url, size = _resolve_cdn_url(hf_url)
        print(f"OK ({size/1e9:.2f} GB)")
        _cdn_cache[zip_name] = (cdn_url, size)
    return _cdn_cache[zip_name]


def download_event(event: str) -> dict[str, int]:
    out_dir = BASE_DIR / event
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_map = {
        "target.zip":     "target",
        "post-event.zip": "post-event",
        "pre-event.zip":  "pre-event",
    }

    totals = {}
    for zip_name, subfolder in zip_map.items():
        dest_dir = out_dir / subfolder
        dest_dir.mkdir(exist_ok=True)

        try:
            cdn_url, file_size = _get_cdn(zip_name)
            entries = _load_or_fetch_entries(zip_name, cdn_url, file_size, event)

            downloaded = skipped = 0
            for i, entry in enumerate(entries, 1):
                fname = Path(entry["filename"]).name
                dest  = dest_dir / fname
                if dest.exists():
                    continue
                size_kb = entry["compressed_size"] / 1024
                print(f"    [{i:>4}/{len(entries)}] {fname}  ({size_kb:.0f} KB) ...", end=" ", flush=True)
                try:
                    _extract_entry(cdn_url, entry, dest_dir)
                    print("done")
                    downloaded += 1
                except CDNExpired:
                    print("CDN expired, refreshing ...", end=" ", flush=True)
                    cdn_url, _ = _get_cdn(zip_name, force_refresh=True)
                    try:
                        _extract_entry(cdn_url, entry, dest_dir)
                        print("done")
                        downloaded += 1
                    except Exception as e2:
                        print(f"SKIP ({e2})")
                        skipped += 1
                except Exception as e:
                    print(f"SKIP ({e})")
                    skipped += 1

            if skipped:
                print(f"  Skipped {skipped} entries")
            totals[subfolder] = downloaded

        except Exception as e:
            print(f"\n  ERROR on {zip_name}: {e}")
            import traceback; traceback.print_exc()
            totals[subfolder] = 0

    return totals


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python scripts/download_events.py turkey-earthquake")
        print("  python scripts/download_events.py turkey-earthquake beirut-explosion")
        print("  python scripts/download_events.py --all")
        print(f"\nAvailable events:\n" + "\n".join(f"  {e}" for e in ALL_EVENTS))
        sys.exit(0)

    if "--all" in args:
        events = ALL_EVENTS
    else:
        events = args
        unknown = [e for e in events if e not in ALL_EVENTS]
        if unknown:
            print(f"Unknown events: {unknown}")
            print(f"Available: {ALL_EVENTS}")
            sys.exit(1)

    print(f"Events to download: {events}")
    print(f"Output base: {BASE_DIR}")
    print(f"Strategy: HTTP range requests (central directory cached to disk)\n")

    grand_total = 0
    for event in events:
        print(f"\n{'='*60}")
        print(f"  Event: {event}")
        print(f"{'='*60}")
        totals = download_event(event)
        grand_total += sum(totals.values())

        for sub in ["pre-event", "post-event", "target"]:
            n = len(list((BASE_DIR / event / sub).glob("*.tif")))
            print(f"  {sub}/  → {n} tiles on disk")

    print(f"\n{'='*60}")
    print(f"Done. {grand_total} new files downloaded across {len(events)} event(s).")


if __name__ == "__main__":
    main()
