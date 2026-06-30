"""
Selective download of Morocco earthquake tiles from HuggingFace BRIGHT dataset.

Uses HTTP range requests to read only the zip central directory, then
downloads only morocco-earthquake entries — avoids pulling the full
pre-event.zip (9.9 GB), post-event.zip (3.3 GB), and target.zip (50 MB).

Usage:
    python scripts/download_morocco.py

Output:
    data/processed/morocco-earthquake/
        pre-event/   morocco-earthquake_XXXXX_pre_disaster.tif
        post-event/  morocco-earthquake_XXXXX_post_disaster.tif
        target/      morocco-earthquake_XXXXX_building_damage.tif

Requires: pip install huggingface_hub httpx
"""
import io
import struct
import sys
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

REPO_ID  = "Kullervo/BRIGHT"
EVENT    = "morocco-earthquake"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed" / "morocco-earthquake"
ZIPS     = ["target.zip", "post-event.zip", "pre-event.zip"]   # smallest first

# ── Zip parsing helpers ───────────────────────────────────────────────────────

def _resolve_cdn_url(hf_url: str) -> tuple[str, int]:
    """Follow HuggingFace redirect to get stable CDN URL + file size."""
    r = httpx.head(hf_url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    size = int(r.headers["content-length"])
    return str(r.url), size


def _range_get(cdn_url: str, start: int, end: int) -> bytes:
    """Download bytes [start, end] inclusive from CDN."""
    r = httpx.get(cdn_url, headers={"Range": f"bytes={start}-{end}"},
                  follow_redirects=True, timeout=60)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"Range request failed: {r.status_code}")
    return r.content


def _find_eocd(cdn_url: str, file_size: int) -> tuple[int, int]:
    """
    Locate the End-Of-Central-Directory record.
    Returns (cd_offset, cd_size).
    Handles both regular zip and zip64.
    """
    # Read the last 64 KB which is enough for EOCD + zip64 locator
    tail_start = max(0, file_size - 65536)
    tail = _range_get(cdn_url, tail_start, file_size - 1)

    # --- zip64 EOCD locator (signature 0x07064b50) ---
    loc_sig = b"PK\x06\x07"
    loc_idx = tail.rfind(loc_sig)
    if loc_idx != -1:
        # zip64 EOCD offset is at locator+8
        eocd64_offset = struct.unpack_from("<Q", tail, loc_idx + 8)[0]
        eocd64 = _range_get(cdn_url, eocd64_offset, eocd64_offset + 55)
        if eocd64[:4] != b"PK\x06\x06":
            raise ValueError("Bad zip64 EOCD signature")
        cd_size   = struct.unpack_from("<Q", eocd64, 40)[0]
        cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]
        return int(cd_offset), int(cd_size)

    # --- regular EOCD (signature 0x06054b50) ---
    eocd_sig = b"PK\x05\x06"
    idx = tail.rfind(eocd_sig)
    if idx == -1:
        raise ValueError("EOCD signature not found — is this a valid zip?")
    cd_size   = struct.unpack_from("<I", tail, idx + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, idx + 16)[0]
    return int(cd_offset), int(cd_size)


def _parse_central_directory(data: bytes, filter_str: str) -> list[dict]:
    """
    Parse central directory bytes, return entries whose filename contains filter_str.
    Each entry dict: filename, comp_method, compressed_size, local_header_offset.
    """
    entries = []
    sig = b"PK\x01\x02"
    pos = 0
    while pos < len(data):
        idx = data.find(sig, pos)
        if idx == -1:
            break
        if idx + 46 > len(data):
            break

        comp_method       = struct.unpack_from("<H", data, idx + 10)[0]
        compressed_size   = struct.unpack_from("<I", data, idx + 20)[0]
        uncompressed_size = struct.unpack_from("<I", data, idx + 24)[0]
        fname_len         = struct.unpack_from("<H", data, idx + 28)[0]
        extra_len         = struct.unpack_from("<H", data, idx + 30)[0]
        comment_len       = struct.unpack_from("<H", data, idx + 32)[0]
        local_hdr_offset  = struct.unpack_from("<I", data, idx + 42)[0]

        fname_bytes = data[idx + 46 : idx + 46 + fname_len]
        fname = fname_bytes.decode("utf-8", errors="replace")

        # Handle zip64 extended info in extra field.
        # Per ZIP spec APPNOTE.TXT §4.5.3: fields appear ONLY when their 32-bit
        # placeholder is 0xFFFFFFFF, in order: original_size, compressed_size,
        # local_hdr_offset, disk_start.  Do NOT assume a fixed-length array.
        if compressed_size == 0xFFFFFFFF or local_hdr_offset == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF:
            extra = data[idx + 46 + fname_len : idx + 46 + fname_len + extra_len]
            ep = 0
            while ep + 4 <= len(extra):
                tag  = struct.unpack_from("<H", extra, ep)[0]
                size = struct.unpack_from("<H", extra, ep + 2)[0]
                if tag == 0x0001:  # zip64 extended information
                    foff = ep + 4   # current read position within the extra field
                    if uncompressed_size == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        uncompressed_size = struct.unpack_from("<Q", extra, foff)[0]
                        foff += 8
                    if compressed_size == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        compressed_size = struct.unpack_from("<Q", extra, foff)[0]
                        foff += 8
                    if local_hdr_offset == 0xFFFFFFFF and foff + 8 <= ep + 4 + size:
                        local_hdr_offset = struct.unpack_from("<Q", extra, foff)[0]
                    break
                ep += 4 + size

        if filter_str in fname and fname.endswith(".tif"):
            entries.append({
                "filename":          fname,
                "comp_method":       comp_method,
                "compressed_size":   int(compressed_size),
                "local_hdr_offset":  int(local_hdr_offset),
            })

        pos = idx + 46 + fname_len + extra_len + comment_len

    return entries


def _extract_entry(cdn_url: str, entry: dict, dest_dir: Path) -> Path:
    """
    Range-request the compressed bytes of a single zip entry, decompress, save.
    """
    off = entry["local_hdr_offset"]
    # Read local file header (30 bytes fixed + variable)
    lhdr = _range_get(cdn_url, off, off + 29)
    if lhdr[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header for {entry['filename']}")
    fname_len = struct.unpack_from("<H", lhdr, 26)[0]
    extra_len = struct.unpack_from("<H", lhdr, 28)[0]
    data_start = off + 30 + fname_len + extra_len

    # Download compressed payload
    data_end = data_start + entry["compressed_size"] - 1
    payload = _range_get(cdn_url, data_start, data_end)

    # Decompress
    method = entry["comp_method"]
    if method == 0:        # stored
        content = payload
    elif method == 8:      # deflated
        content = zlib.decompress(payload, wbits=-15)
    else:
        raise ValueError(f"Unsupported compression method {method} in {entry['filename']}")

    dest = dest_dir / Path(entry["filename"]).name
    dest.write_bytes(content)
    return dest


# ── Main ─────────────────────────────────────────────────────────────────────

def process_zip(zip_name: str, subfolder: str) -> int:
    dest_dir = OUT_DIR / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    hf_url  = hf_hub_url(REPO_ID, zip_name, repo_type="dataset")
    print(f"\n{'='*60}")
    print(f"  {zip_name}")
    print(f"  Resolving CDN URL ...", end=" ", flush=True)
    cdn_url, file_size = _resolve_cdn_url(hf_url)
    print(f"OK  ({file_size/1e9:.2f} GB)")

    print(f"  Reading central directory ...", end=" ", flush=True)
    cd_offset, cd_size = _find_eocd(cdn_url, file_size)
    cd_data = _range_get(cdn_url, cd_offset, cd_offset + cd_size - 1)
    print(f"OK  ({cd_size/1e6:.1f} MB)")

    entries = _parse_central_directory(cd_data, EVENT)
    print(f"  Found {len(entries)} morocco-earthquake tiles")

    downloaded = skipped_errors = 0
    for i, entry in enumerate(entries, 1):
        fname = Path(entry["filename"]).name
        dest  = dest_dir / fname
        if dest.exists():
            print(f"  [{i:>3}/{len(entries)}] skip  {fname}")
            continue
        size_kb = entry["compressed_size"] / 1024
        print(f"  [{i:>3}/{len(entries)}] {fname}  ({size_kb:.0f} KB) ...", end=" ", flush=True)
        try:
            _extract_entry(cdn_url, entry, dest_dir)
            print("done")
            downloaded += 1
        except Exception as e:
            print(f"SKIP ({e})")
            skipped_errors += 1

    if skipped_errors:
        print(f"  Skipped {skipped_errors} entries due to errors")
    return downloaded


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    print(f"Event: {EVENT}")
    print(f"\nStrategy: HTTP range requests — only Morocco bytes downloaded.")
    print(f"Estimated download: ~200 MB total (vs 13+ GB for full dataset)\n")

    # Map zip name → subfolder name under morocco-earthquake/
    zip_map = {
        "target.zip":     "target",
        "post-event.zip": "post-event",
        "pre-event.zip":  "pre-event",
    }
    total = 0
    for zip_name, subfolder in zip_map.items():
        try:
            total += process_zip(zip_name, subfolder)
        except Exception as e:
            print(f"\n  ERROR processing {zip_name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done! Downloaded {total} new files.")
    for sub in ["pre-event", "post-event", "target"]:
        n = len(list((OUT_DIR / sub).glob("*.tif")))
        print(f"  {sub}/  → {n} tiles")


if __name__ == "__main__":
    main()
