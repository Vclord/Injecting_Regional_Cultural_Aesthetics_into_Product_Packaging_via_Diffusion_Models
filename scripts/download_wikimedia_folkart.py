"""
download_wikimedia_folkart.py
=============================
Download Indian folk-art style references from Wikimedia Commons for the
three target traditions, with full licence + attribution metadata.

After the cultural-injection pivot, these images are the STYLE-REFERENCE pool
used for IP-Adapter conditioning. They carry the regional identity of the
project, so quality and provenance matter.

TRADITIONS
----------
  tamil_nadu  -> Tanjore painting    (gold leaf, ornate deities)
  west_bengal -> Kalighat painting   (bold brushstrokes, flat colour)
  bihar       -> Madhubani painting  (dense linework, natural dyes, motifs)

WHAT THIS DOES
--------------
1. For each tradition, queries the confirmed Commons categories (and one level
   of subcategories) for image files.
2. Fetches imageinfo for each file: download URL, dimensions, MIME type, and
   extmetadata (licence, artist, credit, description).
3. Filters out: non-images, SVGs, tiny images, and known junk patterns.
4. Downloads a reasonably-sized version (max 1600px) to
   data/style_references/<tradition>/.
5. Writes data/style_references_metadata.csv with full provenance + licence
   for every downloaded image. THIS IS YOUR ATTRIBUTION RECORD (CC-BY-SA needs it).

LICENCE NOTE
------------
Commons content is openly licensed (CC-BY / CC-BY-SA / public domain). The
'licence' and 'artist' columns in the CSV are your attribution evidence.
Cite "Wikimedia Commons contributors" + per-image artist in your dissertation.

USAGE
-----
    python scripts/download_wikimedia_folkart.py --max-per-tradition 80

DEPENDENCIES
------------
    pip install requests pandas tqdm
"""

from __future__ import annotations
import argparse
import csv
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia REQUIRES a descriptive User-Agent with contact info, or they block you.
USER_AGENT = (
    "MScDissertation-FolkArtCollector/1.0 "
    "(University of Stirling; research use; "
    "contact: vivek00089@students.stir.ac.uk)"
)

# Confirmed category names on Commons (verified to exist).
# Each tradition lists categories to harvest. Subcategories are traversed one level deep.
TRADITION_CATEGORIES = {
    "tamil_nadu": [
        "Tanjore paintings",
    ],
    "west_bengal": [
        "Kalighat painting",
    ],
    "bihar": [
        "Madhubani painting",
    ],
}

# Search-term fallbacks (used in addition to categories to top up yield).
TRADITION_SEARCH_TERMS = {
    "tamil_nadu": ["Tanjore painting", "Thanjavur painting"],
    "west_bengal": ["Kalighat painting", "Kalighat Pat"],
    "bihar": ["Madhubani painting", "Mithila painting"],
}

MIN_DIMENSION = 400          # reject images whose shorter side is below this
MAX_DOWNLOAD_DIM = 1600      # request a thumbnail no larger than this (px)
REQUEST_DELAY_SEC = 0.4      # be polite to the API
TIMEOUT = 30

# Junk filename patterns to skip (logos, icons, maps, diagrams)
JUNK_PATTERNS = ["icon", "logo", "map", "diagram", "locator", "flag", "coat_of_arms"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class ArtRecord:
    tradition: str
    filename: str
    filepath: str
    commons_title: str
    width: int
    height: int
    licence: str
    artist: str
    credit: str
    description: str
    descriptionurl: str
    source: str = "wikimedia_commons"


# ----------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------

def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    r = SESSION.get(COMMONS_API, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_files_in_category(category: str, traverse_subcats: bool = True) -> set[str]:
    """Return set of File: titles in a category (and one level of subcats)."""
    titles: set[str] = set()
    subcats: list[str] = []

    cont = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "500",
            "cmtype": "file|subcat",
            **cont,
        }
        data = api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            title = m["title"]
            if title.startswith("File:"):
                titles.add(title)
            elif title.startswith("Category:") and traverse_subcats:
                subcats.append(title.replace("Category:", ""))
        if "continue" in data:
            cont = data["continue"]
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break

    # one level of subcategory traversal
    for sub in subcats:
        log.info(f"    + subcategory: {sub}")
        titles |= get_files_in_category(sub, traverse_subcats=False)
        time.sleep(REQUEST_DELAY_SEC)

    return titles


def search_files(term: str, limit: int = 60) -> set[str]:
    """Search the File namespace for a term."""
    titles: set[str] = set()
    data = api_get({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srnamespace": "6",   # File namespace
        "srlimit": str(limit),
    })
    for m in data.get("query", {}).get("search", []):
        titles.add(m["title"])
    return titles


def get_imageinfo(title: str) -> Optional[dict]:
    """Fetch imageinfo (url, size, mime, licence metadata) for one File title."""
    data = api_get({
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(MAX_DOWNLOAD_DIM),  # request a scaled thumbnail URL
    })
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        infos = page.get("imageinfo")
        if infos:
            return infos[0]
    return None


def extract_meta(info: dict) -> dict:
    """Pull licence/artist/credit/description from extmetadata, stripping HTML."""
    ext = info.get("extmetadata", {})

    def field(name: str) -> str:
        v = ext.get(name, {}).get("value", "")
        # crude HTML strip
        import re
        v = re.sub(r"<[^>]+>", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v

    return {
        "licence": field("LicenseShortName") or field("License"),
        "artist": field("Artist"),
        "credit": field("Credit"),
        "description": field("ImageDescription")[:300],
    }


def is_junk(title: str) -> bool:
    low = title.lower()
    return any(p in low for p in JUNK_PATTERNS)


def download_image(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = SESSION.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) < 8 * 1024:
            return False
        dest.write_bytes(content)
        return True
    except requests.RequestException as e:
        log.warning(f"    download failed: {e}")
        return False


# ----------------------------------------------------------------------
# Main collection
# ----------------------------------------------------------------------

def collect_tradition(tradition: str, out_root: Path, max_images: int) -> list[ArtRecord]:
    out_dir = out_root / tradition
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {tradition} ===")
    # gather candidate titles from categories
    titles: set[str] = set()
    for cat in TRADITION_CATEGORIES[tradition]:
        log.info(f"  category: {cat}")
        titles |= get_files_in_category(cat)
    # top up with search
    for term in TRADITION_SEARCH_TERMS[tradition]:
        log.info(f"  search: {term}")
        titles |= search_files(term)
        time.sleep(REQUEST_DELAY_SEC)

    log.info(f"  {len(titles)} candidate files found")

    records: list[ArtRecord] = []
    for title in sorted(titles):
        if len(records) >= max_images:
            break
        if is_junk(title):
            continue

        info = get_imageinfo(title)
        time.sleep(REQUEST_DELAY_SEC)
        if not info:
            continue

        mime = info.get("mime", "")
        if not mime.startswith("image/") or "svg" in mime:
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        if min(w, h) < MIN_DIMENSION:
            continue

        # prefer the scaled thumbnail URL if present, else full URL
        dl_url = info.get("thumburl") or info.get("url")
        if not dl_url:
            continue

        # build a safe local filename
        safe = title.replace("File:", "").replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "._-")[:120]
        dest = out_dir / safe
        if not download_image(dl_url, dest):
            continue

        meta = extract_meta(info)
        records.append(ArtRecord(
            tradition=tradition,
            filename=safe,
            filepath=str(dest),
            commons_title=title,
            width=w, height=h,
            descriptionurl=info.get("descriptionurl", ""),
            **meta,
        ))
        if len(records) % 10 == 0:
            log.info(f"    ... {len(records)} downloaded")

    log.info(f"  {tradition}: {len(records)} images saved")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Download folk-art references from Commons.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/style_references"))
    parser.add_argument("--max-per-tradition", type=int, default=80)
    parser.add_argument(
        "--tradition",
        choices=list(TRADITION_CATEGORIES.keys()),
        help="Only download one tradition (default: all three).",
    )
    args = parser.parse_args()

    log.info(f"Output: {args.out_dir.resolve()}")
    log.info(f"Max per tradition: {args.max_per_tradition}")
    log.info("Remember to set your real email in USER_AGENT.")

    traditions = [args.tradition] if args.tradition else list(TRADITION_CATEGORIES.keys())
    all_records: list[ArtRecord] = []
    for t in traditions:
        all_records.extend(collect_tradition(t, args.out_dir, args.max_per_tradition))

    # write metadata CSV
    meta_path = Path("data/style_references_metadata.csv")
    pd.DataFrame([asdict(r) for r in all_records]).to_csv(
        meta_path, index=False, quoting=csv.QUOTE_MINIMAL
    )

    log.info("=" * 60)
    log.info(f"DONE. {len(all_records)} folk-art references downloaded.")
    for t in traditions:
        n = sum(1 for r in all_records if r.tradition == t)
        log.info(f"  {t:14s}: {n}")
    log.info(f"Metadata: {meta_path.resolve()}")
    log.info("Next: eyeball each folder, delete obvious non-painting junk,")
    log.info("      then we preprocess to square crops for IP-Adapter.")


if __name__ == "__main__":
    main()
