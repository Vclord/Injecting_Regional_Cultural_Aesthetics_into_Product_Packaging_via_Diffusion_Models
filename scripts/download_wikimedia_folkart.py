"""
prompt to LLM:
Write a Python script (download_wikimedia_folkart.py) using requests and pandas to collect Indian folk-art images from
Wikimedia Commons. This will act as the style-reference pool for my IP-Adapter, so provenance and image quality are
critical.
Target three specific traditions: Tanjore (Tamil Nadu), Kalighat (West Bengal), and Madhubani (Bihar).
I need the script to:
1.Query the MediaWiki API to pull file titles from their main categories (traverse exactly one level of subcategories
to maximize yield) and use a free-text search fallback.
2.Filter out non-artographic junk. Exclude SVGs, files with dimensions under 400px, and filenames containing keywords
like "icon", "map", "logo", or "flag".
3.Download the images to local folders. To save bandwidth and processing time, request scaled down thumbnails
(max 1600px width) instead of the raw source files.
4.Attribution Ledger: Extract the extmetadata from the API, strip out the messy HTML tags, and save a master CSV
containing the license, artist, credit, and description. I need this to legally satisfy CC-BY-SA requirements.
System Requirements: Enforce a strict 0.4-second time.sleep() delay and apply a custom, descriptive academic User-Agent
to prevent IP bans. Wrap the tool in an argparse CLI with a --max-per-tradition limit.
"""
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

# Wikimedia REQUIRES a descriptive User-Agent with contact info to prevent IP bans.
USER_AGENT = (
    "MScDissertation-FolkArtCollector/1.0 "
    "(University of Stirling; research use; "
    "contact: vivek00089@students.stir.ac.uk)"
)

# Confirmed taxonomy nodes on Wikimedia Commons.
# Subcategories of these nodes are traversed one level deep to maximize yield.
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

# Search-term fallbacks to supplement the formal category traversal.
TRADITION_SEARCH_TERMS = {
    "tamil_nadu": ["Tanjore painting", "Thanjavur painting"],
    "west_bengal": ["Kalighat painting", "Kalighat Pat"],
    "bihar": ["Madhubani painting", "Mithila painting"],
}

# Image quality and API threshold parameters
MIN_DIMENSION = 400          # Reject images too small to provide useful texture to IP-Adapter
MAX_DOWNLOAD_DIM = 1600      # Request scaled thumbnails to save bandwidth and processing time
REQUEST_DELAY_SEC = 0.4      # Throttling to respect Wikimedia API guidelines
TIMEOUT = 30

# Substring heuristics to filter out non-artographic assets common on Wikipedia
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
    """
    Standardised schema for maintaining academic and legal provenance of the
    downloaded dataset. This acts as the CC-BY-SA attribution ledger.
    """
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
    """
    Base wrapper for Wikimedia API GET requests. Automatically appends the JSON format flag.

    Args:
        params (dict): API query parameters.

    Returns:
        dict: The parsed JSON response.
    """
    params = {**params, "format": "json"}
    r = SESSION.get(COMMONS_API, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_files_in_category(category: str, traverse_subcats: bool = True) -> set[str]:
    """
    Retrieves all file titles within a specific Wikimedia category, handling
    pagination tokens automatically. Recursively fetches one layer of subcategories.

    Args:
        category (str): The root category name (without the "Category:" prefix).
        traverse_subcats (bool): Whether to perform depth=1 subcategory traversal.

    Returns:
        set[str]: A unique set of file titles (prefixed with "File:").
    """
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

        # Handle MediaWiki pagination
        if "continue" in data:
            cont = data["continue"]
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break

    # Execute bounded depth=1 recursion for discovered subcategories
    for sub in subcats:
        log.info(f"    + subcategory: {sub}")
        titles |= get_files_in_category(sub, traverse_subcats=False)
        time.sleep(REQUEST_DELAY_SEC)

    return titles


def search_files(term: str, limit: int = 60) -> set[str]:
    """
    Executes a free-text search specifically restricted to the File namespace.

    Args:
        term (str): The search query string.
        limit (int): Maximum number of results to return.

    Returns:
        set[str]: A set of matched file titles.
    """
    titles: set[str] = set()
    data = api_get({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srnamespace": "6",   # MediaWiki namespace ID 6 corresponds to "File"
        "srlimit": str(limit),
    })
    for m in data.get("query", {}).get("search", []):
        titles.add(m["title"])
    return titles


def get_imageinfo(title: str) -> Optional[dict]:
    """
    Fetches the detailed metadata (URL, dimensions, MIME type, licensing) for a single file.
    Requests a scaled down thumbnail URL to conserve local storage and bandwidth.

    Args:
        title (str): The full Wikimedia file title.

    Returns:
        Optional[dict]: The imageinfo dictionary, or None if the request fails.
    """
    data = api_get({
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(MAX_DOWNLOAD_DIM),
    })

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        infos = page.get("imageinfo")
        if infos:
            return infos[0]
    return None


def extract_meta(info: dict) -> dict:
    """
    Parses the messy 'extmetadata' block from the Wikimedia API, applying a
    crude regex sanitisation to strip HTML tags commonly embedded in these fields.

    Args:
        info (dict): The raw imageinfo dictionary.

    Returns:
        dict: A cleaned dictionary containing licence, artist, credit, and description.
    """
    ext = info.get("extmetadata", {})

    def field(name: str) -> str:
        v = ext.get(name, {}).get("value", "")
        import re
        v = re.sub(r"<[^>]+>", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v

    return {
        "licence": field("LicenseShortName") or field("License"),
        "artist": field("Artist"),
        "credit": field("Credit"),
        "description": field("ImageDescription")[:300], # Truncate unusually long descriptions
    }


def is_junk(title: str) -> bool:
    """Evaluates the filename against a heuristic list of non-art patterns."""
    low = title.lower()
    return any(p in low for p in JUNK_PATTERNS)


def download_image(url: str, dest: Path) -> bool:
    """
    Streams the image binary to disk with an 8KB minimum size threshold to
    filter out placeholders or corrupted downloads.

    Args:
        url (str): The direct download URL.
        dest (Path): The local filesystem path.

    Returns:
        bool: True if successfully saved or already exists, False otherwise.
    """
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
    """
    Orchestrates the data collection for a single art tradition. Combines category
    traversal and search queries to build a candidate list, filters for quality,
    and downloads the valid assets.

    Args:
        tradition (str): The key mapping to the target cultural style.
        out_root (Path): The base directory for downloaded assets.
        max_images (int): The halt threshold for this specific tradition.

    Returns:
        list[ArtRecord]: The metadata records for all successfully downloaded images.
    """
    out_dir = out_root / tradition
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {tradition} ===")

    # Phase 1: Gather candidate titles
    titles: set[str] = set()
    for cat in TRADITION_CATEGORIES[tradition]:
        log.info(f"  category: {cat}")
        titles |= get_files_in_category(cat)

    for term in TRADITION_SEARCH_TERMS[tradition]:
        log.info(f"  search: {term}")
        titles |= search_files(term)
        time.sleep(REQUEST_DELAY_SEC)

    log.info(f"  {len(titles)} candidate files found")

    # Phase 2: Filter, Download, and Record Metadata
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

        # Reject vector graphics and non-images
        mime = info.get("mime", "")
        if not mime.startswith("image/") or "svg" in mime:
            continue

        # Reject assets that are structurally too small for model training/conditioning
        w, h = info.get("width", 0), info.get("height", 0)
        if min(w, h) < MIN_DIMENSION:
            continue

        # Prefer the scaled thumbnail URL if present to preserve bandwidth
        dl_url = info.get("thumburl") or info.get("url")
        if not dl_url:
            continue

        # Sanitize the Wikimedia title into a safe local OS filename
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
    """CLI entry point handling parameter parsing and overarching execution logic."""
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

    # Serialize the attribution ledger to disk
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