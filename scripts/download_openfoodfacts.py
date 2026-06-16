"""
download_openfoodfacts.py
=========================
Download Indian snack packaging images from Open Food Facts (OFF) for
MSc dissertation: Reference-Driven Localisation of Indian Snack Packaging.

WHAT THIS DOES
--------------
1. Queries the OFF v2 search API for products that are:
   - sold in India (countries_tags_en=India)
   - in chip/snack categories (configurable)
   - have a front-of-pack image
2. Downloads each product's front image (highest available resolution) to
   data/raw/_unsorted/<product_code>.jpg
3. Saves rich metadata to data/raw/_unsorted/_metadata_raw.csv including
   product name, brand, languages, ingredients-language tags — to help
   you manually sort into states later.
4. Includes a "state hint" column based on a small brand → state mapping
   (you can edit BRAND_STATE_HINTS below).
5. Respects OFF's rate limit (~2 requests/sec); resumes if interrupted.

WHAT THIS DOES *NOT* DO
-----------------------
- It does NOT classify images into states. OFF doesn't have that field.
  After running, you manually sort the _unsorted/ folder into
  tamil_nadu/, west_bengal/, punjab/ using visible scripts + brand hints.

LICENCE
-------
OFF images are CC-BY-SA. Track the source URL (in metadata) and credit
"Open Food Facts contributors" in your dissertation acknowledgements.

USAGE
-----
    cd C:\\Users\\Vivek\\Documents\\dissertation
    sdxl-env\\Scripts\\activate
    python scripts\\download_openfoodfacts.py --max-products 400

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
from tqdm import tqdm


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

OFF_SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
USER_AGENT = (
    "MScDissertation-VivekChandra/1.0 "
    "(Stirling University; research use; "
    "contact: vic00089@students.stir.ac.uk)"
)
# Categories to query. Each gets its own request; results are deduplicated by barcode.
# These are OFF "categories_tags_en" slugs — broad on purpose.
CATEGORIES = [
    "Snacks",
    "Salty-snacks",
    "Chips-and-fries",
    "Crisps",
    "Potato-crisps",
    "Flavoured-potato-crisps",
]
# Hint: brand substrings (lower-case) → likely Indian state.
# Used only to populate a "state_hint" column. NOT used for filtering.
# Add to this freely — it just helps your manual sort.
BRAND_STATE_HINTS = {
    # Gujarat-origin brands
    "balaji":    "gujarat",
    "haldiram":  "gujarat",      # Rajasthan/Delhi origin but pan-Indian; flag for review
    "gopal":     "gujarat",
    # Punjab / North India
    "bikaji":    "rajasthan",    # not in your 3 states but useful hint
    "kurkure":   "pan_india",
    "lays":      "pan_india",
    "bingo":     "pan_india",
    # Tamil Nadu
    "ten ten":   "tamil_nadu",
    "beta":      "tamil_nadu",
    "a1":        "tamil_nadu",
    # West Bengal
    "haldiram":  "pan_india",    # overridden above; just a note
}
# Fields to request from OFF (smaller payload = faster)
OFF_FIELDS = ",".join([
    "code", "product_name", "product_name_en", "brands", "brands_tags",
    "categories_tags", "countries_tags", "languages_tags",
    "ingredients_text", "labels_tags",
    "image_front_url", "image_front_small_url",
    "images",
])

PAGE_SIZE = 100  # OFF max
REQUEST_DELAY_SEC = 0.5  # respect OFF: ~2 req/sec is fine
DOWNLOAD_TIMEOUT_SEC = 15
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
class ProductRecord:
    """One row in the metadata CSV."""
    code: str
    product_name: str
    brands: str
    categories: str
    languages: str
    countries: str
    image_url: str
    image_path: str
    state_hint: str
    source: str = "open_food_facts"
    licence: str = "CC-BY-SA"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def state_hint_from_brand(brands: str) -> str:
    """Best-effort guess at Indian state from brand string."""
    if not brands:
        return ""
    b = brands.lower()
    for key, hint in BRAND_STATE_HINTS.items():
        if key in b:
            return hint
    return ""


def pick_best_image_url(product: dict) -> Optional[str]:
    """
    Choose the highest-resolution front image available.

    OFF gives image_front_url (usually 400px) and an `images` dict with full
    metadata. We try to upgrade to full-resolution if possible.
    """
    front_url = product.get("image_front_url")
    if not front_url:
        return None

    # OFF image URL pattern:
    # https://images.openfoodfacts.org/images/products/.../front_en.RR.400.jpg
    # Where RR is the revision. Replace .400. with .full. for the biggest version.
    if ".400.jpg" in front_url:
        return front_url.replace(".400.jpg", ".full.jpg")
    if ".200.jpg" in front_url:
        return front_url.replace(".200.jpg", ".full.jpg")
    return front_url


def fetch_page(category: str, page: int) -> dict:
    """Fetch one page from OFF search API."""
    params = {
        "countries_tags_en": "India",
        "categories_tags_en": category,
        "fields": OFF_FIELDS,
        "page_size": PAGE_SIZE,
        "page": page,
    }
    r = SESSION.get(OFF_SEARCH_URL, params=params, timeout=DOWNLOAD_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def download_image(url: str, dest: Path) -> bool:
    """Download one image; skip if it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = SESSION.get(url, timeout=DOWNLOAD_TIMEOUT_SEC, stream=True)
        r.raise_for_status()
        # quick size sanity: skip tiny images (<8 KB)
        content = r.content
        if len(content) < 8 * 1024:
            log.debug(f"  Skipping {url}: image too small ({len(content)} bytes)")
            return False
        dest.write_bytes(content)
        return True
    except requests.RequestException as e:
        log.warning(f"  Download failed for {url}: {e}")
        return False


# ----------------------------------------------------------------------
# Main collection loop
# ----------------------------------------------------------------------

def collect(out_dir: Path, max_products: int) -> list[ProductRecord]:
    """Iterate categories and pages until max_products downloaded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_codes: set[str] = set()
    records: list[ProductRecord] = []

    # If a metadata CSV already exists, load it to support resume
    meta_path = out_dir / "_metadata_raw.csv"
    if meta_path.exists():
        existing = pd.read_csv(meta_path, dtype=str).fillna("")
        seen_codes.update(existing["code"].tolist())
        for _, row in existing.iterrows():
            records.append(ProductRecord(**row.to_dict()))
        log.info(f"Resuming: found {len(seen_codes)} existing records.")

    for category in CATEGORIES:
        if len(records) >= max_products:
            break
        log.info(f"Category: {category}")
        page = 1
        while len(records) < max_products:
            try:
                payload = fetch_page(category, page)
            except requests.RequestException as e:
                log.error(f"  Page fetch failed (cat={category}, page={page}): {e}")
                break

            products = payload.get("products", [])
            if not products:
                log.info(f"  No more products in {category} (page {page}).")
                break

            new_this_page = 0
            for product in products:
                code = str(product.get("code", "")).strip()
                if not code or code in seen_codes:
                    continue
                seen_codes.add(code)

                image_url = pick_best_image_url(product)
                if not image_url:
                    continue

                # Download
                image_path = out_dir / f"{code}.jpg"
                if not download_image(image_url, image_path):
                    continue

                brands = product.get("brands", "") or ""
                record = ProductRecord(
                    code=code,
                    product_name=(product.get("product_name") or "").strip(),
                    brands=brands.strip(),
                    categories=";".join(product.get("categories_tags", [])),
                    languages=";".join(product.get("languages_tags", [])),
                    countries=";".join(product.get("countries_tags", [])),
                    image_url=image_url,
                    image_path=str(image_path.relative_to(out_dir.parent.parent)),
                    state_hint=state_hint_from_brand(brands),
                )
                records.append(record)
                new_this_page += 1

                # Periodic save every 20 records
                if len(records) % 20 == 0:
                    save_metadata(records, meta_path)
                    log.info(f"  ... {len(records)} downloaded")

                if len(records) >= max_products:
                    break

                time.sleep(REQUEST_DELAY_SEC)

            log.info(f"  Page {page} done. New on page: {new_this_page}. Total: {len(records)}.")
            page += 1
            if new_this_page == 0:
                break

    save_metadata(records, meta_path)
    return records


def save_metadata(records: list[ProductRecord], path: Path) -> None:
    """Atomic-ish CSV write."""
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/_unsorted"),
        help="Where to drop downloaded images (default: data/raw/_unsorted)",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=400,
        help="Stop after this many successfully-downloaded products (default: 400)",
    )
    args = parser.parse_args()

    log.info(f"Output directory: {args.out_dir.resolve()}")
    log.info(f"Max products: {args.max_products}")
    log.info(f"User-Agent: {USER_AGENT}")
    log.info("Remember to set your real email in USER_AGENT above.")

    records = collect(args.out_dir, args.max_products)

    log.info("=" * 60)
    log.info(f"DONE. {len(records)} products downloaded.")
    log.info(f"Images:   {args.out_dir.resolve()}")
    log.info(f"Metadata: {(args.out_dir / '_metadata_raw.csv').resolve()}")
    log.info("Next step: manually sort images into data/raw/<state>/")
    log.info("Use the 'state_hint' column in the CSV as a starting point.")


if __name__ == "__main__":
    main()