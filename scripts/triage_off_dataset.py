"""
triage_off_dataset.py
=====================
Triage the Open Food Facts (OFF) packaging images for the packaging-domain LoRA.

After the pivot to cultural-injection framing, OFF images are a single
"generic Indian snack packaging" training set. No state sorting needed.
This script handles QUALITY filtering only.

WHAT THIS DOES
--------------
1. Scans data/raw/_unsorted/ for downloaded images.
2. Flags and quarantines images that are:
   - corrupt / unreadable
   - too small (below a min dimension)
   - near-duplicates of another image (perceptual hash)
3. Merges in the OFF metadata (_metadata_raw.csv) so each surviving image
   has provenance (source URL, brand, licence).
4. Writes data/packaging_metadata.csv listing every KEPT image with a
   'manual_review' column for you to mark final keep/reject by eye.
5. Moves rejected images to data/raw/_rejected_auto/ (NOT deleted, so you
   can recover any false positives).

WHAT YOU DO AFTER
-----------------
Open data/packaging_metadata.csv, look at each image, and set the
'manual_review' column to 'keep' or 'reject'. Then run this script again
with --finalise to move your manual rejects out and produce the final list.

USAGE
-----
    # First pass: auto-triage + build review CSV
    python scripts/triage_off_dataset.py

    # After you've filled in 'manual_review' in the CSV:
    python scripts/triage_off_dataset.py --finalise

DEPENDENCIES
------------
    pip install pillow imagehash pandas
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

UNSORTED_DIR = Path("data/raw/_unsorted")
REJECTED_AUTO_DIR = Path("data/raw/_rejected_auto")
REJECTED_MANUAL_DIR = Path("data/raw/_rejected_manual")
KEPT_DIR = Path("data/raw/packaging")          # final clean set lands here on --finalise
RAW_META = UNSORTED_DIR / "_metadata_raw.csv"  # produced by the OFF downloader
REVIEW_CSV = Path("data/packaging_metadata.csv")

MIN_DIMENSION = 256       # reject images whose shorter side is below this (px)
PHASH_THRESHOLD = 6       # lower = stricter duplicate detection (Hamming distance)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_raw_metadata() -> pd.DataFrame:
    """Load OFF metadata if present; otherwise return empty frame."""
    if RAW_META.exists():
        df = pd.read_csv(RAW_META, dtype=str).fillna("")
        # 'code' is the barcode; image files are <code>.jpg
        df["filename"] = df["code"].astype(str) + ".jpg"
        return df
    print(f"  WARNING: {RAW_META} not found. Provenance columns will be blank.")
    return pd.DataFrame()


def image_ok(path: Path) -> tuple[bool, str, str]:
    """
    Validate an image.
    Returns (is_ok, reason_if_not, resolution_string).
    """
    try:
        with Image.open(path) as img:
            img.verify()  # checks for corruption
        # reopen (verify() leaves the file unusable)
        with Image.open(path) as img:
            w, h = img.size
    except Exception as e:
        return False, f"corrupt ({type(e).__name__})", ""

    res = f"{w}x{h}"
    if min(w, h) < MIN_DIMENSION:
        return False, f"too_small ({res})", res
    return True, "", res


def compute_phash(path: Path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


# ----------------------------------------------------------------------
# Pass 1 — auto triage
# ----------------------------------------------------------------------

def auto_triage() -> None:
    REJECTED_AUTO_DIR.mkdir(parents=True, exist_ok=True)
    raw_meta = load_raw_metadata()

    images = sorted(
        [p for p in UNSORTED_DIR.glob("*.jpg")] +
        [p for p in UNSORTED_DIR.glob("*.png")] +
        [p for p in UNSORTED_DIR.glob("*.jpeg")]
    )
    print(f"Found {len(images)} images in {UNSORTED_DIR}")

    kept_rows = []
    hashes: dict = {}           # phash -> first filename seen
    n_corrupt = n_small = n_dup = 0

    for path in images:
        ok, reason, res = image_ok(path)
        if not ok:
            if "corrupt" in reason:
                n_corrupt += 1
            else:
                n_small += 1
            shutil.move(str(path), str(REJECTED_AUTO_DIR / path.name))
            continue

        # duplicate check
        h = compute_phash(path)
        is_dup = False
        if h is not None:
            for existing_h, existing_name in hashes.items():
                if (h - existing_h) <= PHASH_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                n_dup += 1
                shutil.move(str(path), str(REJECTED_AUTO_DIR / path.name))
                continue
            hashes[h] = path.name

        # surviving image -> add to review CSV
        row = {
            "filename": path.name,
            "filepath": str(path),
            "resolution": res,
            "phash": str(h) if h else "",
            "manual_review": "",     # YOU fill this: keep / reject
        }
        # merge provenance from OFF metadata if available
        if not raw_meta.empty:
            match = raw_meta[raw_meta["filename"] == path.name]
            if len(match):
                m = match.iloc[0]
                row.update({
                    "product_name": m.get("product_name", ""),
                    "brands": m.get("brands", ""),
                    "categories": m.get("categories", ""),
                    "image_url": m.get("image_url", ""),
                    "source": m.get("source", "open_food_facts"),
                    "licence": m.get("licence", "CC-BY-SA"),
                })
        kept_rows.append(row)

    df = pd.DataFrame(kept_rows)
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REVIEW_CSV, index=False)

    print("-" * 60)
    print(f"Auto-rejected: {n_corrupt} corrupt, {n_small} too-small, {n_dup} duplicates")
    print(f"Survivors for manual review: {len(kept_rows)}")
    print(f"Review CSV written to: {REVIEW_CSV.resolve()}")
    print(f"Auto-rejects quarantined in: {REJECTED_AUTO_DIR.resolve()}")
    print()
    print("NEXT: open the CSV, view each image, set 'manual_review' to")
    print("      'keep' or 'reject'. Blank rows are treated as 'keep' on")
    print("      --finalise, so you only need to mark the rejects.")


# ----------------------------------------------------------------------
# Pass 2 — finalise after manual review
# ----------------------------------------------------------------------

def finalise() -> None:
    if not REVIEW_CSV.exists():
        print(f"ERROR: {REVIEW_CSV} not found. Run without --finalise first.")
        return

    KEPT_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REVIEW_CSV, dtype=str).fillna("")
    kept, rejected = [], []

    for _, row in df.iterrows():
        verdict = row["manual_review"].strip().lower()
        src = Path(row["filepath"])
        if not src.exists():
            # may already have been moved on a previous run; skip quietly
            continue
        if verdict == "reject":
            shutil.move(str(src), str(REJECTED_MANUAL_DIR / src.name))
            rejected.append(row["filename"])
        else:  # 'keep' or blank
            dst = KEPT_DIR / src.name
            shutil.move(str(src), str(dst))
            row["filepath"] = str(dst)
            kept.append(row)

    final_df = pd.DataFrame(kept)
    final_df.to_csv(REVIEW_CSV, index=False)  # rewrite with updated paths

    print("-" * 60)
    print(f"Final kept:     {len(kept)}  -> {KEPT_DIR.resolve()}")
    print(f"Manual rejects: {len(rejected)} -> {REJECTED_MANUAL_DIR.resolve()}")
    print(f"Updated CSV:    {REVIEW_CSV.resolve()}")
    print()
    print("Your clean packaging training set is now in data/raw/packaging/")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Triage OFF packaging images.")
    parser.add_argument(
        "--finalise",
        action="store_true",
        help="Apply manual_review verdicts from the CSV and move files.",
    )
    args = parser.parse_args()

    if args.finalise:
        finalise()
    else:
        auto_triage()


if __name__ == "__main__":
    main()
