"""
preprocess.py  (v2 — smart padding)
===================================
Same as v1 but replaces edge-extend padding with median-border-colour
padding to avoid the vertical-streak artefacts produced when source
images have sharp, high-contrast content right against the edge.

WHY THIS CHANGE
---------------
Edge-extend padding replicates border pixels outward. For natural photos
with soft edges, this is invisible. For folk-art photographs with sharp
borders (decorated frames, architectural detail, hard transitions to
white walls), it produces streaky bars that IP-Adapter encodes as
spurious "style". Median-colour padding samples the actual border-pixel
colours and pads with a representative solid colour — visible as a
clean band rather than a streak, and doesn't inject directional artefacts.

Other behaviour identical to v1.

USAGE
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --check
    python scripts/preprocess.py --resolution 768
"""

from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RAW_PACKAGING_DIR = Path("data/raw/packaging")
RAW_STYLE_DIR = Path("data/style_references")
PROCESSED_PACKAGING_DIR = Path("data/processed/packaging")
PROCESSED_STYLE_DIR = Path("data/processed/style_references")
SPLITS_DIR = Path("data/splits")
PACKAGING_META = Path("data/packaging_metadata.csv")
STYLE_META = Path("data/style_references_metadata.csv")
UNIFIED_META = Path("data/metadata.csv")

TRADITIONS = ["tamil_nadu", "west_bengal", "bihar"]
DEFAULT_RESOLUTION = 1024
VAL_FRAC = 0.10
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------

def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def resize_keep_aspect(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    if max(w, h) == target:
        return img
    scale = target / float(max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pad_white(img: Image.Image, target: int) -> Image.Image:
    """Pad to target x target with white. For packaging."""
    w, h = img.size
    if (w, h) == (target, target):
        return img
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    offset = ((target - w) // 2, (target - h) // 2)
    canvas.paste(img, offset)
    return canvas


def pad_median_border(img: Image.Image, target: int) -> Image.Image:
    """
    Pad to target x target with the median colour of the image's
    actual border pixels. Avoids streak artefacts from edge-extend
    while still blending more naturally than white.

    For folk-art references.
    """
    w, h = img.size
    if (w, h) == (target, target):
        return img

    arr = np.array(img)
    # Sample border strips (1-pixel wide is enough; PIL median is robust)
    border_pixels = np.concatenate([
        arr[0, :, :].reshape(-1, 3),
        arr[-1, :, :].reshape(-1, 3),
        arr[:, 0, :].reshape(-1, 3),
        arr[:, -1, :].reshape(-1, 3),
    ])
    pad_color = tuple(int(c) for c in np.median(border_pixels, axis=0))

    canvas = Image.new("RGB", (target, target), pad_color)
    offset = ((target - w) // 2, (target - h) // 2)
    canvas.paste(img, offset)
    return canvas


# ----------------------------------------------------------------------
# Processing routines
# ----------------------------------------------------------------------

def process_packaging(resolution: int, dry_run: bool) -> list[dict]:
    out_dir = PROCESSED_PACKAGING_DIR
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(RAW_PACKAGING_DIR.glob("*"))
    sources = [p for p in sources if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    log.info(f"Packaging: {len(sources)} source images")

    rows, errors = [], 0
    for path in sources:
        try:
            img = load_rgb(path)
            img = resize_keep_aspect(img, resolution)
            img = pad_white(img, resolution)
            out_path = out_dir / (path.stem + ".png")
            if not dry_run:
                img.save(out_path, "PNG", optimize=True)
            rows.append({
                "image_id": path.stem, "filepath": str(out_path),
                "kind": "packaging", "tradition": "",
                "original_path": str(path),
            })
        except Exception as e:
            log.warning(f"  failed {path.name}: {e}")
            errors += 1
    log.info(f"Packaging: processed {len(rows)}, errors {errors}")
    return rows


def process_style(resolution: int, dry_run: bool) -> list[dict]:
    rows = []
    for tradition in TRADITIONS:
        src_dir = RAW_STYLE_DIR / tradition
        if not src_dir.exists():
            log.warning(f"  missing tradition folder: {src_dir}")
            continue
        out_dir = PROCESSED_STYLE_DIR / tradition
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        sources = sorted(src_dir.glob("*"))
        sources = [p for p in sources if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        log.info(f"Style [{tradition}]: {len(sources)} source images")

        errors = 0
        before = len(rows)
        for path in sources:
            try:
                img = load_rgb(path)
                img = resize_keep_aspect(img, resolution)
                img = pad_median_border(img, resolution)   # <-- changed from edge-extend
                out_path = out_dir / (path.stem + ".png")
                if not dry_run:
                    img.save(out_path, "PNG", optimize=True)
                rows.append({
                    "image_id": path.stem, "filepath": str(out_path),
                    "kind": "style_reference", "tradition": tradition,
                    "original_path": str(path),
                })
            except Exception as e:
                log.warning(f"  failed {path.name}: {e}")
                errors += 1
        log.info(f"Style [{tradition}]: processed {len(rows) - before}, errors {errors}")
    return rows


# ----------------------------------------------------------------------
# Splits and metadata merging
# ----------------------------------------------------------------------

def build_splits(packaging_rows: list[dict]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    shuffled = packaging_rows.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRAC)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    pd.DataFrame(train).to_csv(SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame(val).to_csv(SPLITS_DIR / "val.csv", index=False)
    log.info(f"Splits: train={len(train)} val={len(val)} (val_frac={VAL_FRAC}, seed={SEED})")


def merge_metadata(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if PACKAGING_META.exists():
        pkg = pd.read_csv(PACKAGING_META, dtype=str).fillna("")
        pkg["image_id"] = pkg["filename"].str.replace(".jpg", "", regex=False)
        keep = [c for c in ["image_id", "product_name", "brands", "image_url", "source", "licence"] if c in pkg.columns]
        df = df.merge(pkg[keep], on="image_id", how="left")
    if STYLE_META.exists():
        sty = pd.read_csv(STYLE_META, dtype=str).fillna("")
        if "filename" in sty.columns:
            sty["image_id"] = sty["filename"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
            sty_lic = sty.rename(columns={"licence": "style_licence", "source": "style_source"})
            keep = [c for c in ["image_id", "commons_title", "artist", "credit", "descriptionurl"] if c in sty_lic.columns]
            if "style_licence" in sty_lic.columns:
                keep.append("style_licence")
            df = df.merge(sty_lic[keep], on="image_id", how="left")
    df = df.fillna("")
    df.to_csv(UNIFIED_META, index=False)
    log.info(f"Unified metadata: {len(df)} rows -> {UNIFIED_META}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    log.info(f"Resolution: {args.resolution}x{args.resolution}")
    log.info(f"Dry-run: {args.check}")

    packaging_rows = process_packaging(args.resolution, args.check)
    style_rows = process_style(args.resolution, args.check)

    if args.check:
        log.info("DRY-RUN complete. Re-run without --check to write files.")
        return

    build_splits(packaging_rows)
    merge_metadata(packaging_rows + style_rows)

    log.info("=" * 60)
    log.info("DONE.")
    log.info(f"  Packaging: {len(packaging_rows)} -> {PROCESSED_PACKAGING_DIR}")
    log.info(f"  Style:     {len(style_rows)} -> {PROCESSED_STYLE_DIR}")
    log.info(f"  Splits:    {SPLITS_DIR}/train.csv, val.csv")
    log.info(f"  Metadata:  {UNIFIED_META}")


if __name__ == "__main__":
    main()
