"""
prompt to LLM:
Write a Python preprocessing script (preprocess.py) using PIL and pandas to standardize my raw image dataset into square
 tensors for diffusion model training.
Ensure all images are strictly sanitized to 3-channel RGB (composite any alpha masks or indexed palettes over a solid
white background to avoid black-background rendering errors). Resize everything proportionally using the Lanczos filter
to a default --resolution of 1024.
I need you to implement two distinct padding strategies to make the images square:
1.Packaging Images: Pad the resized images onto a pure white square canvas.
2.Style References (The Artifact Fix): Do not use standard edge-extend padding here; it creates directional streak
artifacts that the IP-Adapter misinterprets as cultural "style". Instead, write a function that extracts the outermost
boundary pixels of the image, calculates their median color, and uses that solid color to pad the square canvas.
After processing, deterministically split the packaging metadata into a 90/10 train/val set. Finally, use a left join
to merge my previously scraped Open Food Facts and Wikimedia CSVs into a single metadata.csv. Wrap the entire tool in
an argparse CLI, including a --check flag for dry runs.
"""
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

# Standardized square resolution required by diffusion model backbones
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
    """
    Loads an image from disk and strictly enforces a 3-channel RGB format.
    Handles transparency masks (RGBA, LA) and indexed palettes (P) by compositing
    them onto a solid white background to prevent black-background artefacts
    during downstream tensor conversion.

    Args:
        path (Path): Path to the source image file.

    Returns:
        Image.Image: The sanitized RGB PIL Image.
    """
    img = Image.open(path)

    # Handle explicit alpha channels
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg

    # Handle indexed color palettes (often contains transparency)
    if img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg

    return img.convert("RGB")


def resize_keep_aspect(img: Image.Image, target: int) -> Image.Image:
    """
    Downsamples an image such that its longest spatial dimension matches the target
    resolution, preserving the original aspect ratio.

    Args:
        img (Image.Image): The source RGB image.
        target (int): The target length in pixels for the longest dimension.

    Returns:
        Image.Image: The proportionally resized image utilizing the high-quality
                     LANCZOS anti-aliasing filter.
    """
    w, h = img.size
    if max(w, h) == target:
        return img

    scale = target / float(max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pad_white(img: Image.Image, target: int) -> Image.Image:
    """
    Pastes the proportionally resized packaging image onto the center of a pure
    white square canvas. This ensures uniformity (shape [target, target, 3])
    without structurally distorting the original product aspect ratio.

    Args:
        img (Image.Image): The pre-resized packaging image.
        target (int): The target width and height of the final square canvas.

    Returns:
        Image.Image: The centered, white-padded image.
    """
    w, h = img.size
    if (w, h) == (target, target):
        return img

    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    offset = ((target - w) // 2, (target - h) // 2)
    canvas.paste(img, offset)
    return canvas


def pad_median_border(img: Image.Image, target: int) -> Image.Image:
    """
    Pastes a folk-art reference onto a square canvas filled with the median colour
    of the image's extreme border pixels.

    This replaces standard edge-extend padding to prevent the IP-Adapter from
    interpreting highly directional streak artefacts as legitimate cultural "style".

    Args:
        img (Image.Image): The pre-resized folk-art image.
        target (int): The target width and height of the final square canvas.

    Returns:
        Image.Image: The centered image padded with its median border colour.
    """
    w, h = img.size
    if (w, h) == (target, target):
        return img

    arr = np.array(img)

    # Extract the 1-pixel wide outermost spatial boundaries of the image tensor
    border_pixels = np.concatenate([
        arr[0, :, :].reshape(-1, 3),   # Top edge
        arr[-1, :, :].reshape(-1, 3),  # Bottom edge
        arr[:, 0, :].reshape(-1, 3),   # Left edge
        arr[:, -1, :].reshape(-1, 3),  # Right edge
    ])

    # Calculate the median across the N sampled boundary pixels
    pad_color = tuple(int(c) for c in np.median(border_pixels, axis=0))

    canvas = Image.new("RGB", (target, target), pad_color)
    offset = ((target - w) // 2, (target - h) // 2)
    canvas.paste(img, offset)
    return canvas


# ----------------------------------------------------------------------
# Processing routines
# ----------------------------------------------------------------------

def process_packaging(resolution: int, dry_run: bool) -> list[dict]:
    """
    Executes the spatial preprocessing pipeline for the base packaging images,
    transforming them into standardized square assets suitable for model ingestion.

    Args:
        resolution (int): Target pixel dimension (W and H).
        dry_run (bool): If True, skips disk writes for testing purposes.

    Returns:
        list[dict]: A list of metadata dictionaries mapping the new file paths
                    to their original identifiers.
    """
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
                "image_id": path.stem,
                "filepath": str(out_path),
                "kind": "packaging",
                "tradition": "",
                "original_path": str(path),
            })
        except Exception as e:
            log.warning(f"  failed {path.name}: {e}")
            errors += 1

    log.info(f"Packaging: processed {len(rows)}, errors {errors}")
    return rows


def process_style(resolution: int, dry_run: bool) -> list[dict]:
    """
    Executes the spatial preprocessing pipeline for the IP-Adapter style references,
    utilizing the artifact-mitigating median boundary padding strategy.

    Args:
        resolution (int): Target pixel dimension (W and H).
        dry_run (bool): If True, skips disk writes for testing purposes.

    Returns:
        list[dict]: A list of metadata dictionaries recording the style outputs.
    """
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
                img = pad_median_border(img, resolution)

                out_path = out_dir / (path.stem + ".png")
                if not dry_run:
                    img.save(out_path, "PNG", optimize=True)

                rows.append({
                    "image_id": path.stem,
                    "filepath": str(out_path),
                    "kind": "style_reference",
                    "tradition": tradition,
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
    """
    Deterministically partitions the packaging dataset into distinct training
    and validation subsets based on the configured split fraction.

    Args:
        packaging_rows (list[dict]): Processed metadata mappings.
    """
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
    """
    Executes a left join to merge the newly processed dataset structure with the
    historical provenance and attribution metadata extracted during the download phase.

    Args:
        rows (list[dict]): Combined lists of packaging and style records.
    """
    df = pd.DataFrame(rows)

    # Merge Open Food Facts metadata
    if PACKAGING_META.exists():
        pkg = pd.read_csv(PACKAGING_META, dtype=str).fillna("")
        pkg["image_id"] = pkg["filename"].str.replace(".jpg", "", regex=False)
        keep = [c for c in ["image_id", "product_name", "brands", "image_url", "source", "licence"] if c in pkg.columns]
        df = df.merge(pkg[keep], on="image_id", how="left")

    # Merge Wikimedia Commons attribution data
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
    """Command-line interface entry point."""
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