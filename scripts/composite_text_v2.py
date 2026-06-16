"""
composite_text_v2.py
====================
Polished version of composite_text.py with five visual improvements over the
v1 sticker-bar approach:

1. Reduced band opacity (~0.75) so underlying foil texture shows through.
2. Inner-edge gradient bleed (band fades to transparent at the inner edge
   over ~12% of band height) so there is no hard sticker-edge line.
3. Text colour sampled from a contrasting region of the image rather than
   pure black/white derived from band luminance.
4. Tilted text following the dominant edge angle of the pack's top/bottom
   contour (detected via Hough on a Canny edge map; falls back to horizontal).
5. Soft drop shadow with slight blur so text sits on a surface rather than
   floats above it.

All deterministic for a given (tradition, seed). Same vocab as v1 so any
existing rubric scores transfer directly when the same (tradition, seed)
pair is composited.

USAGE
-----
    # Single
    python scripts/composite_text_v2.py \\
        --input outputs/spike/v3_lora_plus/lora_plus_madhubani_s42.png \\
        --tradition madhubani --seed 42

    # Batch
    python scripts/composite_text_v2.py \\
        --input-dir outputs/spike/v3_lora_plus/ \\
        --output-dir outputs/spike/v3_composited_v2/

DEPENDENCIES
------------
    pip install pillow numpy opencv-python requests
"""

from __future__ import annotations
import argparse
import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ----------------------------------------------------------------------
# Fonts (unchanged from v1 — Google Noto Sans, SIL OFL 1.1)
# ----------------------------------------------------------------------

FONT_DIR = Path("fonts")
FONT_URLS = {
    "Latin":      "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    "Tamil":      "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Bold.ttf",
    "Bengali":    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Bold.ttf",
    "Devanagari": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
}


def ensure_fonts() -> dict[str, Path]:
    FONT_DIR.mkdir(exist_ok=True)
    paths = {}
    for name, url in FONT_URLS.items():
        local = FONT_DIR / f"NotoSans-{name}-Bold.ttf"
        if not local.exists():
            logging.info(f"Downloading {name} font...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            local.write_bytes(r.content)
        paths[name] = local
    return paths


# ----------------------------------------------------------------------
# Vocab (unchanged from v1 — review with native speaker before dissertation)
# ----------------------------------------------------------------------

BRAND_WORDS = [
    "Crunchy", "Tasty", "Spicy", "Classic", "Royal",
    "Premium", "Snack", "Crispy", "Zesty", "Indian",
]

TRADITION_VOCAB = {
    "bihar": {
        "script": "Devanagari",
        "flavours": [
            ("मसाला", "masala"), ("नमकीन", "namkeen"),
            ("चटपटा", "chatpata"), ("तीखा", "teekha"),
            ("स्वादिष्ट", "swadisht"),
        ],
    },
    "tamil_nadu": {
        "script": "Tamil",
        "flavours": [
            ("மசாலா", "masala"), ("காரம்", "karam"),
            ("சுவை", "suvai"), ("தக்காளி", "thakkali"),
            ("மிளகு", "milagu"),
        ],
    },
    "west_bengal": {
        "script": "Bengali",
        "flavours": [
            ("মশলা", "moshla"), ("ঝাল", "jhaal"),
            ("নোনতা", "nonta"), ("টক", "tok"),
            ("স্বাদ", "swad"),
        ],
    },
}

TRADITION_TO_REGION = {
    "madhubani": "bihar", "tanjore": "tamil_nadu", "kalighat": "west_bengal",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def luminance(rgb) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def shift_brightness(rgb, delta: int):
    return tuple(max(0, min(255, c + delta)) for c in rgb[:3])


def sample_edge_colour(img_rgb: np.ndarray, edge: str, band_h: int):
    """Median colour of a strip along the named edge of the image."""
    h, w, _ = img_rgb.shape
    if edge == "top":
        strip = img_rgb[:band_h, :, :]
    else:
        strip = img_rgb[h - band_h:, :, :]
    median = np.median(strip.reshape(-1, 3), axis=0)
    return tuple(int(c) for c in median)


def pick_contrasting_text_colour(img_rgb: np.ndarray, band_colour, edge: str, band_h: int):
    """
    Pick a text colour by finding the most distinctive colour in the band region
    that *also* contrasts well with the band colour. This avoids the
    "algorithmically chosen white/black" look.
    """
    h, w, _ = img_rgb.shape
    if edge == "top":
        strip = img_rgb[:band_h, :, :]
    else:
        strip = img_rgb[h - band_h:, :, :]
    # Cluster colours and pick one with high contrast to band_colour
    pixels = strip.reshape(-1, 3).astype(np.float32)
    n = pixels.shape[0]
    if n > 5000:
        idx = np.random.default_rng(0).choice(n, 5000, replace=False)
        pixels = pixels[idx]
    n_clusters = min(5, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centres = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    band_lum = luminance(band_colour)
    best, best_contrast = None, -1
    for c in centres:
        cl = luminance(c)
        contrast = abs(cl - band_lum)
        if contrast > best_contrast:
            best_contrast = contrast
            best = tuple(int(v) for v in c)

    # If contrast is poor (band and image share palette), fall back to luminance-based
    if best_contrast < 80:
        return (245, 245, 245) if band_lum < 128 else (20, 20, 20)
    # Boost saturation/contrast of the chosen colour a touch so text reads cleanly
    if luminance(best) < band_lum:
        best = shift_brightness(best, -25)
    else:
        best = shift_brightness(best, +25)
    return best


def detect_edge_angle(img_rgb: np.ndarray, edge: str, band_h: int) -> float:
    """
    Detect the dominant angle (in degrees) of the pack's top or bottom edge
    via Hough transform on a Canny edge map. Returns 0.0 (horizontal) on
    failure or if the detected angle is implausible (> 8 degrees absolute).
    """
    h, w, _ = img_rgb.shape
    if edge == "top":
        strip = img_rgb[:band_h + 30, :, :]   # include slightly below the band for the edge to be inside
    else:
        strip = img_rgb[h - band_h - 30:, :, :]

    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=int(w * 0.3))
    if lines is None:
        return 0.0

    # Find the longest near-horizontal line
    best_angle = 0.0
    best_score = 0
    for line in lines[:20]:
        rho, theta = line[0]
        deg = (theta * 180 / np.pi) - 90.0   # 0 = horizontal
        if abs(deg) > 10:
            continue
        # crude scoring: lines closer to centre-y of strip count more
        score = 1.0 / (1.0 + abs(rho - strip.shape[0] / 2))
        if score > best_score:
            best_score = score
            best_angle = deg

    # Don't apply implausible angles
    return best_angle if abs(best_angle) <= 8 else 0.0


def render_text_with_shadow(text: str, font: ImageFont.FreeTypeFont,
                            colour: tuple, shadow_colour: tuple,
                            shadow_offset: int, blur_radius: int) -> Image.Image:
    """Render text + soft shadow on a transparent RGBA canvas, tightly cropped."""
    # Probe size first to know canvas dims
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = shadow_offset + blur_radius + 8

    canvas = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))

    # Shadow layer
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((pad + shadow_offset - bbox[0], pad + shadow_offset - bbox[1]),
                                text, font=font, fill=shadow_colour)
    if blur_radius > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    canvas = Image.alpha_composite(canvas, shadow)

    # Text layer
    text_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(text_layer).text((pad - bbox[0], pad - bbox[1]),
                                    text, font=font, fill=colour + (255,))
    canvas = Image.alpha_composite(canvas, text_layer)
    return canvas


# ----------------------------------------------------------------------
# Band drawing — the heart of the v2 upgrade
# ----------------------------------------------------------------------

def draw_band_with_gradient(
    base: Image.Image,
    edge: str,            # "top" or "bottom"
    band_h: int,
    band_colour: tuple,
    band_opacity: int,    # 0–255
    gradient_h: int,      # height of inner-edge fade
) -> Image.Image:
    """
    Draw a coloured band at the specified edge with an inner-edge gradient
    so the band fades smoothly into the image rather than ending in a hard line.
    Returns the base image with the band composited (RGBA-aware).
    """
    w, h = base.size
    band_layer = Image.new("RGBA", (w, band_h), band_colour + (band_opacity,))

    # Build vertical alpha gradient on the inner edge
    alpha = np.full((band_h, w), band_opacity, dtype=np.uint8)
    if edge == "top":
        for i in range(gradient_h):
            row = band_h - gradient_h + i
            if 0 <= row < band_h:
                alpha[row, :] = int(band_opacity * (1.0 - i / gradient_h))
    else:
        for i in range(gradient_h):
            row = i
            alpha[row, :] = int(band_opacity * (i / gradient_h))

    band_arr = np.array(band_layer)
    band_arr[:, :, 3] = alpha
    band_layer = Image.fromarray(band_arr, "RGBA")

    if base.mode != "RGBA":
        base = base.convert("RGBA")
    if edge == "top":
        base.alpha_composite(band_layer, (0, 0))
    else:
        base.alpha_composite(band_layer, (0, h - band_h))
    return base


def composite_one_band(
    base: Image.Image,
    edge: str,
    text: str,
    font_path: Path,
    img_rgb_arr: np.ndarray,
    *,
    band_frac: float = 0.11,
    band_opacity: int = 192,        # ~0.75 alpha
    gradient_frac: float = 0.30,    # gradient covers 30% of band height
    tilt_text: bool = True,
):
    """Composite one band + text at the specified edge. Returns updated base."""
    w, h = base.size
    band_h = int(h * band_frac)

    # Sample band colour from the actual image edge
    band_colour = sample_edge_colour(img_rgb_arr, edge, band_h)
    # Shift slightly so the band doesn't perfectly match the image
    band_lum = luminance(band_colour)
    if band_lum > 128:
        band_colour = shift_brightness(band_colour, -20)
    else:
        band_colour = shift_brightness(band_colour, +20)

    # Pick text colour by contrasting with band against image palette
    text_colour = pick_contrasting_text_colour(img_rgb_arr, band_colour, edge, band_h)

    # Detect edge tilt for rotated text
    angle = detect_edge_angle(img_rgb_arr, edge, band_h) if tilt_text else 0.0

    # Draw the band with gradient
    base = draw_band_with_gradient(
        base, edge, band_h,
        band_colour=band_colour,
        band_opacity=band_opacity,
        gradient_h=int(band_h * gradient_frac),
    )

    # Render the text with shadow on transparent canvas
    font_size = int(band_h * 0.55)
    font = ImageFont.truetype(str(font_path), font_size)
    shadow_colour = (0, 0, 0, 170) if luminance(text_colour) > 128 else (255, 255, 255, 170)
    text_img = render_text_with_shadow(
        text, font, text_colour, shadow_colour,
        shadow_offset=max(2, band_h // 40),
        blur_radius=max(1, band_h // 60),
    )

    # Rotate if we have a non-zero detected angle
    if abs(angle) > 0.3:
        text_img = text_img.rotate(-angle, resample=Image.BICUBIC, expand=True)

    # Paste centred horizontally within the band
    tx = (w - text_img.size[0]) // 2
    if edge == "top":
        ty = (band_h - text_img.size[1]) // 2
    else:
        ty = h - band_h + (band_h - text_img.size[1]) // 2

    base.alpha_composite(text_img, (tx, ty))
    return base


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------

def composite_text_v2(img: Image.Image, tradition: str, seed: int,
                      fonts: dict[str, Path]) -> Image.Image:
    region = TRADITION_TO_REGION.get(tradition, tradition)
    if region not in TRADITION_VOCAB:
        raise ValueError(f"Unknown tradition: {tradition}")
    vocab = TRADITION_VOCAB[region]

    rng = random.Random(seed)
    brand_word = rng.choice(BRAND_WORDS)
    flavour_word, _ = rng.choice(vocab["flavours"])

    img = img.convert("RGB").copy()
    img_arr = np.array(img)

    base = img.convert("RGBA")
    base = composite_one_band(base, "top",    brand_word,    fonts["Latin"],         img_arr)
    base = composite_one_band(base, "bottom", flavour_word,  fonts[vocab["script"]], img_arr)
    return base.convert("RGB")


def infer_tradition_from_name(name: str) -> Optional[str]:
    name = name.lower()
    for trad in ("madhubani", "tanjore", "kalighat"):
        if trad in name:
            return trad
    for region in ("bihar", "tamil_nadu", "west_bengal"):
        if region in name:
            for trad, r in TRADITION_TO_REGION.items():
                if r == region:
                    return trad
    return None


def infer_seed_from_name(name: str) -> int:
    m = re.search(r"_s(\d+)", name)
    if m:
        return int(m.group(1))
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/composited_v2"))
    parser.add_argument("--tradition", type=str, choices=list(TRADITION_TO_REGION.keys()))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    fonts = ensure_fonts()
    logging.info("Fonts ready.")

    if args.input:
        tradition = args.tradition or infer_tradition_from_name(args.input.name)
        if not tradition:
            raise ValueError("Pass --tradition or use a filename with the tradition embedded.")
        seed = args.seed if args.seed is not None else infer_seed_from_name(args.input.name)
        img = Image.open(args.input)
        out = composite_text_v2(img, tradition, seed, fonts)
        if args.output:
            out_path = args.output
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / (args.input.stem + "_v2.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path, "PNG")
        logging.info(f"Saved: {out_path}  (tradition={tradition}, seed={seed})")
    elif args.input_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() == ".png"])
        sources = [p for p in sources if "grid" not in p.name.lower()]
        n_done, n_skipped = 0, 0
        for src in sources:
            tradition = infer_tradition_from_name(src.name)
            if not tradition:
                n_skipped += 1
                continue
            seed = infer_seed_from_name(src.name)
            try:
                out = composite_text_v2(Image.open(src), tradition, seed, fonts)
                out_path = args.output_dir / (src.stem + "_v2.png")
                out.save(out_path, "PNG")
                n_done += 1
            except Exception as e:
                logging.warning(f"  FAILED {src.name}: {e}")
                n_skipped += 1
        logging.info(f"Done. Composited: {n_done}, skipped: {n_skipped}")
        logging.info(f"Output: {args.output_dir.resolve()}")
    else:
        parser.error("Pass --input or --input-dir")


if __name__ == "__main__":
    main()
