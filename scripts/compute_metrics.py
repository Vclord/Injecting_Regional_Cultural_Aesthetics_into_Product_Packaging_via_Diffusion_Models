"""
compute_metrics.py
==================
Quantitative evaluation suite for the dissertation.

Computes four metrics per spike output image:
  1. CLIP-image similarity to OFF packaging reference pool (mean, std)
     → packaging-domain fidelity / "how packet-like is this output?"
     Higher = closer to real packaging.
  2. CLIP-text similarity to a canonical packaging prompt
     → prompt adherence / "does this look like what we asked for?"
     Higher = better adherence.
  3. DINOv2 similarity to tradition-specific folk-art reference pool (mean, std)
     → regional/cultural style alignment
     Higher = closer to the target tradition.
  4. LPIPS perceptual distance to OFF packaging reference pool (mean, std)
     → perceptual quality relative to real packaging
     LOWER = closer to real packaging (LPIPS is a distance, not similarity)

Outputs a long-form CSV with one row per scored image, plus a printed
summary table aggregated by (spike, configuration).

Designed for local execution on RTX 3060 6 GB. Peak VRAM ~4.5 GB across
the loaded models (CLIP ViT-L/14, DINOv2 ViT-B/14, LPIPS VGG).

USAGE
-----
    # Single spike directory
    python scripts/compute_metrics.py --input-dir outputs/spike/v4_controlnet

    # Multiple directories in one run
    python scripts/compute_metrics.py \
        --input-dir outputs/spike/v2_ip_adapter outputs/spike/v3_lora_plus \
                    outputs/spike/v4_controlnet outputs/spike/v5_lora_ablation \
        --output evaluation/quantitative_metrics.csv

    # FLUX comparison
    python scripts/compute_metrics.py \
        --input-dir outputs/flux_comparison \
        --output evaluation/quantitative_metrics_flux.csv

DEPENDENCIES
------------
    pip install torch transformers lpips pillow numpy pandas

NOTES
-----
- First run downloads ~3 GB of model weights to the HuggingFace cache.
- Reference embeddings are computed once at startup and cached in memory.
- Already-processed images (present in the output CSV) are skipped on rerun.
- Grid images, smoke tests, and files starting with '_' are excluded.
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # assume script at scripts/
PACKAGING_POOL_DIR = PROJECT_ROOT / "data" / "processed" / "packaging"
STYLE_POOL_DIRS = {
    "madhubani":  PROJECT_ROOT / "data" / "processed" / "style_references" / "bihar",
    "tanjore":    PROJECT_ROOT / "data" / "processed" / "style_references" / "tamil_nadu",
    "kalighat":   PROJECT_ROOT / "data" / "processed" / "style_references" / "west_bengal",
}

PACKAGING_POOL_SIZE = 30
POOL_RANDOM_SEED = 42

CLIP_TEXT_PROMPT = (
    "Front-facing product photograph of an Indian snack packet, "
    "professional product photography"
)

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DINOV2_MODEL_ID = "facebook/dinov2-base"

TRADITION_TOKENS = {"madhubani", "tanjore", "kalighat"}

# Skip files that are grid summaries, smoke tests, or other meta-artefacts
SKIP_PATTERNS = ("grid", "comparison", "smoke", "scale_sweep")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------------------------------------------------
# Filename and config parsing
# ----------------------------------------------------------------------

@dataclass
class ImageRecord:
    path: Path
    spike: str            # 'v2','v3','v4','v5','flux','other'
    tradition: Optional[str]
    seed: Optional[int]
    config_str: str       # human-readable summary of the pipeline configuration
    lora_scale: Optional[float] = None
    ip_scale: Optional[float] = None
    cn_scale: Optional[float] = None
    model: str = "sdxl"   # 'sdxl' or 'flux'
    variant: Optional[str] = None  # e.g., 'baseline_no_lora', 'with_lora'

def parse_record(p: Path) -> Optional[ImageRecord]:
    """
    Parse a spike output image filename into structured config.
    Uses sibling .json config file if present (v4, v5, FLUX outputs have these);
    falls back to filename heuristics for older spikes (v2, v3).
    """
    name = p.stem
    if any(pat in name.lower() for pat in SKIP_PATTERNS) or name.startswith("_"):
        return None

    # Determine spike from the ENTIRE parent directory path (handles subfolders safely)
    parent_path = str(p.parent).lower()
    spike = "other"
    if "v2" in parent_path or "ip_adapter" in parent_path:
        spike = "v2"
    elif "v3" in parent_path or "lora_plus" in parent_path:
        spike = "v3"
    elif "v4" in parent_path or "controlnet" in parent_path:
        spike = "v4"
    elif "v5" in parent_path or "lora_ablation" in parent_path:
        spike = "v5"
    elif "flux" in parent_path:
        spike = "flux"

    # Tradition from filename
    tradition = next((t for t in TRADITION_TOKENS if t in name.lower()), None)

    # Seed from filename
    m = re.search(r"_?s(?:eed)?[_-]?(\d+)", name.lower())
    seed = int(m.group(1)) if m else None

    # Sibling JSON for clean config
    cfg = {}
    json_path = p.with_suffix(".json")
    if json_path.exists():
        try:
            cfg = json.loads(json_path.read_text())
        except Exception as e:
            logging.warning(f"Failed to parse {json_path.name}: {e}")

    record = ImageRecord(
        path=p, spike=spike, tradition=tradition, seed=seed,
        config_str="",
    )
    record.lora_scale = cfg.get("lora_scale")
    record.ip_scale = cfg.get("ip_adapter_scale") or cfg.get("ip_scale")
    record.cn_scale = cfg.get("controlnet_scale") or cfg.get("cn_scale")
    record.model = cfg.get("model", "sdxl")
    record.variant = cfg.get("variant")

    # Filename-based fallback for older spikes that lack JSON
    if spike == "v4" and record.cn_scale is None:
        m = re.match(r"cn(\d{2,})", name)
        if m:
            raw = m.group(1)
            record.cn_scale = float(f"{raw[0]}.{raw[1:]}") if len(raw) > 1 else float(raw)
    if spike == "v5" and record.lora_scale is None:
        m = re.match(r"lora(\d{2,})", name)
        if m:
            raw = m.group(1)
            record.lora_scale = float(f"{raw[0]}.{raw[1:]}") if len(raw) > 1 else float(raw)

    # Build the human-readable config string
    parts = [spike]
    if record.model != "sdxl":
        parts.append(record.model)
    if record.variant:
        parts.append(record.variant)
    if record.lora_scale is not None:
        parts.append(f"L={record.lora_scale}")
    if record.ip_scale is not None:
        parts.append(f"IP={record.ip_scale}")
    if record.cn_scale is not None:
        parts.append(f"CN={record.cn_scale}")
    record.config_str = " | ".join(parts)
    return record


# ----------------------------------------------------------------------
# Model loaders
# ----------------------------------------------------------------------

def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    logging.info(f"Loading CLIP: {CLIP_MODEL_ID}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, proc

def load_dinov2(device):
    from transformers import AutoModel, AutoImageProcessor
    logging.info(f"Loading DINOv2: {DINOV2_MODEL_ID}")
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    return model, proc

def load_lpips(device):
    import lpips
    logging.info("Loading LPIPS (VGG backbone)")
    model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    # Set requires_grad to False to save memory
    for p in model.parameters():
        p.requires_grad = False
    return model


# ----------------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def clip_image_embeddings(images, model, proc, device, batch_size=8):
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].half()

        feats = model.get_image_features(**inputs)

        # Safe extraction if transformers returns an object instead of a tensor
        if not isinstance(feats, torch.Tensor):
            if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                feats = feats[0]

        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float())
    return torch.cat(embs, dim=0)

@torch.no_grad()
def clip_text_embedding(prompt, model, proc, device):
    inputs = proc(text=[prompt], return_tensors="pt", padding=True).to(device)

    feats = model.get_text_features(**inputs)

    # Safe extraction if transformers returns an object instead of a tensor
    if not isinstance(feats, torch.Tensor):
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            feats = feats.text_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        else:
            feats = feats[0]

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float()

@torch.no_grad()
def dinov2_embeddings(images, model, proc, device, batch_size=8):
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        out = model(**inputs)
        feats = out.last_hidden_state[:, 0, :]   # CLS token
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float())
    return torch.cat(embs, dim=0)


def lpips_distance_to_pool(img_path, ref_paths, model, device, target_size=256):
    """Compute mean ± std LPIPS distance from one image to each ref in the pool."""
    import lpips
    img = Image.open(img_path).convert("RGB").resize((target_size, target_size))
    img_t = lpips.im2tensor(np.array(img)).to(device)

    distances = []
    for ref_path in ref_paths:
        ref = Image.open(ref_path).convert("RGB").resize((target_size, target_size))
        ref_t = lpips.im2tensor(np.array(ref)).to(device)
        with torch.no_grad():
            d = model(img_t, ref_t).item()
        distances.append(d)
    return float(np.mean(distances)), float(np.std(distances))


# ----------------------------------------------------------------------
# Reference pool builders
# ----------------------------------------------------------------------

def build_packaging_pool(rng):
    all_imgs = sorted(PACKAGING_POOL_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No packaging images at {PACKAGING_POOL_DIR}")
    if len(all_imgs) <= PACKAGING_POOL_SIZE:
        return all_imgs
    idx = rng.choice(len(all_imgs), PACKAGING_POOL_SIZE, replace=False)
    return [all_imgs[i] for i in sorted(idx)]

def build_style_pools():
    pools = {}
    for trad, d in STYLE_POOL_DIRS.items():
        imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))
        if not imgs:
            logging.warning(f"No style references for {trad} at {d}")
            pools[trad] = []
        else:
            pools[trad] = imgs
    return pools


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", nargs="+", required=True, type=Path,
                        help="One or more directories containing spike output images")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "evaluation" / "quantitative_metrics.csv",
                        help="Output CSV path (default: evaluation/quantitative_metrics.csv)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all images even if already in the output CSV")
    args = parser.parse_args()

    setup_logging(args.verbose)
    device = args.device
    logging.info(f"Device: {device}")
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)")

    # Collect input images
    image_records = []
    for d in args.input_dir:
        # Smart Path Resolution:
        # If the path doesn't exist relative to where you ran the command,
        # it attempts to find it relative to the main project folder.
        if not d.exists():
            fallback_dir = PROJECT_ROOT / d
            if fallback_dir.exists():
                d = fallback_dir
            else:
                logging.warning(f"Skipping nonexistent input dir: {d}")
                continue

        # Use rglob() instead of glob() to catch images inside nested subfolders
        for p in sorted(d.rglob("*.png")):
            rec = parse_record(p)
            if rec is not None:
                image_records.append(rec)

    logging.info(f"Collected {len(image_records)} input images")
    if not image_records:
        logging.error("No images to process. Exiting.")
        return

    # Load existing CSV for resume support
    existing_files = set()
    if args.output.exists() and not args.force:
        existing_df = pd.read_csv(args.output)
        existing_files = set(existing_df["path"].tolist())
        logging.info(f"Found existing CSV with {len(existing_files)} entries; skipping those.")
    else:
        existing_df = pd.DataFrame()

    todo = [r for r in image_records if str(r.path) not in existing_files]
    logging.info(f"Processing {len(todo)} new images")
    if not todo:
        logging.info("Nothing new to process. CSV is up to date.")
        return

    # Build reference pools
    rng = np.random.default_rng(POOL_RANDOM_SEED)
    pkg_pool = build_packaging_pool(rng)
    style_pools = build_style_pools()
    logging.info(f"Packaging reference pool: {len(pkg_pool)} images")
    for t, imgs in style_pools.items():
        logging.info(f"  Style pool [{t}]: {len(imgs)} images")

    # Load models
    clip_model, clip_proc = load_clip(device)
    dinov2_model, dinov2_proc = load_dinov2(device)
    lpips_model = load_lpips(device)

    # Pre-encode reference pools
    logging.info("Encoding packaging reference pool with CLIP and DINOv2...")
    pkg_imgs = [Image.open(p).convert("RGB") for p in pkg_pool]
    pkg_clip = clip_image_embeddings(pkg_imgs, clip_model, clip_proc, device)
    pkg_dino = dinov2_embeddings(pkg_imgs, dinov2_model, dinov2_proc, device)
    del pkg_imgs

    logging.info("Encoding style reference pools with DINOv2...")
    style_dino = {}
    for trad, imgs in style_pools.items():
        if not imgs:
            style_dino[trad] = None
            continue
        ref_imgs = [Image.open(p).convert("RGB") for p in imgs]
        style_dino[trad] = dinov2_embeddings(ref_imgs, dinov2_model, dinov2_proc, device)

    # Single canonical CLIP-text embedding
    text_emb = clip_text_embedding(CLIP_TEXT_PROMPT, clip_model, clip_proc, device)

    # Process each input image
    results = []
    for i, rec in enumerate(todo, 1):
        try:
            img = Image.open(rec.path).convert("RGB")

            # CLIP image embedding (single image)
            img_clip = clip_image_embeddings([img], clip_model, clip_proc, device)

            # CLIP-image similarity to packaging pool
            clip_pkg_sims = (img_clip @ pkg_clip.T).squeeze(0).numpy()
            clip_img_mean = float(clip_pkg_sims.mean())
            clip_img_std = float(clip_pkg_sims.std())

            # CLIP-text similarity to prompt
            clip_text_sim = float((img_clip @ text_emb.T).item())

            # DINOv2 style similarity
            if rec.tradition and style_dino.get(rec.tradition) is not None:
                img_dino = dinov2_embeddings([img], dinov2_model, dinov2_proc, device)
                style_sims = (img_dino @ style_dino[rec.tradition].T).squeeze(0).numpy()
                dino_mean = float(style_sims.mean())
                dino_std = float(style_sims.std())
            else:
                dino_mean = np.nan
                dino_std = np.nan

            # LPIPS distance to packaging pool
            lpips_mean, lpips_std = lpips_distance_to_pool(
                rec.path, pkg_pool, lpips_model, device
            )

            results.append({
                "path": str(rec.path),
                "filename": rec.path.name,
                "spike": rec.spike,
                "model": rec.model,
                "variant": rec.variant,
                "tradition": rec.tradition,
                "seed": rec.seed,
                "lora_scale": rec.lora_scale,
                "ip_scale": rec.ip_scale,
                "cn_scale": rec.cn_scale,
                "config_str": rec.config_str,
                "clip_image_mean": clip_img_mean,
                "clip_image_std": clip_img_std,
                "clip_text_sim": clip_text_sim,
                "dinov2_style_mean": dino_mean,
                "dinov2_style_std": dino_std,
                "lpips_mean": lpips_mean,
                "lpips_std": lpips_std,
            })

            if i % 10 == 0 or i == len(todo):
                logging.info(f"[{i:>3}/{len(todo)}] {rec.path.name}")
        except Exception as e:
            logging.error(f"Failed on {rec.path}: {e}")
            continue

    # Write CSV (merge with existing if applicable)
    new_df = pd.DataFrame(results)
    if not existing_df.empty:
        new_df = pd.concat([existing_df, new_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(args.output, index=False)
    logging.info(f"Wrote {len(new_df)} rows to {args.output}")

    # Print summary
    print_summary(new_df)


def print_summary(df: pd.DataFrame):
    """Aggregate by (spike, config_str) and print a summary table."""
    if df.empty:
        return

    print("\n" + "=" * 100)
    print(" QUANTITATIVE METRICS SUMMARY")
    print("=" * 100)

    grouped = df.groupby(["spike", "config_str"]).agg(
        n=("filename", "count"),
        clip_img=("clip_image_mean", "mean"),
        clip_text=("clip_text_sim", "mean"),
        dinov2=("dinov2_style_mean", "mean"),
        lpips=("lpips_mean", "mean"),
    ).reset_index()

    print(f"\n{'Spike':<6} {'Configuration':<35} {'N':>4}"
          f" {'CLIP-img':>10} {'CLIP-txt':>10} {'DINOv2':>10} {'LPIPS':>10}")
    print("-" * 100)
    for _, row in grouped.iterrows():
        dinov2_str = f"{row['dinov2']:.4f}" if not pd.isna(row['dinov2']) else "   N/A   "
        print(f"{row['spike']:<6} {row['config_str'][:34]:<35} {row['n']:>4}"
              f" {row['clip_img']:>10.4f} {row['clip_text']:>10.4f}"
              f" {dinov2_str:>10} {row['lpips']:>10.4f}")
    print()
    print("Interpretation:")
    print("  CLIP-img:   higher = closer to OFF packaging pool (packaging fidelity)")
    print("  CLIP-txt:   higher = better adherence to the canonical packaging prompt")
    print("  DINOv2:     higher = closer to tradition reference pool (style fidelity)")
    print("  LPIPS:      LOWER = closer to real packaging (perceptual quality)")
    print()
    print("Per-tradition breakdown for the same metrics is available in the CSV.")


if __name__ == "__main__":
    main()