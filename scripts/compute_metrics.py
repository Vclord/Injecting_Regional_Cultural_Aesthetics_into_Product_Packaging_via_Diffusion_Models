"""
prompt to LLM:
Write a Python evaluation suite (compute_metrics.py) using torch, transformers, and lpips to calculate the quantitative
metrics for my dissertation's generated images. I need to compute four specific metrics per image:
1.Packaging Fidelity (CLIP-image): Cosine similarity between the generated image and a fixed reference pool of 30 real
packaging images.
2.Prompt Adherence (CLIP-text): Cosine similarity against my canonical prompt: "Front-facing product photograph of an
Indian snack packet, professional product photography".
3.Style Alignment (DINOv2): Cosine similarity (using the CLS token) against my tradition-specific reference
pools (Madhubani, Tanjore, Kalighat).
4.Perceptual Quality (LPIPS): Mean LPIPS distance (VGG backbone) to the packaging pool.
Crucial Hardware Constraint: This must run locally on my 6GB VRAM RTX 3060. To prevent out-of-memory errors, you must
load CLIP and DINOv2 in float16, explicitly disable gradients (requires_grad=False for LPIPS and torch.no_grad()
everywhere else), and pre-encode the reference pools into memory once at startup. Wrap this in a CLI that takes input
directories. For metadata (spike version, tradition, seed, LoRA/ControlNet scales), try to load a sibling .json file
first, then fall back to regex on the filename. Append results to a master CSV (skip images that are already in the CSV
to allow resuming), and print an aggregated summary table to the console grouped by experimental configuration.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Assume script is at scripts/
PACKAGING_POOL_DIR = PROJECT_ROOT / "data" / "processed" / "packaging"
STYLE_POOL_DIRS = {
    "madhubani":  PROJECT_ROOT / "data" / "processed" / "style_references" / "bihar",
    "tanjore":    PROJECT_ROOT / "data" / "processed" / "style_references" / "tamil_nadu",
    "kalighat":   PROJECT_ROOT / "data" / "processed" / "style_references" / "west_bengal",
}

# Standardise the number of reference images to ensure consistent variance calculations
PACKAGING_POOL_SIZE = 30
POOL_RANDOM_SEED = 42

# Canonical prompt used to evaluate raw text-to-image adherence
CLIP_TEXT_PROMPT = (
    "Front-facing product photograph of an Indian snack packet, "
    "professional product photography"
)

# Model checkpoints
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DINOV2_MODEL_ID = "facebook/dinov2-base"

TRADITION_TOKENS = {"madhubani", "tanjore", "kalighat"}

# Skip files that are grid summaries, smoke tests, or other meta-artefacts
SKIP_PATTERNS = ("grid", "comparison", "smoke", "scale_sweep")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    """Configures the standard library logger."""
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
    """Standardised data structure for an image waiting to be evaluated."""
    path: Path
    spike: str            # Identifier for the experimental phase (e.g., 'v2', 'v4', 'flux')
    tradition: Optional[str]
    seed: Optional[int]
    config_str: str       # A human-readable summary for the final aggregation table
    lora_scale: Optional[float] = None
    ip_scale: Optional[float] = None
    cn_scale: Optional[float] = None
    model: str = "sdxl"   # Foundation model used ('sdxl' or 'flux')
    variant: Optional[str] = None  # Ablation condition, e.g., 'baseline_no_lora'


def parse_record(p: Path) -> Optional[ImageRecord]:
    """
    Parses metadata for an image, first checking for a sibling JSON config file,
    and falling back to regex filename heuristics for older experiment spikes.

    Args:
        p (Path): Filepath to the output image.

    Returns:
        Optional[ImageRecord]: Structured record, or None if the file should be skipped.
    """
    name = p.stem
    if any(pat in name.lower() for pat in SKIP_PATTERNS) or name.startswith("_"):
        return None

    # Determine spike iteration safely by checking the entire parent path
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

    # Extract styling tradition from filename
    tradition = next((t for t in TRADITION_TOKENS if t in name.lower()), None)

    # Extract generation seed from filename
    m = re.search(r"_?s(?:eed)?[_-]?(\d+)", name.lower())
    seed = int(m.group(1)) if m else None

    # Attempt to load exact generation parameters from sibling JSON if available
    cfg = {}
    json_path = p.with_suffix(".json")
    if json_path.exists():
        try:
            cfg = json.loads(json_path.read_text())
        except Exception as e:
            logging.warning(f"Failed to parse {json_path.name}: {e}")

    record = ImageRecord(
        path=p, spike=spike, tradition=tradition, seed=seed, config_str=""
    )
    record.lora_scale = cfg.get("lora_scale")
    record.ip_scale = cfg.get("ip_adapter_scale") or cfg.get("ip_scale")
    record.cn_scale = cfg.get("controlnet_scale") or cfg.get("cn_scale")
    record.model = cfg.get("model", "sdxl")
    record.variant = cfg.get("variant")

    # Fallback to regex parsing for older spikes that did not save JSON configs
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

    # Construct the display string for the summary table
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
# Model loaders (Optimized for 6GB VRAM)
# ----------------------------------------------------------------------

def load_clip(device: str) -> tuple:
    """Loads CLIP using float16 precision to fit comfortably alongside other models."""
    from transformers import CLIPModel, CLIPProcessor
    logging.info(f"Loading CLIP: {CLIP_MODEL_ID}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, proc


def load_dinov2(device: str) -> tuple:
    """Loads DINOv2 using float16 precision."""
    from transformers import AutoModel, AutoImageProcessor
    logging.info(f"Loading DINOv2: {DINOV2_MODEL_ID}")
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    return model, proc


def load_lpips(device: str):
    """
    Loads LPIPS (VGG backbone).
    Requires_grad is explicitly disabled across all parameters to save memory,
    as we only use this for forward-pass inference.
    """
    import lpips
    logging.info("Loading LPIPS (VGG backbone)")
    model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ----------------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def clip_image_embeddings(images: list[Image.Image], model, proc, device: str, batch_size: int = 8) -> torch.Tensor:
    """
    Computes L2-normalized CLIP image embeddings.

    Returns:
        torch.Tensor: A tensor of shape [num_images, embed_dim] on the CPU.
    """
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].half()

        feats = model.get_image_features(**inputs)

        # Handle different output structures across transformers versions
        if not isinstance(feats, torch.Tensor):
            if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                feats = feats[0]

        # L2 normalize so dot product equals cosine similarity
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float())

    return torch.cat(embs, dim=0)


@torch.no_grad()
def clip_text_embedding(prompt: str, model, proc, device: str) -> torch.Tensor:
    """
    Computes the L2-normalized CLIP text embedding for a given string.

    Returns:
        torch.Tensor: A tensor of shape [1, embed_dim] on the CPU.
    """
    inputs = proc(text=[prompt], return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(**inputs)

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
def dinov2_embeddings(images: list[Image.Image], model, proc, device: str, batch_size: int = 8) -> torch.Tensor:
    """
    Computes L2-normalized DINOv2 embeddings using the CLS token.
    DINOv2 is used here because it is highly sensitive to texture, stroke, and style.

    Returns:
        torch.Tensor: A tensor of shape [num_images, embed_dim] on the CPU.
    """
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].half()

        out = model(**inputs)
        # Extract the classification (CLS) token for global image representation
        feats = out.last_hidden_state[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float())

    return torch.cat(embs, dim=0)


def lpips_distance_to_pool(img_path: Path, ref_paths: list[Path], model, device: str, target_size: int = 256) -> tuple:
    """
    Computes the mean and standard deviation of the LPIPS distance from one image
    to every image in a reference pool.

    LPIPS is a perceptual distance metric; lower scores mean the images are structurally closer.

    Returns:
        tuple: (mean_distance, std_distance)
    """
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

def build_packaging_pool(rng: np.random.Generator) -> list[Path]:
    """Randomly samples a fixed number of authentic packaging images to act as the baseline pool."""
    all_imgs = sorted(PACKAGING_POOL_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No packaging images at {PACKAGING_POOL_DIR}")
    if len(all_imgs) <= PACKAGING_POOL_SIZE:
        return all_imgs

    idx = rng.choice(len(all_imgs), PACKAGING_POOL_SIZE, replace=False)
    return [all_imgs[i] for i in sorted(idx)]


def build_style_pools() -> dict[str, list[Path]]:
    """Gathers all available real folk-art references grouped by tradition."""
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

def main() -> None:
    """
    Main execution pipeline:
    1. Parses arguments and handles directory resolution.
    2. Collects images, skipping ones already scored in previous runs.
    3. Loads and pre-encodes the reference pools (Packaging and Styles) into memory.
    4. Iterates over target images, computing similarities (dot products) and distances.
    5. Saves results to CSV and prints an aggregate summary table.
    """
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

    # 1. Collect target images securely
    image_records = []
    for d in args.input_dir:
        # Smart Path Resolution: resolve relative to script execution or project root
        if not d.exists():
            fallback_dir = PROJECT_ROOT / d
            if fallback_dir.exists():
                d = fallback_dir
            else:
                logging.warning(f"Skipping nonexistent input dir: {d}")
                continue

        # Use rglob to ensure we catch images nested inside subfolders
        for p in sorted(d.rglob("*.png")):
            rec = parse_record(p)
            if rec is not None:
                image_records.append(rec)

    logging.info(f"Collected {len(image_records)} input images")
    if not image_records:
        logging.error("No images to process. Exiting.")
        return

    # 2. Check existing CSV to allow resuming interrupted runs
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

    # 3. Build and pre-encode reference pools into memory
    rng = np.random.default_rng(POOL_RANDOM_SEED)
    pkg_pool = build_packaging_pool(rng)
    style_pools = build_style_pools()

    logging.info(f"Packaging reference pool: {len(pkg_pool)} images")
    for t, imgs in style_pools.items():
        logging.info(f"  Style pool [{t}]: {len(imgs)} images")

    clip_model, clip_proc = load_clip(device)
    dinov2_model, dinov2_proc = load_dinov2(device)
    lpips_model = load_lpips(device)

    logging.info("Encoding packaging reference pool with CLIP and DINOv2...")
    pkg_imgs = [Image.open(p).convert("RGB") for p in pkg_pool]

    # Pre-computed reference tensors: [pool_size, embed_dim]
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

    # Encode the static text prompt exactly once
    text_emb = clip_text_embedding(CLIP_TEXT_PROMPT, clip_model, clip_proc, device)

    # 4. Main Evaluation Loop
    results = []
    for i, rec in enumerate(todo, 1):
        try:
            img = Image.open(rec.path).convert("RGB")

            # Get target image representation: shape [1, embed_dim]
            img_clip = clip_image_embeddings([img], clip_model, clip_proc, device)

            # Cosine similarity to the packaging pool via dot product
            clip_pkg_sims = (img_clip @ pkg_clip.T).squeeze(0).numpy()
            clip_img_mean = float(clip_pkg_sims.mean())
            clip_img_std = float(clip_pkg_sims.std())

            # Cosine similarity to the text prompt
            clip_text_sim = float((img_clip @ text_emb.T).item())

            # Cosine similarity to specific cultural style reference pool
            if rec.tradition and style_dino.get(rec.tradition) is not None:
                img_dino = dinov2_embeddings([img], dinov2_model, dinov2_proc, device)
                style_sims = (img_dino @ style_dino[rec.tradition].T).squeeze(0).numpy()
                dino_mean = float(style_sims.mean())
                dino_std = float(style_sims.std())
            else:
                dino_mean = np.nan
                dino_std = np.nan

            # LPIPS distance requires running individual pairs through VGG
            lpips_mean, lpips_std = lpips_distance_to_pool(
                rec.path, pkg_pool, lpips_model, device
            )

            # Store computed metrics
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

    # 5. Save and Export
    new_df = pd.DataFrame(results)
    if not existing_df.empty:
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(args.output, index=False)
    logging.info(f"Wrote {len(new_df)} rows to {args.output}")

    print_summary(new_df)


def print_summary(df: pd.DataFrame) -> None:
    """
    Groups the metric results by spike and configuration version,
    calculating the mean of each metric to print an easy-to-read summary table.
    """
    if df.empty:
        return

    print("\n" + "=" * 100)
    print(" QUANTITATIVE METRICS SUMMARY")
    print("=" * 100)

    # Aggregate metric means across groups
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
        # Handle cases where DINOv2 style metrics weren't computable (e.g. non-traditional baseline)
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