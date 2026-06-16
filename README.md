# Folk-Art-Conditioned Indian Packaging Generation

Code and evaluation artefacts for the MSc dissertation:
**"Injecting Regional Cultural Aesthetics into Product Packaging via Reference-Conditioned Diffusion Models: A Comparative Study of SDXL and FLUX with LoRA and IP-Adapter Conditioning"**

University of Stirling, MSc Artificial Intelligence, 2026.

---

## What this repository contains

This is a four-component diffusion inference pipeline for generating Indian regional folk-art-styled snack packaging concepts. The pipeline combines:

1. **Packaging-domain LoRA** trained on the Open Food Facts (OFF) Indian snack subset
2. **IP-Adapter Plus** for folk-art style transfer (Madhubani, Tanjore, Kalighat)
3. **Canny ControlNet** for structural conditioning against a real pouch silhouette
4. **Post-hoc PIL text compositing** for regional-script labels (Devanagari, Tamil, Bengali)

A LoRA-only comparison against FLUX.1-schnell is also provided, isolating base-model contribution from auxiliary conditioning.

## Example outputs

See `examples/` for representative final-pipeline outputs across the three traditions.

## Repository structure

| Directory | Contents |
|---|---|
| `data/` | Dataset metadata (URLs, licences, provenance). Images themselves are not redistributed; recover via URLs. |
| `scripts/` | Standalone Python scripts (dataset triage, metric computation, kappa, text compositing) |
| `notebooks/` | Colab notebooks for training and inference |
| `evaluation/` | Rubric, scoring CSVs, quantitative metrics, methodology log |
| `examples/` | Representative pipeline outputs |

## Reproducibility

### Trained LoRA weights
LoRA checkpoints are too large for GitHub. Download from [LINK TO HUGGING FACE OR GOOGLE DRIVE]:
- `sdxl_packaging_lora_r16_steps2000.safetensors` (~200 MB)
- `flux_packaging_lora_r16_res1024_steps2000.safetensors` (~700 MB)

### Image datasets
Image files are not redistributed (respecting source-platform licensing terms). Re-download via the URLs in `data/packaging_metadata.csv` and `data/style_references_metadata.csv`.

### Pre-committed rubric
The scoring rubric at `evaluation/rubric.md` was committed on 2026-05-27 11:53:22 UTC.
- SHA-256 hash: `7d03d195ec821585dbbbe24c919fcdcdd51899f431fa9fc2670e85f633db33a6`
- Verify with: `Get-FileHash -Algorithm SHA256 evaluation/rubric.md` (PowerShell) or `sha256sum evaluation/rubric.md` (Linux/macOS)

## Environment

### SDXL pipeline
```bash
pip install -r requirements.txt
```

### FLUX pipeline (different pinned versions)
```bash
pip install -r requirements-flux.txt
```

The FLUX training pipeline required specific pinned versions to work in the diffusers FLUX implementation; this is documented in `evaluation/methodology_log.md`.

## Citing

If you use this work or its evaluation artefacts, please cite:

> Chandra, V. (2026). *Injecting Regional Cultural Aesthetics into Product Packaging via Reference-Conditioned Diffusion Models: A Comparative Study of SDXL and FLUX with LoRA and IP-Adapter Conditioning.* MSc Dissertation, University of Stirling.

## Licence

Code: Apache-2.0 Licence (see `LICENSE`)
Evaluation data and metadata: CC-BY 4.0
Image attributions: per-image as recorded in `data/*_metadata.csv`

## Contact

Vivek Chandra — vivekchandra726@gmail.com
                vic00089@students.stir.ac.uk
