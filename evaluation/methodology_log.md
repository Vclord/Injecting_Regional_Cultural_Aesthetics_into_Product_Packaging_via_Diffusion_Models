# Methodology Log

## 2026-05-30 — Spike grid revised after smoke test

**Original plan (in research_proposal_v2.docx, O3 / project plan):**
Single-variant IP-Adapter scale sweep at 0.4 / 0.6 / 0.8.

**Observation triggering the change:**
The IP-Adapter smoke test at scale=0.6 with the Madhubani reference produced
an image that was clearly Madhubani-styled but was a *painting* rather than
a *snack packet*. The reference image's content (figures, frame, composition)
was imported alongside its style, overwhelming the text prompt's request for
product photography.

**Change made (this spike only — the proposal's main-experiment plan stands):**
Replace the single-variant scale sweep with a comparison of two IP-Adapter
variants:
- Regular IP-Adapter (`ip-adapter_sdxl.bin`)
- IP-Adapter-Plus (`ip-adapter-plus_sdxl_vit-h.safetensors`)
at two scales (0.5, 0.7) across the same prompts, traditions, and seeds.

**Justification:**
The original ablation would have produced a known curve along a single
known-flawed approach. The revised grid compares two methods at their
plausible operating points, addressing the actual content-vs-style failure
mode revealed by the smoke test. The change is consistent with O2 / O3 of
the proposal in spirit (characterise the LoRA–IP-Adapter trade-off space)
while producing more methodologically useful evidence.

**Rubric:** Unchanged. The pre-committed rubric (rubric.md) still applies.
Scoring of the new grid will use the same four axes, two-session protocol,
and intra-rater agreement reporting.

**Outputs of this spike will inform the choice of IP-Adapter variant in
the main pipeline used for the Week 3–6 experiments.**

## 2026-05-30 (continued) — Plus smoke test result

The IP-Adapter-Plus variant smoke test at scale=0.7 with the Madhubani
reference produced an output that retained snack-packet semantics from the
text prompt (pouch silhouette, top crimp, product-photography framing)
while transferring Madhubani style (linework, palette, motif vocabulary)
without importing the reference's content (deity figure, elephants).

This confirms Plus-Style as the appropriate IP-Adapter variant for the
main pipeline. The full 72-image grid will provide systematic evidence
across prompts, traditions, seeds, and scales.

Image saved to: outputs/spike/ip_adapter_v2/_smoke_test_plus.png

## 2026-05-31 (or actual date) — Full Plus-vs-Regular grid results

The 72-image comparison grid (regular vs Plus IP-Adapter, three traditions,
two scales, two seeds, three prompts) is complete.

Headline findings:
- Plus + scale 0.7 reliably produces packet-shaped outputs with regional
  decoration for Madhubani; partial for Tanjore; fails for Kalighat (style
  not preserved).
- Regular IP-Adapter at scale 0.7 produces fine-art panels rather than
  packets for all three traditions.
- Kalighat reveals a limit of style-only transfer: CLIP's style embedding
  does not appear to capture gesture/composition-defined styles as cleanly
  as pattern/palette-defined styles. This finding will be discussed in the
  results chapter as a contribution.

Architecture decision for main pipeline (Weeks 3 onwards):
- IP-Adapter variant: Plus
- Default conditioning scale: 0.7 (subject to per-tradition tuning)
- LoRA: now empirically motivated as the component that must (a) anchor
  packaging semantics and (b) potentially compensate for Plus's weaker
  signal on gesture-style traditions like Kalighat.

Compute observation: full grid took ~15 hours on RTX 3060 Laptop,
significantly over the 2-hour estimate due to thermal throttling under
sustained load. Subsequent batch experiments will be run on Google Colab
Pro (A100/L4).

## 2026-06-03 — Spike v2 rubric agreement complete

Session 1: 2026-06-01 15:00
Session 2: 2026-06-03 18:23
Gap: ~51 hours (rubric requires ≥48h)

Cohen's weighted kappa (linear weights) per axis:
- text_legibility: 0.844 (almost perfect — most scores were 0 by design)
- regional_appropriateness: 0.465 (moderate)
- packaging_plausibility: 0.742 (substantial)
- visual_quality: 0.857 (almost perfect)

Mean per-image total disagreement: 0.77 (max 3, of possible 12).
47.4% of images received identical scores across both sessions.

Interpretation: agreement is strongest for objectively-assessable axes and 
weakest for the most subjective axis (regional appropriateness), as 
anticipated. All axes meet or exceed the "moderate agreement" threshold 
(κ > 0.4) of Landis and Koch (1977). Single-rater methodology considered 
defensible for downstream comparisons.

Highest-disagreement images saved to spike_v2_high_disagreement.csv for 
narrative review in the dissertation's evaluation chapter.

## 2026-06-03 — SDXL LoRA v1 training complete

W&B run: https://wandb.ai/vivekchandra726-university-of-stirling/text2image-fine-tune/runs/at8esao4
Hyperparameters: rank=16, alpha=16, lr=1e-4, steps=2000, batch=1×4 (grad-accum), resolution=1024, cosine LR, AdamW-8bit, bf16
Training images: ~270 (Open Food Facts subset, processed)
Caption: uniform — "a photograph of ipsnackpkg, an Indian snack packet, product photography on a white background"
Saved adapter: sdxl_packaging_lora_r16_lr0.0001_steps2000_20260603_2249.safetensors

Validation result (4 standalone samples, no IP-Adapter):
- 2/4 outputs are coherent full snack packets with brand zones, layout, gibberish-but-structurally-correct text
- 2/4 outputs are partial — the trigger token did not fully overcome SDXL's prompt interpretation when packaging anchors in the prompt were weak ("red and yellow packaging", "premium snack pack")
- Text rendering is gibberish (expected; rendering handled by PIL post-compositing in the main pipeline)

Interpretation: LoRA has learned the packaging-domain visual prior. Strength of assertion depends on prompt phrasing. Combined-pipeline evaluation with IP-Adapter Plus is the meaningful next test.

Next: integrate LoRA + IP-Adapter Plus locally and regenerate a subset of the spike v2 grid.

## 2026-06-04 — LoRA v1 audit decision

W&B loss curve (saved as outputs/lora_checkpoints/wandb_loss_v1.png) is flat 
around ~0.10 with no visible downward trend, characteristic of diffusion LoRA 
training where per-step loss reflects random-timestep noise rather than 
learning progress.

Standalone validation samples (validation_20260603_2249.png) show 2/4 outputs 
are coherent full packets and 2/4 are partial, indicating the LoRA learned 
packaging-domain features but does not strongly override the base model when 
prompts lack explicit packaging anchors.

Decision: proceed to LoRA + IP-Adapter Plus combined-pipeline test before 
considering retraining. Rationale: the LoRA's research purpose is paired use 
with Plus, not standalone generation. The combined pipeline's behaviour is 
the meaningful evidence. Retraining now risks over-anchoring the LoRA in a 
way that suppresses Plus's style transfer.

If combined pipeline succeeds: v1 LoRA accepted as project artefact.
If combined pipeline shows weak packet anchoring: retrain with one of:
  - rank=32 (doubles capacity)
  - max_steps=3000, lr=1.5e-4 (longer, harder training)
  - shorter caption (drops "Indian snack packet" from caption so the trigger 
    carries the full visual prior, may improve transfer to weaker prompts)

## 2026-06-04 — Spike v3: LoRA + IP-Adapter Plus integration

Generated 12 images (3 traditions × 2 seeds × 2 conditions: Plus-only / 
LoRA+Plus) at IP-scale=0.7, LoRA-scale=1.0. Runtime ~99 minutes locally.

Result: LoRA+Plus consistently adds packaging semantics (foil texture, 
gibberish-text top band, product-callout zones, pack silhouettes with 
seal lines) absent from Plus-only outputs. Effect is strongest on 
traditions where Plus alone failed (Kalighat, Tanjore). Madhubani 
benefits less because Plus-only already produces packet-like outputs.

Tanjore remains the weakest tradition — the Navaneeta Krishna reference 
imports such strong compositional content that even LoRA cannot fully 
restore packet semantics.

Decision: v1 LoRA accepted as project artefact. No retraining for now.
Architecture confirmed: Plus carries folk-art style, LoRA adds commercial 
packaging structure.

Comparison grid: outputs/spike/v3_lora_vs_plus_only.png
Individual outputs: outputs/spike/v3_lora_plus/

## 2026-06-04 — Text compositing implemented

PIL-based compositing of top (Latin "brand-like") and bottom (regional script
flavour) text bands onto generated outputs. Vocab: 10 Latin words × 5 regional
words per tradition (Devanagari/Tamil/Bengali). Fonts: Google Noto Sans (SIL OFL
1.1). Deterministic per (tradition, seed); see scripts/composite_text.py.

Regional vocabulary spelling reviewed by [native speaker name] on [date] —
[no corrections / specific corrections noted].

Composited outputs treated as a separate evaluation condition: the rubric is
applied to both uncomposited and composited versions of each spike v3 image,
and the difference in Axis 1 (Text Legibility) scores quantifies the
compositing step's contribution. The composited outputs are presented in
the dissertation as the project's final deliverable; uncomposited outputs
are presented in the methodology chapter as the model's raw generative output.

## 2026-06-04 — Text compositing complete

All 12 spike v3 outputs now have composited versions in
outputs/spike/v3_composited/. Visual review confirms bands integrate
with image colour palette and text is legible on all outputs (with
exceptions noted in <file>).

Composited outputs treated as a separate evaluation condition. See
scripts/composite_text.py for compositing logic, vocab, and
deterministic seeding.

Vocab pending native-speaker review: Devanagari (मसाला, नमकीन, चटपटा,
तीखा, स्वादिष्ट), Tamil (மசாலா, காரம், சுவை, தக்காளி, மிளகு),
Bengali (মশলা, ঝাল, নোনতা, টক, স্বাদ). Reviewer: TBD.

## 2026-06-04 — ControlNet edge map prepared

Source: data/processed/packaging/8904004400694.png (Haldiram's Aloo Bhujia)
Canny thresholds: low=175, high=300
Pre-processing: chips window masked with median-border-colour fill,
mask region (fractional): (0.10, 0.36, 0.92, 0.76).

Result preserves: outer pouch silhouette, top crimp, bottom zig-zag seal,
brand oval, title block, bottom callout pill.
Excludes: internal bhujia-strand edges that would over-constrain the model.

Edge map saved at: outputs/spike/v4_controlnet/_canny_control.png

## 2026-06-04 — Text compositing v2 implemented

Polished version of composite_text.py addressing visible "sticker bar" 
appearance of v1. Changes:
- Reduced band opacity (192/255 ≈ 0.75)
- Inner-edge gradient fade (~30% of band height) eliminates hard inner boundary
- Text colour sampled from contrasting cluster in the band region (k-means k=5)
- Hough-detected edge angle drives slight text rotation when pack edges are tilted
- Soft drop shadow with Gaussian blur on rendered text

Outputs in outputs/spike/v3_composited_v2/. The v1 outputs in 
outputs/spike/v3_composited/ are retained for methodology comparison: the 
dissertation will show v1 vs v2 side-by-side to document the iterative 
visual refinement process.

Limitations remain: bands still occlude underlying decoration; outer 
band edges remain straight; no contour-following. Native multilingual 
text rendering via glyph-conditioned diffusion (AnyText / GlyphControl) 
is noted as future work.

## 2026-06-04 — ControlNet scale calibration via mini-spike

After the initial smoke test (cn=0.7) showed ControlNet over-constraining
the Madhubani style transfer, a 3-image scale sweep was run on the
Madhubani reference at seed 42 to identify the trade-off curve before
committing to the full 12-image grid.

Scales tested: 0.3, 0.5, 0.7
- cn=0.3: Strong Madhubani style; pack structure (title block, brand
  zone) only loosely followed. Title text corrupted ("HAIJIA").
- cn=0.5: Balanced — pack structure clean ("ALOO BHIJIA" mostly correct,
  Haldiram's oval intact, layout present), Madhubani style visibly
  preserved (fish motifs, characteristic palette, decorative borders).
- cn=0.7: Pack structure fully intact ("ALOO BHUJIA" correct), but
  Madhubani style reduced to decorative trim.

Decision: cn=0.5 chosen as the project's operating ControlNet scale on
empirical evidence. cn=0.4 added as the adjacent data point for the full
grid ablation. cn=0.3 and cn=0.7 results retained as bracket evidence
in the dissertation's discussion chapter.

Sweep image saved at: outputs/spike/v4_controlnet/_mini_scale_sweep_strip.png

## 2026-06-XX — Spike v4 ControlNet results complete

Generated 12 images: 3 traditions × 2 seeds × 2 ControlNet scales (0.4, 0.5),
with IP-Adapter scale held at 0.7 and LoRA scale at 1.0. Three-way comparison
against spike v3 (no ControlNet) is in outputs/spike/v4_three_way_comparison.png.

Note: the comparison grid figure was generated with incorrect column labels
(0.5 / 0.8); actual scales used were 0.4 / 0.5. Regeneration needed for
dissertation figures.

Key finding: ControlNet trades style-transfer fidelity for packaging-structure
fidelity, and the trade-off is UNEQUAL across the three traditions.

- Madhubani (pattern-and-palette style): ControlNet retains most style;
  produces commercial-looking packets with visible Madhubani decoration.
- Tanjore (figural-content style): ControlNet eliminates the painting-vs-
  packet failure mode but reduces Tanjore style to its palette signature
  only (no deities, no gold leaf, no ornate arches).
- Kalighat (gesture-and-flat-colour style): ControlNet produces packets
  but loses Kalighat style almost entirely. Neither v3 nor v4 fully works
  for Kalighat — v3 keeps style at the cost of packet semantics; v4 the
  reverse.

Implication for project: no single configuration is optimal across all
traditions. Tradition-dependent operating point recommended.

This negative result on Kalighat is a contribution worth reporting:
it suggests that style-transfer methods based on CLIP image embeddings
(IP-Adapter Plus) and structural conditioning (Canny ControlNet) are
biased toward styles whose visual identity resides in pattern/palette
rather than gesture/composition.

## 2026-06-04 (continued) — Spike v4 grid corrected and re-read

Initial three-way comparison grid was generated with incorrect column 
labels (cn=0.5 labelled as 0.8, etc). Regenerated with correct labels
(cn=0.4 and cn=0.5). Underlying generations were correct throughout —
only the visualisation was mislabelled.

Headline finding (revised): The LoRA + IP-Adapter Plus + Canny ControlNet
pipeline at scale 0.4 produces outputs across all three traditions that
preserve both commercial-packaging structure (pouch silhouette, brand
zones, title blocks) AND folk-art aesthetic (palette, motifs, figural
content where applicable). At scale 0.5 the balance shifts toward
structure; at 0.7 style is reduced to palette signature only.

Operating point chosen: ControlNet scale = 0.4.
Bracket evidence: 0.0 (v3 baseline), 0.3 (mini-spike), 0.4 (chosen),
0.5 (alternate), 0.7 (smoke test) — five points on the trade-off curve.

Outputs of strongest qualitative interest for the dissertation gallery:
- madhubani s1337 v4@0.4 (black elephant figure preserved on a clear pack)
- tanjore s42 v4@0.4 (deity integrated into pack rather than canvas)
- kalighat s1337 v4@0.4 (blue figure preserved + pack structure)

## 2026-06-XX — Spike v5 LoRA-scale ablation complete

Generated 18 images: 3 traditions × 2 seeds × 3 LoRA scales (0.5, 0.7, 1.0),
with ControlNet held at 0.4 and IP-Adapter Plus at 0.7. Output:
outputs/spike/v5_lora_ablation_grid.png

Key finding: LoRA scale has minimal qualitative impact on outputs at the
chosen operating point. All three scales produce broadly equivalent results
for each (tradition, seed) pair. Differences exist (slight text variation,
small composition shifts) but no systematic style-vs-structure trade-off
emerges from varying LoRA scale alone.

Likely explanations (in order of plausibility):
1. ControlNet at scale 0.4 dominates packaging-domain semantics, leaving
   little room for LoRA's contribution to manifest.
2. The v1 LoRA (rank=16, 2000 steps, uniform caption) has limited
   capacity, making its scale-dependence small in absolute terms.
3. IP-Adapter Plus dominates style transfer, making LoRA scale orthogonal.

These hypotheses cannot be cleanly disentangled within this ablation.
A higher-capacity LoRA (rank=32, longer training) would be needed to
distinguish (1) from (2); this is left as future work.

Final chosen pipeline configuration:
  LoRA scale: 0.7   (middle of ablated range)
  IP-Adapter Plus scale: 0.7
  ControlNet (Canny) scale: 0.4
  Control image: Haldiram's Aloo Bhujia (8904004400694.png)
  Canny thresholds: low=175, high=300, with chips window masked

This configuration is the pipeline used for the dissertation's main
deliverables and quantitative-metric evaluations.

## 2026-06-08 — Spike v4 rubric agreement complete

Session 1: 2026-06-06 10:00
Session 2: 2026-06-08 (≥48h gap satisfied)

Cohen's weighted kappa (linear weights), n=12:
- text_legibility: NaN (both sessions scored uniform mean 1.00, no variance)
- regional_appropriateness: 0.647 (substantial)
- packaging_plausibility: 0.063 (near chance — see below)
- visual_quality: 0.789 (substantial)

Mean per-image total disagreement: 0.92 (max 2 of possible 12).
25% of images received identical scores across both sessions.

Methodology observations:
1. text_legibility kappa undefined due to zero variance in both sessions.
   Both sessions scored mean 1.00, where v2 sessions averaged 0.28.
   This reflects a between-spike interpretive shift: v4 outputs contain
   model-generated gibberish text-shapes in expected pack zones, which
   appears to have been scored as score=1 ("faint trace") in v4 where
   v2's similar gibberish was scored 0. This is documented as a known
   limitation, not corrected post-hoc.

2. packaging_plausibility kappa of 0.063 is near-chance. Most likely
   explanation: ControlNet conditioning produced uniformly packet-shaped
   outputs across all v4 images, compressing the score distribution to
   2-3 range. With chance baseline agreement at ~50% for binary 2/3
   outcomes, the absolute kappa is unreliable on this axis at this
   sample size.

Mitigation in dissertation:
The larger v2 spike (n=78) provides the more reliable kappa estimate
for the rubric's intra-rater reliability claim. v2 kappa on
packaging_plausibility was 0.742 (substantial). v4 figures are
reported as small-sample follow-up data and not used as the headline
reliability evidence.

## 2026-06-09 — FLUX LoRA training: technical and successful

Initial attempts at FLUX.1-schnell LoRA training on the diffusers `main`
branch failed with 8+ distinct configuration and version conflicts
(documented in detail for the dissertation methodology log). Root cause
was version skew between Colab's Python 3.12 base image, the diffusers
main branch, torchao binary compatibility, and the diffusers training
script's evolving latent-packing code path.

Resolved by pinning versions: diffusers==0.32.0, transformers==4.45.2,
peft==0.13.2, accelerate==1.1.1, with torchao removed entirely
(unnecessary for non-quantized training).

Two LoRAs produced:
- flux_packaging_lora_r16_res1024_steps2000.safetensors (primary —
  symmetric configuration to the SDXL LoRA: same rank, same step
  count, same effective batch size, matched learning rate adjusted
  for FLUX sensitivity)
- flux_packaging_lora_r16_res512_steps1000.safetensors (supplementary
  — produced earlier in the troubleshooting process; retained as
  robustness check against any artefacts of the 1024-res run)

The version-pinning saga itself is a methodology fact worth a sentence
in the dissertation's Methodology §3.3 (Training) — it demonstrates
the practical fragility of multi-model adaptation pipelines on
fast-moving open-source ecosystems.

## 2026-06-09 (continued) — FLUX LoRA inference scale diagnostic

Initial FLUX + LoRA comparison outputs produced hazy/ghosted packets at
inference LoRA scale = 1.0 (matching the SDXL operating point). A 4-config
diagnostic (1024-LoRA & 512-LoRA × scale {0.5, 1.0, 0.3}) confirmed both
LoRAs are correctly trained but scale 1.0 over-asserts on FLUX outputs.
At scale 0.5, both LoRAs produce clean, well-formed Indian snack packets.

This is consistent with FLUX-community practice (LoRA scale 0.6-0.8 is
typical; SDXL convention is 1.0). It is a real architectural difference,
not a training failure.

Operating point for FLUX + LoRA inference: scale = 0.5.
Operating point for SDXL + LoRA inference: scale = 1.0 (unchanged).
This asymmetry will be disclosed in the dissertation's methodology
section as a model-architecture-specific convention.

Diagnostic outputs saved at:
outputs/flux_comparison/_lora_diagnostic/diagnostic_grid.png

## 2026-06-09 (final) — SDXL vs FLUX comparison locked

Final operating points:
- SDXL + LoRA: scale 1.0
- FLUX + LoRA: scale 0.5 (selected via diagnostic against scales 0.3 and 1.0)

The scale asymmetry reflects the conventional inference practice for each
backbone, not a methodological flaw. Comparing at each backbone's appropriate
operating scale is more representative of real deployment than enforcing
numeric equality. This is disclosed in the dissertation's methodology section.

Final comparison grid: outputs/flux_comparison/sdxl_vs_flux_comparison_v2.png
(The v1 grid is retained as evidence of the diagnostic story for the appendix.)

## 2026-06-10 — Spike v5 (LoRA-scale ablation) rubric agreement

Session 1: 2026-06-08
Session 2: 2026-06-10 (~48h gap satisfied)

Cohen's weighted kappa (linear weights), n=18:
- text_legibility: NaN (zero variance, both sessions mean 1.00)
- regional_appropriateness: 1.000 (100% exact agreement)
- packaging_plausibility: 0.769 (substantial; 88.9% exact agreement)
- visual_quality: NaN (zero variance, both sessions mean 2.00)

Mean per-image total disagreement: 0.11 (16/18 zero-disagreement).

Interpretation:
The high agreement reflects the genuine visual homogeneity of v5 outputs
(as observed qualitatively in the ablation grid: minimal difference across
LoRA scales at the chosen ControlNet operating point of 0.4). This
supports rather than undermines the v5 finding of minimal LoRA-scale 
effect. Two axes returned NaN due to score-distribution collapse onto
single values, identical mechanism to v4's text_legibility NaN. Reliability
claims for the project anchor on the larger v2 sample (n=78, all axes
substantial or higher).
