# Evaluation Rubric — MSc Dissertation

**Project:** Reference-Driven Localisation of Indian Snack Packaging Using Diffusion Models
**Author:** Vivek Chandra, University of Stirling
**Status:** Pre-committed. This file must not be edited after the first output generation run. Any clarifications go in a separate `rubric\_addenda.md` and are reported as limitations in the dissertation.

\---

## Purpose

This rubric defines the four-axis ordinal scoring scheme used in the structured author-led qualitative evaluation of generated packaging images. It is committed to version control prior to any output generation in order to constrain the evaluator (the author) and to support a defensible single-rater methodology.

## Scoring procedure

For every image included in the qualitative evaluation set:

1. The image is scored on all four axes below, independently, on an ordinal 0–3 scale.
2. Each image is scored in **two independent sessions** separated by **at least 48 hours**.
3. Sessions are conducted in a quiet environment with consistent lighting; the evaluator may not view their previous session's scores before completing the second.
4. Image filenames are anonymised at scoring time so that the conditioning configuration (prompt-only, LoRA-only, IP-Adapter-only, combined) is not visible to the evaluator. A blind-key CSV maps anonymous IDs to configurations and is consulted only after scoring is complete.
5. Intra-rater agreement is reported per axis using Cohen's weighted kappa (linear weights).

## Inclusion criteria for the evaluation set

* 30 images per conditioning configuration, sampled across the three target states (Tamil Nadu, West Bengal, Punjab).
* Sampling is stratified by state and by the same five fixed prompts used across all conditions, to ensure comparability.
* Failed generations (model errors, malformed outputs) are excluded *before* sampling, not after scoring.

\---

## Axis 1 — Text Legibility

Scores the regional-script text rendered on or composited onto the generated packaging. Where post-hoc PIL compositing is used, the axis still applies (it measures the final output's legibility, not the source of the text).

|Score|Descriptor|
|:-:|-|
|**0**|No script-like content present, or content is unreadable noise (visual artefacts, blobs, illegible marks).|
|**1**|Script-like marks are present but are not real characters of any Indian language; the text is decorative pseudo-script.|
|**2**|Some characters appear to be real characters of the intended regional script (Tamil / Bengali / Gurmukhi), but the text as a whole is not parseable as words.|
|**3**|The text is parseable as legitimate words or fragments in the intended regional script, regardless of whether the meaning is appropriate for the product.|

## Axis 2 — Regional Appropriateness

Scores whether the visual aesthetic plausibly corresponds to the intended target state, based on colour palette, typography, motif vocabulary, and overall design language.

**Important constraint:** "Appropriateness" is judged against the *reference pool* for that state in the dataset, not against the evaluator's personal expectations of regional culture. This constraint is intended to reduce essentialist scoring.

|Score|Descriptor|
|:-:|-|
|**0**|Output bears no aesthetic resemblance to the reference pool for the intended state; could equally plausibly belong to any other state, or to no state.|
|**1**|Output shares some surface elements with the reference pool (e.g. a colour or motif) but the overall design language is generic.|
|**2**|Output is recognisably influenced by the reference pool's design language in at least two of: palette, typography style, motif vocabulary.|
|**3**|Output is strongly aligned with the reference pool's design language across palette, typography, and motif vocabulary; a viewer familiar with the reference set would identify it as belonging to that pool.|

## Axis 3 — Brand-Identity Plausibility

Scores whether the output reads as a coherent, plausible commercial product package, considering layout, brand-mark presence, product-information regions, and overall design discipline. Does NOT require resemblance to any specific real brand.

|Score|Descriptor|
|:-:|-|
|**0**|Output does not read as a product package — no recognisable brand mark, no product-information regions, or chaotic layout.|
|**1**|Output reads as a package but lacks coherence: missing or warped key elements (no clear brand area, no product description, indistinct product photography).|
|**2**|Output has the major structural elements of a product package (brand area, product imagery, descriptive text region) but at least one is weak or under-formed.|
|**3**|Output is a coherent, commercially plausible package design with all major structural elements present and well-formed.|

## Axis 4 — Overall Visual Quality

Scores the technical image quality of the output: artefacts, distortions, anatomy of objects (chips/snacks shape), composition, lighting consistency.

|Score|Descriptor|
|:-:|-|
|**0**|Severe artefacts: warped geometry, melted/deformed objects, broken composition, obvious diffusion-model failure modes.|
|**1**|Notable artefacts present but the overall image is parseable; visible diffusion-model issues in textures or object boundaries.|
|**2**|Minor artefacts only; image quality is broadly acceptable; small flaws in textures or edges.|
|**3**|Clean, polished output indistinguishable in technical quality from professional product photography.|

\---

## Aggregation and reporting

* Per-image scores are reported as the **mean of the two sessions** rounded to one decimal place.
* Per-configuration scores are reported as the **median across images** per axis, with interquartile range.
* Intra-rater agreement is reported per axis as **Cohen's weighted kappa** with linear weights, on the raw (un-averaged) session scores.
* Configurations are compared per axis using non-parametric tests (Mann–Whitney U for pairwise; Kruskal–Wallis for omnibus) where statistical comparison is reported.

## Known limitations of this evaluation method

These are acknowledged in advance and discussed in the dissertation's limitations chapter:

1. **Single-rater subjectivity.** The author is the sole evaluator. Intra-rater agreement is reported as a transparency measure but does not substitute for inter-rater agreement.
2. **Author cultural position.** The author's familiarity with the three target states is not uniform. This may bias scoring on Axis 2 (Regional Appropriateness). The reference-pool-anchored definition of Axis 2 partially mitigates but does not eliminate this.
3. **Anchoring to the reference pool.** Axis 2 explicitly measures alignment to the curated reference pool, not to a broader notion of cultural authenticity. The dissertation discusses this design choice.
4. **No human-subject validation.** Scores cannot be claimed to represent how target-market consumers would evaluate the outputs. The evaluation supports methodological comparison across conditioning configurations, not claims about consumer reception.

\---

## Sign-off

Committed to version control on the date of the first git/onedrive commit of this file. The hash/timestamp of that commit is reported in the dissertation methodology chapter.

