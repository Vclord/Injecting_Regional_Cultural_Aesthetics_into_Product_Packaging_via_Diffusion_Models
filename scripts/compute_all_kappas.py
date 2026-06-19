"""
prompt to LLM:
Write a Python script (compute_all_kappas.py) using pandas and sklearn to calculate inter-rater reliability for my
dissertation's evaluation data. I need to compute Cohen's kappa (using linear weights) across my spike CSV files
(spike_v2 through flux). For each file, evaluate the four rubric axes (text legibility, regional appropriateness,
packaging plausibility, and visual quality) by comparing the session1_{axis} and session2_{axis} columns.
Make sure to handle missing data cleanly. Crucially, anticipate the zero-variance edge case: if both grading sessions
yield the exact same score for every row on a given axis, kappa is mathematically undefined, so catch that exception
and return NaN. Print a formatted summary table to the console with the kappa scores, sample sizes, and means for both
sessions. Finally, export a consolidated summary dataframe to kappa_summary_all_spikes.csv so I can drop it straight
into Appendix C.
"""
"""
compute_all_kappas.py
=====================
Compute Cohen's weighted kappa (linear weights) for all spike scoring CSVs.
Output: a summary table ready for Appendix C and dissertation prose.

Usage:
    python compute_all_kappas.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

# Directory containing the evaluation CSV files
EVAL_DIR = Path("evaluation")

# Mapping of spike iterations to their corresponding evaluation results
SPIKE_FILES = {
    "v2":   "spike_v2_scores.csv",
    "v3":   "spike_v3_scores.csv",
    "v4":   "spike_v4_scores.csv",
    "v5":   "spike_v5_lora_scores.csv",
    "flux": "spike_flux_scores.csv",
}

# The four rubric axes evaluated by both grading sessions
AXES = [
    "text_legibility",
    "regional_appropriateness",
    "packaging_plausibility",
    "visual_quality",
]


def compute_axis_kappa(df: pd.DataFrame, axis: str) -> tuple:
    """
    Computes Cohen's weighted kappa (linear weights) for a single evaluation axis.

    Args:
        df (pd.DataFrame): The dataframe containing the scoring data.
        axis (str): The specific evaluation metric to compare (e.g., 'visual_quality').

    Returns:
        tuple: A 4-tuple containing:
            - kappa (float or None): The computed kappa score, NaN if mathematically
              undefined, or None if computation fails or columns are missing.
            - n_scored (int): The number of valid overlapping scores between sessions.
            - mean_s1 (float or None): The mean score from session 1.
            - mean_s2 (float or None): The mean score from session 2.
    """
    col_s1 = f"session1_{axis}"
    col_s2 = f"session2_{axis}"

    # If either column is missing from this spike's CSV, axis is not applicable
    if col_s1 not in df.columns or col_s2 not in df.columns:
        return None, 0, None, None

    # Convert to numeric, coercing any malformed text entries to NaN
    s1 = pd.to_numeric(df[col_s1], errors="coerce")
    s2 = pd.to_numeric(df[col_s2], errors="coerce")

    # Filter out rows where either session failed to provide a valid score
    mask = s1.notna() & s2.notna()
    n = int(mask.sum())
    if n == 0:
        return None, 0, None, None

    s1v, s2v = s1[mask].astype(int), s2[mask].astype(int)

    # Edge case: If both sessions assigned the exact same single value to all images,
    # variance is zero and kappa is mathematically undefined.
    if s1v.nunique() == 1 and s2v.nunique() == 1 and s1v.iloc[0] == s2v.iloc[0]:
        return float("nan"), n, float(s1v.mean()), float(s2v.mean())

    try:
        # Linear weighting penalises adjacent rating disagreements less severely than extreme ones
        k = cohen_kappa_score(s1v, s2v, weights="linear")
    except Exception as e:
        print(f"  kappa computation failed for {axis}: {e}")
        return None, n, float(s1v.mean()), float(s2v.mean())

    return k, n, float(s1v.mean()), float(s2v.mean())


def main() -> None:
    """
    Entry point for the script. Iterates over all specified spike CSVs,
    computes the kappa statistics for each axis, prints a formatted table,
    and exports a consolidated summary for dissertation appendices.
    """
    # Print table header
    print(f"\n{'Spike':<6} {'Axis':<28} {'N':>4}  {'κ':>8}  {'M1':>5}  {'M2':>5}")
    print("-" * 70)

    summary_rows = []

    # Process each spike experiment individually
    for spike_name, fname in SPIKE_FILES.items():
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"{spike_name:<6}  CSV not found: {path}")
            continue

        df = pd.read_csv(path)
        print(f"\n--- {spike_name} ({len(df)} rows in CSV) ---")

        # Initialise the summary dictionary for the current spike
        spike_row = {"spike": spike_name, "n_rows": len(df)}

        # Compute agreement metrics across all 4 rubric axes
        for axis in AXES:
            k, n, m1, m2 = compute_axis_kappa(df, axis)

            # Format kappa output based on availability and mathematical validity
            if k is None and n == 0:
                k_str = "n/a"  # axis not applicable to this spike
            elif k is None or (isinstance(k, float) and np.isnan(k)):
                k_str = "NaN"  # mathematically undefined (zero variance)
            else:
                k_str = f"{k:.3f}"

            # Format mean outputs
            m1_str = f"{m1:.2f}" if m1 is not None else "-"
            m2_str = f"{m2:.2f}" if m2 is not None else "-"

            print(f"{spike_name:<6} {axis:<28} {n:>4}  {k_str:>8}  {m1_str:>5}  {m2_str:>5}")

            # Store clean data for CSV export
            spike_row[f"kappa_{axis}"] = k if isinstance(k, float) and not np.isnan(k) else None
            spike_row[f"n_{axis}"] = n
            spike_row[f"mean_s1_{axis}"] = m1
            spike_row[f"mean_s2_{axis}"] = m2

        summary_rows.append(spike_row)

    # Save consolidated table for Appendix C integration
    summary_df = pd.DataFrame(summary_rows)
    out_path = EVAL_DIR / "kappa_summary_all_spikes.csv"
    summary_df.to_csv(out_path, index=False)

    print(f"\n\nWritten summary to: {out_path}")
    print("\nUse the rows above to fill Appendix C Table C.1.")


if __name__ == "__main__":
    main()