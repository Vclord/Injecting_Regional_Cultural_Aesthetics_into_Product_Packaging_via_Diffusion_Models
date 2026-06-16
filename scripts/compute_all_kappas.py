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

EVAL_DIR = Path("evaluation")

SPIKE_FILES = {
    "v2":   "spike_v2_scores.csv",
    "v3":   "spike_v3_scores.csv",
    "v4":   "spike_v4_scores.csv",
    "v5":   "spike_v5_lora_scores.csv",
    "flux": "spike_flux_scores.csv",
}

AXES = [
    "text_legibility",
    "regional_appropriateness",
    "packaging_plausibility",
    "visual_quality",
]

def compute_axis_kappa(df: pd.DataFrame, axis: str):
    """Compute weighted kappa for one axis. Returns (kappa, n_scored, mean_s1, mean_s2)."""
    col_s1 = f"session1_{axis}"
    col_s2 = f"session2_{axis}"

    # If either column is missing from this spike's CSV, axis not applicable
    if col_s1 not in df.columns or col_s2 not in df.columns:
        return None, 0, None, None

    s1 = pd.to_numeric(df[col_s1], errors="coerce")
    s2 = pd.to_numeric(df[col_s2], errors="coerce")

    mask = s1.notna() & s2.notna()
    n = int(mask.sum())
    if n == 0:
        return None, 0, None, None

    s1v, s2v = s1[mask].astype(int), s2[mask].astype(int)

    # If both sessions assigned the same single value to all images,
    # kappa is mathematically undefined (zero variance).
    if s1v.nunique() == 1 and s2v.nunique() == 1 and s1v.iloc[0] == s2v.iloc[0]:
        return float("nan"), n, float(s1v.mean()), float(s2v.mean())

    try:
        k = cohen_kappa_score(s1v, s2v, weights="linear")
    except Exception as e:
        print(f"  kappa computation failed for {axis}: {e}")
        return None, n, float(s1v.mean()), float(s2v.mean())

    return k, n, float(s1v.mean()), float(s2v.mean())


def main():
    print(f"\n{'Spike':<6} {'Axis':<28} {'N':>4}  {'κ':>8}  {'M1':>5}  {'M2':>5}")
    print("-" * 70)

    summary_rows = []
    for spike_name, fname in SPIKE_FILES.items():
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"{spike_name:<6}  CSV not found: {path}")
            continue

        df = pd.read_csv(path)
        print(f"\n--- {spike_name} ({len(df)} rows in CSV) ---")

        spike_row = {"spike": spike_name, "n_rows": len(df)}
        for axis in AXES:
            k, n, m1, m2 = compute_axis_kappa(df, axis)
            if k is None and n == 0:
                k_str = "n/a"  # axis not applicable to this spike
            elif k is None or (isinstance(k, float) and np.isnan(k)):
                k_str = "NaN"  # mathematically undefined (zero variance)
            else:
                k_str = f"{k:.3f}"
            m1_str = f"{m1:.2f}" if m1 is not None else "-"
            m2_str = f"{m2:.2f}" if m2 is not None else "-"
            print(f"{spike_name:<6} {axis:<28} {n:>4}  {k_str:>8}  {m1_str:>5}  {m2_str:>5}")

            spike_row[f"kappa_{axis}"] = k if isinstance(k, float) and not np.isnan(k) else None
            spike_row[f"n_{axis}"] = n
            spike_row[f"mean_s1_{axis}"] = m1
            spike_row[f"mean_s2_{axis}"] = m2

        summary_rows.append(spike_row)

    # Save consolidated table for Appendix C
    summary_df = pd.DataFrame(summary_rows)
    out_path = EVAL_DIR / "kappa_summary_all_spikes.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n\nWritten summary to: {out_path}")
    print("\nUse the rows above to fill Appendix C Table C.1.")


if __name__ == "__main__":
    main()