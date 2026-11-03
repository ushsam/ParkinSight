
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("Datasets/Processed/Proteomics")
BASELINE_DIR = Path("Datasets/Processed")
OUT_DIR = INPUT_DIR / "CorrelationFiltered"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = {
    "csf": {
        "filtered": INPUT_DIR / "baseline_csf_filtered.csv",
        "raw": BASELINE_DIR / "baseline_csf_raw.csv",
    },
    "plasma": {
        "filtered": INPUT_DIR / "baseline_plasma_filtered.csv",
        "raw": BASELINE_DIR / "baseline_plasma_raw.csv",
    },
}

KEYS = ["PATNO", "EVENT_ID"]
CORRELATION_THRESHOLD = 0.95


# ============================================================
# Correlation filtering
# ============================================================

def filter_correlated_proteins(view_name, filtered_path, raw_path):

    print(f"\n{'=' * 60}")
    print(f"CORRELATION FILTERING: {view_name.upper()}")
    print(f"{'=' * 60}")

    # 1. Load previously filtered data
    df = pd.read_csv(filtered_path)
    raw_df = pd.read_csv(raw_path)

    ids = df[KEYS].copy()

    # Verify participant alignment with original baseline data
    if not df[KEYS].equals(raw_df[KEYS]):
        raise ValueError(
            f"{view_name}: Participant IDs/order do not match "
            "between filtered and raw baseline data."
        )

    features = df.drop(columns=KEYS).select_dtypes(
        include="number"
    ).copy()

    if features.isna().any().any():
        raise ValueError(
            f"{view_name}: Missing values found in filtered data."
        )

    n_initial = features.shape[1]

    print(f"Participants: {len(df)}")
    print(f"Initial proteins: {n_initial}")

    # 2. Calculate protein variances
    variances = features.var(ddof=0)

    # 3. Calculate original missingness for tie-breaking
    raw_features = raw_df.reindex(columns=features.columns)

    missing_rates = raw_features.isna().mean()

    # 4. Calculate Pearson correlation matrix
    print("Calculating pairwise Pearson correlations...")

    corr_matrix = features.corr(method="pearson")

    # Convert to absolute correlations for filtering
    abs_corr = corr_matrix.abs()

    # 5. Generate a complete report of highly correlated pairs
    upper_mask = np.triu(
        np.ones(abs_corr.shape, dtype=bool),
        k=1
    )

    upper_corr = abs_corr.where(upper_mask)

    pair_rows = []

    for protein_a in upper_corr.index:
        correlated = upper_corr.loc[protein_a].dropna()

        correlated = correlated[
            correlated > CORRELATION_THRESHOLD
        ]

        for protein_b, corr_value in correlated.items():
            pair_rows.append({
                "protein_a": protein_a,
                "protein_b": protein_b,
                "pearson_correlation": corr_matrix.loc[
                    protein_a, protein_b
                ],
                "absolute_correlation": corr_value,
            })

    pairs_report = pd.DataFrame(
        pair_rows,
        columns=[
            "protein_a",
            "protein_b",
            "pearson_correlation",
            "absolute_correlation",
        ]
    )

    pairs_report.to_csv(
        OUT_DIR / f"{view_name}_high_correlation_pairs.csv",
        index=False
    )

    # 6. Greedy feature selection
    # Priority:
    #   1. Higher variance
    #   2. Lower original missingness
    #   3. Alphabetical protein name

    priority = pd.DataFrame({
        "protein": features.columns,
        "variance": variances.values,
        "missing_rate": missing_rates.reindex(
            features.columns
        ).values,
    })

    priority = priority.sort_values(
        by=["variance", "missing_rate", "protein"],
        ascending=[False, True, True],
        kind="stable"
    )

    ordered_proteins = priority["protein"].tolist()

    kept = []
    removed = []

    for protein in ordered_proteins:

        if not kept:
            kept.append(protein)
            continue

        correlations = abs_corr.loc[protein, kept]

        high_corr = correlations[
            correlations > CORRELATION_THRESHOLD
        ]

        if high_corr.empty:
            kept.append(protein)

        else:
            # Assign the strongest correlated retained protein
            representative = high_corr.idxmax()

            removed.append({
                "removed_protein": protein,
                "retained_protein": representative,
                "pearson_correlation": corr_matrix.loc[
                    protein, representative
                ],
                "absolute_correlation": abs_corr.loc[
                    protein, representative
                ],
                "removed_variance": variances[protein],
                "retained_variance": variances[representative],
                "removed_missing_rate": missing_rates[protein],
                "retained_missing_rate": missing_rates[
                    representative
                ],
            })

    # 7. Construct final feature matrix
    selected_features = features[kept].copy()

    processed_df = pd.concat(
        [
            ids.reset_index(drop=True),
            selected_features.reset_index(drop=True)
        ],
        axis=1
    )

    # 8. Save filtered dataset
    processed_df.to_csv(
        OUT_DIR / f"baseline_{view_name}_correlation_filtered.csv",
        index=False
    )

    # 9. Save removal report
    removal_report = pd.DataFrame(
        removed,
        columns=[
            "removed_protein",
            "retained_protein",
            "pearson_correlation",
            "absolute_correlation",
            "removed_variance",
            "retained_variance",
            "removed_missing_rate",
            "retained_missing_rate",
        ]
    )

    removal_report.to_csv(
        OUT_DIR / f"{view_name}_correlation_removal_report.csv",
        index=False
    )

    # 10. Summary
    n_removed = len(removed)
    n_retained = len(kept)

    summary = {
        "view": view_name,
        "participants": len(df),
        "initial_proteins": n_initial,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "highly_correlated_pairs": len(pair_rows),
        "proteins_removed": n_removed,
        "proteins_retained": n_retained,
    }

    print("\nResults:")
    print(f"  Highly correlated pairs: {len(pair_rows)}")
    print(f"  Proteins removed: {n_removed}")
    print(f"  Proteins retained: {n_retained}")

    return summary


# ============================================================
# Run CSF and plasma independently
# ============================================================

summaries = []

for view_name, paths in VIEWS.items():

    if not paths["filtered"].exists():
        raise FileNotFoundError(paths["filtered"])

    if not paths["raw"].exists():
        raise FileNotFoundError(paths["raw"])

    summary = filter_correlated_proteins(
        view_name,
        paths["filtered"],
        paths["raw"]
    )

    summaries.append(summary)


# ============================================================
# Save overall summary
# ============================================================

summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUT_DIR / "correlation_filtering_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("CORRELATION FILTERING COMPLETE")
print("=" * 60)

print(summary_df.to_string(index=False))

print(f"\nOutputs saved to: {OUT_DIR}")
