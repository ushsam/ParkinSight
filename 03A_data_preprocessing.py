
from pathlib import Path
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("Datasets/Processed")
OUT_DIR = INPUT_DIR / "Proteomics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = {
    "csf": INPUT_DIR / "baseline_csf_raw.csv",
    "plasma": INPUT_DIR / "baseline_plasma_raw.csv",
}

KEYS = ["PATNO", "EVENT_ID"]

MAX_MISSINGNESS = 0.20


# ============================================================
# Preprocessing function
# ============================================================

def preprocess_view(view_name, file_path):

    print(f"\n{'=' * 55}")
    print(f"Processing {view_name.upper()}")
    print(f"{'=' * 55}")

    # 1. Load baseline data
    df = pd.read_csv(file_path)

    for col in KEYS:
        if col not in df.columns:
            raise ValueError(f"{view_name}: Missing identifier {col}")

    # Preserve participant identifiers
    ids = df[KEYS].copy()

    # Select numeric protein abundance features
    feature_df = df.drop(columns=KEYS)

    # Exclude missingness indicator columns (e.g., *_MISS)
    miss_columns = [
        col for col in feature_df.columns
        if str(col).upper().endswith("_MISS")
    ]

    if miss_columns:
        print(
            f"Excluding {len(miss_columns)} missingness indicator columns."
        )

    feature_df = feature_df.drop(columns=miss_columns)

    # Exclude non-numeric columns
    non_numeric = feature_df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        print(f"Excluding {len(non_numeric)} non-numeric columns.")

    feature_df = feature_df.select_dtypes(
        include="number"
    ).copy()

    if feature_df.empty:
        raise ValueError(f"{view_name}: No numeric features found.")

    n_initial = feature_df.shape[1]

    print(f"Participants: {len(df)}")
    print(f"Initial protein features: {n_initial}")

    # 2. Calculate missingness
    missing_fraction = feature_df.isna().mean()

    # Remove proteins exceeding missingness threshold
    keep_missing = missing_fraction <= MAX_MISSINGNESS

    removed_missing = feature_df.columns[~keep_missing].tolist()

    feature_df = feature_df.loc[:, keep_missing]

    print(f"Removed for missingness: {len(removed_missing)}")

    if feature_df.empty:
        raise ValueError(
            f"{view_name}: No proteins remain after missingness filtering."
        )

    # 3. Median imputation
    missing_before_imputation = int(feature_df.isna().sum().sum())

    medians = feature_df.median()

    feature_df = feature_df.fillna(medians)

    remaining_missing = int(feature_df.isna().sum().sum())

    assert remaining_missing == 0

    print(f"Missing values imputed: {missing_before_imputation}")

    # 4. Remove zero-variance proteins
    variances = feature_df.var(ddof=0)

    keep_variance = variances > 0

    removed_zero_variance = feature_df.columns[
        ~keep_variance
    ].tolist()

    feature_df = feature_df.loc[:, keep_variance]

    print(f"Removed for zero variance: {len(removed_zero_variance)}")

    if feature_df.empty:
        raise ValueError(
            f"{view_name}: No proteins remain after variance filtering."
        )

    # 5. Flag near-zero-variance proteins (do not remove)
    variances = feature_df.var(ddof=0)

    variance_cutoff = variances.quantile(0.05)

    low_variance_flags = variances <= variance_cutoff

    print(f"Flagged for low variance: {low_variance_flags.sum()}")

    # 6. Reattach participant identifiers
    processed_df = pd.concat(
        [
            ids.reset_index(drop=True),
            feature_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ========================================================
    # Feature-level report
    # ========================================================

    report = pd.DataFrame(index=df.drop(columns=KEYS).select_dtypes(
        include="number"
    ).columns)

    report.index.name = "protein"

    report["missing_fraction"] = missing_fraction
    report["variance_after_imputation"] = (
        feature_df.var(ddof=0).reindex(report.index)
    )

    report["filter_status"] = "retained"

    report.loc[
        removed_missing, "filter_status"
    ] = "removed_missingness"

    report.loc[
        removed_zero_variance, "filter_status"
    ] = "removed_zero_variance"

    report["low_variance_flag"] = False

    report.loc[
        low_variance_flags.index[low_variance_flags],
        "low_variance_flag"
    ] = True

    # Save feature-level report
    report.to_csv(
        OUT_DIR / f"{view_name}_filtering_report.csv"
    )

    # Save processed matrix
    processed_df.to_csv(
        OUT_DIR / f"baseline_{view_name}_filtered.csv",
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    n_retained = feature_df.shape[1]

    summary = {
        "view": view_name,
        "participants": len(df),
        "initial_features": n_initial,
        "removed_missingness": len(removed_missing),
        "missing_values_imputed": missing_before_imputation,
        "removed_zero_variance": len(removed_zero_variance),
        "flagged_low_variance": int(low_variance_flags.sum()),
        "final_features": n_retained,
    }

    print("\nFinal results:")
    print(f"  Initial proteins: {n_initial}")
    print(f"  Removed (missingness): {len(removed_missing)}")
    print(f"  Imputed values: {missing_before_imputation}")
    print(f"  Removed (zero variance): {len(removed_zero_variance)}")
    print(f"  Flagged (low variance): {low_variance_flags.sum()}")
    print(f"  Retained proteins: {n_retained}")

    return summary


# ============================================================
# Run CSF and plasma independently
# ============================================================

summaries = []

for view_name, file_path in VIEWS.items():

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    summary = preprocess_view(view_name, file_path)
    summaries.append(summary)


# Save overall summary
summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUT_DIR / "proteomics_preprocessing_summary.csv",
    index=False
)

print("\n" + "=" * 55)
print("PROTEOMICS PREPROCESSING COMPLETE")
print("=" * 55)

print(summary_df.to_string(index=False))

print(f"\nOutput directory: {OUT_DIR}")
