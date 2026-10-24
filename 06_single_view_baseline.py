
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Paths and settings
# ============================================================
ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "Datasets" / "Processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROCESSED_DIR / "SingleView"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = PROCESSED_DIR / "baseline_clinical_raw.csv"
CSF_FILE = PROCESSED_DIR / "baseline_csf_raw.csv"
PLASMA_FILE = PROCESSED_DIR / "baseline_plasma_raw.csv"

RANDOM_STATE = 42
K_VALUES = range(2, 7)

ID_COL = "PATNO"
KEY_COLS = ["PATNO", "EVENT_ID"]

# ============================================================
# 2. Load and normalize participant IDs
# ============================================================

def normalize_id(series):
    """Normalize IDs to strings without accidental trailing .0."""
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\.0$", "", regex=True)
    return result


clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
csf = pd.read_csv(CSF_FILE, low_memory=False)
plasma = pd.read_csv(PLASMA_FILE, low_memory=False)

for name, df in [
    ("clinical", clinical),
    ("CSF", csf),
    ("plasma", plasma),
]:
    if ID_COL not in df.columns:
        raise ValueError(f"{name} data is missing {ID_COL}")

    df[ID_COL] = normalize_id(df[ID_COL])

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{name} baseline data has duplicate participant IDs."
        )

    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 3. Align the same participants across all three views
# ============================================================

common_ids = (
    set(clinical[ID_COL].dropna())
    & set(csf[ID_COL].dropna())
    & set(plasma[ID_COL].dropna())
)

common_ids = sorted(common_ids)

if len(common_ids) != 206:
    warnings.warn(
        f"Expected 206 complete baseline participants; "
        f"found {len(common_ids)}. Verify cohort alignment."
    )

def index_by_id(df):
    return df.set_index(ID_COL).loc[common_ids]

clinical = index_by_id(clinical)
csf = index_by_id(csf)
plasma = index_by_id(plasma)

assert clinical.index.equals(csf.index)
assert clinical.index.equals(plasma.index)

print(f"\nAligned participant count: {len(common_ids)}")

# ============================================================
# 4. Build the clinical/demographic view
# ============================================================

clinical_cols = ["age", "SEX", "EDUCYRS"]

missing_cols = [
    col for col in clinical_cols
    if col not in clinical.columns
]
if missing_cols:
    raise ValueError(
        f"Missing expected clinical columns: {missing_cols}"
    )

clinical_view = clinical[clinical_cols].copy()

for col in ["age", "EDUCYRS", "SEX"]:
    clinical_view[col] = pd.to_numeric(
        clinical_view[col], errors="coerce"
    )

if clinical_view.isna().any().any():
    raise ValueError(
        "Clinical view has missing or nonnumeric values. "
        "Inspect before proceeding."
    )

# SEX is categorical, not a continuous measurement.
sex_view = pd.get_dummies(
    clinical_view["SEX"].astype(int),
    prefix="SEX",
    dtype=float
)

# Use an explicit reference category when both categories are present.
if sex_view.shape[1] < 2:
    raise ValueError(
        "Expected two observed SEX categories; check coding."
    )

clinical_continuous = clinical_view[["age", "EDUCYRS"]]

clinical_scaled = pd.DataFrame(
    StandardScaler().fit_transform(clinical_continuous),
    index=clinical.index,
    columns=["age", "EDUCYRS"]
)

clinical_processed = pd.concat(
    [clinical_scaled, sex_view.set_axis(clinical.index)],
    axis=1
)

# ============================================================
# 5. Preprocess CSF and plasma protein views
# ============================================================

def preprocess_proteomics(df, view_name):
    # Do not use participant IDs or visit identifiers as features.
    feature_cols = [
        col for col in df.columns
        if col not in KEY_COLS
    ]

    X = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove entirely missing and constant features.
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    # Remove features with no variation among observed values.
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError(f"{view_name} has no usable features.")

    # Median imputation, followed by per-protein standardization.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    result = pd.DataFrame(
        X_scaled,
        index=df.index,
        columns=X.columns
    )

    # A final guard against numerical issues.
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            f"{view_name} contains non-finite scaled values."
        )

    summary = {
        "view": view_name,
        "input_features": len(feature_cols),
        "all_missing_removed": len(all_missing),
        "constant_removed": len(constant_cols),
        "retained_features": result.shape[1],
        "missing_cells_before_imputation": int(X.isna().sum().sum()),
    }

    print(f"\n{view_name} preprocessing:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return result, summary


csf_processed, csf_summary = preprocess_proteomics(
    csf, "CSF"
)
plasma_processed, plasma_summary = preprocess_proteomics(
    plasma, "plasma"
)

# ============================================================
# 6. Save the individual processed views
# ============================================================

def save_view(df, filename):
    output = df.copy()
    output.insert(0, ID_COL, output.index)
    output.to_csv(OUTPUT_DIR / filename, index=False)

save_view(
    clinical_processed,
    "baseline_clinical_processed.csv"
)
save_view(
    csf_processed,
    "baseline_csf_processed.csv"
)
save_view(
    plasma_processed,
    "baseline_plasma_processed.csv"
)

# ============================================================
# 7. Early fusion A: concatenate standardized features
# ============================================================

# Each feature has unit variance, but the proteomics views have
# thousands of features and may dominate total distance.
early_unbalanced = pd.concat(
    [
        clinical_processed,
        csf_processed,
        plasma_processed,
    ],
    axis=1
)

save_view(
    early_unbalanced,
    "early_fusion_standardized.csv"
)

# ============================================================
# 8. Early fusion B: balance the three views
# ============================================================

# Dividing each view by sqrt(number of features) makes its
# squared-distance contribution closer to a per-feature average.
# This is a sensitivity-analysis baseline, not a learned weight.
def balance_view(df):
    return df / np.sqrt(df.shape[1])

clinical_balanced = balance_view(clinical_processed)
csf_balanced = balance_view(csf_processed)
plasma_balanced = balance_view(plasma_processed)

early_balanced = pd.concat(
    [
        clinical_balanced,
        csf_balanced,
        plasma_balanced,
    ],
    axis=1
)

save_view(
    early_balanced,
    "early_fusion_view_balanced.csv"
)

# ============================================================
# 9. Initial K-means benchmark
# ============================================================

def run_kmeans_benchmark(X_df, experiment_name):
    X = X_df.to_numpy(dtype=float)
    results = []
    assignments = pd.DataFrame(
        {ID_COL: X_df.index}
    )

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            metric="euclidean"
        )

        counts = pd.Series(labels).value_counts().sort_index()

        results.append({
            "experiment": experiment_name,
            "k": k,
            "silhouette": score,
            "inertia": model.inertia_,
            "smallest_cluster": int(counts.min()),
            "largest_cluster": int(counts.max()),
        })

        assignments[f"k{k}"] = labels

        print(
            f"{experiment_name}: k={k}, "
            f"silhouette={score:.4f}, "
            f"cluster_sizes={counts.to_dict()}"
        )

    assignments.to_csv(
        OUTPUT_DIR
        / f"{experiment_name}_cluster_assignments.csv",
        index=False
    )

    return results


print("\nRunning initial K-means benchmarks...")

benchmark_results = []

benchmark_results.extend(
    run_kmeans_benchmark(
        early_unbalanced,
        "early_unbalanced"
    )
)

benchmark_results.extend(
    run_kmeans_benchmark(
        early_balanced,
        "early_view_balanced"
    )
)

results_df = pd.DataFrame(benchmark_results)
results_df.to_csv(
    OUTPUT_DIR/ "early_fusion_clustering_metrics.csv",
    index=False
)

# ============================================================
# 10. Save preprocessing metadata
# ============================================================

metadata = {
    "n_participants": len(common_ids),
    "clinical_features": clinical_processed.shape[1],
    "csf_features": csf_processed.shape[1],
    "plasma_features": plasma_processed.shape[1],
    "early_fusion_unbalanced_features": early_unbalanced.shape[1],
    "early_fusion_balanced_features": early_balanced.shape[1],
    "clinical_columns_used": clinical_processed.columns.tolist(),
    "excluded_from_clustering": [
        "moca",
        "GP2_proxy_PGS",
        "META5_proxy_PGS",
        "CSFSAA",
        "CSFSAA_assay",
        "PRIMDIAG",
        "COHORT",
        "subgroup",
        "PATNO",
        "EVENT_ID",
    ],
    "random_state": RANDOM_STATE,
    "csf_preprocessing": csf_summary,
    "plasma_preprocessing": plasma_summary,
}

with open(
    OUTPUT_DIR / "early_fusion_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 65)
print("EARLY-FUSION BASELINE COMPLETE")
print("=" * 65)
print(results_df.to_string(index=False))
print("\nSaved processed matrices, assignments, metrics, and metadata to:")
print(OUTPUT_DIR)


