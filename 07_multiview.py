
from pathlib import Path

import json
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "Datasets" / "Processed"

OUTPUT_DIR = PROCESSED_DIR / "MultiView"
VIEWS_DIR = OUTPUT_DIR / "processed_views"
KERNEL_DIR = OUTPUT_DIR / "kernels"
ASSIGNMENTS_DIR = OUTPUT_DIR / "cluster_assignments"
METRICS_DIR = OUTPUT_DIR / "metrics"

for directory in [
    VIEWS_DIR,
    KERNEL_DIR,
    ASSIGNMENTS_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

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
    """Normalize participant IDs to strings."""
    result = series.astype("string").str.strip()
    return result.str.replace(r"\.0$", "", regex=True)


clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
csf = pd.read_csv(CSF_FILE, low_memory=False)
plasma = pd.read_csv(PLASMA_FILE, low_memory=False)

for name, df in [
    ("clinical", clinical),
    ("CSF", csf),
    ("plasma", plasma),
]:
    if ID_COL not in df.columns:
        raise ValueError(f"{name} is missing {ID_COL}")

    df[ID_COL] = normalize_id(df[ID_COL])

    if df[ID_COL].isna().any():
        raise ValueError(f"{name} contains missing participant IDs.")

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate participant IDs."
        )

    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# 3. Align participants across all three views
# ============================================================

common_ids = sorted(
    set(clinical[ID_COL])
    & set(csf[ID_COL])
    & set(plasma[ID_COL])
)

if len(common_ids) != 206:
    warnings.warn(
        f"Expected 206 aligned participants; found {len(common_ids)}."
    )

def index_by_id(df):
    return df.set_index(ID_COL).loc[common_ids]


clinical = index_by_id(clinical)
csf = index_by_id(csf)
plasma = index_by_id(plasma)

assert clinical.index.equals(csf.index)
assert clinical.index.equals(plasma.index)

print(f"\nAligned participants: {len(common_ids)}")

if len(common_ids) < 3:
    raise ValueError("Too few aligned participants for clustering.")


# ============================================================
# 4. Preprocess clinical/demographic view
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

for col in clinical_cols:
    clinical_view[col] = pd.to_numeric(
        clinical_view[col], errors="coerce"
    )

if clinical_view.isna().any().any():
    raise ValueError(
        "Clinical view has missing/nonnumeric values. "
        "Inspect them before proceeding."
    )

# Encode SEX as categorical rather than continuous.
sex_view = pd.get_dummies(
    clinical_view["SEX"].astype(int),
    prefix="SEX",
    dtype=float
)

if sex_view.shape[1] < 2:
    raise ValueError(
        "Fewer than two SEX categories found. Check coding."
    )

clinical_continuous = clinical_view[["age", "EDUCYRS"]]

clinical_scaled = pd.DataFrame(
    StandardScaler().fit_transform(clinical_continuous),
    index=clinical.index,
    columns=["age", "EDUCYRS"]
)

clinical_processed = pd.concat(
    [
        clinical_scaled,
        sex_view.set_axis(clinical.index),
    ],
    axis=1
)

print("\nClinical features:", clinical_processed.shape[1])


# ============================================================
# 5. Preprocess CSF and plasma views independently
# ============================================================

def preprocess_proteomics(df, view_name):
    feature_cols = [
        col for col in df.columns
        if col not in KEY_COLS
    ]

    X = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove features entirely missing across participants.
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    # Remove features with no variation among observed values.
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError(
            f"{view_name} has no usable protein features."
        )

    missing_before = int(X.isna().sum().sum())

    # Median imputation.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Standardize each protein across participants.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # SimpleImputer may change the number of columns depending
    # on sklearn version/configuration. Recover names safely.
    retained_cols = X.columns[
        imputer.get_feature_names_out()
    ] if False else X.columns

    if X_scaled.shape[1] != len(retained_cols):
        raise ValueError(
            f"{view_name}: feature names and imputed matrix "
            "have different lengths."
        )

    result = pd.DataFrame(
        X_scaled,
        index=df.index,
        columns=retained_cols
    )

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            f"{view_name} has non-finite scaled values."
        )

    summary = {
        "view": view_name,
        "input_features": len(feature_cols),
        "all_missing_removed": len(all_missing),
        "constant_removed": len(constant_cols),
        "retained_features": result.shape[1],
        "missing_cells_before_imputation": missing_before,
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
# 6. Save processed views separately
# ============================================================

def save_view(df, path):
    output = df.copy()
    output.insert(0, ID_COL, output.index)
    output.to_csv(path, index=False)


save_view(
    clinical_processed,
    VIEWS_DIR / "baseline_clinical_processed.csv"
)

save_view(
    csf_processed,
    VIEWS_DIR / "baseline_csf_processed.csv"
)

save_view(
    plasma_processed,
    VIEWS_DIR / "baseline_plasma_processed.csv"
)


# ============================================================
# 7. Construct an RBF kernel for each view
# ============================================================

def build_rbf_kernel(X_df):
    """
    Convert a participant-by-feature matrix into a
    participant-by-participant RBF similarity matrix.

    The median heuristic determines the RBF bandwidth.
    """
    X = X_df.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("Kernel input contains non-finite values.")

    D2 = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    upper = D2[np.triu_indices_from(D2, k=1)]
    positive = upper[upper > 0]

    if len(positive) == 0:
        raise ValueError(
            "All participants have identical feature profiles."
        )

    gamma = 1.0 / np.median(positive)
    K = np.exp(-gamma * D2)

    return K, float(gamma)


K_clinical, gamma_clinical = build_rbf_kernel(
    clinical_processed
)

K_csf, gamma_csf = build_rbf_kernel(
    csf_processed
)

K_plasma, gamma_plasma = build_rbf_kernel(
    plasma_processed
)

kernels = {
    "clinical": K_clinical,
    "csf": K_csf,
    "plasma": K_plasma,
}

gamma_values = {
    "clinical": gamma_clinical,
    "csf": gamma_csf,
    "plasma": gamma_plasma,
}

print("\nKernel diagnostics:")

for name, K in kernels.items():
    off_diagonal = K[np.triu_indices_from(K, k=1)]

    print(
        f"{name}: shape={K.shape}, "
        f"gamma={gamma_values[name]:.6g}, "
        f"mean off-diagonal similarity="
        f"{off_diagonal.mean():.4f}"
    )


# ============================================================
# 8. Combine the three kernels
# ============================================================

# Initial baseline: equal weight for each data view.
weights = {
    "clinical": 1 / 3,
    "csf": 1 / 3,
    "plasma": 1 / 3,
}

K_combined = sum(
    weights[name] * kernels[name]
    for name in kernels
)

assert K_combined.shape == (
    len(common_ids), len(common_ids)
)

assert np.allclose(K_combined, K_combined.T)

# Save each kernel with participant IDs on both axes.
for name, K in {
    **kernels,
    "combined_equal": K_combined,
}.items():
    pd.DataFrame(
        K,
        index=common_ids,
        columns=common_ids
    ).to_csv(
        KERNEL_DIR / f"kernel_{name}.csv",
        index_label=ID_COL
    )


# ============================================================
# 9. Spectral clustering and silhouette evaluation
# ============================================================

def run_kernel_benchmark(K, experiment_name):
    results = []

    assignments = pd.DataFrame({
        ID_COL: common_ids
    })

    # Euclidean distances in the feature space represented
    # by the kernel. These are used for silhouette scoring.
    diagonal = np.diag(K)

    D2 = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2 * K
    )

    D = np.sqrt(np.maximum(D2, 0))
    np.fill_diagonal(D, 0)

    for k in K_VALUES:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=20,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(K)

        counts = pd.Series(labels).value_counts()

        score = silhouette_score(
            D,
            labels,
            metric="precomputed"
        )

        results.append({
            "experiment": experiment_name,
            "k": k,
            "silhouette": float(score),
            "smallest_cluster": int(counts.min()),
            "largest_cluster": int(counts.max()),
            "n_clusters_observed": int(counts.size),
        })

        assignments[f"k{k}"] = labels

        print(
            f"{experiment_name}: k={k}, "
            f"silhouette={score:.4f}, "
            f"cluster_sizes={counts.sort_index().to_dict()}"
        )

    assignments.to_csv(
        ASSIGNMENTS_DIR
        / f"{experiment_name}_assignments.csv",
        index=False
    )

    return results


print("\nRunning multiview spectral clustering...")

all_results = []

# Single-view kernel benchmarks
for name, K in kernels.items():
    all_results.extend(
        run_kernel_benchmark(
            K,
            f"kernel_{name}"
        )
    )

# Integrated three-view benchmark
all_results.extend(
    run_kernel_benchmark(
        K_combined,
        "kernel_combined_equal"
    )
)

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    METRICS_DIR / "kernel_clustering_metrics.csv",
    index=False
)

print("\n" + "=" * 65)
print("MULTIVIEW CLUSTERING RESULTS")
print("=" * 65)
print(results_df.to_string(index=False))


# ============================================================
# 10. Save metadata and preprocessing information
# ============================================================

metadata = {
    "n_participants": len(common_ids),
    "participant_ids": common_ids,
    "clinical_features": clinical_processed.columns.tolist(),
    "csf_features": csf_processed.columns.tolist(),
    "plasma_features": plasma_processed.columns.tolist(),
    "kernel_weights": weights,
    "rbf_gamma": gamma_values,
    "cluster_counts_tested": list(K_VALUES),
    "random_state": RANDOM_STATE,
    "clustering_method": "SpectralClustering",
    "kernel_method": "RBF with median-distance heuristic",
    "clinical_preprocessing": {
        "continuous_features": ["age", "EDUCYRS"],
        "categorical_features": ["SEX"],
        "scaling": "StandardScaler for continuous features",
    },
    "csf_preprocessing": csf_summary,
    "plasma_preprocessing": plasma_summary,
    "excluded_from_clustering": [
        "MOCA",
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
}

with open(
    OUTPUT_DIR / "multiview_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\nAll multiview outputs saved to:")
print(OUTPUT_DIR)


# ============================================================
# 11. Kernel agreement and weight sensitivity
# ============================================================

from itertools import combinations

# ------------------------------------------------------------
# 11A. Compare pairwise similarity patterns
# ------------------------------------------------------------

print("\nKernel similarity diagnostics")

upper_idx = np.triu_indices(len(common_ids), k=1)

for name, K in kernels.items():
    values = K[upper_idx]

    print(
        f"{name}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
    )

print("\nCorrelation between kernel similarity patterns")

for name_a, name_b in combinations(kernels.keys(), 2):
    values_a = kernels[name_a][upper_idx]
    values_b = kernels[name_b][upper_idx]

    corr = np.corrcoef(values_a, values_b)[0, 1]

    print(f"{name_a} vs {name_b}: r={corr:.4f}")


# ------------------------------------------------------------
# 11B. Explore prespecified weighting schemes
# ------------------------------------------------------------

weight_schemes = {
    "equal": {
        "clinical": 1/3,
        "csf": 1/3,
        "plasma": 1/3,
    },
    "csf_dominant": {
        "clinical": 0.20,
        "csf": 0.60,
        "plasma": 0.20,
    },
    "csf_clinical": {
        "clinical": 0.40,
        "csf": 0.50,
        "plasma": 0.10,
    },
    "clinical_csf": {
        "clinical": 0.50,
        "csf": 0.40,
        "plasma": 0.10,
    },
    "no_plasma": {
        "clinical": 0.50,
        "csf": 0.50,
        "plasma": 0.00,
    },
}

weight_results = []

for scheme_name, scheme in weight_schemes.items():

    K_weighted = sum(
        scheme[name] * kernels[name]
        for name in kernels
    )

    for k in [2, 3, 4]:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=20,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(K_weighted)

        diag = np.diag(K_weighted)

        D2 = (
            diag[:, None]
            + diag[None, :]
            - 2 * K_weighted
        )

        D = np.sqrt(np.maximum(D2, 0))
        np.fill_diagonal(D, 0)

        counts = np.bincount(labels, minlength=k)

        score = silhouette_score(
            D,
            labels,
            metric="precomputed"
        )

        weight_results.append({
            "scheme": scheme_name,
            "clinical_weight": scheme["clinical"],
            "csf_weight": scheme["csf"],
            "plasma_weight": scheme["plasma"],
            "k": k,
            "silhouette": score,
            "smallest_cluster": int(counts.min()),
            "largest_cluster": int(counts.max()),
        })

weight_results_df = pd.DataFrame(weight_results)

weight_results_df.to_csv(
    METRICS_DIR / "kernel_weight_sensitivity.csv",
    index=False
)

print("\nWeight sensitivity results:")
print(
    weight_results_df.sort_values(
        ["k", "silhouette"],
        ascending=[True, False]
    ).to_string(index=False)
)

from itertools import combinations
from sklearn.metrics import adjusted_rand_score

# Compare the two-cluster assignments from each weighting scheme.
schemes_to_compare = [
    "equal",
    "csf_dominant",
    "csf_clinical",
    "no_plasma",
]

labels_by_scheme = {}

for scheme_name in schemes_to_compare:
    scheme = weight_schemes[scheme_name]

    K_weighted = sum(
        scheme[view_name] * kernels[view_name]
        for view_name in kernels
    )

    model = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=50,
        random_state=RANDOM_STATE,
    )

    labels_by_scheme[scheme_name] = model.fit_predict(K_weighted)

# Pairwise agreement
agreement_rows = []

for name_a, name_b in combinations(schemes_to_compare, 2):
    ari = adjusted_rand_score(
        labels_by_scheme[name_a],
        labels_by_scheme[name_b],
    )

    agreement_rows.append({
        "scheme_a": name_a,
        "scheme_b": name_b,
        "adjusted_rand_index": ari,
    })

agreement_df = pd.DataFrame(agreement_rows)
agreement_df.to_csv(
    METRICS_DIR / "k2_assignment_agreement.csv",
    index=False,
)

print("\nTwo-cluster assignment agreement:")
print(agreement_df.to_string(index=False))

# Save participant-level assignments for downstream analysis
assignment_df = pd.DataFrame(
    {"PATNO": common_ids}
)

for scheme_name, labels in labels_by_scheme.items():
    assignment_df[f"cluster_{scheme_name}"] = labels

assignment_df.to_csv(
    ASSIGNMENTS_DIR / "k2_weight_scheme_assignments.csv",
    index=False,
)