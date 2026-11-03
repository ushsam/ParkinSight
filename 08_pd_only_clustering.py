
"""
12_pd_only_clustering.py

PD-only single-view and multiview clustering analysis.

Workflow:
1. Identify diagnosed PD participants using the curated PPMI subgroup field.
2. Filter baseline clinical, CSF, and plasma data to the PD cohort.
3. Prepare clinical features.
4. Filter and preprocess CSF and plasma proteins.
5. Refit PCA separately for CSF and plasma.
6. Run single-view and multiview spectral clustering.
7. Evaluate clustering across kernel bandwidths and random seeds.
8. Save processed data, cluster assignments, and evaluation results.

Run from the ParkinSight project root:
    python 12_pd_only_clustering.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "Datasets" / "Raw"
PROCESSED_DIR = ROOT / "Datasets" / "Processed"

OUTPUT_DIR = PROCESSED_DIR / "PD_Only"
VIEW_DIR = OUTPUT_DIR / "Views"
PCA_DIR = OUTPUT_DIR / "PCA"
RESULT_DIR = OUTPUT_DIR / "Results"

for folder in [OUTPUT_DIR, VIEW_DIR, PCA_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# Input files
CURATED_FILE = (
    RAW_DIR / "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
)

COHORT_FILE = (
    PROCESSED_DIR / "baseline_cohort_ids.csv"
)

CLINICAL_FILE = (
    PROCESSED_DIR / "baseline_clinical_raw.csv"
)

CSF_FILE = (
    PROCESSED_DIR / "baseline_csf_raw.csv"
)

PLASMA_FILE = (
    PROCESSED_DIR / "baseline_plasma_raw.csv"
)

CURATED_SHEET = "20260810"

# Exact diagnosed PD subgroup labels
PD_SUBGROUPS = [
    "Sporadic PD",
    "LRRK2 PD",
    "GBA PD",
    "PRKN PD",
    "SNCA PD",
    "LRRK2 and GBA PD",
]

# Proteomics preprocessing
MAX_MISSINGNESS = 0.20
CORRELATION_THRESHOLD = 0.95

# PCA sensitivity analysis
PCA_COMPONENTS = [3, 5, 10]

# Clustering parameters
K_VALUES = [2, 3]
GAMMA_MULTIPLIERS = [0.25, 0.5, 1.0]

# 30 seeds for stability analysis
RANDOM_SEEDS = list(range(42, 72))

# Spectral clustering settings
N_INIT = 20

# Multiview weights: clinical, CSF, plasma
WEIGHT_SCHEMES = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "clinical_dominant": (0.6, 0.2, 0.2),
    "csf_dominant": (0.2, 0.6, 0.2),
    "plasma_dominant": (0.2, 0.2, 0.6),
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_patno(series):
    """Standardize PPMI participant IDs."""

    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


def find_column(df, candidates, required=True):
    """
    Find a column using case-insensitive candidate names.
    """

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_baseline_file(path):
    """Load a baseline CSV and standardize PATNO."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    patno_col = find_column(df, ["PATNO"])

    if patno_col != "PATNO":
        df = df.rename(columns={patno_col: "PATNO"})

    df["PATNO"] = clean_patno(df["PATNO"])

    df = df.dropna(subset=["PATNO"])

    if df["PATNO"].duplicated().any():
        duplicates = df.loc[
            df["PATNO"].duplicated(keep=False), "PATNO"
        ].tolist()

        raise ValueError(
            f"Duplicate PATNO values in {path.name}: {duplicates[:10]}"
        )

    return df


def align_to_cohort(df, patnos, name):
    """
    Align a dataset to the specified participant order.
    Raise an error rather than silently dropping participants.
    """

    indexed = df.set_index("PATNO")

    missing = sorted(set(patnos) - set(indexed.index))

    if missing:
        raise ValueError(
            f"{name} is missing {len(missing)} PD participants. "
            f"Examples: {missing[:10]}"
        )

    return indexed.loc[patnos].reset_index()


# ============================================================
# 3. IDENTIFY PD-ONLY COHORT
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: IDENTIFY PD-ONLY COHORT")
print("=" * 60)

cohort = pd.read_csv(COHORT_FILE)

cohort_patno_col = find_column(cohort, ["PATNO"])
cohort["PATNO"] = clean_patno(cohort[cohort_patno_col])

cohort_patnos = cohort["PATNO"].dropna().unique().tolist()

curated = pd.read_excel(
    CURATED_FILE,
    sheet_name=CURATED_SHEET
)

curated["PATNO"] = clean_patno(curated["PATNO"])

# Keep only baseline records from the original 206-person cohort
curated_baseline = curated[
    curated["PATNO"].isin(cohort_patnos)
    & (curated["EVENT_ID"].astype(str).str.strip() == "BL")
].copy()

if curated_baseline["PATNO"].duplicated().any():
    raise ValueError(
        "Multiple baseline records found for at least one participant."
    )

if "subgroup" not in curated_baseline.columns:
    raise ValueError(
        "The curated dataset does not contain a 'subgroup' column."
    )

print("\nOriginal cohort subgroup counts:")
print(curated_baseline["subgroup"].value_counts(dropna=False))

# Select diagnosed PD participants
pd_cohort = curated_baseline[
    curated_baseline["subgroup"].isin(PD_SUBGROUPS)
].copy()

pd_cohort = pd_cohort.sort_values("PATNO").reset_index(drop=True)

pd_patnos = pd_cohort["PATNO"].tolist()

if len(pd_patnos) < 10:
    raise ValueError(
        f"Only {len(pd_patnos)} PD participants found. "
        "Check the subgroup labels and participant IDs."
    )

print("\nPD-only cohort:")
print("Number of participants:", len(pd_cohort))
print(pd_cohort["subgroup"].value_counts())

pd_cohort[
    ["PATNO", "EVENT_ID", "subgroup"]
].to_csv(
    OUTPUT_DIR / "pd_only_cohort.csv",
    index=False
)


# ============================================================
# 4. LOAD AND ALIGN BASELINE DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: LOAD AND ALIGN BASELINE DATA")
print("=" * 60)

clinical_raw = load_baseline_file(CLINICAL_FILE)
csf_raw = load_baseline_file(CSF_FILE)
plasma_raw = load_baseline_file(PLASMA_FILE)

clinical_raw = align_to_cohort(
    clinical_raw, pd_patnos, "Clinical"
)

csf_raw = align_to_cohort(
    csf_raw, pd_patnos, "CSF"
)

plasma_raw = align_to_cohort(
    plasma_raw, pd_patnos, "Plasma"
)

print("Clinical:", clinical_raw.shape)
print("CSF:", csf_raw.shape)
print("Plasma:", plasma_raw.shape)

# Add subgroup labels for downstream characterization.
# Do not use subgroup as a clustering feature.
subgroup_map = pd_cohort.set_index("PATNO")["subgroup"]

clinical_raw["subgroup"] = clinical_raw["PATNO"].map(
    subgroup_map
)


# ============================================================
# 5. PREPARE CLINICAL VIEW
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: PREPARE CLINICAL VIEW")
print("=" * 60)

# Adjust these aliases if your source columns use different names.
clinical_aliases = {
    "age": ["age", "AGE", "AGE_AT_VISIT"],
    "sex": ["SEX", "sex"],
    "education_years": ["EDUCYRS", "education_years"],
    "moca": ["moca", "MOCA", "MOCA_TOTAL"],
    "upsit": ["upsit", "UPSIT", "UPSIT_TOTAL"],
}

clinical_features = pd.DataFrame(
    {"PATNO": clinical_raw["PATNO"]}
)

for feature, aliases in clinical_aliases.items():

    col = find_column(
        clinical_raw,
        aliases,
        required=True
    )

    clinical_features[feature] = clinical_raw[col]

# Numeric features
numeric_features = [
    "age",
    "education_years",
    "moca",
    "upsit",
]

for col in numeric_features:
    clinical_features[col] = pd.to_numeric(
        clinical_features[col],
        errors="coerce"
    )

# Drop clinical features that are entirely missing
numeric_features = [
    col for col in numeric_features
    if clinical_features[col].notna().any()
]

if not numeric_features:
    raise ValueError("No usable numeric clinical features found.")

# Median imputation
clinical_imputer = SimpleImputer(strategy="median")

clinical_numeric = clinical_imputer.fit_transform(
    clinical_features[numeric_features]
)

clinical_numeric = pd.DataFrame(
    clinical_numeric,
    columns=numeric_features,
    index=clinical_features.index
)

# Encode sex as a categorical variable
sex = clinical_features["sex"].astype("string").fillna("Missing")

sex_dummies = pd.get_dummies(
    sex,
    prefix="sex",
    drop_first=True,
    dtype=float
)

clinical_matrix = pd.concat(
    [clinical_numeric, sex_dummies],
    axis=1
)

# Remove constant features
clinical_matrix = clinical_matrix.loc[
    :,
    clinical_matrix.nunique() > 1
]

if clinical_matrix.shape[1] == 0:
    raise ValueError("No variable clinical features remain.")

# Standardize clinical features
clinical_scaler = StandardScaler()

clinical_scaled = clinical_scaler.fit_transform(
    clinical_matrix
)

clinical_view = pd.DataFrame(
    clinical_scaled,
    columns=clinical_matrix.columns,
)

clinical_view.insert(
    0, "PATNO", pd_patnos
)

clinical_view.to_csv(
    VIEW_DIR / "clinical_view.csv",
    index=False
)

print("Clinical features:", clinical_matrix.shape[1])
print("Clinical view:", clinical_view.shape)


# ============================================================
# 6. PROTEOMICS PREPROCESSING
# ============================================================

def prepare_proteomics_view(
    df,
    view_name,
    output_dir,
    max_missingness=0.20,
    correlation_threshold=0.95,
):
    """
    Prepare one proteomics view.

    Steps:
    - Exclude IDs and missingness indicator columns.
    - Retain numeric protein features.
    - Remove proteins with excessive missingness.
    - Median imputation.
    - Remove constant proteins.
    - Greedy correlation filtering.
    """

    print(f"\nPreparing {view_name}...")

    excluded = {
        "PATNO",
        "EVENT_ID",
        "SUBJECT_ID",
        "VISIT_DATE",
        "SUBGROUP",
        "SAMPLE_ID",
        "PLATE_ID",
    }

    protein_cols = []

    for col in df.columns:

        col_upper = str(col).upper()

        if col_upper in excluded:
            continue

        if col_upper.endswith("_MISS"):
            continue

        protein_cols.append(col)

    # Convert candidate protein columns to numeric
    X = df[protein_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove columns with no numeric observations
    X = X.dropna(axis=1, how="all")

    input_count = X.shape[1]

    # Missingness filter
    missingness = X.isna().mean()

    X = X.loc[
        :,
        missingness <= max_missingness
    ]

    after_missingness = X.shape[1]

    # Median imputation
    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Remove constant features
    variances = X_imputed.var(axis=0)

    X_imputed = X_imputed.loc[
        :,
        variances > 0
    ]

    after_constant = X_imputed.shape[1]

    # Greedy correlation filtering
    # Prioritize proteins with lower missingness, then higher variance.
    feature_order = sorted(
        X_imputed.columns,
        key=lambda col: (
            missingness[col],
            -X_imputed[col].var(),
            str(col)
        )
    )

    corr = X_imputed.corr().abs()

    retained = []

    for feature in feature_order:

        if not retained:
            retained.append(feature)
            continue

        correlations = corr.loc[
            feature, retained
        ]

        if correlations.max() < correlation_threshold:
            retained.append(feature)

    X_filtered = X_imputed[retained].copy()

    # Standardize proteins before PCA
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_filtered)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=X_filtered.columns,
        index=X_filtered.index
    )

    # Save filtered protein matrix
    output_file = (
        output_dir / f"baseline_{view_name.lower()}_filtered.csv"
    )

    filtered_output = scaled_df.copy()
    filtered_output.insert(0, "PATNO", pd_patnos)

    filtered_output.to_csv(
        output_file,
        index=False
    )

    summary = {
        "view": view_name,
        "input_features": input_count,
        "after_missingness_filter": after_missingness,
        "after_constant_filter": after_constant,
        "after_correlation_filter": X_filtered.shape[1],
    }

    print(summary)

    return scaled_df, summary


csf_scaled, csf_summary = prepare_proteomics_view(
    csf_raw,
    "CSF",
    VIEW_DIR,
    MAX_MISSINGNESS,
    CORRELATION_THRESHOLD
)

plasma_scaled, plasma_summary = prepare_proteomics_view(
    plasma_raw,
    "Plasma",
    VIEW_DIR,
    MAX_MISSINGNESS,
    CORRELATION_THRESHOLD
)

pd.DataFrame(
    [csf_summary, plasma_summary]
).to_csv(
    OUTPUT_DIR / "proteomics_preprocessing_summary.csv",
    index=False
)


# ============================================================
# 7. REFIT PCA FOR PD-ONLY PROTEOMICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: REFIT PROTEOMICS PCA")
print("=" * 60)

def fit_proteomics_pca(
    X,
    view_name,
    n_components,
    output_dir,
):
    """Fit PCA and return participant-level component scores."""

    max_components = min(
        X.shape[0] - 1,
        X.shape[1]
    )

    if n_components > max_components:
        raise ValueError(
            f"{view_name}: requested {n_components} PCs, "
            f"but maximum is {max_components}."
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="full"
    )

    scores = pca.fit_transform(X)

    columns = [
        f"{view_name}_PC{i + 1}"
        for i in range(n_components)
    ]

    score_df = pd.DataFrame(
        scores,
        columns=columns
    )

    score_df.insert(
        0, "PATNO", pd_patnos
    )

    # Save participant scores
    score_df.to_csv(
        output_dir
        / f"baseline_{view_name.lower()}_pca{n_components}.csv",
        index=False
    )

    explained = pd.DataFrame({
        "view": view_name,
        "component": columns,
        "explained_variance_ratio": (
            pca.explained_variance_ratio_
        ),
        "cumulative_explained_variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    return score_df, explained


pca_views = {}
pca_summaries = []

for n_components in PCA_COMPONENTS:

    csf_scores, csf_explained = fit_proteomics_pca(
        csf_scaled,
        "CSF",
        n_components,
        PCA_DIR
    )

    plasma_scores, plasma_explained = fit_proteomics_pca(
        plasma_scaled,
        "Plasma",
        n_components,
        PCA_DIR
    )

    pca_views[n_components] = {
        "CSF": csf_scores.drop(columns="PATNO").to_numpy(),
        "Plasma": plasma_scores.drop(columns="PATNO").to_numpy(),
    }

    pca_summaries.extend([
        csf_explained,
        plasma_explained
    ])

pd.concat(
    pca_summaries,
    ignore_index=True
).to_csv(
    PCA_DIR / "pca_explained_variance.csv",
    index=False
)


# ============================================================
# 8. KERNEL FUNCTIONS
# ============================================================

def median_heuristic_kernel(X, gamma_multiplier=1.0):
    """
    RBF kernel using the median heuristic.

    gamma = multiplier / median(nonzero squared distances)
    """

    distances = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    nonzero = distances[distances > 1e-12]

    if len(nonzero) == 0:
        raise ValueError("All pairwise distances are zero.")

    median_distance = np.median(nonzero)

    gamma = gamma_multiplier / median_distance

    K = np.exp(-gamma * distances)

    # Frobenius normalization
    norm = np.linalg.norm(K, ord="fro")

    if norm > 0:
        K = K / norm

    return K


def combine_kernels(kernels, weights):
    """Combine normalized kernels using specified weights."""

    combined = np.zeros_like(kernels[0])

    for kernel, weight in zip(kernels, weights):
        combined += weight * kernel

    norm = np.linalg.norm(combined, ord="fro")

    if norm > 0:
        combined /= norm

    return combined


def kernel_distance_matrix(K):
    """Convert a kernel matrix into Euclidean distances."""

    diagonal = np.diag(K)

    D2 = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2 * K
    )

    D2 = np.maximum(D2, 0)

    D = np.sqrt(D2)

    np.fill_diagonal(D, 0)

    return D


def run_spectral_clustering(K, k, random_state):
    """Run spectral clustering and calculate silhouette."""

    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=N_INIT,
        random_state=random_state,
    )

    labels = model.fit_predict(K)

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        silhouette = np.nan
    else:
        distances = kernel_distance_matrix(K)

        silhouette = silhouette_score(
            distances,
            labels,
            metric="precomputed"
        )

    return labels, silhouette


# ============================================================
# 9. DEFINE SINGLE-VIEW AND MULTIVIEW CONFIGURATIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DEFINE CLUSTERING CONFIGURATIONS")
print("=" * 60)

# Clinical kernel is independent of the proteomics PCA dimensions.
clinical_X = clinical_view.drop(
    columns="PATNO"
).to_numpy()

clinical_kernel = median_heuristic_kernel(
    clinical_X,
    gamma_multiplier=1.0
)

# Each specification contains the views and their weights.
configurations = []

# Clinical-only configuration
configurations.append({
    "name": "clinical_only",
    "pca_components": None,
    "views": ["clinical"],
    "weights": [1.0],
})

for n_components in PCA_COMPONENTS:

    configurations.extend([
        {
            "name": f"csf_only_pca{n_components}",
            "pca_components": n_components,
            "views": ["csf"],
            "weights": [1.0],
        },
        {
            "name": f"plasma_only_pca{n_components}",
            "pca_components": n_components,
            "views": ["plasma"],
            "weights": [1.0],
        },
        {
            "name": f"clinical_csf_pca{n_components}",
            "pca_components": n_components,
            "views": ["clinical", "csf"],
            "weights": [0.5, 0.5],
        },
        {
            "name": f"clinical_plasma_pca{n_components}",
            "pca_components": n_components,
            "views": ["clinical", "plasma"],
            "weights": [0.5, 0.5],
        },
        {
            "name": f"csf_plasma_pca{n_components}",
            "pca_components": n_components,
            "views": ["csf", "plasma"],
            "weights": [0.5, 0.5],
        },
    ])

    # Three-view combinations with different weights
    for scheme_name, weights in WEIGHT_SCHEMES.items():

        configurations.append({
            "name": (
                f"all_views_{scheme_name}_pca{n_components}"
            ),
            "pca_components": n_components,
            "views": ["clinical", "csf", "plasma"],
            "weights": list(weights),
        })


# ============================================================
# 10. RUN CLUSTERING AND STABILITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: RUN CLUSTERING AND STABILITY ANALYSIS")
print("=" * 60)

all_results = []
all_seed42_labels = []

total_configurations = (
    len(configurations)
    * len(GAMMA_MULTIPLIERS)
    * len(K_VALUES)
)

completed = 0

for config in configurations:

    # Construct kernels for each view
    kernels = {}

    for view in config["views"]:

        if view == "clinical":
            X = clinical_X

        elif view == "csf":
            X = pca_views[
                config["pca_components"]
            ]["CSF"]

        elif view == "plasma":
            X = pca_views[
                config["pca_components"]
            ]["Plasma"]

        else:
            raise ValueError(f"Unknown view: {view}")

        kernels[view] = X

    for gamma_multiplier in GAMMA_MULTIPLIERS:

        # Recalculate each view's kernel at this bandwidth
        current_kernels = [
            median_heuristic_kernel(
                kernels[view],
                gamma_multiplier
            )
            for view in config["views"]
        ]

        # Combine views
        if len(current_kernels) == 1:
            combined_kernel = current_kernels[0]
        else:
            combined_kernel = combine_kernels(
                current_kernels,
                config["weights"]
            )

        for k in K_VALUES:

            completed += 1

            print(
                f"\n[{completed}/{total_configurations}] "
                f"{config['name']} | "
                f"gamma={gamma_multiplier} | k={k}"
            )

            seed_labels = []
            seed_silhouettes = []

            for seed in RANDOM_SEEDS:

                labels, silhouette = run_spectral_clustering(
                    combined_kernel,
                    k,
                    seed
                )

                seed_labels.append(labels)
                seed_silhouettes.append(silhouette)

                # Save seed 42 assignments for each configuration
                if seed == 42:

                    label_df = pd.DataFrame({
                        "PATNO": pd_patnos,
                        "cluster": labels,
                    })

                    label_df["configuration"] = config["name"]
                    label_df["gamma_multiplier"] = gamma_multiplier
                    label_df["k"] = k

                    all_seed42_labels.append(label_df)

            # Pairwise ARI across random seeds
            ari_values = [
                adjusted_rand_score(
                    seed_labels[i],
                    seed_labels[j]
                )
                for i, j in combinations(
                    range(len(seed_labels)), 2
                )
            ]

            # Cluster sizes from seed 42
            seed42 = seed_labels[
                RANDOM_SEEDS.index(42)
            ]

            cluster_sizes = pd.Series(
                seed42
            ).value_counts().sort_index()

            result = {
                "configuration": config["name"],
                "views": "+".join(config["views"]),
                "pca_components": config["pca_components"],
                "weights": str(config["weights"]),
                "gamma_multiplier": gamma_multiplier,
                "k": k,

                "mean_silhouette": np.nanmean(
                    seed_silhouettes
                ),
                "sd_silhouette": np.nanstd(
                    seed_silhouettes
                ),

                "mean_stability_ARI": np.mean(
                    ari_values
                ),
                "sd_stability_ARI": np.std(
                    ari_values
                ),
                "min_stability_ARI": np.min(
                    ari_values
                ),
                "max_stability_ARI": np.max(
                    ari_values
                ),

                "cluster_sizes_seed42": str(
                    cluster_sizes.to_dict()
                ),
            }

            all_results.append(result)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: SAVE RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=[
        "mean_stability_ARI",
        "mean_silhouette"
    ],
    ascending=False
)

results_df.to_csv(
    RESULT_DIR / "pd_only_clustering_summary.csv",
    index=False
)

labels_df = pd.concat(
    all_seed42_labels,
    ignore_index=True
)

labels_df.to_csv(
    RESULT_DIR / "pd_only_cluster_assignments_seed42.csv",
    index=False
)

# Save top 20 configurations for easier inspection
results_df.head(20).to_csv(
    RESULT_DIR / "pd_only_top20_configurations.csv",
    index=False
)


# ============================================================
# 12. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PD-ONLY CLUSTERING COMPLETE")
print("=" * 60)

print("\nPD-only participants:", len(pd_patnos))

print("\nProteomics preprocessing:")
print(pd.DataFrame([
    csf_summary,
    plasma_summary
]).to_string(index=False))

print("\nTop 10 configurations:")
print(
    results_df[
        [
            "configuration",
            "gamma_multiplier",
            "k",
            "mean_silhouette",
            "mean_stability_ARI",
            "cluster_sizes_seed42",
        ]
    ].head(10).to_string(index=False)
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")
for path in sorted(OUTPUT_DIR.rglob("*.csv")):
    print(" -", path.relative_to(ROOT))
