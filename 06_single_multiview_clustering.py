
"""
Single-view and multiview clustering with sensitivity analysis.

Tests:
1. Single-view and multiview clustering
2. Kernel bandwidth sensitivity
3. Multiview kernel weight sensitivity
4. Random-seed clustering stability

Run from the project root:
    python 08_single_multiview_clustering.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    adjusted_rand_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

# For the original 206-participant analysis:
INPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering"

# For a PD-only analysis, change INPUT_DIR to the folder
# containing the aligned clinical_view.csv, csf_view.csv,
# and plasma_view.csv files.

OUTPUT_DIR = INPUT_DIR / "SensitivityAnalysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEEDS = list(range(42, 72))
REFERENCE_SEED = 42

CLUSTER_COUNTS = [2, 3, 4, 5]

# Multipliers applied to the median-heuristic gamma
GAMMA_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]

N_INIT = 20

VIEW_FILES = {
    "clinical": "clinical_view.csv",
    "csf": "csf_view.csv",
    "plasma": "plasma_view.csv",
}

# View combinations
EXPERIMENTS = {
    "clinical": ["clinical"],
    "csf": ["csf"],
    "plasma": ["plasma"],
    "clinical_csf": ["clinical", "csf"],
    "clinical_plasma": ["clinical", "plasma"],
    "csf_plasma": ["csf", "plasma"],
    "all_views": ["clinical", "csf", "plasma"],
}

# Equal weights for two-view combinations
PAIRWISE_WEIGHTS = {
    "equal": [0.5, 0.5],
}

# Weight sensitivity for three-view combination
THREE_VIEW_WEIGHTS = {
    "equal": [1/3, 1/3, 1/3],
    "clinical_dominant": [0.6, 0.2, 0.2],
    "csf_dominant": [0.2, 0.6, 0.2],
    "plasma_dominant": [0.2, 0.2, 0.6],
}


# ==================================================
# 2. LOAD AND VALIDATE VIEWS
# ==================================================

KEYS = ["PATNO", "EVENT_ID"]

views = {}
participant_ids = None

for view_name, filename in VIEW_FILES.items():

    path = INPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    df = pd.read_csv(path)

    if not all(col in df.columns for col in KEYS):
        raise ValueError(
            f"{view_name} must contain {KEYS}"
        )

    if df.duplicated("PATNO").any():
        raise ValueError(
            f"Duplicate participants in {view_name}"
        )

    ids = df[KEYS].copy().reset_index(drop=True)

    if participant_ids is None:
        participant_ids = ids

    elif not participant_ids.equals(ids):
        raise ValueError(
            f"Participant alignment error in {view_name}"
        )

    X = df.drop(columns=KEYS).apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isnull().any().any():
        raise ValueError(
            f"Missing or nonnumeric values in {view_name}"
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Non-finite values in {view_name}"
        )

    if X.shape[1] == 0:
        raise ValueError(
            f"No features found in {view_name}"
        )

    views[view_name] = X.to_numpy(dtype=float)

    print(
        f"{view_name}: "
        f"{X.shape[0]} participants, "
        f"{X.shape[1]} features"
    )

n_samples = len(participant_ids)

if n_samples <= max(CLUSTER_COUNTS):
    raise ValueError(
        "Not enough participants for requested cluster counts."
    )

print("\nTotal participants:", n_samples)


# ==================================================
# 3. BUILD RBF KERNELS
# ==================================================

def calculate_base_gamma(X):
    """
    Calculate median-heuristic gamma.
    """

    squared_distances = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    nonzero_distances = squared_distances[
        squared_distances > 1e-12
    ]

    if len(nonzero_distances) == 0:
        raise ValueError(
            "All participants have identical features."
        )

    median_distance = np.median(nonzero_distances)

    return 1.0 / median_distance


def build_rbf_kernel(X, gamma):
    """
    Build RBF kernel and Frobenius-normalize it.
    """

    K = rbf_kernel(
        X,
        gamma=gamma
    )

    K = (K + K.T) / 2

    norm = np.linalg.norm(K, ord="fro")

    if norm == 0:
        raise ValueError("Kernel has zero norm.")

    K = K / norm

    return K


base_gammas = {
    name: calculate_base_gamma(X)
    for name, X in views.items()
}

print("\nBase median-heuristic gamma:")
for name, gamma in base_gammas.items():
    print(f"{name}: {gamma:.6f}")


# ==================================================
# 4. KERNEL COMBINATION
# ==================================================

def combine_kernels(kernel_list, weights):
    """
    Combine kernels using specified weights.
    """

    if len(kernel_list) != len(weights):
        raise ValueError(
            "Number of kernels and weights do not match."
        )

    combined = np.zeros_like(kernel_list[0])

    for K, weight in zip(kernel_list, weights):
        combined += weight * K

    combined = (combined + combined.T) / 2

    norm = np.linalg.norm(combined, ord="fro")

    if norm > 0:
        combined = combined / norm

    return combined


# ==================================================
# 5. KERNEL DISTANCE MATRIX
# ==================================================

def kernel_to_distance(K):
    """
    Convert kernel similarity to Euclidean distance.
    """

    diagonal = np.diag(K)

    squared_distances = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2 * K
    )

    squared_distances = np.maximum(
        squared_distances, 0
    )

    distances = np.sqrt(squared_distances)

    np.fill_diagonal(distances, 0)

    return distances


# ==================================================
# 6. DEFINE EXPERIMENT CONFIGURATIONS
# ==================================================

configurations = []

for experiment_name, view_names in EXPERIMENTS.items():

    if len(view_names) == 1:

        configurations.append({
            "experiment": experiment_name,
            "views": view_names,
            "weight_name": "single_view",
            "weights": [1.0],
        })

    elif len(view_names) == 2:

        for weight_name, weights in PAIRWISE_WEIGHTS.items():

            configurations.append({
                "experiment": experiment_name,
                "views": view_names,
                "weight_name": weight_name,
                "weights": weights,
            })

    else:

        for weight_name, weights in THREE_VIEW_WEIGHTS.items():

            configurations.append({
                "experiment": experiment_name,
                "views": view_names,
                "weight_name": weight_name,
                "weights": weights,
            })


# ==================================================
# 7. RUN SENSITIVITY ANALYSIS
# ==================================================

results = []
all_labels = []
kernel_info = []

total_runs = (
    len(configurations)
    * len(GAMMA_MULTIPLIERS)
    * len(CLUSTER_COUNTS)
)

completed = 0

for gamma_multiplier in GAMMA_MULTIPLIERS:

    print(
        "\n" + "=" * 60
        + f"\nGAMMA MULTIPLIER: {gamma_multiplier}"
        + "\n" + "=" * 60
    )

    # Build kernels for each view using the same
    # gamma multiplier applied to its own base gamma.
    current_kernels = {}

    for view_name, X in views.items():

        gamma = (
            base_gammas[view_name]
            * gamma_multiplier
        )

        K = build_rbf_kernel(
            X,
            gamma
        )

        current_kernels[view_name] = K

        kernel_info.append({
            "view": view_name,
            "gamma_multiplier": gamma_multiplier,
            "base_gamma": base_gammas[view_name],
            "actual_gamma": gamma,
            "kernel_frobenius_norm": np.linalg.norm(
                K, ord="fro"
            ),
        })

    # Run each view combination
    for config in configurations:

        experiment_name = config["experiment"]
        view_names = config["views"]
        weights = config["weights"]

        kernels_to_combine = [
            current_kernels[name]
            for name in view_names
        ]

        K = combine_kernels(
            kernels_to_combine,
            weights
        )

        D = kernel_to_distance(K)

        for k in CLUSTER_COUNTS:

            completed += 1

            print(
                f"\n[{completed}/{total_runs}] "
                f"{experiment_name} | "
                f"{config['weight_name']} | "
                f"gamma={gamma_multiplier} | "
                f"k={k}"
            )

            seed_labels = []
            seed_silhouettes = []

            for seed in RANDOM_SEEDS:

                model = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    assign_labels="kmeans",
                    n_init=N_INIT,
                    random_state=seed,
                )

                labels = model.fit_predict(K)

                silhouette = silhouette_score(
                    D,
                    labels,
                    metric="precomputed"
                )

                seed_labels.append(labels)
                seed_silhouettes.append(silhouette)

                # Save labels from reference seed
                if seed == REFERENCE_SEED:

                    label_df = participant_ids.copy()

                    label_df["cluster"] = labels
                    label_df["experiment"] = experiment_name
                    label_df["weight_scheme"] = (
                        config["weight_name"]
                    )
                    label_df["gamma_multiplier"] = (
                        gamma_multiplier
                    )
                    label_df["k"] = k

                    all_labels.append(label_df)

            # --------------------------------------
            # Random-seed stability
            # --------------------------------------

            ari_values = [
                adjusted_rand_score(
                    seed_labels[i],
                    seed_labels[j]
                )
                for i, j in combinations(
                    range(len(seed_labels)), 2
                )
            ]

            # Reference seed cluster sizes
            reference_labels = seed_labels[
                RANDOM_SEEDS.index(REFERENCE_SEED)
            ]

            cluster_sizes = (
                pd.Series(reference_labels)
                .value_counts()
                .sort_index()
            )

            result = {
                "experiment": experiment_name,
                "views": "+".join(view_names),
                "weight_scheme": config["weight_name"],
                "weights": str(weights),
                "gamma_multiplier": gamma_multiplier,
                "k": k,

                "mean_silhouette": np.mean(
                    seed_silhouettes
                ),
                "sd_silhouette": np.std(
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

                "smallest_cluster": cluster_sizes.min(),
                "largest_cluster": cluster_sizes.max(),
                "cluster_sizes": str(
                    cluster_sizes.to_dict()
                ),
            }

            results.append(result)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    [
        "mean_silhouette",
        "mean_stability_ARI"
    ],
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "sensitivity_analysis_results.csv",
    index=False
)

pd.DataFrame(kernel_info).to_csv(
    OUTPUT_DIR / "kernel_parameters.csv",
    index=False
)

pd.concat(
    all_labels,
    ignore_index=True
).to_csv(
    OUTPUT_DIR / "sensitivity_cluster_assignments.csv",
    index=False
)

# Save top 20 configurations
results_df.head(20).to_csv(
    OUTPUT_DIR / "top20_configurations.csv",
    index=False
)


# ==================================================
# 9. SAVE BEST CONFIGURATION PER EXPERIMENT
# ==================================================

best_per_experiment = (
    results_df
    .sort_values(
        [
            "mean_silhouette",
            "mean_stability_ARI"
        ],
        ascending=False
    )
    .groupby(
        ["experiment", "weight_scheme"],
        as_index=False
    )
    .first()
)

best_per_experiment.to_csv(
    OUTPUT_DIR / "best_per_experiment.csv",
    index=False
)


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 60)

print("\nTop 20 configurations:")

print(
    results_df[
        [
            "experiment",
            "weight_scheme",
            "gamma_multiplier",
            "k",
            "mean_silhouette",
            "mean_stability_ARI",
            "smallest_cluster",
            "largest_cluster",
        ]
    ].head(20).to_string(index=False)
)

print("\nBest configuration per experiment:")

print(
    best_per_experiment[
        [
            "experiment",
            "weight_scheme",
            "gamma_multiplier",
            "k",
            "mean_silhouette",
            "mean_stability_ARI",
        ]
    ].to_string(index=False)
)

print("\nOutput directory:")
print(OUTPUT_DIR)
