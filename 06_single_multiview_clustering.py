
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    pairwise_distances,
    silhouette_score
)
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering"
OUTPUT_DIR = INPUT_DIR / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLUSTER_COUNTS = [2, 3, 4, 5]

VIEW_FILES = {
    "clinical": "clinical_view.csv",
    "csf": "csf_view.csv",
    "plasma": "plasma_view.csv",
}

EXPERIMENTS = {
    "clinical": ["clinical"],
    "csf": ["csf"],
    "plasma": ["plasma"],
    "clinical_csf": ["clinical", "csf"],
    "clinical_plasma": ["clinical", "plasma"],
    "csf_plasma": ["csf", "plasma"],
    "all_views": ["clinical", "csf", "plasma"],
}


# --------------------------------------------------
# 2. Load and validate views
# --------------------------------------------------

KEYS = ["PATNO", "EVENT_ID"]

views = {}
participant_ids = None

for view_name, filename in VIEW_FILES.items():

    df = pd.read_csv(INPUT_DIR / filename)

    if df.duplicated("PATNO").any():
        raise ValueError(f"Duplicate participants in {view_name}")

    ids = df[KEYS].copy()

    if participant_ids is None:
        participant_ids = ids
    elif not participant_ids.equals(ids):
        raise ValueError(
            f"Participant alignment error in {view_name}"
        )

    X = df.drop(columns=KEYS).apply(
        pd.to_numeric, errors="coerce"
    )

    if X.isnull().any().any():
        raise ValueError(
            f"Missing or nonnumeric values in {view_name}"
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Non-finite values in {view_name}"
        )

    views[view_name] = X.to_numpy(dtype=float)

    print(
        f"{view_name}: "
        f"{X.shape[0]} participants, "
        f"{X.shape[1]} features"
    )

n_samples = len(participant_ids)

if n_samples <= max(CLUSTER_COUNTS):
    raise ValueError("Not enough participants for requested k values.")


# --------------------------------------------------
# 3. Construct RBF kernel for each view
# --------------------------------------------------

def build_rbf_kernel(X):

    # Median heuristic for gamma
    squared_distances = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    nonzero_distances = squared_distances[
        squared_distances > 0
    ]

    if len(nonzero_distances) == 0:
        raise ValueError("All participants have identical features.")

    median_distance = np.median(nonzero_distances)
    gamma = 1.0 / median_distance

    K = rbf_kernel(X, gamma=gamma)

    # Symmetrize for numerical stability
    K = (K + K.T) / 2

    # Frobenius normalization
    norm = np.linalg.norm(K, ord="fro")

    if norm == 0:
        raise ValueError("Kernel has zero norm.")

    K = K / norm

    return K, gamma


kernels = {}
kernel_info = []

for view_name, X in views.items():

    K, gamma = build_rbf_kernel(X)

    kernels[view_name] = K

    kernel_info.append({
        "view": view_name,
        "gamma": gamma,
        "kernel_frobenius_norm": np.linalg.norm(K, "fro"),
    })


# --------------------------------------------------
# 4. Combine kernels using equal weights
# --------------------------------------------------

def combine_kernels(view_names):

    weight = 1.0 / len(view_names)

    combined = sum(
        weight * kernels[name]
        for name in view_names
    )

    # Ensure symmetry
    combined = (combined + combined.T) / 2

    return combined


# --------------------------------------------------
# 5. Convert kernel to distance matrix
# --------------------------------------------------

def kernel_to_distance(K):

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


# --------------------------------------------------
# 6. Run spectral clustering
# --------------------------------------------------

results = []
all_labels = []

for experiment_name, view_names in EXPERIMENTS.items():

    print(f"\nRunning experiment: {experiment_name}")

    K = combine_kernels(view_names)
    D = kernel_to_distance(K)

    for k in CLUSTER_COUNTS:

        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=20,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(K)

        silhouette = silhouette_score(
            D,
            labels,
            metric="precomputed"
        )

        cluster_sizes = pd.Series(labels).value_counts()

        result = {
            "experiment": experiment_name,
            "views": "+".join(view_names),
            "k": k,
            "silhouette": silhouette,
            "smallest_cluster": cluster_sizes.min(),
            "largest_cluster": cluster_sizes.max(),
        }

        results.append(result)

        label_df = participant_ids.copy()
        label_df["cluster"] = labels

        label_file = (
            OUTPUT_DIR /
            f"{experiment_name}_k{k}_labels.csv"
        )

        label_df.to_csv(label_file, index=False)

        all_labels.append(label_df.assign(
            experiment=experiment_name,
            k=k
        ))

        print(
            f"  k={k}: "
            f"silhouette={silhouette:.4f}, "
            f"sizes={sorted(cluster_sizes.tolist())}"
        )


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["k", "silhouette"],
    ascending=[True, False]
)

results_df.to_csv(
    OUTPUT_DIR / "clustering_results.csv",
    index=False
)

pd.DataFrame(kernel_info).to_csv(
    OUTPUT_DIR / "kernel_parameters.csv",
    index=False
)

pd.concat(all_labels, ignore_index=True).to_csv(
    OUTPUT_DIR / "all_clustering_labels.csv",
    index=False
)

print("\n===================================")
print("CLUSTERING EXPERIMENTS COMPLETE")
print("===================================")

print("\nResults:")
print(
    results_df[
        ["experiment", "k", "silhouette",
         "smallest_cluster", "largest_cluster"]
    ].to_string(index=False)
)

print("\nSaved results to:", OUTPUT_DIR)
