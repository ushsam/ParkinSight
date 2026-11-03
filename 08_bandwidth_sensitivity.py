
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.cluster import SpectralClustering


# --------------------------------------------------
# Paths and experiment settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering"
OUTPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering" / "BandwidthSensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
K_VALUES = [2, 3, 4]

RANDOM_STATE = 42
N_INIT = 20


# --------------------------------------------------
# Load prepared views
# --------------------------------------------------

def load_view(filename):
    df = pd.read_csv(INPUT_DIR / filename)

    ids = df[["PATNO", "EVENT_ID"]].copy()
    X = df.drop(columns=["PATNO", "EVENT_ID"]).to_numpy(dtype=float)

    return ids, X


ids, clinical = load_view("clinical_view.csv")
_, csf = load_view("csf_view.csv")
_, plasma = load_view("plasma_view.csv")

# Confirm participant alignment
assert len(ids) == len(clinical) == len(csf) == len(plasma)


# --------------------------------------------------
# Kernel construction
# --------------------------------------------------

def calculate_base_gamma(X):
    distances = euclidean_distances(X, squared=True)

    # Exclude diagonal zeros
    nonzero_distances = distances[distances > 0]

    median_distance = np.median(nonzero_distances)

    return 1.0 / median_distance


def build_kernel(X, gamma_multiplier):
    base_gamma = calculate_base_gamma(X)
    gamma = base_gamma * gamma_multiplier

    K = rbf_kernel(X, gamma=gamma)

    # Match the Frobenius normalization in the
    # existing clustering pipeline
    K = K / np.linalg.norm(K, ord="fro")

    return K, gamma


def combine_kernels(kernels):
    return np.mean(kernels, axis=0)


# --------------------------------------------------
# Kernel-based silhouette
# --------------------------------------------------

def kernel_to_distance(K):
    diagonal = np.diag(K)

    squared_distances = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2 * K
    )

    squared_distances = np.maximum(squared_distances, 0)

    return np.sqrt(squared_distances)


# --------------------------------------------------
# Run sensitivity analysis
# --------------------------------------------------

views = {
    "clinical": clinical,
    "csf": csf,
    "plasma": plasma,
}

models = {
    "clinical": ["clinical"],
    "csf": ["csf"],
    "plasma": ["plasma"],
    "all_views": ["clinical", "csf", "plasma"],
}

results = []
all_labels = {}

for multiplier in GAMMA_MULTIPLIERS:

    print(f"\nGamma multiplier: {multiplier}")

    kernels = {}
    gamma_values = {}

    for view_name, X in views.items():
        K, gamma = build_kernel(X, multiplier)

        kernels[view_name] = K
        gamma_values[view_name] = gamma

    for model_name, included_views in models.items():

        selected_kernels = [
            kernels[name] for name in included_views
        ]

        if len(selected_kernels) == 1:
            K_combined = selected_kernels[0]
        else:
            K_combined = combine_kernels(selected_kernels)

        distance_matrix = kernel_to_distance(K_combined)

        for k in K_VALUES:

            clustering = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                n_init=N_INIT,
                random_state=RANDOM_STATE,
            )

            labels = clustering.fit_predict(K_combined)

            silhouette = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed",
            )

            cluster_sizes = np.bincount(labels)
            label_key = (model_name, k, multiplier)

            all_labels[label_key] = labels

            results.append({
                "model": model_name,
                "gamma_multiplier": multiplier,
                "k": k,
                "silhouette": silhouette,
                "cluster_sizes": str(sorted(
                    cluster_sizes.tolist()
                )),
                "smallest_cluster": cluster_sizes.min(),
                "largest_cluster": cluster_sizes.max(),
                "gamma_clinical": gamma_values["clinical"],
                "gamma_csf": gamma_values["csf"],
                "gamma_plasma": gamma_values["plasma"],
            })

            print(
                f"{model_name:12s} | k={k} | "
                f"silhouette={silhouette:.4f} | "
                f"sizes={sorted(cluster_sizes.tolist())} | "
            )

for result in results:

    model_name = result["model"]
    k = result["k"]
    multiplier = result["gamma_multiplier"]

    baseline_labels = all_labels[(model_name, k, 1.0)]
    current_labels = all_labels[(model_name, k, multiplier)]

    result["ARI_vs_baseline"] = adjusted_rand_score(
        baseline_labels,
        current_labels
    )

# --------------------------------------------------
# Save results and participant labels
# --------------------------------------------------


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "bandwidth_sensitivity_results.csv",
    index=False,
)

labels_df = ids.copy()

for (model_name, k, multiplier), labels in all_labels.items():
    column = f"{model_name}_k{k}_gamma{multiplier}"
    labels_df[column] = labels

labels_df.to_csv(
    OUTPUT_DIR / "bandwidth_sensitivity_labels.csv",
    index=False,
)

print("\nSensitivity analysis complete.")
print(f"Results saved to: {OUTPUT_DIR}")
