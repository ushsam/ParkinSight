
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


# --------------------------------------------------
# Paths and experiment settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering"

OUTPUT_DIR = (
    ROOT / "Datasets" / "Processed"
    / "Clustering" / "KernelWeightSensitivity"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA_MULTIPLIERS = [0.25, 0.5]
K_VALUES = [2, 3]

RANDOM_STATE = 42
N_INIT = 20


# --------------------------------------------------
# Kernel weighting configurations
# --------------------------------------------------

WEIGHT_CONFIGS = {
    "equal": {
        "clinical": 1/3,
        "csf": 1/3,
        "plasma": 1/3,
    },
    "clinical_dominant": {
        "clinical": 0.60,
        "csf": 0.20,
        "plasma": 0.20,
    },
    "csf_dominant": {
        "clinical": 0.20,
        "csf": 0.60,
        "plasma": 0.20,
    },
    "plasma_dominant": {
        "clinical": 0.20,
        "csf": 0.20,
        "plasma": 0.60,
    },
    "clinical_csf": {
        "clinical": 0.50,
        "csf": 0.50,
        "plasma": 0.00,
    },
    "clinical_plasma": {
        "clinical": 0.50,
        "csf": 0.00,
        "plasma": 0.50,
    },
    "csf_plasma": {
        "clinical": 0.00,
        "csf": 0.50,
        "plasma": 0.50,
    },
}


# --------------------------------------------------
# Load prepared views
# --------------------------------------------------

def load_view(filename):
    df = pd.read_csv(INPUT_DIR / filename)

    ids = df[["PATNO", "EVENT_ID"]].copy()

    X = df.drop(
        columns=["PATNO", "EVENT_ID"]
    ).to_numpy(dtype=float)

    return ids, X


ids, clinical = load_view("clinical_view.csv")
csf_ids, csf = load_view("csf_view.csv")
plasma_ids, plasma = load_view("plasma_view.csv")

# Confirm identical participant ordering
assert ids.equals(csf_ids)
assert ids.equals(plasma_ids)

views = {
    "clinical": clinical,
    "csf": csf,
    "plasma": plasma,
}


# --------------------------------------------------
# Construct individual RBF kernels
# --------------------------------------------------

def calculate_base_gamma(X):
    distances = euclidean_distances(X, squared=True)

    nonzero_distances = distances[distances > 0]

    median_distance = np.median(nonzero_distances)

    return 1.0 / median_distance


def build_kernel(X, gamma_multiplier):
    base_gamma = calculate_base_gamma(X)

    gamma = base_gamma * gamma_multiplier

    K = rbf_kernel(X, gamma=gamma)

    # Frobenius normalization
    K = K / np.linalg.norm(K, ord="fro")

    return K, gamma


# --------------------------------------------------
# Combine kernels using specified weights
# --------------------------------------------------

def combine_kernels(kernels, weights):
    combined = np.zeros_like(kernels["clinical"])

    for view_name, weight in weights.items():
        combined += weight * kernels[view_name]

    return combined


# --------------------------------------------------
# Convert kernel to distance matrix
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
# Run kernel-weight sensitivity analysis
# --------------------------------------------------

results = []
all_labels = {}

for multiplier in GAMMA_MULTIPLIERS:

    print(f"\nGamma multiplier: {multiplier}")

    # Construct each view's kernel once per gamma
    kernels = {}
    gamma_values = {}

    for view_name, X in views.items():

        K, gamma = build_kernel(X, multiplier)

        kernels[view_name] = K
        gamma_values[view_name] = gamma

    for config_name, weights in WEIGHT_CONFIGS.items():

        K_combined = combine_kernels(kernels, weights)

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

            label_key = (config_name, multiplier, k)

            all_labels[label_key] = labels

            results.append({
                "configuration": config_name,
                "gamma_multiplier": multiplier,
                "k": k,
                "clinical_weight": weights["clinical"],
                "csf_weight": weights["csf"],
                "plasma_weight": weights["plasma"],
                "silhouette": silhouette,
                "cluster_sizes": str(
                    sorted(cluster_sizes.tolist())
                ),
                "smallest_cluster": cluster_sizes.min(),
                "largest_cluster": cluster_sizes.max(),
                "ARI_vs_equal": np.nan,
                "gamma_clinical": gamma_values["clinical"],
                "gamma_csf": gamma_values["csf"],
                "gamma_plasma": gamma_values["plasma"],
            })

            print(
                f"{config_name:20s} | "
                f"k={k} | "
                f"silhouette={silhouette:.4f} | "
                f"sizes={sorted(cluster_sizes.tolist())}"
            )


# --------------------------------------------------
# Calculate ARI against equal-weight clustering
# --------------------------------------------------

for result in results:

    config_name = result["configuration"]
    multiplier = result["gamma_multiplier"]
    k = result["k"]

    baseline_labels = all_labels[
        ("equal", multiplier, k)
    ]

    current_labels = all_labels[
        (config_name, multiplier, k)
    ]

    result["ARI_vs_equal"] = adjusted_rand_score(
        baseline_labels,
        current_labels,
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "kernel_weight_sensitivity_results.csv",
    index=False,
)

# Save participant-level cluster assignments
labels_df = ids.copy()

for (config_name, multiplier, k), labels in all_labels.items():

    column = f"{config_name}_gamma{multiplier}_k{k}"

    labels_df[column] = labels

labels_df.to_csv(
    OUTPUT_DIR / "kernel_weight_sensitivity_labels.csv",
    index=False,
)

print("\nKernel-weight sensitivity analysis complete.")
print(f"Results saved to: {OUTPUT_DIR}")
