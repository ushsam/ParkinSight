
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Datasets" / "Processed" / "Clustering"

OUTPUT_DIR = INPUT_DIR / "KernelDiagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["PATNO", "EVENT_ID"]

VIEW_FILES = {
    "clinical": "clinical_view.csv",
    "csf": "csf_view.csv",
    "plasma": "plasma_view.csv",
}


# --------------------------------------------------
# 2. Load and validate views
# --------------------------------------------------

views = {}
participant_ids = None

for name, filename in VIEW_FILES.items():

    df = pd.read_csv(INPUT_DIR / filename)

    if df.duplicated("PATNO").any():
        raise ValueError(f"Duplicate PATNO in {name}")

    ids = df[KEYS].copy()

    if participant_ids is None:
        participant_ids = ids
    elif not participant_ids.equals(ids):
        raise ValueError(f"Participant alignment error: {name}")

    X = df.drop(columns=KEYS).apply(
        pd.to_numeric, errors="coerce"
    )

    if X.isnull().any().any():
        raise ValueError(f"Missing values in {name}")

    X = X.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in {name}")

    views[name] = X

    print(
        f"{name}: {X.shape[0]} participants, "
        f"{X.shape[1]} features"
    )

n = len(participant_ids)


# --------------------------------------------------
# 3. Construct RBF kernels
# --------------------------------------------------

def build_kernel(X):

    squared_distances = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    nonzero = squared_distances[
        squared_distances > 0
    ]

    if len(nonzero) == 0:
        raise ValueError("All participants have identical features.")

    median_distance = np.median(nonzero)
    gamma = 1.0 / median_distance

    K = rbf_kernel(X, gamma=gamma)

    K = (K + K.T) / 2

    return K, gamma, median_distance, squared_distances


kernels = {}
kernel_parameters = []
similarity_results = []

for name, X in views.items():

    K, gamma, median_distance, D2 = build_kernel(X)

    kernels[name] = K

    # Extract off-diagonal similarities
    off_diag = K[np.triu_indices(n, k=1)]

    similarity_results.append({
        "view": name,
        "mean_similarity": np.mean(off_diag),
        "median_similarity": np.median(off_diag),
        "std_similarity": np.std(off_diag),
        "min_similarity": np.min(off_diag),
        "max_similarity": np.max(off_diag),
        "q05": np.quantile(off_diag, 0.05),
        "q25": np.quantile(off_diag, 0.25),
        "q75": np.quantile(off_diag, 0.75),
        "q95": np.quantile(off_diag, 0.95),
        "pct_similarity_below_0.01": np.mean(
            off_diag < 0.01
        ) * 100,
        "pct_similarity_above_0.90": np.mean(
            off_diag > 0.90
        ) * 100,
    })

    kernel_parameters.append({
        "view": name,
        "gamma": gamma,
        "median_squared_distance": median_distance,
        "mean_squared_distance": np.mean(
            D2[np.triu_indices(n, k=1)]
        ),
    })


# --------------------------------------------------
# 4. Centered kernel alignment
# --------------------------------------------------

def center_kernel(K):

    H = np.eye(K.shape[0]) - (
        np.ones(K.shape) / K.shape[0]
    )

    return H @ K @ H


def centered_alignment(K1, K2):

    K1c = center_kernel(K1)
    K2c = center_kernel(K2)

    numerator = np.sum(K1c * K2c)

    denominator = (
        np.linalg.norm(K1c, "fro")
        * np.linalg.norm(K2c, "fro")
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator


view_names = list(kernels.keys())

alignment_matrix = pd.DataFrame(
    index=view_names,
    columns=view_names,
    dtype=float
)

for v1 in view_names:
    for v2 in view_names:
        alignment_matrix.loc[v1, v2] = (
            centered_alignment(
                kernels[v1],
                kernels[v2]
            )
        )

alignment_matrix.to_csv(
    OUTPUT_DIR / "centered_kernel_alignment.csv"
)


# --------------------------------------------------
# 5. Pairwise similarity comparison
# --------------------------------------------------

pairwise_results = []

for i in range(len(view_names)):
    for j in range(i + 1, len(view_names)):

        v1 = view_names[i]
        v2 = view_names[j]

        K1 = kernels[v1]
        K2 = kernels[v2]

        upper = np.triu_indices(n, k=1)

        sim1 = K1[upper]
        sim2 = K2[upper]

        pearson = np.corrcoef(sim1, sim2)[0, 1]

        spearman = spearmanr(
            sim1, sim2
        ).statistic

        pairwise_results.append({
            "view_1": v1,
            "view_2": v2,
            "similarity_pearson": pearson,
            "similarity_spearman": spearman,
        })

pd.DataFrame(pairwise_results).to_csv(
    OUTPUT_DIR / "pairwise_kernel_correlations.csv",
    index=False
)


# --------------------------------------------------
# 6. Kernel eigenvalue diagnostics
# --------------------------------------------------

eigen_results = []

for name, K in kernels.items():

    Kc = center_kernel(K)

    eigenvalues = np.linalg.eigvalsh(Kc)
    eigenvalues = np.maximum(eigenvalues, 0)

    total = eigenvalues.sum()

    if total > 0:
        proportions = eigenvalues / total
        proportions = proportions[proportions > 0]

        effective_rank = np.exp(
            -np.sum(
                proportions * np.log(proportions)
            )
        )
    else:
        effective_rank = 0

    eigen_results.append({
        "view": name,
        "minimum_eigenvalue": np.min(
            np.linalg.eigvalsh(K)
        ),
        "centered_minimum_eigenvalue": np.min(
            np.linalg.eigvalsh(Kc)
        ),
        "effective_rank": effective_rank,
        "largest_centered_eigenvalue_share": (
            eigenvalues.max() / total
            if total > 0 else np.nan
        ),
    })

pd.DataFrame(eigen_results).to_csv(
    OUTPUT_DIR / "kernel_eigenvalue_diagnostics.csv",
    index=False
)


# --------------------------------------------------
# 7. Save diagnostic tables
# --------------------------------------------------

similarity_df = pd.DataFrame(similarity_results)
parameters_df = pd.DataFrame(kernel_parameters)

similarity_df.to_csv(
    OUTPUT_DIR / "within_view_similarity.csv",
    index=False
)

parameters_df.to_csv(
    OUTPUT_DIR / "kernel_parameters.csv",
    index=False
)


# --------------------------------------------------
# 8. Plot similarity distributions
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, K in kernels.items():

    off_diag = K[np.triu_indices(n, k=1)]

    plt.hist(
        off_diag,
        bins=40,
        alpha=0.45,
        label=name,
        density=True
    )

plt.xlabel("RBF similarity")
plt.ylabel("Density")
plt.title("Within-view RBF Similarity Distributions")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "similarity_distributions.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 9. Plot centered kernel alignment
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(
    alignment_matrix.astype(float),
    vmin=0,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Centered kernel alignment")

plt.xticks(
    range(len(view_names)),
    view_names
)

plt.yticks(
    range(len(view_names)),
    view_names
)

plt.title("Between-view Kernel Alignment")

for i in range(len(view_names)):
    for j in range(len(view_names)):
        plt.text(
            j, i,
            f"{alignment_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kernel_alignment.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 10. Print summary
# --------------------------------------------------

print("\n======================================")
print("KERNEL DIAGNOSTICS COMPLETE")
print("======================================")

print("\nWithin-view similarity:")
print(similarity_df.round(4).to_string(index=False))

print("\nKernel alignment:")
print(alignment_matrix.round(4))

print("\nPairwise kernel correlations:")
print(
    pd.DataFrame(pairwise_results)
    .round(4)
    .to_string(index=False)
)

print("\nEigenvalue diagnostics:")
print(
    pd.DataFrame(eigen_results)
    .round(4)
    .to_string(index=False)
)

print("\nSaved diagnostics to:", OUTPUT_DIR)
