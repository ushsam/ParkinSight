
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT / "Datasets" / "Processed" /
    "Proteomics" / "CorrelationFiltered"
)

OUTPUT_DIR = (
    ROOT / "Datasets" / "Processed" /
    "Proteomics" / "PCA"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = {
    "csf": INPUT_DIR / "baseline_csf_correlation_filtered.csv",
    "plasma": INPUT_DIR / "baseline_plasma_correlation_filtered.csv",
}

KEYS = ["PATNO", "EVENT_ID"]

VARIANCE_TARGETS = [0.80, 0.90, 0.95]


# ============================================================
# PCA preprocessing and analysis
# ============================================================

def run_pca(view_name, file_path):

    print("\n" + "=" * 60)
    print(f"PROCESSING {view_name.upper()} PCA")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load correlation-filtered matrix
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    for col in KEYS:
        if col not in df.columns:
            raise ValueError(
                f"{view_name}: Missing identifier {col}"
            )

    ids = df[KEYS].copy()

    feature_df = df.drop(columns=KEYS)

    feature_df = feature_df.select_dtypes(
        include="number"
    ).copy()

    if feature_df.empty:
        raise ValueError(
            f"{view_name}: No numeric protein features found."
        )

    if feature_df.isna().any().any():
        raise ValueError(
            f"{view_name}: Missing values detected. "
            "Resolve missingness before PCA."
        )

    # Remove any zero-variance features as a final safeguard
    variances = feature_df.var(ddof=0)

    zero_variance = variances[variances <= 0].index.tolist()

    if zero_variance:
        print(
            f"Removing {len(zero_variance)} zero-variance features."
        )
        feature_df = feature_df.drop(columns=zero_variance)

    n_samples, n_features = feature_df.shape

    print(f"Participants: {n_samples}")
    print(f"Protein features: {n_features}")

    # --------------------------------------------------------
    # 2. Standardize protein features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(feature_df)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_df.columns,
        index=feature_df.index,
    )

    # Save standardized protein matrix
    standardized_df = pd.concat(
        [
            ids.reset_index(drop=True),
            scaled_df.reset_index(drop=True),
        ],
        axis=1,
    )

    standardized_df.to_csv(
        OUTPUT_DIR / f"baseline_{view_name}_standardized.csv",
        index=False,
    )

    print("Standardization completed.")

    # --------------------------------------------------------
    # 3. Run PCA
    # --------------------------------------------------------

    # Maximum meaningful number of components is n_samples - 1
    max_components = min(n_samples - 1, n_features)

    pca = PCA(
        n_components=max_components,
        svd_solver="full",
    )

    scores = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    cumulative_variance = np.cumsum(explained_variance)

    print(f"Number of PCA components: {max_components}")

    # --------------------------------------------------------
    # 4. Explained variance report
    # --------------------------------------------------------

    variance_df = pd.DataFrame({
        "component": [
            f"PC{i}" for i in range(1, max_components + 1)
        ],
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance": cumulative_variance,
    })

    variance_df.to_csv(
        OUTPUT_DIR / f"{view_name}_explained_variance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Number of components at variance targets
    # --------------------------------------------------------

    target_results = []

    print("\nComponents required by variance target:")

    for target in VARIANCE_TARGETS:

        n_required = int(
            np.searchsorted(cumulative_variance, target) + 1
        )

        target_results.append({
            "view": view_name,
            "variance_target": target,
            "components_required": n_required,
            "actual_cumulative_variance": (
                cumulative_variance[n_required - 1]
            ),
        })

        print(
            f"  {target:.0%} variance: "
            f"{n_required} components "
            f"({cumulative_variance[n_required - 1]:.2%})"
        )

    target_df = pd.DataFrame(target_results)

    target_df.to_csv(
        OUTPUT_DIR / f"{view_name}_variance_targets.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Save participant PCA scores
    # --------------------------------------------------------

    score_columns = [
        f"PC{i}" for i in range(1, max_components + 1)
    ]

    scores_df = pd.DataFrame(
        scores,
        columns=score_columns,
    )

    scores_df = pd.concat(
        [
            ids.reset_index(drop=True),
            scores_df,
        ],
        axis=1,
    )

    scores_df.to_csv(
        OUTPUT_DIR / f"baseline_{view_name}_pca_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Save protein loadings
    # --------------------------------------------------------

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_df.columns,
        columns=score_columns,
    )

    loadings_df.index.name = "protein"

    loadings_df.to_csv(
        OUTPUT_DIR / f"{view_name}_pca_loadings.csv"
    )

    # --------------------------------------------------------
    # 8. Scree plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, max_components + 1),
        explained_variance,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{view_name.upper()} PCA Scree Plot")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{view_name}_scree_plot.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 9. Cumulative explained variance plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, max_components + 1),
        cumulative_variance,
        marker="o",
        markersize=3,
    )

    for target in VARIANCE_TARGETS:
        plt.axhline(
            target,
            linestyle="--",
            label=f"{target:.0%} variance",
        )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(
        f"{view_name.upper()} Cumulative Explained Variance"
    )

    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{view_name}_cumulative_variance.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # 10. Summary
    # --------------------------------------------------------

    summary = {
        "view": view_name,
        "participants": n_samples,
        "initial_proteins": n_features + len(zero_variance),
        "proteins_removed_zero_variance": len(zero_variance),
        "final_proteins": n_features,
        "maximum_components": max_components,
    }

    print("\nPCA completed.")

    return summary, target_df


# ============================================================
# Run CSF and plasma independently
# ============================================================

summaries = []
target_summaries = []

for view_name, file_path in VIEWS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    summary, targets = run_pca(view_name, file_path)

    summaries.append(summary)
    target_summaries.append(targets)


# ============================================================
# Save combined summaries
# ============================================================

summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUTPUT_DIR / "pca_summary.csv",
    index=False,
)

combined_targets = pd.concat(
    target_summaries,
    ignore_index=True,
)

combined_targets.to_csv(
    OUTPUT_DIR / "combined_variance_targets.csv",
    index=False,
)


print("\n" + "=" * 60)
print("CSF AND PLASMA PCA COMPLETE")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nVariance targets:")
print(combined_targets.to_string(index=False))

print(f"\nOutput directory: {OUTPUT_DIR}")


ROOT = Path(__file__).resolve().parent

PCA_DIR = (
    ROOT / "Datasets" / "Processed" /
    "Proteomics" / "PCA"
)

OUTPUT_DIR = PCA_DIR / "PCA80"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "csf": 83,
    "plasma": 74,
}

for view, n_components in CONFIG.items():

    input_file = PCA_DIR / f"baseline_{view}_pca_scores.csv"
    df = pd.read_csv(input_file)

    # Preserve participant identifiers
    id_cols = ["PATNO", "EVENT_ID"]
    pc_cols = [
        col for col in df.columns
        if col.startswith("PC")
    ]

    selected_cols = id_cols + pc_cols[:n_components]
    pca_80 = df[selected_cols].copy()

    output_file = (
        OUTPUT_DIR /
        f"baseline_{view}_pca80.csv"
    )

    pca_80.to_csv(output_file, index=False)

    print(
        f"{view.upper()}: "
        f"{len(pca_80)} participants, "
        f"{n_components} components saved"
    )

print("\n80% PCA representations prepared.")

