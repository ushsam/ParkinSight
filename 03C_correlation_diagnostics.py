# 04_correlation_diagnostics.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "Datasets" / "Processed" / "Proteomics"
OUTPUT_DIR = INPUT_DIR / "CorrelationDiagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = {
    "CSF": {
        "filtered": INPUT_DIR / "baseline_csf_filtered.csv",
        "raw": INPUT_DIR / "baseline_csf_raw.csv",
    },
    "Plasma": {
        "filtered": INPUT_DIR / "baseline_plasma_filtered.csv",
        "raw": INPUT_DIR / "baseline_plasma_raw.csv",
    },
}

THRESHOLDS = [0.90, 0.95, 0.99]

# Threshold used to define redundancy groups
GROUP_THRESHOLD = 0.95


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_protein_matrix(filtered_path):
    """Load a filtered matrix and retain numeric protein features."""

    df = pd.read_csv(filtered_path)

    id_columns = ["PATNO", "EVENT_ID"]
    available_ids = [c for c in id_columns if c in df.columns]

    feature_df = df.drop(columns=available_ids, errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Remove any remaining all-missing or constant columns
    feature_df = feature_df.dropna(axis=1, how="all")
    feature_df = feature_df.loc[:, feature_df.nunique(dropna=True) > 1]

    # Median imputation for correlation calculations
    feature_df = feature_df.fillna(feature_df.median())

    return df[available_ids].copy(), feature_df


def get_missingness(raw_path, protein_names):
    """Calculate missingness in the original baseline matrix."""

    if not raw_path.exists():
        return pd.Series(0.0, index=protein_names)

    raw = pd.read_csv(raw_path)

    available = [p for p in protein_names if p in raw.columns]

    if not available:
        return pd.Series(0.0, index=protein_names)

    numeric = raw[available].apply(pd.to_numeric, errors="coerce")
    missingness = numeric.isna().mean()

    return missingness.reindex(protein_names).fillna(0.0)


def find_exact_duplicates(df):
    """Identify protein columns with identical values."""

    duplicate_map = {}
    seen = {}

    for protein in df.columns:
        # Use values as a tuple to identify identical columns
        values = tuple(df[protein].to_numpy())

        if values in seen:
            duplicate_map[protein] = seen[values]
        else:
            seen[values] = protein

    rows = [
        {
            "duplicate_protein": duplicate,
            "representative_protein": representative,
        }
        for duplicate, representative in duplicate_map.items()
    ]

    return pd.DataFrame(
        rows,
        columns=["duplicate_protein", "representative_protein"],
    )


def greedy_filter(abs_corr, variances, missingness, threshold):
    """
    Reproduce the greedy correlation filtering approach.

    Priority:
    1. Higher variance
    2. Lower missingness
    3. Alphabetical protein name
    """

    protein_names = abs_corr.index.tolist()

    priority = pd.DataFrame({
        "protein": protein_names,
        "variance": variances.reindex(protein_names).values,
        "missingness": missingness.reindex(protein_names).values,
    })

    priority = priority.sort_values(
        by=["variance", "missingness", "protein"],
        ascending=[False, True, True],
    )

    corr_array = abs_corr.to_numpy()
    positions = {name: i for i, name in enumerate(protein_names)}

    retained = []
    removed = []

    for protein in priority["protein"]:
        position = positions[protein]

        if not retained:
            retained.append(protein)
            continue

        retained_positions = [positions[p] for p in retained]

        correlations = corr_array[position, retained_positions]
        max_corr = np.max(correlations)

        if max_corr <= threshold:
            retained.append(protein)
        else:
            best_idx = np.argmax(correlations)

            removed.append({
                "removed_protein": protein,
                "most_correlated_retained_protein": retained[best_idx],
                "absolute_correlation": correlations[best_idx],
                "threshold": threshold,
            })

    return retained, pd.DataFrame(removed)


def find_redundancy_groups(abs_corr, threshold):
    """
    Identify connected components of proteins whose absolute
    pairwise correlation is >= threshold.
    """

    proteins = abs_corr.index.to_list()
    n = len(proteins)

    corr_array = abs_corr.to_numpy()

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_values = corr_array[upper_i, upper_j]

    edge_mask = upper_values >= threshold

    edge_i = upper_i[edge_mask]
    edge_j = upper_j[edge_mask]
    edge_values = upper_values[edge_mask]

    # Union-find data structure
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for i, j in zip(edge_i, edge_j):
        union(int(i), int(j))

    components = {}

    for i, protein in enumerate(proteins):
        root = find(i)
        components.setdefault(root, []).append(protein)

    # Keep only groups containing at least two proteins
    groups = [
        members
        for members in components.values()
        if len(members) > 1
    ]

    groups.sort(key=len, reverse=True)

    membership_rows = []
    summary_rows = []

    for group_number, members in enumerate(groups, start=1):
        for protein in members:
            membership_rows.append({
                "group_id": group_number,
                "protein": protein,
                "group_size": len(members),
            })

        summary_rows.append({
            "group_id": group_number,
            "group_size": len(members),
            "proteins": "; ".join(members),
        })

    # Save all direct high-correlation relationships
    pair_rows = pd.DataFrame({
        "protein_1": [proteins[i] for i in edge_i],
        "protein_2": [proteins[j] for j in edge_j],
        "absolute_correlation": edge_values,
    })

    membership = pd.DataFrame(membership_rows)
    summary = pd.DataFrame(summary_rows)

    return pair_rows, membership, summary


# ============================================================
# 3. PROCESS EACH VIEW
# ============================================================

all_summaries = []

for view, paths in VIEWS.items():

    print("\n" + "=" * 60)
    print(f"PROCESSING {view.upper()}")
    print("=" * 60)

    if not paths["filtered"].exists():
        print(f"Input file not found: {paths['filtered']}")
        continue

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ids, df = load_protein_matrix(paths["filtered"])

    n_participants, n_proteins = df.shape

    print(f"Participants: {n_participants}")
    print(f"Protein features: {n_proteins}")

    proteins = df.columns.tolist()

    missingness = get_missingness(paths["raw"], proteins)
    variances = df.var(axis=0)

    # --------------------------------------------------------
    # Exact duplicate analysis
    # --------------------------------------------------------

    exact_duplicates = find_exact_duplicates(df)

    exact_duplicates.to_csv(
        OUTPUT_DIR / f"{view.lower()}_exact_duplicates.csv",
        index=False,
    )

    print(f"Exact duplicate proteins: {len(exact_duplicates)}")

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    print("Calculating pairwise correlations...")

    corr = df.corr(method="pearson")
    abs_corr = corr.abs()

    # Get unique upper-triangle correlations
    upper_i, upper_j = np.triu_indices(n_proteins, k=1)
    upper_values = abs_corr.to_numpy()[upper_i, upper_j]

    # --------------------------------------------------------
    # Correlation distribution
    # --------------------------------------------------------

    distribution = pd.DataFrame({
        "absolute_correlation": upper_values
    })

    distribution.to_csv(
        OUTPUT_DIR / f"{view.lower()}_correlation_distribution.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        upper_values,
        bins=50,
        edgecolor="black",
    )

    plt.axvline(
        GROUP_THRESHOLD,
        linestyle="--",
        label=f"Group threshold = {GROUP_THRESHOLD}",
    )

    plt.xlabel("Absolute Pearson correlation")
    plt.ylabel("Number of protein pairs")
    plt.title(f"{view} Protein Correlation Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{view.lower()}_correlation_distribution.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Threshold sensitivity analysis
    # --------------------------------------------------------

    threshold_results = []

    for threshold in THRESHOLDS:

        pair_count = int(np.sum(upper_values >= threshold))

        retained, removed = greedy_filter(
            abs_corr,
            variances,
            missingness,
            threshold,
        )

        threshold_results.append({
            "view": view,
            "threshold": threshold,
            "highly_correlated_pairs": pair_count,
            "proteins_removed": len(removed),
            "proteins_retained": len(retained),
            "retention_percentage": (
                len(retained) / n_proteins * 100
            ),
        })

        print(
            f"Threshold {threshold:.2f}: "
            f"{pair_count:,} correlated pairs, "
            f"{len(removed):,} removed, "
            f"{len(retained):,} retained"
        )

    threshold_df = pd.DataFrame(threshold_results)

    threshold_df.to_csv(
        OUTPUT_DIR / f"{view.lower()}_threshold_comparison.csv",
        index=False,
    )

    all_summaries.append(threshold_df)

    # --------------------------------------------------------
    # High-correlation pairs at GROUP_THRESHOLD
    # --------------------------------------------------------

    pair_mask = upper_values >= GROUP_THRESHOLD

    high_corr_pairs = pd.DataFrame({
        "protein_1": df.columns[upper_i[pair_mask]],
        "protein_2": df.columns[upper_j[pair_mask]],
        "absolute_correlation": upper_values[pair_mask],
    })

    high_corr_pairs = high_corr_pairs.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    high_corr_pairs.to_csv(
        OUTPUT_DIR / f"{view.lower()}_high_correlation_pairs.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Protein redundancy groups
    # --------------------------------------------------------

    pair_report, group_membership, group_summary = (
        find_redundancy_groups(abs_corr, GROUP_THRESHOLD)
    )

    pair_report.to_csv(
        OUTPUT_DIR / f"{view.lower()}_redundancy_group_pairs.csv",
        index=False,
    )

    group_membership.to_csv(
        OUTPUT_DIR / f"{view.lower()}_redundancy_group_membership.csv",
        index=False,
    )

    group_summary.to_csv(
        OUTPUT_DIR / f"{view.lower()}_redundancy_group_summary.csv",
        index=False,
    )

        # --------------------------------------------------------
    # Inspect largest redundancy groups
    # --------------------------------------------------------

    if not group_membership.empty:

        # Save members of the five largest groups
        largest_groups = (
            group_summary
            .head(5)["group_id"]
            .tolist()
        )

        largest_members = group_membership[
            group_membership["group_id"].isin(largest_groups)
        ].copy()

        largest_members.to_csv(
            OUTPUT_DIR /
            f"{view.lower()}_largest_redundancy_groups.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Protein connectivity (number of high-correlation links)
    # --------------------------------------------------------

    if not pair_report.empty:

        degree = pd.concat([
            pair_report["protein_1"],
            pair_report["protein_2"],
        ]).value_counts()

        degree_report = pd.DataFrame({
            "protein": degree.index,
            "high_correlation_connections": degree.values,
        })

        degree_report.to_csv(
            OUTPUT_DIR /
            f"{view.lower()}_protein_connectivity.csv",
            index=False,
        )

        print("\nMost highly connected proteins:")
        print(degree_report.head(15).to_string(index=False))

    # --------------------------------------------------------
    # Inspect exact duplicates
    # --------------------------------------------------------

    if not exact_duplicates.empty:

        print("\nExample exact duplicate pairs:")
        print(exact_duplicates.head(15).to_string(index=False))
        
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n_pairs = n_proteins * (n_proteins - 1) // 2

    print("\nRedundancy analysis:")
    print(f"Total possible protein pairs: {n_pairs:,}")
    print(
        f"Pairs with |r| >= {GROUP_THRESHOLD}: "
        f"{len(pair_report):,}"
    )
    print(f"Redundancy groups: {len(group_summary)}")

    if not group_summary.empty:
        print(
            f"Largest redundancy group: "
            f"{group_summary['group_size'].max()} proteins"
        )
        print("\nLargest groups:")
        print(
            group_summary[
                ["group_id", "group_size"]
            ].head(10).to_string(index=False)
        )

    print("\nCompleted:", view)


# ============================================================
# 4. COMBINED THRESHOLD SUMMARY
# ============================================================

if all_summaries:

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR / "combined_correlation_threshold_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETED")
    print("=" * 60)

    print(f"Output directory: {OUTPUT_DIR}")
    print(combined.to_string(index=False))