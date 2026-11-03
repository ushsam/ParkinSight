
import pandas as pd
from pathlib import Path

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "Datasets" / "Raw"
PROCESSED_DIR = ROOT / "Datasets" / "Processed"

PROTEOMICS_DIR = PROCESSED_DIR / "Proteomics" / "PCA" / "PCA80"
CLUSTER_DIR = PROCESSED_DIR / "Clustering" / "Results"

OUTPUT_DIR = PROCESSED_DIR / "Longitudinal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = (
    RAW_DIR / "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
)

CLINICAL_SHEET = "20260810"

BASELINE_CLINICAL_FILE = (
    PROCESSED_DIR / "baseline_clinical_raw.csv"
)

CSF_FILE = PROTEOMICS_DIR / "baseline_csf_pca80.csv"
PLASMA_FILE = PROTEOMICS_DIR / "baseline_plasma_pca80.csv"

# CHANGE THIS to your selected clustering output
CLUSTER_FILE = CLUSTER_DIR / "all_views_k2_labels.csv"

# Change this if your actual assessment date column has another name
DATE_COLUMN = "visit_date"

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_patno(series):
    """Standardize participant IDs across Excel and CSV files."""
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


def find_column(df, column_name):
    """Find a column without depending on capitalization."""
    matches = [
        col for col in df.columns
        if col.strip().lower() == column_name.lower()
    ]

    if not matches:
        return None

    return matches[0]


def load_baseline_file(path, name):
    """Load a baseline file and validate participant identifiers."""
    df = pd.read_csv(path)

    if "PATNO" not in df.columns:
        raise ValueError(f"PATNO missing from {name}")

    df["PATNO"] = clean_patno(df["PATNO"])

    if "EVENT_ID" in df.columns:
        df["EVENT_ID"] = df["EVENT_ID"].astype(str).str.strip()
        df = df[df["EVENT_ID"] == "BL"].copy()

    if df["PATNO"].duplicated().any():
        raise ValueError(f"Duplicate baseline PATNO values in {name}")

    return df


# ============================================================
# 3. LOAD BASELINE COHORT
# ============================================================

cohort_file = PROCESSED_DIR / "baseline_cohort_ids.csv"
cohort = pd.read_csv(cohort_file)

cohort["PATNO"] = clean_patno(cohort["PATNO"])
cohort_ids = set(cohort["PATNO"].dropna())

print("Baseline cohort:", len(cohort_ids))


# ============================================================
# 4. LOAD BASELINE CLUSTER ASSIGNMENTS
# ============================================================

labels = pd.read_csv(CLUSTER_FILE)

labels["PATNO"] = clean_patno(labels["PATNO"])
labels["EVENT_ID"] = labels["EVENT_ID"].astype(str).str.strip()

clusters = labels.loc[
    labels["EVENT_ID"] == "BL",
    ["PATNO", "cluster"]
].copy()

clusters = clusters.rename(
    columns={"cluster": "baseline_cluster"}
)

clusters = clusters.dropna(subset=["PATNO"])

if clusters["PATNO"].duplicated().any():
    raise ValueError("Duplicate baseline cluster assignments found.")

clusters["baseline_cluster"] = clusters["baseline_cluster"].astype(int)

if set(clusters["PATNO"]) != cohort_ids:
    raise ValueError(
        "Cluster assignments do not match the 206-person baseline cohort. "
        f"Missing: {len(cohort_ids - set(clusters['PATNO']))}; "
        f"Extra: {len(set(clusters['PATNO']) - cohort_ids)}"
    )

clusters.to_csv(
    OUTPUT_DIR / "baseline_cluster_assignments.csv",
    index=False
)


# ============================================================
# 5. BUILD LONGITUDINAL MOCA DATASET
# ============================================================

print("\nLoading full clinical dataset...")

clinical = pd.read_excel(
    CLINICAL_FILE,
    sheet_name=CLINICAL_SHEET
)

clinical.columns = clinical.columns.str.strip()

# Identify important columns
patno_col = find_column(clinical, "PATNO")
event_col = find_column(clinical, "EVENT_ID")
moca_col = find_column(clinical, "MOCA")
date_col = find_column(clinical, DATE_COLUMN)

required = {
    "PATNO": patno_col,
    "EVENT_ID": event_col,
    "MOCA": moca_col,
    "Assessment date": date_col
}

missing = [name for name, col in required.items() if col is None]

if missing:
    raise ValueError(
        f"Required columns not found: {missing}\n"
        f"Available columns: {clinical.columns.tolist()}"
    )

# Standardize identifiers
clinical["PATNO"] = clean_patno(clinical[patno_col])
clinical["EVENT_ID"] = (
    clinical[event_col].astype(str).str.strip()
)

# Keep only the original baseline cohort
clinical = clinical[
    clinical["PATNO"].isin(cohort_ids)
].copy()

# Standardize outcome and date
clinical["moca"] = pd.to_numeric(
    clinical[moca_col],
    errors="coerce"
)

clinical["assessment_date"] = pd.to_datetime(
    clinical[date_col],
    format="%m/%Y",
    errors="coerce"
)

# Check participant-visit uniqueness
duplicates = clinical.duplicated(
    subset=["PATNO", "EVENT_ID"],
    keep=False
)

if duplicates.any():
    raise ValueError(
        "Duplicate PATNO + EVENT_ID records found. "
        "Inspect these records before proceeding."
    )

# Calculate baseline date for each participant
baseline_dates = (
    clinical.loc[
        clinical["EVENT_ID"] == "BL",
        ["PATNO", "assessment_date"]
    ]
    .dropna(subset=["assessment_date"])
    .set_index("PATNO")["assessment_date"]
)

clinical["baseline_date"] = clinical["PATNO"].map(
    baseline_dates
)

# Calculate elapsed time in months
clinical["months_from_baseline"] = (
    (clinical["assessment_date"] - clinical["baseline_date"])
    .dt.total_seconds()
    / (60 * 60 * 24 * 30.4375)
)

# Attach fixed baseline cluster to every visit
longitudinal = clinical.merge(
    clusters,
    on="PATNO",
    how="inner",
    validate="many_to_one"
)

# Keep analysis variables
longitudinal = longitudinal[
    [
        "PATNO",
        "EVENT_ID",
        "assessment_date",
        "months_from_baseline",
        "moca",
        "baseline_cluster"
    ]
].copy()

longitudinal = longitudinal.sort_values(
    ["PATNO", "assessment_date"]
).reset_index(drop=True)

longitudinal.to_csv(
    OUTPUT_DIR / "longitudinal_moca.csv",
    index=False
)


# ============================================================
# 6. BUILD BASELINE PREDICTOR DATASET
# ============================================================

print("\nBuilding baseline predictor dataset...")

baseline_clinical = load_baseline_file(
    BASELINE_CLINICAL_FILE,
    "baseline clinical data"
)

# Select baseline demographic and clinical variables
clinical_variables = [
    "PATNO",
    "age",
    "SEX",
    "EDUCYRS",
    "duration_yrs",
    "moca"
]

available_clinical = [
    col for col in clinical_variables
    if col in baseline_clinical.columns
]

baseline_predictors = baseline_clinical[
    available_clinical
].copy()

# Rename MoCA to make its baseline role explicit
moca_col = find_column(baseline_predictors, "moca")

if moca_col:
    baseline_predictors = baseline_predictors.rename(
        columns={moca_col: "baseline_moca"}
    )

# Rename duration for clarity
duration_col = find_column(
    baseline_predictors, "duration_yrs"
)

if duration_col:
    baseline_predictors = baseline_predictors.rename(
        columns={duration_col: "baseline_disease_duration_yrs"}
    )

# Load baseline CSF and plasma PCA features
csf = load_baseline_file(CSF_FILE, "CSF PCA")
plasma = load_baseline_file(PLASMA_FILE, "plasma PCA")

# Keep participant ID and PCA features
csf = csf.drop(columns=["EVENT_ID"], errors="ignore")
plasma = plasma.drop(columns=["EVENT_ID"], errors="ignore")

csf = csf.rename(
    columns={
        col: f"CSF_{col}"
        for col in csf.columns
        if col != "PATNO"
    }
)

plasma = plasma.rename(
    columns={
        col: f"Plasma_{col}"
        for col in plasma.columns
        if col != "PATNO"
    }
)

# Merge all baseline predictors
baseline_predictors = baseline_predictors.merge(
    clusters,
    on="PATNO",
    how="inner",
    validate="one_to_one"
)

baseline_predictors = baseline_predictors.merge(
    csf,
    on="PATNO",
    how="inner",
    validate="one_to_one"
)

baseline_predictors = baseline_predictors.merge(
    plasma,
    on="PATNO",
    how="inner",
    validate="one_to_one"
)

if len(baseline_predictors) != len(cohort_ids):
    raise ValueError(
        "Baseline predictor dataset does not contain all "
        "baseline cohort participants."
    )

baseline_predictors = baseline_predictors.sort_values(
    "PATNO"
).reset_index(drop=True)

baseline_predictors.to_csv(
    OUTPUT_DIR / "baseline_predictors.csv",
    index=False
)


# ============================================================
# 7. QUALITY CHECKS AND SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("LONGITUDINAL DATASET SUMMARY")
print("=" * 55)

print("Rows:", len(longitudinal))
print("Participants:", longitudinal["PATNO"].nunique())
print("MoCA measurements:", longitudinal["moca"].notna().sum())

moca_counts = (
    longitudinal.dropna(subset=["moca"])
    .groupby("PATNO")["moca"]
    .count()
)

print("Participants with 2+ MoCA measurements:", (moca_counts >= 2).sum())
print("Participants with 3+ MoCA measurements:", (moca_counts >= 3).sum())

print("\nVisit counts:")
print(longitudinal["EVENT_ID"].value_counts())

print("\nCluster counts:")
print(clusters["baseline_cluster"].value_counts())

print("\n" + "=" * 55)
print("BASELINE PREDICTOR DATASET SUMMARY")
print("=" * 55)

print("Participants:", len(baseline_predictors))
print("Total columns:", len(baseline_predictors.columns))

print("\nMissing values:")
print(
    baseline_predictors.isna().sum()[
        baseline_predictors.isna().sum() > 0
    ]
)

print("\nSaved files:")
print(OUTPUT_DIR / "baseline_cluster_assignments.csv")
print(OUTPUT_DIR / "longitudinal_moca.csv")
print(OUTPUT_DIR / "baseline_predictors.csv")
