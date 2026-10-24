
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Datasets/Raw")
OUT_DIR = Path("Datasets/Processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = DATA_DIR / "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
CSF_FILE = DATA_DIR / "ppmi_project_9000_csf.xlsx"
PLASMA_FILE = DATA_DIR / "ppmi_project_9000_plasma.xlsx"

KEYS = ["PATNO", "EVENT_ID"]


def normalize_id(series):
    values = series.astype("string").str.strip()
    values = values.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return values.replace("", pd.NA)


def load_excel(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)

    for col in KEYS:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing {col}")
        df[col] = normalize_id(df[col])

    return df


# 1. Load datasets
clinical = load_excel(CLINICAL_FILE, "20260810")
csf = load_excel(CSF_FILE, "Results")
plasma = load_excel(PLASMA_FILE, "Results")

# 2. Select baseline records
clinical_bl = clinical.loc[clinical["EVENT_ID"] == "BL"].copy()
csf_bl = csf.loc[csf["EVENT_ID"] == "BL"].copy()
plasma_bl = plasma.loc[plasma["EVENT_ID"] == "BL"].copy()

# 3. Validate uniqueness before merging
for name, df in [
    ("Clinical baseline", clinical_bl),
    ("CSF baseline", csf_bl),
    ("Plasma baseline", plasma_bl),
]:
    if df["PATNO"].isna().any():
        raise ValueError(f"{name}: missing PATNO values")

    if df["PATNO"].duplicated().any():
        raise ValueError(
            f"{name}: multiple BL rows for a participant. "
            "Investigate before merging."
        )

    print(f"{name}: {len(df):,} rows, "
          f"{df['PATNO'].nunique():,} participants")

# 4. Determine baseline overlap
clinical_ids = set(clinical_bl["PATNO"])
csf_ids = set(csf_bl["PATNO"])
plasma_ids = set(plasma_bl["PATNO"])

common_ids = sorted(clinical_ids & csf_ids & plasma_ids)

print("\nBASELINE COHORT OVERLAP")
print(f"Clinical baseline: {len(clinical_ids):,}")
print(f"CSF baseline: {len(csf_ids):,}")
print(f"Plasma baseline: {len(plasma_ids):,}")
print(f"All three at baseline: {len(common_ids):,}")
print(f"CSF baseline without plasma baseline: "
      f"{len(csf_ids - plasma_ids):,}")
print(f"Plasma baseline without CSF baseline: "
      f"{len(plasma_ids - csf_ids):,}")

if not common_ids:
    raise ValueError(
        "No shared baseline participants. "
        "Check ID formatting and EVENT_ID values."
    )

# 5. Restrict each view to the exact same participant cohort
clinical_cohort = (
    clinical_bl[clinical_bl["PATNO"].isin(common_ids)]
    .sort_values("PATNO")
    .reset_index(drop=True)
)

csf_cohort = (
    csf_bl[csf_bl["PATNO"].isin(common_ids)]
    .sort_values("PATNO")
    .reset_index(drop=True)
)

plasma_cohort = (
    plasma_bl[plasma_bl["PATNO"].isin(common_ids)]
    .sort_values("PATNO")
    .reset_index(drop=True)
)

# 6. Verify participant alignment
assert clinical_cohort["PATNO"].tolist() == common_ids
assert csf_cohort["PATNO"].tolist() == common_ids
assert plasma_cohort["PATNO"].tolist() == common_ids

assert clinical_cohort["PATNO"].is_unique
assert csf_cohort["PATNO"].is_unique
assert plasma_cohort["PATNO"].is_unique

# 7. Report missingness by view
missingness_rows = []

for view_name, df in [
    ("clinical", clinical_cohort),
    ("csf", csf_cohort),
    ("plasma", plasma_cohort),
]:
    for col in df.columns:
        missingness_rows.append({
            "view": view_name,
            "variable": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_fraction": float(df[col].isna().mean()),
        })

missingness = pd.DataFrame(missingness_rows)
missingness.to_csv(OUT_DIR / "baseline_missingness.csv", index=False)

# 8. Save aligned raw views separately
# These are raw cohort extracts, not final clustering features.
clinical_cohort.to_csv(
    OUT_DIR / "baseline_clinical_raw.csv", index=False
)
csf_cohort.to_csv(
    OUT_DIR / "baseline_csf_raw.csv", index=False
)
plasma_cohort.to_csv(
    OUT_DIR / "baseline_plasma_raw.csv", index=False
)

pd.DataFrame({"PATNO": common_ids}).to_csv(
    OUT_DIR / "baseline_cohort_ids.csv", index=False
)

print("\nSAVED FILES")
for filename in [
    "baseline_cohort_ids.csv",
    "baseline_clinical_raw.csv",
    "baseline_csf_raw.csv",
    "baseline_plasma_raw.csv",
    "baseline_missingness.csv",
]:
    print(f"  {OUT_DIR / filename}")

print("\nFINAL SHAPES")
print("Clinical:", clinical_cohort.shape)
print("CSF:", csf_cohort.shape)
print("Plasma:", plasma_cohort.shape)
print("\nBaseline cohort construction complete.")
