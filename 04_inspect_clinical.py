
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Datasets/Raw")
OUT_DIR = Path("Datasets/Processed")

clinical = pd.read_csv(
    OUT_DIR / "baseline_clinical_raw.csv",
    low_memory=False
)

dictionary = pd.read_excel(
    DATA_DIR / "PPMI_Curated_Data_Cut_Public_20260810.xlsx",
    sheet_name="Data dictionary"
)

# Normalize column names used to match the dictionary
dictionary["Variable"] = dictionary["Variable"].astype("string").str.strip()

# Calculate baseline missingness
summary = pd.DataFrame({
    "variable": clinical.columns,
    "dtype": [str(clinical[c].dtype) for c in clinical.columns],
    "missing_count": [clinical[c].isna().sum() for c in clinical.columns],
    "missing_fraction": [clinical[c].isna().mean() for c in clinical.columns],
    "n_unique": [clinical[c].nunique(dropna=True) for c in clinical.columns],
})

# Attach variable descriptions and categories
dict_cols = [
    c for c in [
        "Variable", "Category", "Description",
        "Derived Variable", "Original Variable(s)",
        "Derivation Notes"
    ]
    if c in dictionary.columns
]

summary = summary.merge(
    dictionary[dict_cols].drop_duplicates("Variable"),
    left_on="variable",
    right_on="Variable",
    how="left"
).drop(columns=["Variable"], errors="ignore")

# Flag variables that should not define the clustering kernel
reserved_patterns = (
    r"^PATNO$|^EVENT_ID$|^SITE$|^visit_date$|"
    r"\bMOCA\b|^moca$|PRS|PGS|SAA|"
    r"Death|^age_death$"
)

summary["reserved_or_review"] = summary["variable"].str.contains(
    reserved_patterns, case=False, regex=True, na=False
)

# Flag high missingness for review, not automatic exclusion
summary["high_missingness_review"] = (
    summary["missing_fraction"] > 0.20
)

summary = summary.sort_values(
    ["reserved_or_review", "missing_fraction", "variable"],
    ascending=[True, True, True]
)

summary.to_csv(
    OUT_DIR / "clinical_feature_review.csv",
    index=False
)

print("Baseline participants:", len(clinical))
print("Clinical variables:", len(clinical.columns))
print("Variables with >20% missingness:",
      int(summary["high_missingness_review"].sum()))

print("\nDEMOGRAPHIC CANDIDATES")
demographic_names = [
    "age", "age_at_visit", "SEX", "EDUCYRS",
    "COHORT", "subgroup", "PRIMDIAG"
]
print(
    summary[summary["variable"].isin(demographic_names)][
        ["variable", "dtype", "missing_fraction",
         "Category", "Description"]
    ].to_string(index=False)
)

print("\nVARIABLES WITH HIGHEST MISSINGNESS")
print(
    summary.sort_values("missing_fraction", ascending=False)[
        ["variable", "missing_count", "missing_fraction",
         "Category", "Description"]
    ].head(30).to_string(index=False)
)

print("\nMOCA / PRS / SAA FIELDS")
mask = summary["variable"].str.contains(
    r"MOCA|PRS|PGS|SAA",
    case=False, regex=True, na=False
)
print(
    summary.loc[mask, [
        "variable", "missing_fraction", "Category", "Description"
    ]].to_string(index=False)
)

print("\nSaved:", OUT_DIR / "clinical_feature_review.csv")
