
from pathlib import Path
import pandas as pd

OUT_DIR = Path("Datasets/Processed")

df = pd.read_csv(
    OUT_DIR / "baseline_clinical_raw.csv",
    low_memory=False
)

candidates = [
    "age",
    "SEX",
    "EDUCYRS",
    "PRIMDIAG",
    "COHORT",
    "subgroup",
    "age_at_visit",
    "moca",
    "GP2_proxy_PGS",
    "META5_proxy_PGS",
    "CSFSAA",
    "CSFSAA_assay",
]

# Discover possible baseline motor-severity variables.
motor_candidates = [
    c for c in df.columns
    if any(term in c.lower() for term in [
        "updrs", "hoehn", "yahr", "motor"
    ])
]

print("BASELINE FEATURE AUDIT")
print("=" * 70)

for col in candidates:
    if col not in df.columns:
        print(f"\n{col}: NOT FOUND")
        continue

    s = df[col]
    print(f"\n{col}")
    print(f"  dtype: {s.dtype}")
    print(f"  missing: {s.isna().sum()} ({s.isna().mean():.1%})")
    print(f"  unique nonmissing values: {s.nunique()}")

    if s.nunique(dropna=True) <= 20:
        print("  value counts:")
        print(s.value_counts(dropna=False).head(20).to_string())
    elif pd.api.types.is_numeric_dtype(s):
        print("  descriptive statistics:")
        print(s.describe().round(3).to_string())

print("\nPOTENTIAL MOTOR-SEVERITY VARIABLES")
print("=" * 70)

for col in motor_candidates:
    s = df[col]
    print(
        f"{col}: missing={s.isna().mean():.1%}, "
        f"unique={s.nunique(dropna=True)}, dtype={s.dtype}"
    )

# Save a short review table
review_cols = [
    c for c in candidates + motor_candidates
    if c in df.columns
]

review = pd.DataFrame({
    "variable": review_cols,
    "dtype": [str(df[c].dtype) for c in review_cols],
    "missing_fraction": [df[c].isna().mean() for c in review_cols],
    "unique_nonmissing": [df[c].nunique(dropna=True) for c in review_cols],
})

review.to_csv(
    OUT_DIR / "clinical_candidate_audit.csv",
    index=False
)

print("\nSaved:", OUT_DIR / "clinical_candidate_audit.csv")
