
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

PROCESSED = ROOT / "Datasets" / "Processed"
PROTEOMICS = PROCESSED / "Proteomics" / "PCA" / "PCA80"

OUTPUT = PROCESSED / "Clustering"
OUTPUT.mkdir(parents=True, exist_ok=True)

KEYS = ["PATNO", "EVENT_ID"]

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

clinical = pd.read_csv(
    PROCESSED / "baseline_clinical_raw.csv"
)

csf = pd.read_csv(
    PROTEOMICS / "baseline_csf_pca80.csv"
)

plasma = pd.read_csv(
    PROTEOMICS / "baseline_plasma_pca80.csv"
)

# --------------------------------------------------
# 2. Select clinical features
# --------------------------------------------------

clinical_features = [
    "age",
    "SEX",
    "EDUCYRS",
    "moca",
    "upsit",
]

clinical = clinical[KEYS + clinical_features].copy()

# Convert selected numeric variables
numeric_features = [
    "age",
    "EDUCYRS",
    "moca",
    "upsit",
]

for col in numeric_features:
    clinical[col] = pd.to_numeric(
        clinical[col], errors="coerce"
    )

# Clean categorical sex values
clinical["SEX"] = clinical["SEX"].astype("string").str.strip()
clinical["SEX"] = clinical["SEX"].replace(
    {"": pd.NA, "nan": pd.NA}
)

# --------------------------------------------------
# 3. Missingness report
# --------------------------------------------------

print("\nClinical missingness:")
print(clinical[clinical_features].isnull().sum())

# Median imputation for numeric variables
for col in numeric_features:
    clinical[col] = clinical[col].fillna(
        clinical[col].median()
    )

# Mode imputation for sex
if clinical["SEX"].isnull().any():
    clinical["SEX"] = clinical["SEX"].fillna(
        clinical["SEX"].mode()[0]
    )

# --------------------------------------------------
# 4. Encode sex
# --------------------------------------------------

clinical = pd.get_dummies(
    clinical,
    columns=["SEX"],
    drop_first=True,
    dtype=int
)

# --------------------------------------------------
# 5. Standardize clinical variables
# --------------------------------------------------

clinical_ids = clinical[KEYS].copy()

clinical_values = clinical.drop(
    columns=KEYS
)

scaler = StandardScaler()

clinical_scaled = pd.DataFrame(
    scaler.fit_transform(clinical_values),
    columns=clinical_values.columns,
    index=clinical_values.index
)

clinical_final = pd.concat(
    [clinical_ids, clinical_scaled],
    axis=1
)

# --------------------------------------------------
# 6. Align all three views
# --------------------------------------------------

def sort_and_validate(df, name):
    if df.duplicated("PATNO").any():
        raise ValueError(f"Duplicate PATNO in {name}")

    return df.sort_values("PATNO").reset_index(drop=True)

clinical_final = sort_and_validate(
    clinical_final, "Clinical"
)

csf = sort_and_validate(csf, "CSF")
plasma = sort_and_validate(plasma, "Plasma")

# Verify identical participant IDs
ids = clinical_final["PATNO"]

if not ids.equals(csf["PATNO"]):
    raise ValueError("Clinical and CSF participants do not match.")

if not ids.equals(plasma["PATNO"]):
    raise ValueError("Clinical and plasma participants do not match.")

if not (
    clinical_final["EVENT_ID"].equals(csf["EVENT_ID"])
    and clinical_final["EVENT_ID"].equals(plasma["EVENT_ID"])
):
    raise ValueError("Event IDs do not match across views.")

# --------------------------------------------------
# 7. Save final views
# --------------------------------------------------

clinical_final.to_csv(
    OUTPUT / "clinical_view.csv", index=False
)

csf.to_csv(
    OUTPUT / "csf_view.csv", index=False
)

plasma.to_csv(
    OUTPUT / "plasma_view.csv", index=False
)

# --------------------------------------------------
# 8. Summary
# --------------------------------------------------

print("\nThree-view preprocessing complete.")
print("Participants:", len(ids))

print("\nClinical view:", clinical_final.shape)
print("CSF view:", csf.shape)
print("Plasma view:", plasma.shape)

print("\nSaved files to:", OUTPUT)
