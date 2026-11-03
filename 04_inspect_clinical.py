
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "Datasets" / "Processed" /
    "baseline_clinical_raw.csv"
)

df = pd.read_csv(INPUT_FILE)

print("Clinical dataset shape:", df.shape)
print("\nAvailable columns:\n")

for col in df.columns:
    print(col)

# Search for candidate variables
keywords = [
    "AGE", "SEX", "EDUC", "DURATION",
    "MOCA", "HY", "HOEHN", "UPSIT",
    "OLFACT"
]

print("\n\nPotential clustering features:")

for keyword in keywords:
    matches = [
        col for col in df.columns
        if keyword.lower() in col.lower()
    ]

    print(f"\n{keyword}:")
    for col in matches:
        print("  ", col)

# Check participant and event identifiers
print("\nUnique participants:", df["PATNO"].nunique())
print("Event IDs:", df["EVENT_ID"].value_counts().to_dict())
