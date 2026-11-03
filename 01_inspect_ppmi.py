
from pathlib import Path
import pandas as pd

# Run this script from the project root
DATA_DIR = Path("Datasets/Raw")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_DIR.resolve()}. "
        "Run this script from your project root."
    )

# 1. List available datasets
files = sorted(
    p for p in DATA_DIR.iterdir()
    if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".parquet"}
)

print(f"Data directory: {DATA_DIR.resolve()}")
print(f"Found {len(files)} data files\n")

for file in files:
    print(f"- {file.name}")

print("\n" + "=" * 70)

# 2. Inspect each file without modifying the source data
for file in files:
    print(f"\nFILE: {file.name}")

    try:
        if file.suffix.lower() in {".xlsx", ".xls"}:
            excel = pd.ExcelFile(file)
            print("Sheets:", excel.sheet_names)

            for sheet in excel.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet, nrows=5)
                print(f"\n  Sheet: {sheet}")
                print(f"  Preview shape: {df.shape}")
                print("  Columns:", df.columns.tolist()[:25])

        elif file.suffix.lower() == ".csv":
            df = pd.read_csv(file, nrows=5, low_memory=False)
            print("  Preview shape:", df.shape)
            print("  Columns:", df.columns.tolist()[:25])

        elif file.suffix.lower() == ".parquet":
            df = pd.read_parquet(file)
            print("  Shape:", df.shape)
            print("  Columns:", df.columns.tolist()[:25])

    except Exception as exc:
        print(f"  Inspection error: {exc}")


file = "Datasets/Raw/PPMI_Curated_Data_Cut_Public_20260810.xlsx"

clinical = pd.read_excel(
    file,
    sheet_name="20260810"
)

# Find columns containing date-related terms
date_columns = [
    col for col in clinical.columns
    if any(term in col.upper() for term in ["DATE", "DT"])
]

print("Potential date columns:")
print(date_columns)

# Inspect sample values
for col in date_columns:
    print(f"\n{col}:")
    print(clinical[col].dropna().head(10).tolist())
