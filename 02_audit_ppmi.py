
from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path("Datasets/Raw")

FILES = {
    "clinical": (
        "PPMI_Curated_Data_Cut_Public_20260810.xlsx",
        "20260810",
    ),
    "csf": ("ppmi_project_9000_csf.xlsx", "Results"),
    "plasma": ("ppmi_project_9000_plasma.xlsx", "Results"),
}

KEYS = ["PATNO", "EVENT_ID"]


def normalize_id(series):
    """Normalize identifiers without converting missing values to strings."""
    result = series.astype("string").str.strip()
    # Excel sometimes represents integer IDs as values such as 1001.0
    result = result.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    result = result.replace("", pd.NA)
    return result


def load_dataset(name, filename, sheet):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet)

    missing_columns = [col for col in KEYS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{name}: missing columns {missing_columns}")

    for col in KEYS:
        df[col] = normalize_id(df[col])

    return df


def report_dataset(name, df):
    print(f"\n{'=' * 72}")
    print(f"{name.upper()}")
    print(f"{'=' * 72}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Unique participants: {df['PATNO'].nunique():,}")
    print(f"Rows missing PATNO: {df['PATNO'].isna().sum():,}")
    print(f"Rows missing EVENT_ID: {df['EVENT_ID'].isna().sum():,}")

    valid = df.dropna(subset=KEYS)
    duplicate_mask = valid.duplicated(KEYS, keep=False)

    print(f"Unique participant-visit pairs: {len(valid.drop_duplicates(KEYS)):,}")
    print(f"Rows belonging to duplicated pairs: {duplicate_mask.sum():,}")
    print(f"Number of duplicated pairs: "
          f"{valid.duplicated(KEYS).sum():,}")

    print("\nMost frequent EVENT_ID values:")
    print(df["EVENT_ID"].value_counts(dropna=False).head(15).to_string())


def valid_keys(df):
    return set(
        map(tuple, df.dropna(subset=KEYS)[KEYS]
            .drop_duplicates().itertuples(index=False, name=None))
    )


def find_columns(df, patterns):
    return [
        col for col in df.columns
        if any(re.search(pattern, col, re.IGNORECASE)
               for pattern in patterns)
    ]


# 1. Load the three datasets
clinical, csf, plasma = (
    load_dataset(name, filename, sheet)
    for name, (filename, sheet) in FILES.items()
)

# 2. Basic data-quality audit
for name, df in [
    ("Clinical", clinical),
    ("CSF", csf),
    ("Plasma", plasma),
]:
    report_dataset(name, df)

# 3. Compare participant coverage
clinical_patnos = set(clinical["PATNO"].dropna())
csf_patnos = set(csf["PATNO"].dropna())
plasma_patnos = set(plasma["PATNO"].dropna())

print(f"\n{'=' * 72}")
print("PARTICIPANT OVERLAP")
print(f"{'=' * 72}")
print(f"Clinical participants: {len(clinical_patnos):,}")
print(f"CSF participants: {len(csf_patnos):,}")
print(f"Plasma participants: {len(plasma_patnos):,}")
print(f"CSF AND plasma: {len(csf_patnos & plasma_patnos):,}")
print(f"CSF only, relative to plasma: "
      f"{len(csf_patnos - plasma_patnos):,}")
print(f"Plasma only, relative to CSF: "
      f"{len(plasma_patnos - csf_patnos):,}")
print(f"Participants in all three datasets: "
      f"{len(clinical_patnos & csf_patnos & plasma_patnos):,}")

# 4. Compare participant-visit pairs
clinical_keys = valid_keys(clinical)
csf_keys = valid_keys(csf)
plasma_keys = valid_keys(plasma)

print(f"\n{'=' * 72}")
print("PARTICIPANT-VISIT OVERLAP")
print(f"{'=' * 72}")
print(f"Clinical visit pairs: {len(clinical_keys):,}")
print(f"CSF visit pairs: {len(csf_keys):,}")
print(f"Plasma visit pairs: {len(plasma_keys):,}")
print(f"CSF AND plasma visit pairs: "
      f"{len(csf_keys & plasma_keys):,}")
print(f"All three share the same visit: "
      f"{len(clinical_keys & csf_keys & plasma_keys):,}")

print(f"\nCSF visits without a matching clinical visit: "
      f"{len(csf_keys - clinical_keys):,}")
print(f"Plasma visits without a matching clinical visit: "
      f"{len(plasma_keys - clinical_keys):,}")

# 5. Identify columns relevant to the project
patterns = [
    r"\bMOCA\b",
    r"PRS|POLYGENIC|PGS",
    r"SAA|SEED.?AMPLIFICATION|ALPHA.?SYNUCLEIN",
    r"AGE|SEX|EDUC",
    r"PATNO|EVENT_ID",
]

print(f"\n{'=' * 72}")
print("RELEVANT CLINICAL COLUMNS")
print(f"{'=' * 72}")

for pattern in patterns:
    matches = find_columns(clinical, [pattern])
    print(f"\nPattern {pattern}: {len(matches)} columns")
    print(matches[:60])

# 6. Check proteomics feature structure
for name, df in [("CSF", csf), ("Plasma", plasma)]:
    protein_cols = [col for col in df.columns if col not in KEYS]
    print(f"\n{name} feature columns: {len(protein_cols):,}")
    print("First 10:", protein_cols[:10])

    # Fraction missing across all feature cells
    print(f"Overall feature-cell missingness: "
          f"{df[protein_cols].isna().mean().mean():.2%}")

    all_missing = df[protein_cols].isna().all(axis=0)
    print(f"Completely missing feature columns: {all_missing.sum():,}")

print("\nAudit complete. No source files were modified.")
