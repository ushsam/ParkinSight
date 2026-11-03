# ParkinSight
## Parkinson’s Disease Multiview Proteomics Clustering

## 1. Project Setup

Create the following folder structure:

```text
ParkinSight/
├── README.md
├── 01_extract_baseline_cohort.py
├── 02_preprocess_proteomics.py
├── 03_correlation_filtering.py
├── 04_correlation_diagnostics.py
├── 05_proteomics_pca.py
├── 06_prepare_pca_80.py
├── 07_prepare_clustering_views.py
├── 08_single_multiview_clustering.py
├── 09_kernel_diagnostics.py
├── 10_bandwidth_sensitivity.py
├── 11_kernel_weight_sensitivity.py
└── Datasets/
    ├── Raw/
    │   ├── PPMI_Curated_Data_Cut_Public_20260810.xlsx
    │   ├── ppmi_project_9000_csf.xlsx
    │   └── ppmi_project_9000_plasma.xlsx
    └── Processed/
        ├── baseline_*.csv
        ├── Proteomics/
        │   ├── CorrelationFiltered/
        │   └── PCA/
        │       └── PCA80/
        └── Clustering/
            ├── Results/
            ├── KernelDiagnostics/
            ├── BandwidthSensitivity/
            └── KernelWeightSensitivity/
```

- **Raw:** Contains the original PPMI clinical and Project 9000 proteomics files.
- **Processed:** Stores extracted cohorts, cleaned data, dimensionality-reduction outputs, prepared clustering views, and analysis results.

Scripts read source data from `Datasets/Raw/` and save generated outputs under `Datasets/Processed/`. Check the path definitions if the project directory changes.

## 2. Project Goal

Identify potential Parkinson’s disease subtypes by comparing single-view and multiview clustering approaches using three baseline data views:

- **Clinical:** Basic demographic and clinical characteristics.
- **CSF:** Cerebrospinal fluid proteomics.
- **Plasma:** Plasma proteomics.

The current clustering inputs exclude PRS, SAA status, longitudinal MoCA measurements, and UPDRS. These variables are reserved for downstream cluster characterization and evaluation.

The longer-term goal is to investigate whether identified clusters differ in genetic risk, alpha-synuclein pathology, and longitudinal cognitive decline.

## 3. Analysis Workflow

Run the numbered Python scripts from the project root in order.

### Baseline cohort and preprocessing

1. **`03_extract_baseline_cohort.py`** – Extracts baseline (`BL`) clinical, CSF, and plasma data and identifies participants with all three data views.
2. **`03A_preprocess_proteomics.py`** – Removes non-protein metadata and missingness indicator columns, filters features with excessive missingness, imputes remaining missing values, and removes constant features.
3. **`03B_correlation_filtering.py`** – Removes highly correlated protein features using a 0.95 correlation threshold.
4. **`03C_correlation_diagnostics.py`** – Evaluates correlation distributions, threshold sensitivity, duplicate features, and redundant protein groups.

### Dimensionality reduction and clustering views

5. **`03D_proteomics_pca.py`** – Standardizes protein features and performs separate PCA for CSF and plasma.
6. **`04_inspect_clinical.py`** – Inspects for potential clinical features to include
7. **`05_prepare_clustering_views.py`** – Prepares and aligns the final clinical, CSF, and plasma clustering matrices.

### Clustering and sensitivity analysis

8. **`06_single_multiview_clustering.py`** – Constructs RBF kernels, performs single-view and multiview spectral clustering, and evaluates silhouette scores and cluster sizes.
9. **`07_kernel_diagnostics.py`** – Examines within-view similarity distributions, kernel alignment, pairwise kernel correlations, and eigenvalue properties.
10. **`08_bandwidth_sensitivity.py`** – Tests different RBF kernel bandwidth multipliers to assess their effect on clustering results.
11. **`09_kernel_weight_sensitivity.py`** – Compares equal, view-dominant, and two-view kernel combinations.

The Project 9000 Olink NPX values are already normalized. The workflow therefore does not repeat Olink normalization. Standardization is applied before PCA so that proteins contribute on comparable scales.

## 4. Current Analysis Status

### Cohort and prepared views

The baseline cohort contains **206 participants** with complete clinical, CSF, and plasma data.

| View | Features used for clustering |
|---|---:|
| Clinical | 5 |
| CSF | 83 principal components |
| Plasma | 74 principal components |

The clinical view includes age, sex, education years, MoCA, and UPSIT. Duration of disease was excluded from the primary clustering view because of substantial missingness.

For proteomics, correlation filtering retained 1,447 CSF proteins and 1,380 plasma proteins before PCA.

### Initial clustering results

The initial equal-weight kernel integration did not outperform clinical-only clustering based on silhouette scores.

| Model | k = 2 silhouette | k = 3 silhouette |
|---|---:|---:|
| Clinical only | 0.1547 | 0.1881 |
| CSF only | 0.1040 | 0.1581 |
| Plasma only | 0.0804 | 0.0682 |
| Equal-weight multiview | 0.0444 | 0.0366 |

### Sensitivity analysis

Lower RBF bandwidth multipliers generally improved clustering separation.

The strongest current candidate results include:

| Candidate model | Clusters | Silhouette |
|---|---:|---:|
| Clinical only, γ × 0.25 | 2 | 0.2016 |
| Clinical only, γ × 0.25 | 3 | 0.1932 |
| Clinical-dominant multiview, γ × 0.25 | 3 | 0.1206 |
| CSF-dominant multiview, γ × 0.25 | 2 | 0.1083 |

Weight sensitivity analysis showed that clinical-dominant and CSF-dominant combinations generally performed better than equal weighting. However, the multiview solutions still had lower silhouette scores than the strongest clinical-only solutions.

These findings suggest that the views capture different participant similarity structures, but integrating them does not automatically improve cluster separation. The current results are exploratory and do not yet establish biologically meaningful subtypes.

## 5. Next Steps

1. **Clustering stability:** Repeat each candidate clustering solution across multiple random seeds and compare assignments using pairwise adjusted Rand indices (ARI).
2. **Candidate selection:** Compare stability, silhouette scores, and cluster sizes before selecting solutions for further analysis.
3. **Cluster characterization:** Examine variables excluded from clustering, including disease duration, PRS, and SAA status.
4. **Longitudinal analysis:** Use mixed-effects models to evaluate whether MoCA decline differs across candidate clusters.

**Current status:** Baseline cohort extraction, proteomics preprocessing, correlation filtering, PCA, single-view and multiview clustering, kernel diagnostics, bandwidth sensitivity, and kernel-weight sensitivity analyses are complete. Clustering stability testing is the next step. All identified clusters remain exploratory pending stability assessment and downstream biological validation.