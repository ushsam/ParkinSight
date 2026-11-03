# ParkinSight
## Parkinson’s Disease Multiview Proteomics Clustering

## 1. Project Setup

Create the following folder structure:

```text
ParkinSight/
├── README.md
├── 01_inspect_ppmi.py
├── 02_audit_ppmi.py
├── 03_construct_baseline_cohort.py
├── 03A_data_preprocessing.py
├── 03B_correlation_filtering.py
├── 03C_correlation_diagnostics.py
├── 03D_proteomics_pca.py
├── 04_inspect_clinical.py
├── 05_prepare_clustering_views.py
├── 06_single_multiview_clustering.py
├── 07_build_longitudinal_dataset.py
├── 08_pd_only_clustering.py
└── Datasets/
    ├── Raw/
    │   ├── PPMI_Curated_Data_Cut_Public_20260810.xlsx
    │   ├── ppmi_project_9000_csf.xlsx
    │   └── ppmi_project_9000_plasma.xlsx
    └── Processed/
        ├── baseline_*.csv
        ├── Proteomics/
        │   ├── CorrelationFiltered/
        │   ├── PCA/
        │   │   └── PCA80/
        │   └── ...
        ├── Clustering/
        │   ├── Results/
        |   ├── SensitivityAnalysis/
        │   
        └── PD_Only/
            ├── Views/
            ├── PCA/
            └── Results/
```

- **Raw:** Contains the original PPMI clinical and Project 9000 proteomics files.
- **Processed:** Stores extracted cohorts, cleaned data, dimensionality-reduction outputs, prepared clustering views, and analysis results.

Scripts read source data from `Datasets/Raw/` and save generated outputs under `Datasets/Processed/`. Check the path definitions if the project directory changes.

## 2. Project Goal

Identify potential Parkinson's disease (PD) subtypes by comparing single-view and multiview unsupervised clustering approaches using three baseline data views:

- **Clinical:** Demographic and basic clinical characteristics.
- **CSF:** Cerebrospinal fluid proteomics.
- **Plasma:** Plasma proteomics.

The primary analysis focuses on participants diagnosed with PD. The broader Project 9000 cohort is retained as a secondary analysis to compare clustering patterns across the full participant population.

PRS, SAA status, longitudinal MoCA measurements, and UPDRS are excluded from the clustering inputs and reserved for downstream cluster characterization and evaluation.

The longer-term goal is to investigate whether identified clusters differ in genetic risk, alpha-synuclein pathology, and longitudinal cognitive decline.

## 3. Analysis Workflow

Run the numbered Python scripts from the project root in order. The full-cohort and PD-only analyses use the processed baseline data but have separate clustering outputs.

### Baseline cohort and preprocessing

1. **`03_construct_baseline_cohort.py`** – Extracts baseline (`BL`) clinical, CSF, and plasma data and identifies participants with complete data across all three views.
2. **`03A_data_preprocessing.py`** – Removes non-protein metadata and missingness indicator columns, filters features with excessive missingness, imputes remaining missing values, and removes constant features.
3. **`03B_correlation_filtering.py`** – Removes highly correlated protein features using a 0.95 correlation threshold.
4. **`03C_correlation_diagnostics.py`** – Evaluates correlation distributions, threshold sensitivity, duplicate features, and redundant protein groups.

### Dimensionality reduction and clustering views

5. **`03D_proteomics_pca.py`** – Standardizes protein features and performs separate PCA for CSF and plasma.
6. **`05_prepare_clustering_views.py`** – Prepares and aligns the final clinical, CSF, and plasma clustering matrices.

### Full-cohort clustering and sensitivity analysis

7. **`06_single_multiview_clustering.py`** – Constructs RBF kernels and performs single-view and multiview spectral clustering, performs sensitivity analysis on bandwidth, kernel weights, and stability testing

### PD-only clustering
8. **`08_pd_only_clustering.py`** – Restricts the complete-case baseline cohort to participants diagnosed with PD, preprocesses the proteomic views within this cohort, performs PCA, and evaluates single-view and multiview clustering configurations.

The PD-only analysis evaluates multiple PCA dimensions (3, 5, and 10 components), RBF bandwidth multipliers, cluster counts, and kernel weighting schemes. Results are saved separately under `Datasets/Processed/PD_Only/`.

**Proteomics preprocessing note:** Project 9000 Olink NPX values are already normalized. The workflow does not repeat Olink normalization. Protein features are standardized before PCA so that they contribute on comparable scales.

## 4. Current Analysis Status

### Cohort definition

The initial complete-case Project 9000 baseline cohort contains **206 participants** with clinical, CSF, and plasma data.

The cohort is divided into two analysis populations:

- **Primary analysis:** Participants diagnosed with PD, including eligible genetic PD subgroups.
- **Secondary analysis:** The complete 206-participant cohort, including PD, healthy controls, RBD, hyposmia, and other available participant groups.

The PD-only analysis is intended to reduce heterogeneity associated with mixing diagnosed PD participants with other clinical populations. Cluster assignments will be characterized using subgroup and biological information after clustering.

### Full-cohort prepared views

| View | Features used for clustering |
|---|---:|
| Clinical | 5 |
| CSF | 83 principal components |
| Plasma | 74 principal components |

The clinical view includes age, sex, education years, MoCA, and UPSIT. Disease duration was excluded from the primary clustering view because of substantial missingness.

For proteomics, correlation filtering retained 1,447 CSF proteins and 1,380 plasma proteins before PCA.

### Initial full-cohort clustering results

The initial equal-weight kernel integration did not outperform clinical-only clustering based on silhouette scores.

| Model | k = 2 silhouette | k = 3 silhouette |
|---|---:|---:|
| Clinical only | 0.1547 | 0.1881 |
| CSF only | 0.1040 | 0.1581 |
| Plasma only | 0.0804 | 0.0682 |
| Equal-weight multiview | 0.0444 | 0.0366 |

### Full-cohort sensitivity analysis

The sensitivity analysis evaluated RBF bandwidth multipliers of 0.25, 0.5, 1, 2, and 4, with cluster counts from 2 to 5.

Selected results from the latest sensitivity analysis:

| Model | Gamma multiplier | k | Silhouette | Cluster sizes | Mean stability ARI |
|---|---:|---:|---:|---|---:|
| Clinical only | 0.50 | 3 | 0.2111 | 77 / 63 / 66 | 1.000 |
| Clinical only | 0.25 | 4 | 0.2131 | 68 / 21 / 50 / 67 | 0.948 |
| Clinical only | 0.25 | 2 | 0.2016 | 124 / 82 | 1.000 |
| CSF only | 0.50 | 3 | 0.2181 | 194 / 1 / 11 | 1.000 |
| CSF only | 0.25 | 2 | 0.1626 | 105 / 101 | 0.966 |

Clinical-only clustering produced relatively balanced clusters and comparatively strong silhouette scores. The clinical-only, three-cluster solution at gamma multiplier 0.50 is a leading candidate for further characterization.

Although the CSF-only, three-cluster solution at gamma multiplier 0.50 achieved the highest silhouette score, it produced a singleton cluster. It should therefore not be selected as a primary solution without further investigation.

The stability ARI values measure consistency across random initializations. They do not establish robustness to changes in participants, features, or preprocessing.

### PD-only clustering

The PD-only analysis uses a smaller cohort and refits proteomic preprocessing and PCA within the PD population rather than reusing full-cohort PCA representations.

The sensitivity analysis evaluates:

- Clinical-only, CSF-only, and plasma-only clustering.
- Pairwise and three-view multiview combinations.
- CSF and plasma PCA representations with 3, 5, and 10 components.
- RBF bandwidth multipliers of 0.25, 0.5, and 1.0.
- Two- and three-cluster solutions.
- Equal and view-dominant kernel weights.
- Multiple random initializations.

Preliminary results indicate that CSF-only clustering can produce relatively strong separation, although the highest-scoring configurations may contain singleton clusters. A CSF-only five-component, two-cluster solution achieved a silhouette score of approximately 0.33, with cluster sizes of 29 and 37.

These results are exploratory. The relatively small PD-only cohort requires careful evaluation of cluster stability and biological interpretability before selecting a final solution.

## 5. Next Steps

1. **Candidate selection:** Compare silhouette scores, cluster sizes, and initialization stability to identify reasonable candidate solutions.
2. **Cluster characterization:** Compare baseline clinical characteristics across candidate clusters using original, unscaled clinical measurements.
3. **Proteomic characterization:** Identify proteins that differ across clusters and examine whether the differences are consistent across CSF and plasma.
4. **Robustness analysis:** Evaluate participant-level subsampling and feature perturbation to determine whether cluster assignments remain consistent.
5. **Single-view versus multiview comparison:** Determine whether integrating clinical, CSF, and plasma information provides meaningful additional structure beyond individual views.
6. **Biological evaluation:** Examine excluded variables such as PRS and SAA status to assess potential biological differences between clusters.
7. **Longitudinal analysis:** Use mixed-effects models to evaluate whether MoCA decline differs across candidate clusters among diagnosed PD participants.

## 6. Interpretation and Limitations

All clustering results are exploratory and should not yet be interpreted as established biological subtypes.

Important considerations include:

- The PD-only cohort is relatively small, limiting the complexity of clustering configurations that can be reliably evaluated.
- High silhouette scores can result from small or singleton clusters.
- Stability across random seeds does not guarantee stability under participant or feature perturbations.
- Clinical variables may dominate clustering because they capture disease severity and demographic differences rather than distinct underlying biology.
- Multiview integration does not necessarily improve clustering separation, even when individual views capture complementary information.

**Current status:** Baseline cohort extraction, proteomics preprocessing, correlation filtering, PCA, full-cohort single-view and multiview clustering, kernel diagnostics, and sensitivity analyses have been completed. The PD-only clustering workflow has also been developed and preliminary candidate solutions identified. Cluster characterization, robustness testing, and downstream biological and longitudinal validation are the next priorities.
