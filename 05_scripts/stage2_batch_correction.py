from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / '01_raw_outputs'
DATA_PATH = PROJECT_ROOT / '02_merged_data'
MERGED_PATH = DATA_PATH
META_PATH = PROJECT_ROOT / '03_metadata_tables'
FIG_PATH = PROJECT_ROOT / '04_figures'
BASE_PATH = PROJECT_ROOT

for _path in [DATA_PATH, META_PATH, FIG_PATH]:
    _path.mkdir(parents=True, exist_ok=True)


def install_packages(*packages):
    if packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

# # ASD Research: Multilayer Microbiome Feature Integration from Diagnosis to Behavioral Prediction
# ## Stage 2: Batch Effect Correction and Data Standardization
#
# ---
#
# **Version**: V4.7 Revised Edition  
# **Date**: February 2026  
#
# ### Goals of This Stage
#
# 1. **CLR normalization**: Apply Centered Log-Ratio transformation to all data
# 2. **Batch effect correction**: Use ComBat to correct the discovery cohort (7 batches)
# 3. **Preserve raw data**: Keep raw relative abundance for alpha-diversity analysis
#
# ### Input Data (from Stage 1)
#
# ```python
# stage1_data = {
#     'discovery_data_filtered': {...},  # Four-level discovery cohort data
#     'moscow_data_filtered': {...},     # Four-level Moscow cohort data
#     'discovery_group': Series,         # Group information
#     'discovery_study': Series,         # Batch information (7 StudyIDs)
#     'moscow_group': Series,            # Moscow group labels
#     'local_cohort_samples': list,      # Changchun sample IDs
#     'local_group': Series              # Changchun group labels
# }
# ```
#
# ### Output Data (stage2_normalized_data.pkl)
#
# ```python
# {
#     'discovery_data_corrected': {...},  # Batch-corrected discovery cohort
#     'discovery_data_raw': {...},        # Raw data (for alpha diversity)
#     'moscow_data_clr': {...},           # Moscow CLR data
#     'discovery_group': Series,          # Group information (passed through)
#     'discovery_study': Series,          # Batch information (passed through)
#     ...
# }
# ```

# ## 2.1 Environment Setup

# ============================================================
# Install required dependencies
# ============================================================
install_packages('pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn')
install_packages('combat')
# Python version of ComBat

print("✓ Dependency installation completed")

# ============================================================
# Import required libraries
# ============================================================
import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try importing ComBat
try:
    from combat.pycombat import pycombat
    COMBAT_AVAILABLE = True
    print("✓ ComBat (pycombat) import succeeded")
except ImportError:
    try:
        from pycombat import pycombat
        COMBAT_AVAILABLE = True
        print("✓ ComBat (pycombat) import succeeded")
    except ImportError:
        COMBAT_AVAILABLE = False
        print("⚠ ComBat is unavailable; a fallback method will be used")

print("✓ Library import completed")

# ============================================================
# Mount Google Drive
# ============================================================


# Define paths
FIG_PATH = os.path.join(BASE_PATH, '04_figures')

# ============================================================
# Load Stage 1 output data
# ============================================================
print("=" * 60)
print("Loading Stage 1 data")
print("=" * 60)

stage1_path = os.path.join(str(DATA_PATH), 'stage1_preprocessed_data.pkl')
with open(stage1_path, 'rb') as f:
    stage1_data = pickle.load(f)

# Check the version
version = stage1_data.get('version', 'unknown')
print(f"Stage 1 data version: {version}")

# Extract data
discovery_data = stage1_data['discovery_data_filtered']
moscow_data = stage1_data['moscow_data_filtered']

discovery_group = stage1_data['discovery_group']
discovery_study = stage1_data['discovery_study']
moscow_group = stage1_data['moscow_group']

local_cohort_samples = stage1_data['local_cohort_samples']
local_group = stage1_data['local_group']

print(f"\n【data dimensions】")
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    df = discovery_data.get(dtype)
    if df is not None:
        print(f"  discovery cohort {dtype}: {df.shape}")

print(f"\n【group information】")
print(f"  discovery cohort: {len(discovery_group)} samples")
print(f"  discovery cohortgroup: {discovery_group.value_counts().to_dict()}")
print(f"  Number of batches: {discovery_study.nunique()}")
print(f"  Changchun sub-cohort: {len(local_cohort_samples)} samples")
print(f"  Moscow cohort: {len(moscow_group)} samples")

# ## 2.2 CLR Transformation

# ============================================================
# Define the CLR transformation function
# ============================================================
def clr_transform(df, name="data", pseudocount=1e-6):
    """
    Perform Centered Log-Ratio (CLR) transformation

    Parameters:
    -----------
    df : pd.DataFrame
        Relative abundance data (behavioral features, columns are samples)
    name : str
        Dataset name (used for logging)
    pseudocount : float
        Pseudocount to avoid log(0)

    Returns:
    --------
    pd.DataFrame : CLR-transformed data
    """
    if df is None:
        return None

    # Add a pseudocount
    df_pseudo = df + pseudocount

    # Compute the geometric mean for each sample
    log_data = np.log(df_pseudo)
    geometric_mean = log_data.mean(axis=0)

    # CLR transformation
    clr_data = log_data - geometric_mean

    print(f"  {name}: {df.shape} → CLRcompleted")

    return clr_data

print("✓ CLR transformation function definition completed")

# ============================================================
# Run CLR transformation
# ============================================================
print("=" * 60)
print("Running CLR transformation")
print("=" * 60)

# Preserve raw data (for alpha-diversity analysis)
discovery_data_raw = {}
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    df = discovery_data.get(dtype)
    if df is not None:
        discovery_data_raw[dtype] = df.copy()

moscow_data_raw = {}
for dtype in ['taxa', 'pathways']:
    df = moscow_data.get(dtype)
    if df is not None:
        moscow_data_raw[dtype] = df.copy()

print("✓ Raw data saved (for alpha diversity)")

# CLR transformation - discovery cohort
print("\n--- Discovery cohort CLR transformation ---")
discovery_data_clr = {}
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    df = discovery_data.get(dtype)
    discovery_data_clr[dtype] = clr_transform(df, f"discovery cohort-{dtype}")

# CLR transformation - Moscow cohort
print("\n--- Moscow cohort CLR transformation ---")
moscow_data_clr = {}
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    df = moscow_data.get(dtype)
    moscow_data_clr[dtype] = clr_transform(df, f"Moscow-{dtype}")

print("\n✓ CLR transformation completed")

# ## 2.3 Batch Effect Correction

# ============================================================
# 1. Install the standard neuroCombat package (if not already installed)
# ============================================================
try:
    from neuroCombat import neuroCombat
except ImportError:
    print("Installing neuroCombat...")
    install_packages('neuroCombat')
    from neuroCombat import neuroCombat

# ============================================================
# 2. Define the batch correction function based on neuroCombat (fixed version)
# ============================================================
def combat_batch_correction(data_df, batch_series, group_series=None):
    """
    Perform batch effect correction with neuroCombat (standard approach)
    """
    if data_df is None:
        return None

    # 1. Sample alignment
    common_samples = data_df.columns.intersection(batch_series.index)
    if group_series is not None:
        common_samples = common_samples.intersection(group_series.index)

    # Ensure a consistent data order (Features x Samples)
    data_aligned = data_df[common_samples]
    batch_aligned = batch_series.loc[common_samples]

    print(f"    Aligned sample count: {len(common_samples)}")
    print(f"    Number of batches: {batch_aligned.nunique()}")

    # 2. Prepare neuroCombat inputs
    # Data matrix (numpy array): Features x Samples
    dat = data_aligned.values

    # Covariate matrix (DataFrame): Samples x Covariates
    covars = pd.DataFrame({'batch': batch_aligned})
    categorical_cols = []

    # If a covariate (Group) is available, include it to preserve biological differences
    if group_series is not None:
        group_aligned = group_series.loc[common_samples]
        covars['group'] = group_aligned.values
        categorical_cols = ['group']

    # 3. Run neuroCombat
    try:
        # neuroCombat returns a dictionary, and 'data' is the corrected matrix
        results = neuroCombat(dat=dat,
                              covars=covars,
                              batch_col='batch',
                              categorical_cols=categorical_cols)

        corrected_data = results['data']

        # 4. Convert back to a DataFrame (preserving index and column names)
        corrected_df = pd.DataFrame(corrected_data,
                                    index=data_aligned.index,
                                    columns=data_aligned.columns)

        print(f"    ✓ neuroCombat correction succeeded")
        return corrected_df

    except Exception as e:
        print(f"    ⚠ neuroCombat failed: {str(e)}")
        # Fallback: mean correction (use only in extreme failures; neuroCombat is usually stable)
        print("    -> Falling back to simple mean-centering correction...")
        corrected = data_aligned.copy()
        for batch in batch_aligned.unique():
            idx = batch_aligned[batch_aligned == batch].index
            batch_mean = data_aligned[idx].mean(axis=1)
            global_mean = data_aligned.mean(axis=1)
            corrected[idx] = data_aligned[idx].subtract(batch_mean - global_mean, axis=0)
        return corrected

print("✓ Batch correction function (neuroCombat version) definition completed")

# ============================================================
# Apply batch correction to the discovery cohort (7 batches)
# ============================================================
print("=" * 60)
print("Running batch correction（discovery cohort：7Chinese cohorts）")
print("=" * 60)
print("\n📌 Notes：")
print("   - Batch variable: StudyID (7)")
print("   - Covariate: Group (preserve ASD/TD biological differences)")
print("   - The Moscow cohort is not included in batch correction")

discovery_data_corrected = {}

# Taxa
print("\n--- Taxa level ---")
discovery_data_corrected['taxa'] = combat_batch_correction(
    discovery_data_clr['taxa'], discovery_study, discovery_group
)

# Pathways
print("\n--- Pathways level ---")
discovery_data_corrected['pathways'] = combat_batch_correction(
    discovery_data_clr['pathways'], discovery_study, discovery_group
)

# Genes
print("\n--- Genes level ---")
discovery_data_corrected['genes'] = combat_batch_correction(
    discovery_data_clr['genes'], discovery_study, discovery_group
)

# ECs
print("\n--- ECs level ---")
discovery_data_corrected['ecs'] = combat_batch_correction(
    discovery_data_clr['ecs'], discovery_study, discovery_group
)

print("\n✓ batch correctioncompleted")

# ## 2.4 Visualize Batch Correction Effects

# ============================================================
# PCA Visualization: Before vs After Batch Correction (English)
# ============================================================
print("=" * 60)
print("PCA Visualization (English Labels)")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Select 'taxa' level for visualization
taxa_clr = discovery_data_clr.get('taxa')
taxa_corrected = discovery_data_corrected.get('taxa')

if taxa_clr is not None and taxa_corrected is not None:
    # Ensure sample alignment
    common_samples = taxa_clr.columns.intersection(taxa_corrected.columns)
    common_samples = common_samples.intersection(discovery_study.index)
    common_samples = common_samples.intersection(discovery_group.index)

    # -------------------------------------------------------
    # 1. Before Correction - Colored by Batch (StudyID)
    # -------------------------------------------------------
    ax1 = axes[0, 0]
    X_before = taxa_clr[common_samples].T.values
    pca = PCA(n_components=2)
    X_pca_before = pca.fit_transform(X_before)

    batch_labels = discovery_study.loc[common_samples]
    # Loop through batches to plot
    for batch in batch_labels.unique():
        mask = batch_labels == batch
        ax1.scatter(X_pca_before[mask, 0], X_pca_before[mask, 1],
                   label=batch, alpha=0.7, s=30)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Before Correction - Colored by Batch') # English Title
    # Move legend outside to prevent blocking points
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, borderaxespad=0.)

    # -------------------------------------------------------
    # 2. Before Correction - Colored by Group (ASD/TD)
    # -------------------------------------------------------
    ax2 = axes[0, 1]
    group_labels = discovery_group.loc[common_samples]
    colors = {'ASD': '#E74C3C', 'TD': '#3498DB'}

    for group in ['ASD', 'TD']:
        mask = group_labels == group
        ax2.scatter(X_pca_before[mask, 0], X_pca_before[mask, 1],
                   c=colors[group], label=group, alpha=0.7, s=30)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Before Correction - Colored by Group') # English Title
    ax2.legend(loc='upper right')

    # -------------------------------------------------------
    # 3. After Correction - Colored by Batch (StudyID)
    # -------------------------------------------------------
    ax3 = axes[1, 0]
    X_after = taxa_corrected[common_samples].T.values
    X_pca_after = pca.fit_transform(X_after)

    for batch in batch_labels.unique():
        mask = batch_labels == batch
        ax3.scatter(X_pca_after[mask, 0], X_pca_after[mask, 1],
                   label=batch, alpha=0.7, s=30)

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('After Correction - Colored by Batch') # English Title
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, borderaxespad=0.)

    # -------------------------------------------------------
    # 4. After Correction - Colored by Group (ASD/TD)
    # -------------------------------------------------------
    ax4 = axes[1, 1]
    for group in ['ASD', 'TD']:
        mask = group_labels == group
        ax4.scatter(X_pca_after[mask, 0], X_pca_after[mask, 1],
                   c=colors[group], label=group, alpha=0.7, s=30)

    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('After Correction - Colored by Group') # English Title
    ax4.legend(loc='upper right')

plt.tight_layout()
# Save figures
plt.savefig(os.path.join(str(FIG_PATH), 'Fig_S1_BatchCorrection_PCA.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(str(FIG_PATH), 'Fig_S1_BatchCorrection_PCA.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ PCA Figure saved (English version)")

# ## 2.5 Save Stage 2 Output Data

# ============================================================
# Save Stage 2 output data
# ============================================================
print("=" * 60)
print("Saving Stage 2 output data")
print("=" * 60)

stage2_output = {
    # ==== Discovery cohort data (after batch correction) ====
    'discovery_data_corrected': discovery_data_corrected,

    # ==== Discovery cohort raw data (for alpha diversity) ====
    'discovery_data_raw': discovery_data_raw,

    # ==== Discovery cohort CLR data (before batch correction) ====
    'discovery_data_clr': discovery_data_clr,

    # ==== External validation cohort (Moscow, CLR only) ====
    'moscow_data_clr': moscow_data_clr,

    # ==== Moscow raw data ====
    'moscow_data_raw': moscow_data_raw,

    # ==== Metadata (passed from Stage 1) ====
    'discovery_group': discovery_group,
    'discovery_study': discovery_study,
    'moscow_group': moscow_group,
    'local_cohort_samples': local_cohort_samples,
    'local_group': local_group,

    # ==== Version information ====
    'version': 'V4.7',
    'description': 'Batch-corrected discovery cohort + Moscow cohort CLR'
}

# Save
output_path = os.path.join(str(DATA_PATH), 'stage2_normalized_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(stage2_output, f)

print(f"✓ data saved to: {output_path}")

# Verify
print("\n【Output data validation】")
for key in ['discovery_data_corrected', 'discovery_data_raw', 'moscow_data_clr',
            'discovery_group', 'discovery_study', 'moscow_group',
            'local_cohort_samples', 'local_group']:
    value = stage2_output.get(key)
    if isinstance(value, dict):
        print(f"  ✓ {key}: dict")
        for k, v in value.items():
            if v is not None and hasattr(v, 'shape'):
                print(f"      - {k}: {v.shape}")
    elif isinstance(value, pd.Series):
        print(f"  ✓ {key}: Series, len={len(value)}")
    elif isinstance(value, list):
        print(f"  ✓ {key}: list, len={len(value)}")
        print(f"  ✓ {key}: {type(value).__name__}")

# ============================================================
# Stage 2 summary report
# ============================================================
print("\n" + "=" * 60)
print("Stage 2: Batch effect correction and data standardization - completed!")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    Stage2 Processingsummary (V4.7)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Data standardization]                                                  │
│  • Method: CLR (Centered Log-Ratio) transformation                         │
│  • Pseudocount: 1e-6                                                 │
│                                                                 │
│  【batch effect correction】                                                │
│  • Method: ComBat                                                 │
│  • discovery cohort: 7-batch correction                                        │
│  • Covariate: Group (preserve ASD/TD differences)                              │
│  • Moscow cohort: not corrected                                       │
│                                                                 │
│  [Output variables] for Stages 3-5                                      │
│  • discovery_data_corrected: batch-corrected data                     │
│  • discovery_data_raw: raw data (for alpha diversity)                    │
│  • moscow_data_clr: Moscow CLR data                              │
│  • discovery_group/study: group and batch information                        │
│  • local_cohort_samples: Changchunsample IDs                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

# Data statistics
if discovery_data_corrected.get('taxa') is not None:
    print(f"discovery cohort sample count: {discovery_data_corrected['taxa'].shape[1]}")
if moscow_data_clr.get('taxa') is not None:
    print(f"Moscow cohort sample count: {moscow_data_clr['taxa'].shape[1]}")

print("\n" + "=" * 60)
print("✓ Stage 2 completed. Please continue with Stage 3")
print("=" * 60)
