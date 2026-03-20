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
# ## Stage 1: Data Integration and Preprocessing
#
# ---
#
# **Version**: V4.7 Revised Edition  
# **Date**: February 2026  
#
# ### V4.7 Study Design
#
# **Discovery Cohort**: Chinese multicenter (6 public cohorts) + Changchun cohort = **471 samples**  
# **External Validation**: Moscow cohort = **74 samples** (cross-ethnic validation)
#
# ### Goals of This Stage
#
# 1. Load raw MetaPhlAn/HUMAnN outputs from each cohort
# 2. **Directly merge the discovery cohorts** (Chinese multicenter + Changchun)
# 3. Perform feature filtering (low prevalence, low abundance)
# 4. Extract group and batch information
# 5. Save preprocessed data for Stage 2
#
# ### Output Data (stage1_preprocessed_data.pkl)
#
# ```python
# {
#     'discovery_data_filtered': {...},  # Four-level discovery cohort data
#     'moscow_data_filtered': {...},     # Four-level Moscow cohort data
#     'discovery_group': Series,         # Discovery cohort groups (ASD/TD)
#     'discovery_study': Series,         # Discovery cohort batches (7 StudyIDs)
#     'moscow_group': Series,            # Moscow cohort groups
#     'local_cohort_samples': list,      # Changchun sample ID list
#     'local_group': Series,             # Changchun sub-cohort groups
#     'metadata': dict                   # Metadata tables
# }
# ```

# ## 1.1 Environment Setup

# ============================================================
# Install required dependencies
# ============================================================
install_packages('pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn')

print("✓ Dependency installation completed")

# ============================================================
# Import required libraries
# ============================================================
import os
import glob
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("✓ Library import completed")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")

# ============================================================
# Mount Google Drive
# ============================================================

# ============================================================
# Define project paths
# ============================================================

# Create required directories
for path in [MERGED_PATH, FIG_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

print("✓ Path setup completed")
print(f"  BASE_PATH: {BASE_PATH}")
print(f"  RAW_PATH: {RAW_PATH}")

# ## 1.2 Define Cohort Groups

# ============================================================
# Define cohort groups (V4.7 revision)
# ============================================================

# Six public Chinese cohorts
CHINA_PUBLIC_COHORTS = [
    'Study_Dan2020',      # Hong Kong
    'Study_Wang2019',     # Mainland
    'Study_Xu2023',       # Mainland
    'Study_Zhang2020',    # Mainland
    'Study_Tong2022',     # Mainland
    'Study_CUHK'          # Hong Kong
]

# Changchun in-house cohort (included in the discovery cohort)
LOCAL_COHORT = 'Local_Cohort'

# Discovery cohort = public cohorts + Changchun cohort (7 total)
DISCOVERY_COHORTS = CHINA_PUBLIC_COHORTS + [LOCAL_COHORT]

# External validation cohort (Moscow, cross-ethnic validation)
MOSCOW_COHORT = 'Study_Kovtun2020'

# Data types and their file patterns (dictionary format)
DATA_TYPES = {
    'taxa': {
        'subdir': 'taxa',
        'patterns': ['*_taxonomic_profile.tsv', '*_metaphlan.tsv', '*_profile.tsv', '*.tsv']
    },
    'pathways': {
        'subdir': 'pathways',
        'patterns': ['*_pathabundance_relab.tsv', '*_pathabundance.tsv', '*.tsv']
    },
    'genes': {
        'subdir': 'genes',
        'patterns': ['*_genefamilies_relab.tsv', '*_genefamilies.tsv', '*.tsv']
    },
    'ecs': {
        'subdir': 'ecs',
        'patterns': ['*_ecs_relab.tsv', '*_ecs.tsv', '*.tsv']
    }
}

# Data type list (used for iteration)
DATA_TYPE_LIST = ['taxa', 'pathways', 'genes', 'ecs']

print("【V4.7cohort design】")
print(f"  discovery cohort: {len(DISCOVERY_COHORTS)}Chinese cohorts")
for c in DISCOVERY_COHORTS:
    print(f"    - {c}")
print(f"  external validation: {MOSCOW_COHORT} (cross-ethnic)")

# ## 1.3 Define Data Loading Functions

# ============================================================
# Define a function to read MetaPhlAn species abundance files
# ============================================================
def read_metaphlan_file(filepath):
    """Read species abundance files generated by MetaPhlAn."""
    try:
        # Try different reading strategies
        with open(filepath, 'r') as f:
            first_lines = [f.readline() for _ in range(10)]

        # Find the header row
        skip_rows = 0
        for i, line in enumerate(first_lines):
            if line.startswith('#') and not line.startswith('#clade'):
                skip_rows = i + 1
            elif line.startswith('clade_name') or line.startswith('#clade'):
                skip_rows = i
                break

        df = pd.read_csv(filepath, sep='\t', skiprows=skip_rows)

        # Standardize column names
        if 'clade_name' in df.columns or '#clade_name' in df.columns:
            name_col = 'clade_name' if 'clade_name' in df.columns else '#clade_name'
            abund_col = 'relative_abundance' if 'relative_abundance' in df.columns else df.columns[1]

            result = df[[name_col, abund_col]].copy()
            result.columns = ['feature', 'abundance']
            return result
            # Assume the first column is the feature name and the second column is abundance
            result = df.iloc[:, :2].copy()
            result.columns = ['feature', 'abundance']
            return result

    except Exception as e:
        print(f"    Read failed: {str(e)[:50]}")
        return None

print("✓ MetaPhlAnreader function definition completed")

# ============================================================
# Define a function to read HUMAnN functional abundance files
# ============================================================
def read_humann_file(filepath):
    """Read functional abundance files generated by HUMAnN."""
    try:
        df = pd.read_csv(filepath, sep='\t')

        # The first column contains feature names
        name_col = df.columns[0]
        abund_col = df.columns[1] if len(df.columns) > 1 else None

        if abund_col is None:
            return None

        result = df[[name_col, abund_col]].copy()
        result.columns = ['feature', 'abundance']

        # Remove species stratification information (the part after '|')
        result['feature'] = result['feature'].apply(lambda x: x.split('|')[0] if '|' in str(x) else x)

        # Aggregate identical features
        result = result.groupby('feature')['abundance'].sum().reset_index()

        return result

    except Exception as e:
        print(f"    Read failed: {str(e)[:50]}")
        return None

print("✓ HUMAnNreader function definition completed")

# ============================================================
# Define a function to extract sample IDs from filenames
# ============================================================
def extract_sample_id(filename):
    """Extract the sample ID from a filename."""
    # Remove the file extension
    base = os.path.splitext(filename)[0]

    # Remove common suffixes
    suffixes = ['_taxonomic_profile', '_metaphlan', '_profile',
                '_pathabundance_relab', '_pathabundance',
                '_genefamilies_relab', '_genefamilies',
                '_ecs_relab', '_ecs']

    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    return base

print("✓ Sample ID extraction function definition completed")

# ============================================================
# Define a function to merge data from a single cohort
# ============================================================
def merge_cohort_data(cohort_name, data_type):
    """Merge all sample data from a single cohort."""
    cohort_path = os.path.join(str(RAW_PATH), cohort_name)

    if not os.path.exists(cohort_path):
        print(f"  ⚠ Cohort directory does not exist: {cohort_name}")
        return None, {}

    # Get the data subdirectory
    data_info = DATA_TYPES[data_type]
    data_dir = os.path.join(cohort_path, data_info['subdir'])

    if not os.path.exists(data_dir):
        print(f"  ⚠ data directory does not exist: {data_dir}")
        return None, {}

    # Find data files
    files = []
    for pattern in data_info['patterns']:
        found = glob.glob(os.path.join(data_dir, pattern))
        if found:
            files = found
            break

    if not files:
        print(f"  ? Data files not found: {cohort_name}/{data_type}")
        return None, {}

    # Read and merge
    all_data = {}
    sample_cohort_map = {}

    for filepath in files:
        filename = os.path.basename(filepath)
        sample_id = extract_sample_id(filename)

        # Select the loading function based on the data type
        if data_type == 'taxa':
            df = read_metaphlan_file(filepath)
            df = read_humann_file(filepath)

        if df is not None and len(df) > 0:
            all_data[sample_id] = df.set_index('feature')['abundance']
            sample_cohort_map[sample_id] = cohort_name

    if not all_data:
        return None, {}

    # Merge into a DataFrame
    merged_df = pd.DataFrame(all_data)
    merged_df = merged_df.fillna(0)

    print(f"  ✓ {cohort_name}: {merged_df.shape[1]} samples, {merged_df.shape[0]} features")

    return merged_df, sample_cohort_map

print("✓ Cohort merge function definition completed")

# ============================================================
# Define a function to merge data from multiple cohorts
# ============================================================
def merge_multiple_cohorts(cohort_list, data_type):
    """Merge data from multiple cohorts."""
    all_dfs = []
    all_sample_map = {}

    for cohort in cohort_list:
        df, sample_map = merge_cohort_data(cohort, data_type)
        if df is not None:
            all_dfs.append(df)
            all_sample_map.update(sample_map)

    if not all_dfs:
        return None, {}

    # Merge all cohorts and keep shared features
    # Use an outer join to keep all features and fill missing values with 0
    merged = pd.concat(all_dfs, axis=1, join='outer')
    merged = merged.fillna(0)

    return merged, all_sample_map

print("✓ Multi-cohort merge function definition completed")

# ## 1.4 Run Data Merging

# ============================================================
# Merge discovery cohort data (7 Chinese cohorts)
# ============================================================
print("=" * 60)
print("Merging discovery cohort data（V4.7: 7Chinese cohorts）")
print("=" * 60)

discovery_data = {}
discovery_sample_cohort_map = {}

for dtype in DATA_TYPE_LIST:
    print(f"\n--- Processing {dtype} data ---")
    df, sample_map = merge_multiple_cohorts(DISCOVERY_COHORTS, dtype)

    if df is not None:
        discovery_data[dtype] = df
        discovery_sample_cohort_map.update(sample_map)
        print(f"  Merge completed: {df.shape[0]} features × {df.shape[1]} samples")
        discovery_data[dtype] = None
        print(f"  ⚠️ Merge failed")

print(f"\n【discovery cohortsummary】")
print(f"  Total sample count: {len(discovery_sample_cohort_map)}")
for dtype, df in discovery_data.items():
    if df is not None:
        print(f"  {dtype}: {df.shape[1]} samples, {df.shape[0]} features")

# ============================================================
# Merge Moscow cohort data (external validation)
# ============================================================
print("=" * 60)
print("Merging Moscow cohort data（cross-ethnicexternal validation）")
print("=" * 60)

moscow_data = {}
moscow_sample_cohort_map = {}

for dtype in DATA_TYPE_LIST:
    print(f"\n--- Processing {dtype} data ---")
    df, sample_map = merge_cohort_data(MOSCOW_COHORT, dtype)

    if df is not None:
        moscow_data[dtype] = df
        moscow_sample_cohort_map.update(sample_map)
        moscow_data[dtype] = None

print(f"\n【Moscow cohortsummary】")
print(f"  Total sample count: {len(moscow_sample_cohort_map)}")
for dtype, df in moscow_data.items():
    if df is not None:
        print(f"  {dtype}: {df.shape[1]} samples, {df.shape[0]} features")

# ## 1.5 Load Metadata

# ============================================================
# 1.5 Load metadata (V4.7 final fixed version)
# ============================================================
import pandas as pd
import os

print("=" * 60)
print("Loading metadata tables (V4.7 final fixed version)")
print("=" * 60)

metadata = {}
analytical_meta = None
sample_id_col = None
group_col = None
study_col = None

# 1. Define file paths
meta_files = {
    'analytical_metadata': 'Table3_Analytical_Metadata.csv',
    'sra_metadata': 'Table2_SRA_Metadata_Raw.csv'
}

for key, filename in meta_files.items():
    filepath = os.path.join(str(META_PATH), filename)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            metadata[key] = df
            print(f"  ✓ Successfully read {filename}: {df.shape[0]} rows")
            # Print column names for debugging (if this still fails, we can inspect the actual column names)
            print(f"    Column preview: {df.columns.tolist()[:10]}")

            if key == 'analytical_metadata':
                analytical_meta = df
        except Exception as e:
            print(f"  ? Failed to read {filename}: {e}")

# 2. Intelligently detect column names (with added GroupID support)
if analytical_meta is not None:
    cols = analytical_meta.columns.tolist()

    # --- Detect SampleID ---
    potential_sample_cols = ['Sample_ID', 'SampleID', 'Run', 'run', 'Run_s', 'run_accession', 'sample_name', 'sample_id', 'Accession']
    for candidate in potential_sample_cols:
        if candidate in cols:
            sample_id_col = candidate
            print(f"    -> Detected sample ID column: '{sample_id_col}'")
            break

    # --- Detect Group (key update: added GroupID support) ---
    potential_group_cols = [
        'GroupID', 'Group_ID', 'Group', 'group',  # <--- most likely column names placed first
        'Diagnosis', 'diagnosis', 'disease_status', 'status', 'Status',
        'Assay_Type', 'Condition', 'condition', 'Class', 'class'
    ]
    for candidate in potential_group_cols:
        if candidate in cols:
            group_col = candidate
            print(f"    -> Detected group column: '{group_col}'")
            break

    # --- Detect StudyID ---
    potential_study_cols = ['Study_ID', 'StudyID', 'BioProject', 'bioproject', 'Project', 'project', 'study_id', 'Study', 'study', 'Cohort', 'cohort']
    for candidate in potential_study_cols:
        if candidate in cols:
            study_col = candidate
            print(f"    -> Detected cohort column: '{study_col}'")
            break

# ============================================================
# 1.6 Extract Group Information
# ============================================================
print("\n" + "=" * 60)
print("Extracting group information")
print("=" * 60)

# Initialize
discovery_group = pd.Series(dtype=str)
discovery_study = pd.Series(dtype=str)
moscow_group = pd.Series(dtype=str)
local_group = pd.Series(dtype=str)

if analytical_meta is not None and sample_id_col and group_col:
    # 1. Standardize group labels
    analytical_meta[group_col] = analytical_meta[group_col].replace({
        'Autism': 'ASD', 'autism': 'ASD', 'ASD': 'ASD', 'asd': 'ASD',
        'Control': 'TD', 'control': 'TD', 'Healthy': 'TD', 'TD': 'TD',
        'Typically Developing': 'TD'
    })

    # 2. Extract discovery cohort groups (6 public cohorts)
    public_samples = [s for s, c in discovery_sample_cohort_map.items() if c != 'Local_Cohort']
    matched_meta = analytical_meta[analytical_meta[sample_id_col].isin(public_samples)]
    discovery_group = matched_meta.set_index(sample_id_col)[group_col]

    print(f"  ✓ Public cohort matching succeeded: {len(discovery_group)} samples")

    # 3. Process the Changchun cohort (Local_Cohort) with automatic filename inference
    local_samples = [s for s, c in discovery_sample_cohort_map.items() if c == 'Local_Cohort']
    if len(local_samples) > 0:
        local_in_meta = analytical_meta[analytical_meta[sample_id_col].isin(local_samples)]

        if len(local_in_meta) > 0:
            local_group = local_in_meta.set_index(sample_id_col)[group_col]
            print("  ⚠ No Changchun samples were found in the table; enabling filename inference mode")
            inferred_groups = {}
            for s in local_samples:
                # Simple rule: if the filename contains ASD, label it as ASD; otherwise, label it as TD
                if 'ASD' in s.upper():
                    inferred_groups[s] = 'ASD'
                    inferred_groups[s] = 'TD'
            local_group = pd.Series(inferred_groups)

        # Add Changchun groups into the discovery cohort
        new_local = local_group[~local_group.index.isin(discovery_group.index)]
        discovery_group = pd.concat([discovery_group, new_local])

    # 4. Build the StudyID series
    discovery_study = pd.Series(discovery_sample_cohort_map)
    discovery_study = discovery_study[discovery_study.index.isin(discovery_group.index)]
    discovery_group = discovery_group.loc[discovery_study.index] # align

    # 5. Extract Moscow cohort groups
    moscow_samples = list(moscow_sample_cohort_map.keys())
    moscow_in_meta = analytical_meta[analytical_meta[sample_id_col].isin(moscow_samples)]
    moscow_group = moscow_in_meta.set_index(sample_id_col)[group_col]

    print(f"\n【Final group statistics】")
    print(f"  discovery cohort (n={len(discovery_group)}): {discovery_group.value_counts().to_dict()}")
    print(f"  Moscow cohort (n={len(moscow_group)}): {moscow_group.value_counts().to_dict()}")

    print("❌ Still unable to detect key columns！")
    print(f"  Current detection status -> SampleID: {sample_id_col}, Group: {group_col}, StudyID: {study_col}")
    if analytical_meta is not None:
        print(f"  Actual CSV column names: {analytical_meta.columns.tolist()}")

# ## 1.6 Data Quality Control and Feature Filtering

# ============================================================
# Define the feature filtering function (V4.7 rigorous version)
# ============================================================
def filter_features_rigorous(df, group_series,
                             prevalence_threshold=0.05,
                             abundance_threshold=1e-5,
                             detection_limit=1e-7):
    """
    Rigorous feature-filtering function with group-sensitive filtering support

    Core logic:
    Keep features meeting either of the following conditions:
    1. (ASD-group prevalence >= threshold) OR (TD-group prevalence >= threshold)
       AND
    2. Mean relative abundance >= threshold (optional; some studies use prevalence only)

    Parameters:
    -----------
    df : pd.DataFrame
        Feature matrix (behavioral features, columns are samples)
    group_series : pd.Series
        Sample group information (index is sample ID and values are group labels)
    prevalence_threshold : float
        Minimum prevalence threshold (default 0.05, i.e. 5%)
    abundance_threshold : float
        Minimum mean abundance threshold (default 1e-5)
    detection_limit : float
        Minimum abundance threshold for feature presence (default 1e-7, excluding noise)
    """
    if df is None or group_series is None:
        print("⚠ Input data are empty")
        return None

    # 1. Ensure sample alignment
    common_samples = df.columns.intersection(group_series.index)
    if len(common_samples) < len(df.columns):
        print(f"⚠ Warning: Only {len(common_samples)}/{len(df.columns)} samples have group information，only these samples will be used for filtering")

    df_aligned = df[common_samples]
    groups = group_series.loc[common_samples]

    original_features = df.shape[0]

    # 2. Compute the mean abundance (Global Mean Abundance)
    # Only features with extremely low mean abundance are removed globally
    mean_abundance = df_aligned.mean(axis=1)
    mask_abundance = mean_abundance >= abundance_threshold

    # 3. Compute group-wise prevalence
    # Presence criterion: abundance > detection_limit (more rigorous than simply > 0)
    present_matrix = (df_aligned > detection_limit)

    # Compute prevalence for each group separately
    groups_list = groups.unique()
    prevalence_mask = pd.Series(False, index=df.index)

    print("  Group prevalence check:")
    for g in groups_list:
        # Get samples from the current group
        group_samples = groups[groups == g].index
        # Compute prevalence in the current group
        group_prev = present_matrix[group_samples].sum(axis=1) / len(group_samples)
        # Update the mask: keep the feature if it meets the threshold in any group (logical OR)
        prevalence_mask = prevalence_mask | (group_prev >= prevalence_threshold)

        n_pass = (group_prev >= prevalence_threshold).sum()
        print(f"    - {g} group (n={len(group_samples)}): {n_pass} features passed")

    # 4. Combined filtering (logical AND)
    # Keep features that are (prevalent in any group) AND (meet the global mean abundance threshold)
    final_keep_mask = prevalence_mask & mask_abundance

    df_filtered = df.loc[final_keep_mask]

    # 5. Output the report
    filtered_features = df_filtered.shape[0]
    removed_count = original_features - filtered_features

    print("-" * 40)
    print(f"Feature filtering results:")
    print(f"  Original features: {original_features}")
    print(f"  Retained features: {filtered_features}")
    print(f"  Removed features: {removed_count} ({(removed_count/original_features)*100:.1f}%)")
    print(f"  Threshold settings: Prev>={prevalence_threshold:.0%}, Abund>={abundance_threshold}, Detect>{detection_limit}")
    print("-" * 40)

    return df_filtered

print("✓ Rigorous feature-filtering function definition completed")

# ============================================================
# Run feature filtering (V4.7 rigorous version, group-sensitive)
# ============================================================
print("=" * 60)
print("Running feature filtering (retain group-specific markers)")
print("=" * 60)

# 1. Discovery cohort filtering
# Pass in discovery_group to ensure that features enriched only in the ASD or TD group are retained
print("\n--- discovery cohort (Chinese multicenter + Changchun) ---")
discovery_data_filtered = {}
for dtype, df in discovery_data.items():
    print(f"\n[Processing {dtype} data]:")
    if df is not None:
        discovery_data_filtered[dtype] = filter_features_rigorous(
            df,
            discovery_group,          # <--- Key change: pass in group information
            prevalence_threshold=0.05, # keep the 5% threshold
            abundance_threshold=1e-5   # keep the 1e-5 abundance threshold
        )
        discovery_data_filtered[dtype] = None

# 2. Moscow cohort filtering
# Also pass in moscow_group to remove cohort-specific noise
print("\n--- Moscow cohort (cross-ethnic validation) ---")
moscow_data_filtered = {}
for dtype, df in moscow_data.items():
    print(f"\n[Processing {dtype} data]:")
    if df is not None:
        moscow_data_filtered[dtype] = filter_features_rigorous(
            df,
            moscow_group,             # <--- Key change: pass in group information
            prevalence_threshold=0.05,
            abundance_threshold=1e-5
        )
        moscow_data_filtered[dtype] = None

print("\n" + "="*60)
print("✓ feature filteringcompleted")
print("  Note：Stage 5 modeling will use the feature intersection of the two cohorts,")
print("  but at this stage we filter them independently to preserve the purity of each cohort's data.")

# ## 1.7 Save Data

# ============================================================
# Save Stage 1 output data
# ============================================================
print("=" * 60)
print("Saving Stage 1 output data")
print("=" * 60)

stage1_output = {
    # ==== Filtered data ====
    'discovery_data_filtered': discovery_data_filtered,  # discovery cohort
    'moscow_data_filtered': moscow_data_filtered,        # Moscow cohort

    # ==== Group information (critical) ====
    'discovery_group': discovery_group,      # discovery cohortgroup (ASD/TD)
    'discovery_study': discovery_study,      # discovery cohortbatches (7 StudyIDs)
    'moscow_group': moscow_group,            # Moscow cohortgroup

    # ==== Changchun cohort labels (used in Stage 6) ====
    'local_cohort_samples': local_cohort_samples,  # Changchunsample IDslist
    'local_group': local_group,                     # Changchun sub-cohortgroup

    # ==== Sample-to-cohort mapping ====
    'discovery_sample_cohort_map': discovery_sample_cohort_map,
    'moscow_sample_cohort_map': moscow_sample_cohort_map,

    # ==== Metadata ====
    'metadata': metadata,
    'column_names': {
        'sample_id_col': sample_id_col,
        'group_col': group_col,
        'study_col': study_col
    },

    # ==== Version information ====
    'version': 'V4.7',
    'description': 'discovery cohort(7Chinese cohorts) + external validation(Moscow)'
}

# Save
output_path = os.path.join(MERGED_PATH, 'stage1_preprocessed_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(stage1_output, f)

print(f"✓ data saved to: {output_path}")

# Validate saved contents
print("\n【Output data structure check】")
for key in ['discovery_data_filtered', 'moscow_data_filtered',
            'discovery_group', 'discovery_study', 'moscow_group',
            'local_cohort_samples', 'local_group']:
    value = stage1_output.get(key)
    if isinstance(value, dict):
        print(f"  ✓ {key}: dict with {len(value)} items")
    elif isinstance(value, pd.Series):
        print(f"  ✓ {key}: Series with {len(value)} items")
    elif isinstance(value, list):
        print(f"  ✓ {key}: list with {len(value)} items")
    elif value is None:
        print(f"  ⚠ {key}: None")
        print(f"  ✓ {key}: {type(value).__name__}")

# ============================================================
# Stage 1 summary report
# ============================================================
print("\n" + "=" * 60)
print("Stage 1: Data integration and preprocessing - completed!")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    Stage1 Processingsummary (V4.7)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【discovery cohort】7Chinese cohorts（direct merge）                             │
│  • 6 public cohorts + Changchun cohort                                       │
│  • Use: Stage2-5 (batch correction、analysis and modeling)                        │
│                                                                 │
│  【external validationcohort】Moscow cohort                                      │
│  • Use: Stage5 cross-ethnicexternal validation                                  │
│                                                                 │
│  [Changchun sub-cohort labels] saved                                        │
│  • Use: Stage 6 behavior prediction analysis                                    │
│                                                                 │
│  [Output variables] for Stage 2                                        │
│  • discovery_data_filtered: four-layer discovery cohort data                   │
│  • discovery_group: group information                                    │
│  • discovery_study: batch information（7 StudyIDs）                     │
│  • moscow_data_filtered: Moscow cohort data                         │
│  • moscow_group: Moscowgroup                                     │
│  • local_cohort_samples: Changchunsample IDs                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n[Data statistics]")
print(f"  discovery cohort sample count: {len(discovery_sample_cohort_map)}")
print(f"  Moscow cohort sample count: {len(moscow_sample_cohort_map)}")
print(f"  Changchun sub-cohort sample count: {len(local_cohort_samples)}")

print("\n" + "=" * 60)
print("✓ Stage 1 completed. Please continue with Stage 2")
print("=" * 60)
