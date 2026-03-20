# MicroASD

Code repository for a microbiome-based autism spectrum disorder (ASD) study spanning data integration, ecological analysis, differential analysis, machine learning, behavioral association, and subtype analysis.

## Repository Layout

```text
MicroASD/
├── 01_raw_outputs/          # Raw MetaPhlAn/HUMAnN outputs
├── 02_merged_data/          # Intermediate and final serialized analysis outputs
├── 03_metadata_tables/      # Metadata and clinical tables
├── 04_figures/              # Exported figures
└── 05_scripts/              # Notebooks and refactored Python scripts
```

## Refactored Stage Scripts

- `05_scripts/stage1_data_integration.py`
- `05_scripts/stage2_batch_correction.py`
- `05_scripts/stage3_ecological_analysis.py`
- `05_scripts/stage4_differential_analysis.py`
- `05_scripts/stage5_machine_learning.py`
- `05_scripts/stage6_behavioural_analysis.py`
- `05_scripts/stage7_subtype_analysis.py`

These `.py` files were refactored from the original notebooks while preserving the analysis logic. Fixed Colab/Google Drive paths were replaced with project-root-relative paths so the scripts can run from a cloned repository.

## Expected Data Directories

Create or populate the following folders before running the pipeline:

- `01_raw_outputs`
- `02_merged_data`
- `03_metadata_tables`
- `04_figures`

The scripts automatically resolve paths relative to the repository root.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Original notebooks are retained in `05_scripts/` for provenance.
- The refactored scripts preserve the original stage ordering and output conventions.
- Some scripts still install optional packages at runtime if missing, matching the notebook behavior.
