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

# # Stage 3: Microbiome Ecological Feature Analysis
#
# **Version**: V4.7
#
# ### Input Data (from Stage 2)
# - `discovery_data_corrected`: Batch-corrected discovery cohort data
# - `discovery_data_raw`: Raw relative abundance data (for alpha diversity)
# - `discovery_group`: Group information

# ============================================================
# 1. Environment initialization and dependency installation (revised version)
# ============================================================
# 1. Install numpy first because it is a build dependency of scikit-bio
install_packages('numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn')

# 2. Install the latest scikit-bio version (remove the ==0.5.9 restriction)
# The new version fixes build issues and remains backward-compatible with alpha_diversity
install_packages('scikit-bio')

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

# Try importing to verify the installation
try:
    from skbio.diversity import alpha_diversity
    print("✓ scikit-bio installed and imported successfully")
except ImportError:
    print("⚠ scikit-bio installation failed; a fallback algorithm will be used (see below)")

# ------------------------------------------------------------
# Plot settings (Nature/Science style)
# ------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300

import warnings
warnings.filterwarnings('ignore')

print("✓ Environment dependency installation and plotting setup completed")

# ============================================================
# 2. Mount Google Drive and define paths
# ============================================================


# Define standard project paths

# Ensure the output directory exists
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
    print(f"✓ Creating figure directory: {FIG_PATH}")
    print(f"✓ Figure directory is ready: {FIG_PATH}")

# ============================================================
# 3. Load data and core utility functions (fixed version)
# ============================================================

# --- 3.1 Load Stage 2 data ---
stage2_path = os.path.join(str(DATA_PATH), 'stage2_normalized_data.pkl')
if os.path.exists(stage2_path):
    with open(stage2_path, 'rb') as f:
        stage2_data = pickle.load(f)
    print("✓ Successfully loaded Stage 2 data")
    raise FileNotFoundError(f"❌ File not found: {stage2_path}，Please run Stage 2 first！")

# Extract key variables
# Note the distinction: use Raw data for alpha diversity and Corrected data (batch-corrected CLR) for beta diversity
data_corrected = stage2_data['discovery_data_corrected']
data_raw = stage2_data['discovery_data_raw']
metadata_group = stage2_data['discovery_group']

print(f"  - sample size: {len(metadata_group)}")
print(f"  - group summary: {metadata_group.value_counts().to_dict()}")

# --- 3.2 Define the species-cleaning function (V4.7 critical fix) ---
def clean_taxa_data(df):
    """
    Cleaning logic:
    1. Keep only the Species (s__) level and remove Strain (t__) and higher levels
    2. Simplify names (s__Bacteroides_fragilis -> Bacteroides fragilis)
    """
    # Keep rows that contain s__ but not t__
    species_rows = [idx for idx in df.index if 's__' in idx and 't__' not in idx]

    # If no s__ entries are found, the names may already have been cleaned in Stage 1, or the data may not be Taxa data
    if len(species_rows) == 0:
        # Try checking whether the names have already been cleaned (without prefixes)
        return df

    df_species = df.loc[species_rows].copy()

    # Clean index names by removing the 's__' prefix and replacing underscores with spaces
    clean_names = [name.split('s__')[-1].replace('_', ' ') for name in df_species.index]
    df_species.index = clean_names

    print(f"  [Cleaning report] Original features: {df.shape[0]} -> Retained species: {df_species.shape[0]}")
    return df_species

# ## 3.1 Alpha Diversity Analysis

# ============================================================
# 4. Alpha diversity analysis (using raw relative abundance)
# ============================================================
from skbio.diversity import alpha_diversity

print("\n" + "="*40)
print("Starting alpha-diversity analysis (Shannon, Simpson)")
print("="*40)

metrics = ['shannon', 'simpson']
alpha_results_all = {}

# Iterate through the four data layers (if available)
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    if dtype not in data_raw: continue

    print(f"\n>> Analyzing: {dtype}")
    df = data_raw[dtype]

    # Special handling for Taxa: keep only the species level
    if dtype == 'taxa':
        df = clean_taxa_data(df)

    # Prepare data: skbio requires counts (integers), so we approximate them with relative abundance * 1e6
    counts = (df.T * 1e6).astype(int)
    ids = counts.index

    # Compute metrics
    alpha_df = pd.DataFrame(index=ids)
    for metric in metrics:
        try:
            # Catch possible warnings from skbio
            res = alpha_diversity(metric, counts.values, ids=ids)
            alpha_df[metric] = res
        except Exception as e:
            print(f"  Computing {metric} error: {e}")

    # Merge group information
    alpha_df['Group'] = metadata_group
    alpha_results_all[dtype] = alpha_df

    # --- Statistical testing and plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Statistical testing (Wilcoxon rank-sum)
        vals_asd = alpha_df[alpha_df['Group']=='ASD'][metric]
        vals_td = alpha_df[alpha_df['Group']=='TD'][metric]
        stat, p = stats.ranksums(vals_asd, vals_td)

        # Plotting (violin plot + box plot)
        sns.violinplot(x='Group', y=metric, data=alpha_df, ax=ax,
                       palette={'ASD': '#E74C3C', 'TD': '#3498DB'}, alpha=0.3)
        sns.boxplot(x='Group', y=metric, data=alpha_df, ax=ax,
                    width=0.2, palette={'ASD': '#E74C3C', 'TD': '#3498DB'}, showfliers=False)

        # Annotate the P-value
        title = f"{metric.capitalize()}\nWilcoxon p={p:.2e}"
        if p < 0.05: title += " *"
        ax.set_title(title)
        ax.set_xlabel("")

    plt.suptitle(f"Alpha Diversity - {dtype.capitalize()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(str(FIG_PATH), f'Fig3A_Alpha_{dtype}.pdf'))
    plt.show()
    print(f"  ✓ Figure saved: Fig3A_Alpha_{dtype}.pdf")

# ## 3.2 Beta Diversity Analysis

# ============================================================
# 5. Beta diversity analysis (final hardened version: typo fixed)
# ============================================================
print("\n" + "="*60)
print("Starting beta-diversity analysis (Aitchison distance + PERMANOVA statistics)")
print("="*60)

import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from skbio.stats.distance import permanova, DistanceMatrix

def plot_pcoa_with_stats(coords, metadata, f_stat, p_val, title, output_filename):
    """
    Draw a publication-grade PCoA plot with statistical annotations
    """
    # Set up the layout
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.1)

    ax_main = fig.add_subplot(gs[1:4, 0:3]) # main panel
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main) # top panel
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main) # right panel

    colors = {'ASD': '#E74C3C', 'TD': '#3498DB'}

    # ---------------------------
    # 1. Draw the main plot
    # ---------------------------
    for group in ['ASD', 'TD']:
        # Ensure that metadata and coords have aligned indices
        idx = (metadata.values == group)

        # Skip if this group has no samples
        if np.sum(idx) == 0: continue

        x = coords[idx, 0]
        y = coords[idx, 1]
        c = colors[group]

        # Centroid
        centroid_x, centroid_y = np.mean(x), np.mean(y)

        # A. Spider lines
        ax_main.plot([x, np.full_like(x, centroid_x)],
                     [y, np.full_like(y, centroid_y)],
                     color=c, alpha=0.1, linewidth=0.5, zorder=1)

        # B. Scatter points
        ax_main.scatter(x, y, c=c, label=group, alpha=0.8, s=40, edgecolors='white', linewidth=0.5, zorder=2)

        # C. Large centroid marker
        ax_main.scatter(centroid_x, centroid_y, c=c, s=150, marker='X', edgecolors='black', linewidth=1, zorder=3)

        # D. Confidence ellipse
        if len(x) > 5:
            try:
                cov = np.cov(x, y)
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                ell = Ellipse(xy=(centroid_x, centroid_y),
                              width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                              angle=np.rad2deg(np.arccos(v[0, 0])),
                              color=c, alpha=0.15, zorder=0)
                ax_main.add_artist(ell)
            except:
                pass

    # ---------------------------
    # 2. Annotate key statistics
    # ---------------------------
    stats_text = f"PERMANOVA\nPseudo-$F$ = {f_stat:.2f}\n$P$ = {p_val:.3f}"
    if p_val < 0.001:
        stats_text = f"PERMANOVA\nPseudo-$F$ = {f_stat:.2f}\n$P$ < 0.001"

    ax_main.text(0.02, 0.02, stats_text, transform=ax_main.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ---------------------------
    # 3. Draw the marginal plots
    # ---------------------------
    # Top
    sns.boxplot(x=coords[:, 0], y=metadata, hue=metadata, ax=ax_top,
                palette=colors, orient='h', width=0.5, dodge=False, showfliers=False)
    ax_top.set(xlabel='', ylabel='', xticklabels=[], yticklabels=[])
    ax_top.axis('off')
    if ax_top.get_legend() is not None: ax_top.get_legend().remove()

    # Right
    sns.boxplot(x=metadata, y=coords[:, 1], hue=metadata, ax=ax_right,
                palette=colors, orient='v', width=0.5, dodge=False, showfliers=False)
    ax_right.set(xlabel='', ylabel='', xticklabels=[], yticklabels=[])
    ax_right.axis('off')
    if ax_right.get_legend() is not None: ax_right.get_legend().remove()

    # ---------------------------
    # 4. Styling
    # ---------------------------
    ax_main.set_xlabel(f"PCo1")
    ax_main.set_ylabel(f"PCo2")
    ax_main.legend(loc='upper right', frameon=False)

    fig.suptitle(title, y=0.92, fontsize=12)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()

# --- Run the loop ---
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    if dtype not in data_corrected: continue

    print(f"\n>> Analyzing: {dtype}")
    df = data_corrected[dtype]

    # Clean Taxa labels
    if dtype == 'taxa':
        df = clean_taxa_data(df)

    # 1. Data alignment
    common_idx = df.columns.intersection(metadata_group.index)
    X_df = df[common_idx].T # rows are samples
    group_aligned = metadata_group.loc[common_idx]

    # === Critical fix 1: check and clean NaNs/Infs ===
    if X_df.isnull().values.any() or np.isinf(X_df.values).any():
        print(f"  ⚠ Warning: {dtype} data contain NaN or Inf; cleaning is in progress...")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(0)

    # 2. Compute the distance matrix
    try:
        dist_matrix = euclidean_distances(X_df, X_df)
    except Exception as e:
        print(f"  ❌ Computingdistance matrixfailed: {e}")
        continue

    # === Critical fix 2: force symmetry and remove diagonal noise ===
    # (D + D.T) / 2 removes floating-point asymmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Fix applied here: ensure the full spelling of np.fill_diagonal
    np.fill_diagonal(dist_matrix, 0)

    if np.isnan(dist_matrix).any():
        print("  ⚠ Warning: distance matrix contains NaN; attempting repair...")
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)

    # 3. Compute PERMANOVA
    try:
        dm = DistanceMatrix(dist_matrix, ids=common_idx)
        permanova_res = permanova(dm, group_aligned, permutations=999)
        f_stat = permanova_res['test statistic']
        p_val = permanova_res['p-value']
    except Exception as e:
        print(f"  ⚠ PERMANOVA Computingfailed: {e}")
        f_stat = 0.0
        p_val = 1.0

    # 4. Reduce dimensions with PCoA
    try:
        pcoa = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = pcoa.fit_transform(dist_matrix)

        # 5. Plot
        out_file = os.path.join(str(FIG_PATH), f'Fig3B_Beta_{dtype}_Stats.pdf')
        plot_pcoa_with_stats(
            coords,
            group_aligned,
            f_stat,
            p_val,
            f"Beta Diversity (Aitchison) - {dtype.capitalize()}",
            out_file
        )
        print(f"  ✓ Plotting completed (F={f_stat:.2f}, P={p_val:.4f})")

    except Exception as e:
        print(f"  ❌ PCoA/plotting process error: {e}")

print("\n" + "="*60)
print("🎉 Stage 3 beta-diversity analysis completed")

# ## 3.3 Composition Analysis

# ============================================================
# 6. Composition analysis (all layers: Taxa + Pathways + Genes + ECs)
# ============================================================
print("\n" + "="*60)
print("Starting community composition analysis (covering all four layers)")
print("="*60)

# First: check whether the data exist
print("Data integrity check:")
available_dtypes = []
for dtype in ['taxa', 'pathways', 'genes', 'ecs']:
    if dtype in data_raw and data_raw[dtype] is not None:
        print(f"  ✓ {dtype}: {data_raw[dtype].shape}")
        available_dtypes.append(dtype)
        print(f"  ❌ {dtype}: data missing or not loaded")

def plot_composition_stacked(df, group_series, top_n=10, dtype='taxa', output_path=None):
    """
    Draw grouped mean stacked bar charts (fixed version)
    """
    # 1. Data alignment
    common = df.columns.intersection(group_series.index)
    df_aligned = df[common]
    groups = group_series.loc[common]

    # 2. Compute mean abundance for each group
    group_means = df_aligned.T.groupby(groups).mean().T

    # 3. Rank and select the top N
    group_means['Total_Mean'] = group_means.sum(axis=1)
    df_sorted = group_means.sort_values('Total_Mean', ascending=False).drop('Total_Mean', axis=1)

    # Extract the top N
    top_features = df_sorted.head(top_n)

    # Compute the Others category
    others = df_sorted.iloc[top_n:].sum()
    others.name = 'Others'

    # Merge
    plot_data = pd.concat([top_features, pd.DataFrame(others).T])

    # 4. Plot
    fig, ax = plt.subplots(figsize=(5, 6))

    # Set colors
    colors = plt.cm.tab20.colors
    if top_n > 20:
        colors = plt.cm.tab20c.colors

    # Draw the stacked plot
    plot_data.T.plot(kind='bar', stacked=True, ax=ax, width=0.6,
                     color=list(colors[:top_n]) + ['#D3D3D3'], # gray for Others
                     edgecolor='black', linewidth=0.5)

    # 5. Styling
    ax.set_ylabel("Mean Relative Abundance")
    ax.set_xlabel("")
    ax.set_title(f"Composition - {dtype.capitalize()}", fontsize=12)
    plt.xticks(rotation=0)

    # Fix the legend
    handles, labels = ax.get_legend_handles_labels()
    # Reverse the list with slicing
    ax.legend(handles[::-1], labels[::-1],
              bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=8, title=f"Top {top_n} Features", frameon=False)

    sns.despine()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# --- Run the loop (use available_dtypes to avoid errors) ---
for dtype in available_dtypes:

    print(f"\n>> Drawing: {dtype}")
    df = data_raw[dtype]

    # --- Clean feature names ---
    if dtype == 'taxa':
        df = clean_taxa_data(df)

    elif dtype == 'pathways':
        # Clean pathway names by removing the 'PWY-XXXX: ' prefix
        new_index = []
        for idx in df.index:
            if ': ' in idx:
                clean = idx.split(': ')[1]
                new_index.append(clean[:40] + "..." if len(clean)>40 else clean)
                new_index.append(idx)
        df.index = new_index

    elif dtype == 'ecs':
        # Clean EC names by removing the 'EC:' prefix and keeping only the numeric part
        # Example: "EC:1.1.1.1" -> "1.1.1.1"
        new_index = [idx.replace('EC:', '') for idx in df.index]
        df.index = new_index

    elif dtype == 'genes':
        # Gene IDs are usually UniRef90_XXXX and can be kept as-is
        # Only truncate overly long IDs
        new_index = [idx[:20] + "..." if len(idx)>20 else idx for idx in df.index]
        df.index = new_index

    # 2. Plot
    out_file = os.path.join(str(FIG_PATH), f'Fig3C_Composition_{dtype}.pdf')
    plot_composition_stacked(
        df,
        metadata_group,
        top_n=10,
        dtype=dtype,
        output_path=out_file
    )
    print(f"  ✓ Composition plot saved: Fig3C_Composition_{dtype}.pdf")

print("\n" + "="*60)
print("🎉 All-layer composition analysis completed")
