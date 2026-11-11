import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# Example 1: Histogram
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

# Ensure repo root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# Importing jo_stuff automatically patches matplotlib and seaborn

# ==============================================================================
# 1. SYNTHETIC MULTI-OMICS / CLINICAL DATA GENERATION
# ==============================================================================
np.random.seed(101)
n_samples = 220

# Latent biological drivers
driver_metabolic = np.random.normal(0, 1.2, n_samples)
driver_inflammation = np.random.normal(0, 1.0, n_samples)
driver_hypoxia = np.random.normal(0, 0.9, n_samples)

features = {
    # Genomic / Epigenetic markers
    "TP53_mut": 0.75 * driver_hypoxia
    + 0.3 * driver_inflammation
    + np.random.normal(0, 0.4, n_samples),
    "KRAS_expr": 0.8 * driver_metabolic
    + 0.4 * driver_hypoxia
    + np.random.normal(0, 0.4, n_samples),
    "BRCA_meth": -0.6 * driver_hypoxia + np.random.normal(0, 0.5, n_samples),
    "MYC_amp": 0.65 * driver_metabolic
    + 0.5 * driver_inflammation
    + np.random.normal(0, 0.4, n_samples),
    # Immune / Cytokine panel
    "IL6_serum": 0.85 * driver_inflammation + np.random.normal(0, 0.35, n_samples),
    "TNFa_serum": 0.8 * driver_inflammation
    + 0.2 * driver_metabolic
    + np.random.normal(0, 0.35, n_samples),
    "CD8_infiltr": -0.55 * driver_inflammation
    + 0.4 * driver_hypoxia
    + np.random.normal(0, 0.5, n_samples),
    "PDL1_score": 0.7 * driver_inflammation
    + 0.45 * driver_hypoxia
    + np.random.normal(0, 0.4, n_samples),
    # Metabolic & Histopathology
    "Lactate_dehyd": 0.6 * driver_metabolic
    + 0.7 * driver_hypoxia
    + np.random.normal(0, 0.3, n_samples),
    "Gluc_uptake": 0.85 * driver_metabolic + np.random.normal(0, 0.35, n_samples),
    "Necrosis_pct": 0.5 * driver_hypoxia
    + 0.6 * driver_inflammation
    + np.random.normal(0, 0.4, n_samples),
    "Vasc_density": -0.7 * driver_hypoxia
    - 0.2 * driver_metabolic
    + np.random.normal(0, 0.45, n_samples),
}

df = pd.DataFrame(features)
cols = list(df.columns)
n_features = len(cols)

# Metadata / Module assignments
module_map = {
    "TP53_mut": "Genomics",
    "KRAS_expr": "Genomics",
    "BRCA_meth": "Genomics",
    "MYC_amp": "Genomics",
    "IL6_serum": "Immune",
    "TNFa_serum": "Immune",
    "CD8_infiltr": "Immune",
    "PDL1_score": "Immune",
    "Lactate_dehyd": "Metabolism",
    "Gluc_uptake": "Metabolism",
    "Necrosis_pct": "Metabolism",
    "Vasc_density": "Metabolism",
}
module_colors = {"Genomics": "#3B4992", "Immune": "#EE0000", "Metabolism": "#008B45"}
feature_colors = [module_colors[module_map[c]] for c in cols]

# ==============================================================================
# 2. STATISTICAL ENGINE (r, p-values, FDR adjusted)
# ==============================================================================
corr_matrix = np.zeros((n_features, n_features))
pval_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        r, p = pearsonr(df[cols[i]], df[cols[j]])
        corr_matrix[i, j] = r
        pval_matrix[i, j] = p

corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
# Upper triangle mask (keep diagonal and lower triangle visible)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Marginal summary metrics
mean_abs_corr = np.abs(corr_matrix - np.eye(n_features)).sum(axis=1) / (n_features - 1)
missing_rate = np.random.uniform(0.01, 0.08, n_features)  # Simulated QA metric

# ==============================================================================
# 3. FIGURE CANVAS ARCHITECTURE (GridSpec)
# ==============================================================================
plt.style.use("seaborn-v0_8-white")
fig = plt.figure(figsize=(15, 12), dpi=300)

gs = gridspec.GridSpec(
    nrows=3,
    ncols=4,
    width_ratios=[0.025, 0.72, 0.14, 0.03],
    height_ratios=[0.11, 0.74, 0.03],
    wspace=0.06,
    hspace=0.06,
)

ax_module_y = fig.add_subplot(gs[1, 0])  # Left category strip
ax_main = fig.add_subplot(gs[1, 1])  # Lower-tri heatmap
ax_margin_x = fig.add_subplot(gs[0, 1])  # Top bar plot (Mean Absolute Correlation)
ax_margin_y = fig.add_subplot(gs[1, 2])  # Right bar plot (Data Quality / Missingness)
ax_cbar = fig.add_subplot(gs[1, 3])  # Diverging colorbar

# Hide empty corner slots
for slot in [(0, 0), (0, 2), (0, 3), (2, 0), (2, 1), (2, 2), (2, 3)]:
    ax_hide = fig.add_subplot(gs[slot[0], slot[1]])
    ax_hide.axis("off")

# ==============================================================================
# 4. LOWER TRIANGULAR HEATMAP + SIGNIFICANCE LABELS + BLOCK CLUSTERING
# ==============================================================================
# High-contrast diverging palette
cmap_custom = LinearSegmentedColormap.from_list(
    "bio_diverging",
    ["#1A365D", "#2B6CB0", "#EBF8FF", "#FFF5F5", "#C53030", "#742A2A"],
    N=256,
)

sns.heatmap(
    corr_df,
    mask=mask,
    cmap=cmap_custom,
    vmin=-1.0,
    vmax=1.0,
    center=0.0,
    square=True,
    linewidths=1.0,
    linecolor="#FFFFFF",
    cbar=True,
    cbar_ax=ax_cbar,
    cbar_kws={"label": "Pearson Correlation ($r$)", "shrink": 0.8},
    ax=ax_main,
)

# Overlay p-value significance stars directly inside lower-triangle cells
for i in range(n_features):
    for j in range(n_features):
        if j <= i and i != j:
            p = pval_matrix[i, j]
            r = corr_matrix[i, j]
            symbol = (
                "***"
                if p < 0.001
                else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            )

            # Text contrast logic
            text_color = "#FFFFFF" if abs(r) > 0.45 else "#1A202C"

            ax_main.text(
                j + 0.5,
                i + 0.58,
                f"{r:.2f}\n{symbol}",
                ha="center",
                va="center",
                fontsize=7.5,
                weight="semibold",
                color=text_color,
                linespacing=0.85,
            )

# Draw cluster / functional module bounding boxes (4x4 blocks)
for start_idx in [0, 4, 8]:
    rect = Rectangle(
        (start_idx, start_idx),
        4,
        4,
        fill=False,
        edgecolor="#1A202C",
        linewidth=2.0,
        linestyle="--",
    )
    ax_main.add_patch(rect)

# ==============================================================================
# 5. INTEGRATED MARGINAL PANELS & ANNOTATION TRACKS
# ==============================================================================
# Top Panel: Mean connectivity per biomarker
ax_margin_x.bar(
    range(n_features),
    mean_abs_corr,
    color=feature_colors,
    edgecolor="#FFFFFF",
    width=0.8,
    alpha=0.9,
)
ax_margin_x.set_xlim(-0.5, n_features - 0.5)
ax_margin_x.set_ylim(0, max(mean_abs_corr) * 1.25)
ax_margin_x.set_ylabel("Mean |$r$|", fontsize=9, weight="bold")
ax_margin_x.tick_params(axis="x", labelbottom=False, bottom=False)
ax_margin_x.tick_params(axis="y", labelsize=8)
ax_margin_x.grid(axis="y", linestyle=":", alpha=0.6)
sns.despine(ax=ax_margin_x, top=True, right=True, bottom=True)

# Right Panel: QA Data Missingness Rate
y_positions = np.arange(n_features)
ax_margin_y.barh(
    y_positions,
    missing_rate * 100,
    color="#718096",
    edgecolor="#FFFFFF",
    height=0.75,
    alpha=0.8,
)
ax_margin_y.set_ylim(n_features - 0.5, -0.5)  # Invert Y to align with heatmap
ax_margin_y.set_xlabel("Missing (%)", fontsize=9, weight="bold")
ax_margin_y.tick_params(axis="y", labelleft=False, left=False)
ax_margin_y.tick_params(axis="x", labelsize=8)
ax_margin_y.grid(axis="x", linestyle=":", alpha=0.6)
sns.despine(ax=ax_margin_y, top=True, right=True, left=True)

# Left Side Track: Module classification strip
color_matrix = np.array([[feature_colors[i]] for i in range(n_features)])
for i in range(n_features):
    ax_module_y.add_patch(
        Rectangle((0, i), 1, 1, color=feature_colors[i], ec="white", lw=1)
    )
ax_module_y.set_xlim(0, 1)
ax_module_y.set_ylim(n_features, 0)
ax_module_y.axis("off")

# Format Heatmap Tick Labels
ax_main.set_xticklabels(cols, rotation=45, ha="right", fontsize=9.5, weight="semibold")
ax_main.set_yticklabels(cols, rotation=0, fontsize=9.5, weight="semibold")

# ==============================================================================
# 6. LEGEND, COLORBAR REFINEMENT & TITLE
# ==============================================================================
# Custom module legend
legend_elements = [
    plt.Line2D(
        [0], [0], marker="s", color="w", label=k, markerfacecolor=v, markersize=10
    )
    for k, v in module_colors.items()
]
legend_elements.append(
    plt.Line2D(
        [0], [0], color="#1A202C", lw=1.5, linestyle="--", label="Functional Modules"
    )
)

ax_main.legend(
    handles=legend_elements,
    title="Biological Subsystem",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
    frameon=True,
    framealpha=0.95,
    facecolor="#F7FAFC",
    edgecolor="#E2E8F0",
    fontsize=8.5,
    title_fontsize=9.5,
)

# Header Annotations
fig.suptitle(
    "Multi-Omic Biomarker Interaction Matrix & Data Quality Profile",
    fontsize=16,
    weight="bold",
    x=0.08,
    y=0.98,
    ha="left",
    color="#0F172A",
)

# Statistical footnote
fig.text(
    0.08,
    0.02,
    "Significance levels (Pearson product-moment): *p < 0.05, **p < 0.01, ***p < 0.001. "
    "Diagonal values (r = 1.00) retained for intra-module scale anchoring.",
    fontsize=8.5,
    style="italic",
    color="#475569",
)

plt.show()
