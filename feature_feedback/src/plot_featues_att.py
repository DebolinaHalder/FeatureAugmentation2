#%%
import seaborn as sns
sns.set_context("talk",font_scale=2.5)

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.ticker import FuncFormatter

# =========================================================
# DATASETS & FEATURE TYPES
# =========================================================
datasets = ["cora", "citeseer", "photo", "cornell", "texas"]
feature_types = ["f_gat", "b_gat"]

feat_name = ["GCN(Feat.)","GCN (Both)"]

# =========================================================
# FORMATTER
# =========================================================
def no_leading_zero(x, pos):
    if abs(x) < 1e-12:   # handle zero safely
        return "0"
    s = f"{x:.4f}".rstrip('0').rstrip('.')
    if s.startswith("0"):
        s = s[1:]
    elif s.startswith("-0"):
        s = "-" + s[2:]
    return s

# =========================================================
# PLOT SETUP (3 rows × 5 columns)
# =========================================================
fig, axes = plt.subplots(2, 5, figsize=(37, 13),sharex=True)

# =========================================================
# LOOP OVER FEATURE TYPES (ROWS)
# =========================================================
for row, feat_type in enumerate(feature_types):

    for col, name in enumerate(datasets):

        # -------------------------------
        # LOAD DATA
        # -------------------------------
        with open(f'atten_GAT_{name}.pickle', 'rb') as handle:
            attn_mat = pickle.load(handle)
        
        
        
        with open(f'features_{name}_{feat_type}.pickle', 'rb') as handle:
            features = pickle.load(handle)

        attn_mat = np.asarray(attn_mat)
        features = np.asarray(features)

        # -------------------------------
        # COSINE SIMILARITY
        # -------------------------------
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        X_norm = features / (norms + 1e-12)
        sim_mat = X_norm @ X_norm.T

        

        edge_mask = attn_mat > 0   # keeps only real edges

        attn_mat = attn_mat[edge_mask]
        sim_mat = sim_mat[edge_mask]

        # -------------------------------
        # ALL NODE PAIRS
        # -------------------------------
        alpha_all = attn_mat.flatten()
        sim_all = sim_mat.flatten()
        

        mask = np.isfinite(alpha_all) & np.isfinite(sim_all)
        alpha_all = alpha_all[mask]
        sim_all = sim_all[mask]

        

        # -------------------------------
        # BINNING
        # -------------------------------
        num_bins = 100
        bins = np.linspace(sim_all.min(), sim_all.max(), num_bins)

        bin_indices = np.digitize(sim_all, bins)

        avg_alpha = []
        bin_centers = []

        for k in range(1, len(bins)):
            bin_mask = bin_indices == k
            if np.sum(bin_mask) > 0:
                avg_alpha.append(alpha_all[bin_mask].mean())
                bin_centers.append((bins[k] + bins[k-1]) / 2)

        avg_alpha = np.array(avg_alpha)
        bin_centers = np.array(bin_centers)

        # -------------------------------
        # PLOT
        # -------------------------------
        ax = axes[row, col]
        ax.scatter(bin_centers, avg_alpha, marker='o', linestyle='-', color='tab:blue')

        # Titles only on top row
        if row == 0:
            ax.set_title(name.capitalize(),fontsize = 52)

        # Row labels (feature types)
        if col == 0:
            ax.set_ylabel(feat_name[row])

        # Style
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=38)
        ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# =========================================================
# COMMON LABELS
# =========================================================
fig.supxlabel("Feature Similarity ($sim$)")
fig.supylabel(r'$\mathbb{E}[\alpha_{ij} \mid sim]$')

# =========================================================
# FINALIZE
# =========================================================
plt.tight_layout()
#plt.savefig("attention_GAT_vs_similarity_grid1.pdf", dpi=300)
plt.show()
# %%

# %%
#%%
import seaborn as sns
sns.set_context("talk", font_scale=2.5)

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.spatial.distance import cdist

# =========================================================
# DATASETS & FEATURE TYPES
# =========================================================
datasets = ["cora", "citeseer", "photo", "cornell", "texas"]
feature_types = ["f_gat", "b_gat"]

feat_name = ["GCN(Feat.)", "GCN (Both)"]

# =========================================================
# FORMATTER
# =========================================================
def no_leading_zero(x, pos):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.4f}".rstrip('0').rstrip('.')
    if s.startswith("0"):
        s = s[1:]
    elif s.startswith("-0"):
        s = "-" + s[2:]
    return s

# =========================================================
# PLOT SETUP (2 rows × 5 columns)
# =========================================================
fig, axes = plt.subplots(2, 5, figsize=(37, 13), sharex=True)

# =========================================================
# LOOP OVER FEATURE TYPES (ROWS)
# =========================================================
for row, feat_type in enumerate(feature_types):

    for col, name in enumerate(datasets):

        # -------------------------------
        # LOAD DATA
        # -------------------------------
        with open(f'atten_GAT_{name}.pickle', 'rb') as handle:
            attn_mat = pickle.load(handle)

        with open(f'features_{name}_{feat_type}.pickle', 'rb') as handle:
            features = pickle.load(handle)

        attn_mat = np.asarray(attn_mat)
        features = np.asarray(features)

        # -------------------------------
        # COSINE SIMILARITY
        # -------------------------------
        dists = cdist(features, features, metric='euclidean')

        # bandwidth (important!)
        
        from scipy.spatial.distance import cdist
        dists = cdist(features, features, metric='euclidean')
        sigma = 1
        sim_mat = np.exp(-(dists ** 2) / (2 * sigma ** 2))

        # -------------------------------
        # EDGE MASK
        # -------------------------------
        edge_mask = attn_mat > 0

        attn_vals = attn_mat[edge_mask]
        sim_vals = sim_mat[edge_mask]

        # -------------------------------
        # CLEAN
        # -------------------------------
        mask = np.isfinite(attn_vals) & np.isfinite(sim_vals)
        alpha_all = attn_vals[mask]
        sim_all = sim_vals[mask]

        # -------------------------------
        # BINNING
        # -------------------------------
        num_bins = 100
        bins = np.linspace(sim_all.min(), sim_all.max(), num_bins)
        bin_indices = np.digitize(sim_all, bins)

        avg_alpha = []
        bin_centers = []

        for k in range(1, len(bins)):
            bin_mask = bin_indices == k
            if np.sum(bin_mask) > 0:
                avg_alpha.append(alpha_all[bin_mask].mean())
                bin_centers.append((bins[k] + bins[k-1]) / 2)

        avg_alpha = np.array(avg_alpha)
        bin_centers = np.array(bin_centers)

        # -------------------------------
        # REGRESSION (LINEAR FIT)
        # -------------------------------
        if len(bin_centers) > 1:
            coeffs = np.polyfit(bin_centers, avg_alpha, deg=1)
            slope, intercept = coeffs

            x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 200)
            y_fit = np.polyval(coeffs, x_fit)
        else:
            x_fit, y_fit, slope = None, None, None

        # -------------------------------
        # PLOT
        # -------------------------------
        ax = axes[row, col]

        # scatter
        ax.scatter(bin_centers, avg_alpha, marker='o', color='tab:blue')

        # regression line
        if x_fit is not None:
            ax.plot(x_fit, y_fit, linestyle='--', linewidth=3)

            # show slope
            #ax.text(0.05, 1, f"slope={slope:.2f}",
            #        transform=ax.transAxes, fontsize=50)

        # Titles only on top row
        if row == 0:
            ax.set_title(name.capitalize(), fontsize=52, pad=55)

        # Row labels
        if col == 0:
            ax.set_ylabel(feat_name[row])

        # Style
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=38)
        ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# =========================================================
# COMMON LABELS
# =========================================================
fig.supxlabel("Feature Similarity ($sim$)")
fig.supylabel(r'$\mathbb{E}[\alpha_{ij} \mid sim]$')

# =========================================================
# FINALIZE
# =========================================================
plt.tight_layout()
plt.savefig("attention_GAT_vs_similarity_grid_regression.pdf", dpi=300)
plt.show()
#%%

import numpy as np
from scipy.stats import spearmanr
from dgl.data import CoraGraphDataset, CornellDataset
from scipy.stats import pearsonr
import numpy as np

def spearman_matrix(A, B):

    corrs = pearsonr(A,B)

data = CornellDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()
edges = list(zip(src, dst))

# -------------------------------
# Load attention weights
# -------------------------------
with open('attenedge_GAT_cornell.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()
edge_to_attn = {(u, v): w for (u, v), w in zip(zip(src, dst), attn_weights)}
# -------------------------------
# Load MI graph
# -------------------------------
with open('adj_cornell_b_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

edges_to_mi = list(zip(src_adj, dst_adj))

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# ALIGN MI weights to original edges
# -------------------------------
edge_to_mi = {(u, v): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)}
edge_to_mi.update({(v, u): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)})

mi_on_g = []
attn_on_g = []

for (u, v), a in zip(edges_to_mi, mi_weights):
    mi_on_g.append(edge_to_mi.get((u, v)))  # 0 if missing
    attn_on_g.append(edge_to_mi.get((u, v),0.0))

mi_on_g = np.array(mi_on_g)
attn_on_g = np.array(attn_on_g)
num_nodes = len(labels)
row_sum = np.zeros(num_nodes)

for u, w in zip(src_adj, mi_on_g):
    row_sum[u] += w

row_sum[row_sum == 0] = 1.0

mi_on_g = np.array([
    w / row_sum[u] for u, w in zip(src_adj, mi_on_g)
])

# %%
