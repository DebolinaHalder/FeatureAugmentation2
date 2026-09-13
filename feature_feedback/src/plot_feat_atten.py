#%%
import seaborn as sns
sns.set_context("talk")

import matplotlib.pyplot as plt
import pickle
import numpy as np
from dgl.data import CoraGraphDataset

# =========================================================
# LOAD DATA
# =========================================================
data_cora = CoraGraphDataset()
g_cora = data_cora[0]

with open('atten_GAT_cora.pickle', 'rb') as handle:
    attn_mat = pickle.load(handle)

with open('features_cora_f_gt.pickle', 'rb') as handle:
    features = pickle.load(handle)

attn_mat = np.asarray(attn_mat)
features = np.asarray(features)

# =========================================================
# COSINE SIMILARITY
# =========================================================
norms = np.linalg.norm(features, axis=1, keepdims=True)
X_norm = features / (norms + 1e-12)
sim_mat = X_norm @ X_norm.T

# =========================================================
# USE ALL NODE PAIRS
# =========================================================
alpha_all = attn_mat.flatten()
sim_all = sim_mat.flatten()

# Remove NaNs / inf just in case
mask = np.isfinite(alpha_all) & np.isfinite(sim_all)
alpha_all = alpha_all[mask]
sim_all = sim_all[mask]

# =========================================================
# BINNED EXPECTATION: E[alpha | similarity]
# =========================================================
num_bins = 30
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

# =========================================================
# PLOT 1: Binned Expectation (MAIN FIGURE)
# =========================================================
plt.figure(figsize=(6,5))

plt.plot(bin_centers, avg_alpha, marker='o')
plt.xlabel(r'Feature Similarity $sim(z_i, z_j)$')
plt.ylabel(r'$\mathbb{E}[\alpha_{ij} \mid sim]$')
plt.title("Attention vs Feature Similarity (Binned)")

plt.tight_layout()
plt.show()

# =========================================================
# PLOT 2: DENSITY (OPTIONAL BUT NICE)
# =========================================================

# %%
