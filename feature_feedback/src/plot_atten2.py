#%%
from matplotlib import cm
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
random.seed(0)
import dgl
from matplotlib.colors import Normalize
from dgl.data import CoraGraphDataset,CornellDataset

#%%
f_size = 30

with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

# -------------------------------
# Attention graph edges + weights
# -------------------------------
weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

# threshold (optional, keep if you want sparsity)
thresh = 0.2
mask = weights > thresh

weights = weights[mask]
src_adj = src_adj[mask]
dst_adj = dst_adj[mask]

# -------------------------------
# Build weighted graph
# -------------------------------
nx_g_adj = nx.Graph()
nx_g_adj.add_nodes_from(range(g.num_nodes()))

edges_with_attr = list(zip(src_adj, dst_adj, weights))
nx_g_adj.add_weighted_edges_from(edges_with_attr)

# -------------------------------
# Cluster-aware layout (FIXED)
# -------------------------------
nx_layout = nx.Graph()
nx_layout.add_nodes_from(nx_g_adj.nodes())

for u, v in nx_g_adj.edges():
    if labels[u] == labels[v]:
        w = 5.0   # strong intra-cluster
    else:
        w = 0.002  # weak inter-cluster
    nx_layout.add_edge(u, v, weight=w)

pos = nx.spring_layout(
    nx_g_adj,
    weight='weight',
    seed=38,
    k=0.1,
    iterations=200
)

# -------------------------------
# Color + normalization
# -------------------------------
cmap = plt.cm.YlGnBu
norm = Normalize(vmin=weights.min(), vmax=weights.max())

max_w = weights.max()

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(25, 20))

# sort edges → ensures high-weight edges on top
edges_sorted = sorted(edges_with_attr, key=lambda x: x[2])

for u, v, w in edges_sorted:
    w_norm = (w / max_w)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.5 + 3 * (w_norm**2),
        alpha=0.1 + 0.9 * w_norm,
        color=cmap(norm(w)),
        zorder=1 + w_norm
    )

# nodes
nx.draw_networkx_nodes(
    nx_g_adj,
    pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

plt.title("Attention-Weighted Graph (Cora)", fontsize=f_size)
plt.axis("off")

# -------------------------------
# Colorbar
# -------------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, fraction=0.02, pad=0.02)
cbar.set_label("Edge Attention Weight")

#plt.savefig("cora_attention_graph_clean.pdf", dpi=300)
plt.show()
#%%
#%%
from matplotlib import cm
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
random.seed(0)
import dgl
from matplotlib.colors import Normalize
from dgl.data import CoraGraphDataset

#%%
f_size = 33

# -------------------------------
# Load base graph
# -------------------------------
data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()
edges = list(zip(src, dst))

# -------------------------------
# Load attention weights
# -------------------------------
with open('attenedge_GAT_cora.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()

# -------------------------------
# Load MI graph
# -------------------------------
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# ALIGN MI weights to original edges
# -------------------------------
edge_to_mi = {(u, v): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)}
edge_to_mi.update({(v, u): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)})

mi_on_g = []
attn_on_g = []

for (u, v), a in zip(edges, attn_weights):
    mi_on_g.append(edge_to_mi.get((u, v), 0.0))  # 0 if missing
    attn_on_g.append(a)

mi_on_g = np.array(mi_on_g)
attn_on_g = np.array(attn_on_g)

# -------------------------------
# Build graph
# -------------------------------
nx_g = nx.Graph()
nx_g.add_nodes_from(range(g.num_nodes()))
nx_g.add_edges_from(edges)

# -------------------------------
# Shared layout (cluster-aware)
# -------------------------------
nx_layout = nx.Graph()
nx_layout.add_nodes_from(nx_g.nodes())

for u, v in nx_g.edges():
    if labels[u] == labels[v]:
        w = 5.0
    else:
        w = 0.05
    nx_layout.add_edge(u, v, weight=w)

pos = nx.spring_layout(nx_layout, weight='weight', seed=42, k=0.3, iterations=200)

# -------------------------------
# Color setup
# -------------------------------
cmap = plt.cm.YlGnBu

norm_mi = Normalize(vmin=mi_on_g.min(), vmax=mi_on_g.max())
norm_attn = Normalize(vmin=attn_on_g.min(), vmax=attn_on_g.max())

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(40, 18))

# ===============================
# (1) MI graph
# ===============================
plt.sca(ax[0])

edges_mi = list(zip(src, dst, mi_on_g))
edges_sorted = sorted(edges_mi, key=lambda x: x[2])

for u, v, w in edges_sorted:
    if w == 0:
        continue

    w_norm = (w - mi_on_g.min()) / (mi_on_g.max() - mi_on_g.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_norm**2),
        alpha=0.1 + 0.9 * w_norm,
        color=cmap(norm_mi(w)),
        zorder=1 + w_norm
    )

nx.draw_networkx_nodes(
    nx_g, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[0].set_title("MI-weighted Graph", fontsize=f_size)
ax[0].axis("off")

# ===============================
# (2) Attention graph
# ===============================
plt.sca(ax[1])

edges_attn = list(zip(src, dst, attn_on_g))
edges_sorted = sorted(edges_attn, key=lambda x: x[2])

for u, v, w in edges_sorted:
    w_norm = (w - attn_on_g.min()) / (attn_on_g.max() - attn_on_g.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_norm**2),
        alpha=0.1 + 0.9 * w_norm,
        color=cmap(norm_attn(w)),
        zorder=1 + w_norm
    )

nx.draw_networkx_nodes(
    nx_g, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[1].set_title("GAT Attention Graph", fontsize=f_size)
ax[1].axis("off")

# -------------------------------
# Separate colorbars
# -------------------------------
sm1 = cm.ScalarMappable(norm=norm_mi, cmap=cmap)
sm1.set_array([])
fig.colorbar(sm1, ax=ax[0], fraction=0.03, pad=0.02).set_label("MI Weight")

sm2 = cm.ScalarMappable(norm=norm_attn, cmap=cmap)
sm2.set_array([])
fig.colorbar(sm2, ax=ax[1], fraction=0.03, pad=0.02).set_label("Attention Weight")

plt.tight_layout()
plt.show()
#%%
#%%
from matplotlib import cm
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
random.seed(0)
import dgl
from matplotlib.colors import Normalize
from dgl.data import CoraGraphDataset, CornellDataset

#%%
sns.set_context("talk",font_scale=3.2)
f_size = 28

# -------------------------------
# Load base graph
# -------------------------------
data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()
edges = list(zip(src, dst))

num_nodes = g.num_nodes()

# -------------------------------
# Load attention weights
# -------------------------------
with open('attenedge_GAT_cora.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()

# -------------------------------
# Load MI graph
# -------------------------------
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# ALIGN MI weights to original edges
# -------------------------------
edge_to_mi = {(u, v): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)}
edge_to_mi.update({(v, u): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)})

mi_on_g = np.array([edge_to_mi.get((u, v), 0.0) for (u, v) in edges])

# -------------------------------
# SIMPLE ROW NORMALIZATION (each row sums to 1)
# -------------------------------
row_sum = np.zeros(num_nodes)

for u, w in zip(src, mi_on_g):
    row_sum[u] += w

row_sum[row_sum == 0] = 1.0

mi_on_g = np.array([
    w / row_sum[u] for u, w in zip(src, mi_on_g)
])

attn_on_g = attn_weights.copy()

# -------------------------------
# Build graph
# -------------------------------
nx_g = nx.Graph()
nx_g.add_nodes_from(range(num_nodes))
nx_g.add_edges_from(edges)

# -------------------------------
# Layout (cluster-aware)
# -------------------------------
nx_layout = nx.Graph()
nx_layout.add_nodes_from(nx_g.nodes())

for u, v in nx_g.edges():
    if labels[u] == labels[v]:
        w = 5.0
    else:
        w = 0.05
    nx_layout.add_edge(u, v, weight=w)

pos = nx.spring_layout(nx_layout, weight='weight', seed=42, k=0.3, iterations=200)

# -------------------------------
# Color setup
# -------------------------------
cmap = plt.cm.YlGnBu

norm_mi = Normalize(vmin=mi_on_g.min(), vmax=mi_on_g.max())
norm_attn = Normalize(vmin=attn_on_g.min(), vmax=attn_on_g.max())

# -------------------------------
# Plot
# -------------------------------
#%%
fig, ax = plt.subplots(1, 2, figsize=(40, 18))

# ===============================
# (1) MI normalized graph
# ===============================
plt.sca(ax[0])

edges_mi = list(zip(src, dst, mi_on_g))
edges_sorted = sorted(edges_mi, key=lambda x: x[2])

for u, v, w in edges_sorted:
    if w <= 0:
        continue

    w_scaled = (w - mi_on_g.min()) / (mi_on_g.max() - mi_on_g.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_scaled**2),
        alpha=0.1 + 0.9 * w_scaled,
        color=cmap(norm_mi(w)),
        zorder=1 + w_scaled
    )

nx.draw_networkx_nodes(
    nx_g, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[0].set_title("Learned Edge Weights (Row-normalized)")
ax[0].axis("off")

# ===============================
# (2) Attention graph
# ===============================
plt.sca(ax[1])

edges_attn = list(zip(src, dst, attn_on_g))
edges_sorted = sorted(edges_attn, key=lambda x: x[2])

for u, v, w in edges_sorted:
    w_scaled = (w - attn_on_g.min()) / (attn_on_g.max() - attn_on_g.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_scaled**2),
        alpha=0.1 + 0.9 * w_scaled,
        color=cmap(norm_attn(w)),
        zorder=1 + w_scaled
    )

nx.draw_networkx_nodes(
    nx_g, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[1].set_title("GAT Attention")
ax[1].axis("off")

# -------------------------------
# Colorbars (separate)
# -------------------------------
sm1 = cm.ScalarMappable(norm=norm_mi, cmap=cmap)
sm1.set_array([])
fig.colorbar(sm1, ax=ax[0], fraction=0.03, pad=0.02)

sm2 = cm.ScalarMappable(norm=norm_attn, cmap=cmap)
sm2.set_array([])
fig.colorbar(sm2, ax=ax[1], fraction=0.03, pad=0.02)

plt.tight_layout()
plt.savefig("adj_weights_cora.pdf",bbox_inches = "tight")
plt.show()
#%%
import numpy as np
from scipy.stats import spearmanr

def spearman_matrix(A, B):
    """
    Compute Spearman correlation between two nxn matrices (global).
    """
    assert A.shape == B.shape, "Matrices must have same shape"
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    corr, _ = spearmanr(A_flat, B_flat)
    return corr

data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()
edges = list(zip(src, dst))

# -------------------------------
# Load attention weights
# -------------------------------
with open('attenedge_GAT_cora.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()

# -------------------------------
# Load MI graph
# -------------------------------
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# ALIGN MI weights to original edges
# -------------------------------
edge_to_mi = {(u, v): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)}
edge_to_mi.update({(v, u): w for (u, v), w in zip(zip(src_adj, dst_adj), mi_weights)})

mi_on_g = []
attn_on_g = []

for (u, v), a in zip(edges, attn_weights):
    mi_on_g.append(edge_to_mi.get((u, v), 0.0))  # 0 if missing
    attn_on_g.append(a)

mi_on_g = np.array(mi_on_g)
attn_on_g = np.array(attn_on_g)

spearman_matrix(mi_on_g,attn_on_g)

# %%
#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from dgl.data import CoraGraphDataset

sns.set_context("talk", font_scale=3.5)

# -------------------------------
# Load base graph (for nodes + layout)
# -------------------------------
data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()
num_nodes = g.num_nodes()

# -------------------------------
# Load GAT attention weights
# -------------------------------
with open('attenedge_GAT_cora.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()

src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()

# -------------------------------
# Row-normalize attention (safe)
# -------------------------------
row_sum_attn = np.zeros(num_nodes)
for u, w in zip(src, attn_weights):
    row_sum_attn[u] += w
row_sum_attn[row_sum_attn == 0] = 1.0

attn_norm = np.array([
    w / row_sum_attn[u] for u, w in zip(src, attn_weights)
])

# -------------------------------
# Load MI graph (SEPARATE GRAPH)
# -------------------------------
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_mi, dst_mi = g_adj.edges()
src_mi = src_mi.cpu().numpy()
dst_mi = dst_mi.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# Row-normalize MI weights (independent)
# -------------------------------
row_sum_mi = np.zeros(num_nodes)
for u, w in zip(src_mi, mi_weights):
    row_sum_mi[u] += w
row_sum_mi[row_sum_mi == 0] = 1.0

mi_norm = np.array([
    w / row_sum_mi[u] for u, w in zip(src_mi, mi_weights)
])

# -------------------------------
# Build graphs
# -------------------------------
nx_g_attn = nx.Graph()
nx_g_attn.add_nodes_from(range(num_nodes))
nx_g_attn.add_edges_from(zip(src, dst))

nx_g_mi = nx.Graph()
nx_g_mi.add_nodes_from(range(num_nodes))
nx_g_mi.add_edges_from(zip(src_mi, dst_mi))

# -------------------------------
# Shared layout (based on base graph)
# -------------------------------
nx_layout = nx.Graph()
nx_layout.add_nodes_from(range(num_nodes))

for u, v in zip(src, dst):
    if labels[u] == labels[v]:
        w = 5.0
    else:
        w = 0.05
    nx_layout.add_edge(u, v, weight=w)

pos = nx.spring_layout(nx_layout, weight='weight', seed=42, k=0.3, iterations=200)

# -------------------------------
# Color setup
# -------------------------------
cmap = plt.cm.YlGnBu
norm_mi = Normalize(vmin=mi_norm.min(), vmax=mi_norm.max())
norm_attn = Normalize(vmin=attn_norm.min(), vmax=attn_norm.max())

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(40, 18))

# ===============================
# (1) MI graph
# ===============================
plt.sca(ax[0])

edges_mi = list(zip(src_mi, dst_mi, mi_norm))
edges_mi_sorted = sorted(edges_mi, key=lambda x: x[2])

for u, v, w in edges_mi_sorted:
    if w <= 0:
        continue

    w_scaled = (w - mi_norm.min()) / (mi_norm.max() - mi_norm.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_scaled**2),
        alpha=0.1 + 0.9 * w_scaled,
        color=cmap(norm_mi(w)),
        zorder=1 + w_scaled
    )

nx.draw_networkx_nodes(
    nx_g_mi, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[0].set_title("Learned Edge Weights (Row-normalized)")
ax[0].axis("off")

# ===============================
# (2) Attention graph
# ===============================
plt.sca(ax[1])

edges_attn = list(zip(src, dst, attn_norm))
edges_attn_sorted = sorted(edges_attn, key=lambda x: x[2])

for u, v, w in edges_attn_sorted:
    w_scaled = (w - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8)

    plt.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.3 + 3 * (w_scaled**2),
        alpha=0.1 + 0.9 * w_scaled,
        color=cmap(norm_attn(w)),
        zorder=1 + w_scaled
    )

nx.draw_networkx_nodes(
    nx_g_attn, pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=40
)

ax[1].set_title("GAT Attention")
ax[1].axis("off")

# -------------------------------
# Colorbars
# -------------------------------
sm1 = cm.ScalarMappable(norm=norm_mi, cmap=cmap)
sm1.set_array([])
fig.colorbar(sm1, ax=ax[0], fraction=0.03, pad=0.02)

sm2 = cm.ScalarMappable(norm=norm_attn, cmap=cmap)
sm2.set_array([])
fig.colorbar(sm2, ax=ax[1], fraction=0.03, pad=0.02)

plt.tight_layout()
plt.savefig("mi_vs_attention_cora2.pdf", bbox_inches="tight")
plt.show()

# %%
