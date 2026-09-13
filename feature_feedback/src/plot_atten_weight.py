#%%
from matplotlib import cm
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import pickle
random.seed(0)
import dgl
import networkx as nx
from matplotlib.colors import Normalize, LinearSegmentedColormap
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, AmazonCoBuyPhotoDataset, CornellDataset,TexasDataset
#%%
f_size = 30
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

data = CoraGraphDataset()
g = data[0]

labels = g.ndata["label"].cpu().numpy()

# -------------------------------
# Original graph edges
# -------------------------------
src_g, dst_g = g.edges()
src_g = src_g.cpu().numpy()
dst_g = dst_g.cpu().numpy()

nx_g = nx.Graph()
nx_g.add_nodes_from(range(g.num_nodes()))
edge_list_g = list(zip(src_g, dst_g))
nx_g.add_edges_from(edge_list_g)

# -------------------------------
# Attention graph edges + weights
# -------------------------------
weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

# threshold
thresh = 0.2
mask = weights > thresh

# keep only strong edges
weights = weights[mask]
src_adj = src_adj[mask]
dst_adj = dst_adj[mask]

nx_g_adj = nx.Graph()
nx_g_adj.add_nodes_from(range(g.num_nodes()))
edge_list_adj = list(zip(src_adj, dst_adj))
nx_g_adj.add_edges_from(edge_list_adj)

# -------------------------------
# Better width scaling
# -------------------------------
max_w = max(weights)
scaled_widths = [
    1*w/max_w for w in weights
]

# -------------------------------
# Custom yellow -> black cmap
# -------------------------------
cmap = plt.cm.YlGnBu

norm = Normalize(vmin=weights.min(), vmax=weights.max())
edge_colors = cmap(norm(weights))

# -------------------------------
# Use same layout for both plots
# -------------------------------
# use original graph for layout
nx_layout = nx.Graph()
nx_layout.add_nodes_from(nx_g.nodes())

for u, v in nx_g.edges():
    if labels[u] == labels[v]:
        w = 5.0   # strong pull inside cluster
    else:
        w = 0.03   # weak pull between clusters
    nx_layout.add_edge(u, v, weight=w)

# layout using weights
pos = nx.spring_layout(
    nx_g, seed = 42
)

# -------------------------------
# Plot side by side
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(50, 20))

# -------------------------------
# Left: original graph
# -------------------------------
plt.sca(ax[0])

nx.draw_networkx_edges(
    nx_g,
    pos,
    edgelist=edge_list_g,
    width=3,
    edge_color="gray",
    alpha=0.9
)

nx.draw_networkx_nodes(
    nx_g,
    pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=50
)

ax[0].set_title("Original Graph", fontsize=f_size)
ax[0].axis("off")

# -------------------------------
# Right: attention graph
# -------------------------------
#%%
plt.sca(ax[1])

# sort edges by weight (low → high)
edges_with_attr = list(zip(src_adj, dst_adj, weights))
edges_sorted = sorted(edges_with_attr, key=lambda x: x[2])

for u, v, w in edges_sorted:
    nx.draw_networkx_edges(
        nx_g_adj,
        pos,
        edgelist=[(u, v)],
        width=1 * w / max_w,
        edge_color=[cmap(norm(w))],
        alpha=0.85,
    )

nx.draw_networkx_nodes(
    nx_g_adj,
    pos,
    node_color=labels,
    cmap=plt.cm.tab10,
    node_size=20
)

ax[1].set_title("Attention-Weighted Graph", fontsize=f_size)
ax[1].axis("off")

# -------------------------------
# Colorbar
# -------------------------------
plt.tight_layout(rect=[0, 0, 0.92, 1])

# dedicated colorbar axis [left, bottom, width, height]
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig.colorbar(sm, cax=cbar_ax, label="Edge Attention Weight")

#plt.savefig("cornell_attention_graph2.pdf", dpi=300)
plt.show()
# %%
#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import dgl
from dgl.data import CoraGraphDataset

# -------------------------------
# Load data
# -------------------------------
data = CoraGraphDataset()
g = data[0]

# original edges (for attention)
src_g, dst_g = g.edges()
src_g = src_g.cpu().numpy()
dst_g = dst_g.cpu().numpy()

edges_g = list(zip(src_g, dst_g))

# -------------------------------
# Load attention weights (E,)
# -------------------------------
with open('attenedge_GAT_cora.pickle', 'rb') as handle:
    attn_weights = np.array(pickle.load(handle)).squeeze()

# sanity
assert len(attn_weights) == len(edges_g)

# -------------------------------
# Load MI graph
# -------------------------------
with open('adj_cora_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

edges_adj = list(zip(src_adj, dst_adj))

# -------------------------------
# ALIGN edges (IMPORTANT)
# -------------------------------
# map (u,v) → MI weight
edge_to_mi = {(u, v): w for (u, v), w in zip(edges_adj, mi_weights)}

# if graph is undirected, also add reverse
edge_to_mi.update({(v, u): w for (u, v), w in zip(edges_adj, mi_weights)})

mi_list = []
attn_list = []

for (u, v), attn in zip(edges_g, attn_weights):
    if (u, v) in edge_to_mi:
        mi_list.append(edge_to_mi[(u, v)])
        attn_list.append(attn)

mi_list = np.array(mi_list)
attn_list = np.array(attn_list)

# -------------------------------
# Normalize (important for viz)
# -------------------------------
mi_norm = (mi_list - mi_list.min()) / (mi_list.max() - mi_list.min() + 1e-8)
attn_norm = (attn_list - attn_list.min()) / (attn_list.max() - attn_list.min() + 1e-8)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    mi_norm,
    attn_norm,
    s=10,
    alpha=0.4
)

plt.xlabel("MI Weight (normalized)")
plt.ylabel("GAT Attention (normalized)")
plt.title("Attention vs MI Weight")

# optional: diagonal reference
plt.plot([0,1],[0,1], linestyle='--')

plt.grid(alpha=0.3)
plt.show()
#%%
