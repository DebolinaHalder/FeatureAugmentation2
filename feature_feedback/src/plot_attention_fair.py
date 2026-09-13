#%%
from matplotlib import cm
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import dgl
from matplotlib.colors import Normalize
from utils2 import load_german
from sim_graph import get_graph
from matplotlib.lines import Line2D

#%%
sns.set_context("talk", font_scale=1.5)

# -------------------------------
# Load base graph
# -------------------------------
np.random.seed(10)
dataset = 'sim'
method = 'partition'
sens_attr = "s"
sizes = [600, 400]
probs = [[0.35, 0.005], [0.005, 0.35]]
nb_classes = 'binary'
seed = 10
label_number = 300
p = 0.7
sens_number = 0.4

adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, g = get_graph(
    sizes, probs, nb_classes, method, seed, label_number, sens_number, p
)

g = dgl.from_scipy(adj)

# -------------------------------
# Extract original graph edges
# -------------------------------
src, dst = g.edges()
src = src.cpu().numpy()
dst = dst.cpu().numpy()
edges = list(zip(src, dst))
num_nodes = g.num_nodes()

# -------------------------------
# Sample nodes
# -------------------------------

sampled_nodes = np.random.choice(num_nodes, size=40, replace=False)
sampled_nodes = set(sampled_nodes)

edges_sampled = [(u, v) for (u, v) in edges if u in sampled_nodes and v in sampled_nodes]

# -------------------------------
# Load learned graph (IMPORTANT)
# -------------------------------
with open('adj_german_a_gat.pickle', 'rb') as handle:
    g_adj = pickle.load(handle)

src_adj, dst_adj = g_adj.edges()
src_adj = src_adj.cpu().numpy()
dst_adj = dst_adj.cpu().numpy()

mi_weights = g_adj.edata['feat'].detach().cpu().numpy().squeeze()

# -------------------------------
# Use learned edges directly
# -------------------------------
edges_learned = [
    (u, v, w)
    for u, v, w in zip(src_adj, dst_adj, mi_weights)
    if u in sampled_nodes and v in sampled_nodes
]

# -------------------------------
# Build graphs
# -------------------------------
nx_g = nx.Graph()
nx_g.add_nodes_from(sampled_nodes)
nx_g.add_edges_from(edges_sampled)

nx_g_learned = nx.Graph()
for u, v, w in edges_learned:
    nx_g_learned.add_edge(u, v, w=w)

# -------------------------------
# Layout (same for both)
# -------------------------------
pos = nx.spring_layout(nx_g, seed=42)

# -------------------------------
# Colors and groups
# -------------------------------
labels_np = labels.cpu().numpy() if hasattr(labels, "cpu") else np.array(labels)
sens_np = sens.cpu().numpy() if hasattr(sens, "cpu") else np.array(sens)

label_colors = np.where(labels_np == 0, '#C44E52', '#55A868')

group0 = [n for n in sampled_nodes if sens_np[n] == 0]
group1 = [n for n in sampled_nodes if sens_np[n] == 1]

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# ===============================
# (1) Learned Graph
# ===============================
plt.sca(ax[1])

edges_plot = list(nx_g_learned.edges(data=True))
weights = np.array([d["w"] for (_, _, d) in edges_plot])

# robust normalization (fix visibility)
low, high = np.percentile(weights, [5, 95]) if len(weights) > 0 else (0, 1)
weights_clip = np.clip(weights, low, high)

norm = Normalize(vmin=low, vmax=high)
cmap = cm.YlGnBu

# draw edges
for (u, v, d), w in zip(edges_plot, weights_clip):
    w_scaled = (w - low) / (high - low + 1e-8)

    ax[1].plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=0.2 + 1 * w_scaled,
        color=cmap(norm(w)),
        alpha=0.2 + 0.8 * w_scaled,
        zorder=1 + w_scaled
    )

# nodes
nx.draw_networkx_nodes(
    nx_g, pos,
    nodelist=group0,
    node_color=[label_colors[n] for n in group0],
    node_shape='o',
    node_size=300,
    ax=ax[0]
)

nx.draw_networkx_nodes(
    nx_g, pos,
    nodelist=group1,
    node_color=[label_colors[n] for n in group1],
    node_shape='D',
    node_size=300,
    ax=ax[0]
)

ax[1].set_title("Learned Edge Weights")
ax[1].axis("off")

# ===============================
# (2) Original Graph
# ===============================
plt.sca(ax[0])

for u, v in edges_sampled:
    ax[0].plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=1,
        alpha=0.3,
        color='gray'
    )

nx.draw_networkx_nodes(
    nx_g, pos,
    nodelist=group0,
    node_color=[label_colors[n] for n in group0],
    node_shape='o',
    node_size=300,
    ax=ax[1]
)

nx.draw_networkx_nodes(
    nx_g, pos,
    nodelist=group1,
    node_color=[label_colors[n] for n in group1],
    node_shape='D',
    node_size=300,
    ax=ax[1]
)

ax[0].set_title("Original Graph")
ax[0].axis("off")

# -------------------------------
# Colorbar
# -------------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax[1], fraction=0.03, pad=0.02)

# -------------------------------
# Legend
# -------------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Negative',
           markerfacecolor="#C44E52", markeredgecolor='black', markersize=25),
    Line2D([0], [0], marker='o', color='w', label='Positive',
           markerfacecolor="#55A868", markeredgecolor='black', markersize=25),
    Line2D([0], [0], marker='o', color='black', label='Female',
           markerfacecolor='gray', linestyle='None', markersize=25),
    Line2D([0], [0], marker='D', color='black', label='Male',
           markerfacecolor='gray', linestyle='None', markersize=20),
]

fig.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.savefig("learned_vs_original_graph_german.pdf", bbox_inches='tight')
plt.show()
# %%
