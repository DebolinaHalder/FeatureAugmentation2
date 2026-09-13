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
from matplotlib.ticker import FormatStrFormatter
import dgl
import networkx as nx
from sim_graph import get_graph
from matplotlib.colors import Normalize, LinearSegmentedColormap
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, AmazonCoBuyPhotoDataset, CornellDataset,TexasDataset
# %%

def get_color_labels(labels):
    color = []
    for i in range(len(labels)):
        if labels[i] == 0:
            color.append('#C44E52')
        elif labels[i] == 1:
            color.append('#5AAE61')
    return color


def get_color_sens(sens):
    color = []
    for i in range(len(sens)):
        if sens[i] == 1:
            color.append("blue")
        elif sens[i] == 0:
            color.append("magenta")
    return color
#%%
import matplotlib.pyplot as plt
import pickle

#%%
dataset = 'sim'
method = 'partition'
sens_attr = "s"
sizes = [600, 400]
probs = [[0.35, 0.005], [0.005, 0.35]]
nb_classes = 'binary'
seed = 0
label_number = 300
p = 0.7
sens_number = 0.4
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g = get_graph(sizes, probs, nb_classes,method,seed,label_number,sens_number,p)


import dgl
from utils import feature_norm
g = dgl.DGLGraph()
g=dgl.from_scipy(adj)



# ======================
# LOAD FEATURE SET 1 (existing)
# ======================
with open('our_fea_sim_fgnn.pickle', 'rb') as handle:
    f1_fgnn = pickle.load(handle) * 1e3

with open('our_fea_sim_fvgnn.pickle', 'rb') as handle:
    f1_fvgnn = pickle.load(handle)

with open('our_fea_sim_nifti.pickle', 'rb') as handle:
    f1_nifti = pickle.load(handle)

# ======================
# LOAD FEATURE SET 2 (new)
# ======================
with open('our_both_sim_fairgnn.pickle', 'rb') as handle:
    f2_fgnn = pickle.load(handle)

with open('our_both_sim_fairvgnn.pickle', 'rb') as handle:
    f2_fvgnn = pickle.load(handle) * 1e4

with open('our_both_sim_nifti.pickle', 'rb') as handle:
    f2_nifti = pickle.load(handle)

# ======================
# ORGANIZE DATA
# ======================
methods = [
    ("FairGNN", f1_fgnn, f2_fgnn),
    ("FairVGNN", f1_fvgnn, f2_fvgnn),
    ("NIFTY", f1_nifti, f2_nifti),
]

# ======================
# CREATE FIGURE (4 x 4)
# ======================
sns.set_context("talk",font_scale=2)
fig, axes = plt.subplots(3, 4, figsize=(30, 18))

for i, (name, f1, f2) in enumerate(methods):

    # -------- LEFT SIDE (Feature) --------
    axes[i, 0].scatter(
        f1[:, 0], f1[:, 1],
        c=get_color_labels(labels),
        s=150, alpha=0.7
    )

    axes[i, 1].scatter(
        f1[:, 0], f1[:, 1],
        c=get_color_sens(sens),
        s=150, alpha=0.7
    )

    # -------- RIGHT SIDE (New / Both) --------
    axes[i, 2].scatter(
        f2[:, 0], f2[:, 1],
        c=get_color_labels(labels),
        s=150, alpha=0.7
    )

    axes[i, 3].scatter(
        f2[:, 0], f2[:, 1],
        c=get_color_sens(sens),
        s=150, alpha=0.7
    )

    # Row label (method name)
    axes[i, 0].set_ylabel(name)
    for j in range(4):
        axes[i, j].spines['top'].set_visible(False)
        axes[i, j].spines['right'].set_visible(False)

# ======================
# COLUMN TITLES (only once)
# ======================
axes[0, 0].set_title("GCN (Feat.)")
axes[0, 1].set_title("GCN (Feat.)")
axes[0, 2].set_title("GCN (Both)")
axes[0, 3].set_title("GCN (Both)")

# ======================
# CLEANUP (remove ticks)
# ======================
for row in axes:
    for ax in row:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', labelsize=38)
        ax.set_xlabel("")
        ax.set_ylabel(ax.get_ylabel())  # keep only left labels

fig.supxlabel("Learned feature 1")
fig.supylabel("Learned feature 2")

# ======================
# LABEL LEGEND
# ======================
from matplotlib.lines import Line2D
label_legend = [
    Line2D([0], [0], marker='o', color='w',
           label='Positive',
           markerfacecolor='green',
           markersize=30),

    Line2D([0], [0], marker='o', color='w',
           label='Negative',
           markerfacecolor='red',
           markersize=30),
]

# ======================
# SENSITIVE LEGEND
# ======================

sens_legend = [
    Line2D([0], [0], marker='o', color='w',
           label='Male',
           markerfacecolor='blue',
           markersize=30),

    Line2D([0], [0], marker='o', color='w',
           label='Female',
           markerfacecolor='magenta',
           markersize=30),
]
fig.legend(
    handles=label_legend,
    loc='upper center',
    bbox_to_anchor=(0.33, 1.05),
    ncol=2,
    fontsize = 40,
    frameon=False
)

# Second legend (sensitive)
fig.legend(
    handles=sens_legend,
    loc='upper center',
    bbox_to_anchor=(0.73, 1.05),
    ncol=2,
    fontsize = 40,
    frameon=False
)
plt.tight_layout()
plt.savefig("feature_fairness.pdf", dpi=300, bbox_inches='tight')
plt.show()
# %%
