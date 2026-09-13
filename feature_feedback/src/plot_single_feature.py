#%%
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt
import pickle
import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, AmazonCoBuyPhotoDataset, CornellDataset,TexasDataset
# %%
def get_color(labels):
    color = []
    for i in range(len(labels)):
        if labels[i] == 0:
            color.append('#C44E52')
        elif labels[i] == 1:
            color.append('#5AAE61')
        elif labels[i] == 2:
            color.append('b')
        elif labels[i] == 3:
            color.append('c')
        elif labels[i] == 4:
            color.append('m')
        elif labels[i] == 5:
            color.append('lime')
        elif labels[i] == 6:
            color.append('indigo')
        else:
            color.append('y')
    return color
#%%

data_cora  = CoraGraphDataset()
g_cora = data_cora[0]
labels_cora = g_cora.ndata["label"]
colors_cora = get_color(labels_cora)
idxs = [i for i in range(len(labels_cora))]
with open('features_cora_f_gat.pickle', 'rb') as handle:
    features1 = pickle.load(handle)

with open('output_gat_seed0.pickle', 'rb') as handle:
    gcn_out = pickle.load(handle).detach()

with open('output_gcn_seed0.pickle', 'rb') as handle:
    gat_out = pickle.load(handle).detach()

with open('our_output_gcn_seed0.pickle', 'rb') as handle:
    our_out = pickle.load(handle).detach()

#%%
pred_gcn = gcn_out.argmax(1).numpy()
pred_our = our_out.numpy()
pred_gat = gat_out.argmax(1).numpy()

mask = (pred_gat[idxs] == pred_our[idxs]) & (pred_gat[idxs] != pred_gcn[idxs])
idx_val = torch.tensor(idxs)

indices = idx_val[mask].numpy()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk",font_scale=0.6)
# convert indices to list if tensor
if hasattr(indices, "tolist"):
    indices = indices.tolist()

# all points
plt.scatter(features1[:,0],features1[:,1],c=colors_cora,s=30)

# highlighted points
plt.scatter(features1[indices,0], features1[indices,1],
            facecolors='none', edgecolors='black',  # circle outline
            s=40, linewidths=1.5, label=r'GCN $\neq$ GAT, GCN(Feat.) $=$ GAT')

plt.legend(frameon=False,loc='upper left',bbox_to_anchor=(0, 1.1))
plt.xlabel("Learned Feature 1")
plt.ylabel("Learned Feature 2")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("single_feature_cora.pdf",bbox_inches ="tight")
plt.show()

# %%
