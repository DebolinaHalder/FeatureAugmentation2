#%%
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt
import pickle
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
sns.set_context("talk")
fig, ax = plt.subplots(3,5,figsize=(20,10))
data_cora  = CoraGraphDataset()
g_cora = data_cora[0]
labels_cora = g_cora.ndata["label"]
colors_cora = get_color(labels_cora)

with open('features_cora_f_gt.pickle', 'rb') as handle:
    features1 = pickle.load(handle)
with open('features_cora_b_gt.pickle', 'rb') as handle:
    features2 = pickle.load(handle)
with open('features_cora_mlp_gt.pickle', 'rb') as handle:
    features3 = pickle.load(handle)
ax[0][0].scatter(features1[:,0],features1[:,1],c=colors_cora,s=10)
ax[1][0].scatter(features2[:,0],features2[:,1],c=colors_cora,s=10)
ax[2][0].scatter(features3[:,0],features3[:,1],c=colors_cora,s=10)


data_citeseer  = CiteseerGraphDataset()
g_citeseer = data_citeseer[0]
labels_citeseer = g_citeseer.ndata["label"]
colors_citeseer = get_color(labels_citeseer)

with open('features_citeseer_f_gt.pickle', 'rb') as handle:
    features1 = pickle.load(handle)
with open('features_citeseer_b_gt.pickle', 'rb') as handle:
    features2 = pickle.load(handle)
with open('features_citeseer_mlp_gt.pickle', 'rb') as handle:
    features3 = pickle.load(handle)
ax[0][1].scatter(features1[:,0],features1[:,1],c=colors_citeseer,s=10)
ax[1][1].scatter(features2[:,0],features2[:,1],c=colors_citeseer,s=10)
ax[2][1].scatter(features3[:,0],features3[:,1],c=colors_citeseer,s=10)


data_amazon  = AmazonCoBuyPhotoDataset()
g_amazon = data_amazon[0]
labels_amazon = g_amazon.ndata["label"]
colors_amazon = get_color(labels_amazon)

with open('features_photo_f_gt.pickle', 'rb') as handle:
    features1 = pickle.load(handle)
with open('features_photo_b_gt.pickle', 'rb') as handle:
    features2 = pickle.load(handle)
with open('features_photo_mlp_gt.pickle', 'rb') as handle:
    features3 = pickle.load(handle)
ax[0][2].scatter(features1[:,0],features1[:,1],c=colors_amazon,s=10)
ax[1][2].scatter(features2[:,0],features2[:,1],c=colors_amazon,s=10)
ax[2][2].scatter(features3[:,0],features3[:,1],c=colors_amazon,s=10)

data_cornell  = CornellDataset()
g_cornell = data_cornell[0]
labels_cornell = g_cornell.ndata["label"]
colors_cornell = get_color(labels_cornell)

with open('features_cornell_f_gt.pickle', 'rb') as handle:
    features1 = pickle.load(handle)
with open('features_cornell_b_gt.pickle', 'rb') as handle:
    features2 = pickle.load(handle)
with open('features_cornell_mlp_gt.pickle', 'rb') as handle:
    features3 = pickle.load(handle)
ax[0][3].scatter(features1[:,0],features1[:,1],c=colors_cornell,s=10)
ax[1][3].scatter(features2[:,0],features2[:,1],c=colors_cornell,s=10)
ax[2][3].scatter(features3[:,0],features3[:,1],c=colors_cornell,s=10)


data_texas  = TexasDataset()
g_texas = data_texas[0]
labels_texas = g_texas.ndata["label"]
colors_texas = get_color(labels_texas)

with open('features_texas_f_gt.pickle', 'rb') as handle:
    features1 = pickle.load(handle)
with open('features_texas_b_gt.pickle', 'rb') as handle:
    features2 = pickle.load(handle)
with open('features_texas_mlp_gt.pickle', 'rb') as handle:
    features3 = pickle.load(handle)
ax[0][4].scatter(features1[:,0],features1[:,1],c=colors_texas,s=10)
ax[1][4].scatter(features2[:,0],features2[:,1],c=colors_texas,s=10)
ax[2][4].scatter(features3[:,0],features3[:,1],c=colors_texas,s=10)

for a in ax.flat:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
fsize = 30
ax[0][0].set_title("Cora",fontsize = fsize)
ax[0][1].set_title("Citeseer",fontsize = fsize)
ax[0][2].set_title("Photo",fontsize = fsize)
ax[0][3].set_title("Cornell",fontsize = fsize)
ax[0][4].set_title("Texas",fontsize = fsize)

ax[0][0].set_ylabel("GCN (feat.)",fontsize = fsize)
ax[1][0].set_ylabel("GCN (Both)",fontsize = fsize)
ax[2][0].set_ylabel("MLP (Feat.)",fontsize = fsize)

plt.tight_layout()
fig.supxlabel("Learned Feature 1", fontsize=fsize, y = -0.02)
fig.supylabel("Learned Feature 2", fontsize=fsize, x = -0.02)

plt.savefig("feature_all_gt.pdf", bbox_inches='tight')
# %%
