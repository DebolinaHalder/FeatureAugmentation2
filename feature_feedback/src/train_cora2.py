#%%
import time
import argparse
import numpy as np
from sim_graph import get_graph
import torch
import torch.nn.functional as F   
import torch.optim as optim
from utils import load_data, accuracy,load_pokec, feature_norm
from utils import load_data, accuracy,load_pokec
from models.baselina import Baseline
import pandas as pd
from utils2 import load_german,load_credit,load_bail
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import dgl
from sim_graph import get_graph
import random
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score, average_precision_score
import networkx as nx
from dgl.data import citation_graph as citegrh
from models.feature_generator import Feature_generator
from dgl.data import CornellDataset, TexasDataset, CoraGraphDataset, CiteseerGraphDataset,AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset


#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    cuda = True
    fastmode = False
    epochs = 150   
    lr = 1e-2
    lr2 = 1e-2
    lr3 = 1e-2
    weight_decay = 0
    hidden = 18
    num_hidden = 18
    dropout = 0.1
    alpha = 1e-1
    beta = 0
    gamma = 0
    dist_alpha = 1
    model = "GCN"
    dataset = 'sim'
    name = 'p2'
    sens_number = 0.4
    
    num_heads = 1
    num_out_heads = 1
    num_layers = 1
    residual = False
    attn_drop = 0
    negative_slope = 0.2
    acc= 0.0
    roc = 0.0
    label_number = 120
    loss = 'weight'
    tau = 0.4
    k = 4
    seed = 5
    decayRate = 0.6
    new_features = 1
    
    num_classes = 0
    temperature = 4
    num_steps = 1
    graph_skip_conn = 0.02
    alpha_new = 0.4
    graph_beta = 0.8
    delta = 0.5
    topk = 60
    inner_steps = 5
    edge_threshold = 0.2

#%%
# Training settings
args = Args()

# %%
#data = citegrh.load_cora()
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, CornellDataset, TexasDataset
data  = CoraGraphDataset()
g = data[0]

#%%
features, labels = g.ndata["feat"], g.ndata["label"]
adj = g.adj_external(scipy_fmt='csr')

features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(labels)

# nodes with valid labels
label_idx = np.where(labels >= 0)[0]

# shuffle
perm = np.random.permutation(label_idx)

n = len(perm)

train_ratio = 0.3
val_ratio = 0.2
test_ratio = 0.2

n_train = int(train_ratio * n)
n_val = int(val_ratio * n)

idx_train = torch.LongTensor(perm[:n_train])
idx_val = torch.LongTensor(perm[n_train:n_train + n_val])
idx_test = torch.LongTensor(perm[n_train + n_val:])
#labels = (labels > 0).type_as(labels)

#%%
dataset = 'sim'
method = 'partition'
sens_attr = "s"
sizes = [600, 400]
probs = [[0.8, 0.5], [0.5, 0.8]]
nb_classes = 'binary'
seed = args.seed     
label_number = 200
p = 0.7
sens_number = args.sens_number

adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g = get_graph(sizes, probs, nb_classes,method,seed,label_number,sens_number,p)
g = dgl.DGLGraph()
g=dgl.from_scipy(adj)


# %%
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
args.num_classes = len(np.unique(labels))
result_df = pd.DataFrame(columns=['p','dp','dp_std','eq','eq_std','ndkl','ndkl_std'])
seeds = [args.seed]




#for i in p:
accu = []
roc = []
f1 = []
dp = {}
eq = {}


#args.sens_number = i

#%%
loss_base = []
acc_val_base = []
acc_test_base = []
#print(args.sens_number)
args.model = "GAT"
#features = torch.cat((features,labels.view(-1,1)), 1)
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Baseline(nfeat = features.shape[1], args = args)
    
    
        
    
    t_total = time.time()
    best_result = {}
    best_accu = 0
    training_loss = []
    validation_error = []
    validation_f1 = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(torch.from_numpy(adj.toarray()).float().to(device),features,labels,idx_train)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        model.eval()  
        rep,output = model(torch.from_numpy(adj.toarray()).float().to(device), features)
        output_org = torch.argmax(output, dim=1)
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        #output_org = (output.squeeze()>0).type_as(labels)
        f1_val = f1_score(labels[idx_val].cpu().numpy(),output_org[idx_val],average='weighted')
        acc_val = ((output[idx_val].argmax(1) == labels[idx_val]).type(torch.float).sum())/len(idx_val)
        #roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val].detach().cpu().numpy())
        #validation_f1.append(roc_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test],average='weighted')
        acc_test = ((output[idx_test].argmax(1) == labels[idx_test]).type(torch.float).sum())/len(idx_test)
        #roc_test = average_precision_score(labels[idx_test].cpu().numpy(),output_org[idx_test].detach().cpu().numpy())
        
        if best_accu <= acc_val:
        
            
            best_accu = acc_val

            best_result['acc'] = acc_test.item()
            #best_result['roc'] = roc_test
            best_result['f1'] = f1_test
            #best_result['parity'] = parity
            #best_result['equality'] = equality
            best_output = output
            best_rep = rep
                

        print("=================================")

        print('Epoch: {:04d}'.format(epoch+1),
            
            'cls: {:.4f}'.format(cls_loss.item()),
            
            'acc_val: {:.4f}'.format(acc_val.item()),
            "f1_val: {:.4f}".format(f1_val))
        print("Test:",
                "accuracy: {:.4f}".format(acc_test.item()),
                "f1: {:.4f}".format(f1_test))
        loss_base.append(cls_loss.item())
        acc_val_base.append(acc_val.item())
        acc_test_base.append(acc_test.item())
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    f1.append(best_result['f1'])
    with open('output_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('rep_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result

#%%
src, dst = g.edges()
attn = model.GNN.attentions[1].squeeze().detach().cpu().numpy()
with open('attenedge_GAT_cornell.pickle', 'wb') as handle:
   pickle.dump(attn, handle, protocol=pickle.HIGHEST_PROTOCOL)  # shape: (E,)
A = np.zeros((g.num_nodes(), g.num_nodes()))
for i in range(len(src)):
    A[src[i], dst[i]] = attn[i]
with open('atten_GAT_cora.pickle', 'wb') as handle:
   pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
import seaborn as sns
sns.set_context("talk")
src, dst = g.edges()
attn = model.GNN.attentions[0].squeeze().detach().cpu().numpy()  # shape: (E,)
A = np.zeros((g.num_nodes(), g.num_nodes()))
for i in range(len(src)):
    A[src[i], dst[i]] = attn[i]

# Sort by label
sort_idx = torch.argsort(labels).cpu().numpy()
A_sorted = A[sort_idx, :][:, sort_idx]

plt.figure(figsize=(8, 6))
sns.heatmap(A_sorted, cmap='Oranges', xticklabels=False, yticklabels=False)
#plt.title("GAT Attention Matrix (Sorted by Class Labels)")
#plt.savefig("GAT_attentions.pdf")

# %%
file_output = open("output_gcn_seed0.pickle",'rb')
file_rep = open("rep_gcn_seed0.pickle",'rb')
#rep = pickle.load(file)
torch.set_flush_denormal(True)
loss_f = []
kdloss_f = []
acc_val_f = []
acc_test_f = []
rep = torch.rand(features.shape[0],args.num_hidden)
args.model = "GCN"
teacher_output = pickle.load(file_output)
teacher_rep = pickle.load(file_rep)
torch.autograd.set_detect_anomaly(True)
#new_feature = torch.cat((features,torch.rand((len(features),args.new_features))), 1)
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Feature_generator(nfeat = features.shape[1],x=features,adj=torch.from_numpy(adj.toarray()).float().to(device), rep = rep,args = args)
    
    
    t_total = time.time()
    best_result = {}
    best_accu = 0
    training_loss = []    
    validation_error = []
    validation_f1 = []
    end_epoch = 150
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(torch.from_numpy(adj.toarray()).float().to(device),torch.from_numpy(adj.toarray()),features,labels,idx_train,end_epoch,epoch,teacher_output,teacher_rep)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        kdloss_f.append(model.dist_loss.item())
        model.eval()
        output, weights = model(torch.from_numpy(adj.toarray()).float().to(device),features)
        output_org = torch.argmax(output, dim=1)
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        #output_org = (output.squeeze()>0).type_as(labels)
        f1_val = f1_score(labels[idx_val].cpu().numpy(),output_org[idx_val],average='weighted')
        acc_val = ((output[idx_val].argmax(1) == labels[idx_val]).type(torch.float).sum())/len(idx_val)
        validation_f1.append(f1_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test],average='weighted')
        acc_test = ((output[idx_test].argmax(1) == labels[idx_test]).type(torch.float).sum())/len(idx_test)
        
        
        
        if best_accu <= acc_val :
            best_accu = acc_val

            best_result['acc'] = acc_test.item()
            best_result['f1'] = f1_test
            #best_result['parity'] = parity
            #best_result['equality'] = equality
            best_output_m2d = output_org
            best_weight = weights
            best_x = model.x_t
            best_adj = model.adj_t
            
            

        print("=================================")

        print('Epoch: {:04d}'.format(epoch+1),
            
            'cls: {:.4f}'.format(cls_loss.item()),
            
            'acc_val: {:.4f}'.format(acc_val.item()),
            "roc_val: {:.4f}".format(f1_val))
        print("Test:",
                "accuracy: {:.4f}".format(acc_test.item()),
                "roc: {:.4f}".format(f1_test))
        loss_f.append(cls_loss.item())
        acc_val_f.append(acc_val.item())
        acc_test_f.append(acc_test.item())
        
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    f1.append(best_result['f1'])
    with open('our_output_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_output_m2d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('our_rep_gcn_seed0.pickle', 'wb') as handle:
    #    pickle.dump(best_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result

#%%
with open('adj_cornell_a_gat.pickle', 'wb') as handle:
   pickle.dump(best_g, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt
x = best_x
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
        color.append('k')
    elif labels[i] == 6:
        color.append('indigo')
    else:
        color.append('y')
fig, ax = plt.subplots()
ax.scatter(x[:,-1].detach().numpy(), (x[:,-2].detach().numpy()),c =np.array(color),s = 20, alpha = 0.7)
''' 
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Label = 0',
           markerfacecolor='r', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='label = 1',
           markerfacecolor='g', markersize=15),
]
ax.legend(handles=legend_elements, frameon=False, fontsize = 15)
'''

#np.zeros_like
#ax.bar(np.arange(len(labels)),model.x_t[:,-1].detach().numpy(),color =color)
#plt.savefig("features_GAT_cora_labels.pdf")

#%%
with open('features_cornell_a_gt.pickle', 'wb') as handle:
   pickle.dump(x[:,-2:], handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
def fair_metric(output,idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2,figsize = (20,5))
x_index = np.arange(args.epochs)
ax[0].plot(x_index,loss_f,color ='red')

ax[1].plot(x_index,kdloss_f,color ='red')

ax[1].legend()
ax[0].set_title("train loss")
ax[1].set_title("Dist loss")

#%%
import seaborn as sns
sns.set_context("talk")
src, dst = model.g_t.edges()
attn = best_weight  # shape: (E,)
A = np.zeros((g.num_nodes(), g.num_nodes()))
for i in range(len(src)):
    A[src[i], dst[i]] = attn[i]

# Sort by label
sort_idx = torch.argsort(labels).cpu().numpy()
A_sorted = A[sort_idx, :][:, sort_idx]

plt.figure(figsize=(8, 6))
sns.heatmap(A_sorted, cmap='viridis', xticklabels=False, yticklabels=False)
#plt.title("GAT Attention Matrix (Sorted by Class Labels)")
#plt.savefig("Learned_attention_cora.pdf")
#%%
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx

random.seed(0)
n = 60
random_nodes = random.sample(list(g.nodes()), n)
random_nodes = [int(t) for t in random_nodes]
g_nx_full = g.to_networkx()
subG = g_nx_full.subgraph(random_nodes)


src, dst = model.g_t.edges()
edge_weights = best_weight
g_nx = nx.Graph()

for u, v, w in zip(src.tolist(), dst.tolist(), edge_weights):
    if w >= 0:
        g_nx.add_edge(u, v, w=w)

subG_t = g_nx.subgraph(random_nodes).copy()

node_colors = [
    "tab:red" if labels[n] == 0 else "tab:green"
    for n in random_nodes
]

from collections import defaultdict
fontsize = 25

groups = defaultdict(list)
for n in random_nodes:
    groups[(labels[n], sens[n])].append(n)



pos = nx.spring_layout(subG, seed=5)
sns.set_context("talk")
fig, axes = plt.subplots(1, 1, figsize=(7, 7))

# -------- LEFT: Original Graph --------
#axes.set_title("Original Graph", fontsize = fontsize)

for (label, s_attr), nodes in groups.items():
    nx.draw_networkx_nodes(
        subG,
        pos,
        nodelist=nodes,
        node_color=["#D55E5E" if label == 0 else "#5AAE61"],
        node_shape='o',
        node_size=200,
        edgecolors="black",
        ax=axes
    )

nx.draw_networkx_edges(
    subG,
    pos,
    alpha=0.9,
    width=1,
    arrows=False,
    edge_color = "#7A8CA3",
    ax=axes
)

axes.axis("off")

from matplotlib.lines import Line2D
soft_red = "#C44E52"
soft_green = "#55A868"
legend_elements = [
    # Label legend (color)
    Line2D([0], [0], marker='o', color='w',
           label='Label 0',
           linestyle='None',
           markerfacecolor=soft_red,
           markeredgecolor='black',
           markersize=15),

    Line2D([0], [0], marker='o', color='w',
           linestyle='None',
           label='Label 1',
           markerfacecolor=soft_green,
           markeredgecolor='black',
           markersize=15),
]

fig.legend(
    handles=legend_elements,
    loc='upper right',
    #bbox_to_anchor=(0.75, 0.5),   # center of figure
    ncol=1, 
    fontsize = fontsize,                     # vertical stack
    frameon=False
)

plt.tight_layout()
plt.savefig("original_graph_gat.pdf")
#%%
from collections import defaultdict
fontsize = 25

groups = defaultdict(list)
for n in random_nodes:
    groups[(labels[n], sens[n])].append(n)



pos = nx.spring_layout(subG, seed=5)
sns.set_context("talk")
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# -------- LEFT: Original Graph --------
axes[0].set_title("Original Graph", fontsize = fontsize)

for (label, s_attr), nodes in groups.items():
    nx.draw_networkx_nodes(
        subG,
        pos,

        nodelist=nodes,
        node_color=["#D55E5E" if label == 0 else "#5AAE61"],
        node_shape='o',
        node_size=200,
        edgecolors="black",
        ax=axes[0]
    )

nx.draw_networkx_edges(
    subG,
    pos,
    alpha=0.4,
    width=1.5,
    arrows=False,
    edge_color = "#7A8CA3",
    ax=axes[0]
)

axes[0].axis("off")


# -------- RIGHT: Weighted Graph --------
axes[1].set_title("Weighted Graph" , fontsize = fontsize)

for (label, s_attr), nodes in groups.items():
    nx.draw_networkx_nodes(
        subG_t,
        pos,
        nodelist=nodes,
        node_color=["#D55E5E" if label == 0 else "#5AAE61"],
        node_shape='o',
        edgecolors="black",
        node_size=200,
        ax=axes[1]
    )

# Edge widths proportional to weight
edges = subG_t.edges(data=True)
weights = [d["w"].item() for (_, _, d) in edges]

# normalize for visualization


# Normalize weights
from matplotlib.colors import Normalize
norm = Normalize(vmin=min(weights), vmax=max(weights))
cmap = plt.cm.viridis
max_w = max(weights) if len(weights) > 0 else 1
scaled_widths = [1 * (w / max_w) for w in weights]

nx.draw_networkx_edges(
    subG_t,
    pos,
    edge_color='#7A8CA3',
    #edge_cmap=(cmap),
    #edge_vmin=min(weights),
    #edge_vmax=max(weights),
    width=scaled_widths,
    arrows=False,
    alpha=0.6,
    ax=axes[1]
)

axes[1].axis("off")

from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
soft_red = "#C44E52"
soft_green = "#55A868"
legend_elements = [
    # Label legend (color)
    Line2D([0], [0], marker='o', color='w',
           label='Label 0',
           linestyle='None',
           markerfacecolor=soft_red,
           markeredgecolor='black',
           markersize=15),

    Line2D([0], [0], marker='o', color='w',
           linestyle='None',
           label='Label 1',
           markerfacecolor=soft_green,
           markeredgecolor='black',
           markersize=15)
]
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#divider = make_axes_locatable(axes[1])
#cax = divider.append_axes("right", size="5%", pad=0.1)
#sm = ScalarMappable(norm=norm, cmap=cmap)
#sm.set_array([])  # required for matplotlib
#fig.colorbar(sm, cax=cax)
fig.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.45, 0.85),   # center of figure
    ncol=1, 
    fontsize = fontsize,                     # vertical stack
    frameon=False
)


plt.tight_layout()
plt.savefig("updated_edge_GAT_width.pdf")
plt.show()


# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3,figsize = (20,5))
x_index = np.arange(args.epochs)
ax[0].plot(x_index,loss_base,color ='green')
#ax[0].plot(x_index,loss_base_gat,color ='blue')
ax[0].plot(x_index,loss_f,color ='red')
ax[1].plot(x_index,acc_val_base,color ='green')
#ax[1].plot(x_index,acc_val_base_gat,color ='blue')
ax[1].plot(x_index,acc_val_f,color ='red')
ax[2].plot(x_index,acc_test_base,label = "base GCN",color ='green')
#ax[2].plot(x_index,acc_test_base_gat,label = "base GAT",color ='blue')
ax[2].plot(x_index,acc_test_f,label = "ours",color ='red')
ax[2].legend()
ax[0].set_title("train loss")
ax[1].set_title("Validation accuracy")
ax[2].set_title("Test Accuracy")
#plt.savefig("seed0_GCNcllusters.pdf")



#%%
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt
x = model.x_t
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
        color.append('k')
    elif labels[i] == 6:
        color.append('orange')
    else:
        color.append('y')
fig, ax = plt.subplots()
ax.scatter(x[:,-1].detach().numpy(), (x[:,-2].detach().numpy()),c =np.array(color),s = 20, alpha = 0.7)
''' 
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Label = 0',
           markerfacecolor='r', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='label = 1',
           markerfacecolor='g', markersize=15),
]
ax.legend(handles=legend_elements, frameon=False, fontsize = 15)
'''

#np.zeros_like
#ax.bar(np.arange(len(labels)),model.x_t[:,-1].detach().numpy(),color =color)
#plt.savefig("features_GAT_cora_labels.pdf")
     # %%
file = open("output_gat_cornell.pickle",'rb')
labels = pickle.load(file)

file = open("rep_gat_cora.pickle",'rb')
rep = pickle.load(file)


# %%
args.model = "GCN"
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Baseline(nfeat = features.shape[1], args = args)
    
    
    t_total = time.time()
    best_result = {}
    best_accu = 0
    training_loss = []    
    validation_error = []
    validation_f1 = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(g,features,labels,idx_train)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        model.eval()
        rep,output = model(g, features)
        #output_org = torch.argmax(output, dim=1)
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        output_org = (output.squeeze()>0.5).type_as(labels)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val].detach().cpu().numpy())
        validation_f1.append(roc_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
        acc_test = accuracy(output_org[idx_test], labels[idx_test])
        roc_test = average_precision_score(labels[idx_test].cpu().numpy(),output_org[idx_test].detach().cpu().numpy())
        
        if acc_val > args.acc and roc_val > args.roc:
        
            if best_accu <= acc_val :
                best_accu = acc_val

                best_result['acc'] = acc_test.item()
                best_result['roc'] = roc_test
                best_result['f1'] = f1_test
                #best_result['parity'] = parity
                #best_result['equality'] = equality
                best_output = output_org
                best_rep = rep
                
                

            print("=================================")

            print('Epoch: {:04d}'.format(epoch+1),
                
                'cls: {:.4f}'.format(cls_loss.item()),
                
                'acc_val: {:.4f}'.format(acc_val.item()),
                "roc_val: {:.4f}".format(roc_val))
            print("Test:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "roc: {:.4f}".format(roc_test))
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    roc.append(best_result['roc'])
    f1.append(best_result['f1'])
    
    
best_result
# %%
color = ['red' if l == 0 else 'green' for l in labels]
G = dgl.to_networkx(g)
plt.figure(figsize=[15,7])
nx.draw_networkx(G, node_color = color)
# %%
