#%%
import time
import argparse
import numpy as np
from sim_graph import get_graph
import torch
import torch.nn.functional as F   
import torch.optim as optim

from utils import load_data, accuracy,load_pokec,feature_norm
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
torch.set_default_dtype(torch.float64)

#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    cuda = True
    fastmode = False
    epochs = 1000       
    lr = 1e-3
    lr2 = 1e-2
    lr3 = 1e-2
    weight_decay = 1e-3
    hidden = 10
    num_hidden = 10
    dropout = 0
    alpha = 1e-3
    beta = 1e-3
    gamma = 1e-3
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
    label_number = 400
    loss = 'weight'
    tau = 0.4
    k = 1
    seed = 5
    decayRate = 0.6
    new_features = 1
    num_classes = 2
    temperature = 6
    num_steps = 1
    graph_skip_conn = 0.2
    graph_beta = 0
    alpha_new = 0.1
    delta = 0.5
    inner_steps = 1
    edge_threshold = 0
#%%
# Training settings
args = Args()



#%%
dataset = 'sim'
method = 'partition'
sens_attr = "s"
sizes = [600, 400]
probs = [[0.35, 0.005], [0.005, 0.35]]
nb_classes = 'binary'
seed = args.seed
label_number = 300
p = 0.7
sens_number = args.sens_number
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g = get_graph(sizes, probs, nb_classes,method,seed,label_number,sens_number,p)

#%%
dataset = 'german'
sens_attr = "Gender"
predict_attr = 'GoodCustomer'
label_number = 400
sens_number = args.sens_number
seed = args.seed
path = '../dataset/german/'
adj, features, labels,labels_onehot, idx_train, idx_val, idx_test,sens,idx_sens_train = load_german(dataset,
                                                                                sens_attr,
                                                                                predict_attr,
                                                                                path=path,
                                                                                label_number=label_number,
                                                                                sens_number=sens_number,
                                                                                seed=seed)

#%%
dataset = 'nba'
sens_attr = "country"
predict_attr = "SALARY"
label_number = 100
sens_number = 50
seed = 20
seed = 1
path = "../dataset/NBA"
test_idx = True
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                sens_attr,
                                                                                predict_attr,
                                                                                path=path,
                                                                                label_number=label_number,
                                                                                sens_number=sens_number,
                                                                                seed=seed,test_idx=test_idx)
features = feature_norm(features)

import dgl
from utils import feature_norm
g = dgl.DGLGraph()
g=dgl.from_scipy(adj)
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
# %%
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
args.num_classes = 1
result_df = pd.DataFrame(columns=['p','dp','dp_std','eq','eq_std','ndkl','ndkl_std'])
seeds = [args.seed]



#for i in p:
accu = []
roc = []
f1 = []
dp = {}
eq = {}


#args.sens_number = i



# %%
file_output = open("output_simedge_seed0.pickle",'rb')
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
new_feature = torch.cat((features,torch.rand((len(features),args.new_features))), 1)
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Feature_generator(nfeat = features.shape[1],x=features.double(),adj=torch.from_numpy(adj.toarray()).float().to(device), rep = rep,args = args)  
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
        model.optimize(torch.from_numpy(adj.toarray()).float().to(device),torch.from_numpy(adj.toarray()),features.double(),labels.unsqueeze(1).float(),idx_train,end_epoch,epoch,teacher_output,teacher_rep)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        kdloss_f.append(model.dist_loss.item())
        model.eval()
        output, weights = model(torch.from_numpy(adj.toarray()).float().to(device),features.double())
        output_org = (output.squeeze()>0).type_as(labels).cpu().numpy()
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        #output_org = (output.squeeze()>0).type_as(labels)
        f1_val = f1_score(labels[idx_val].cpu().numpy(),output_org[idx_val])
        acc_val = accuracy(output[idx_val],labels[idx_val])
        validation_f1.append(f1_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
        acc_test = acc_val = accuracy(output[idx_test],labels[idx_test])
        parity,equality = fair_metric(output,idx_test)
        
        
        if best_accu <= acc_val:
            best_accu = acc_val

            best_result['acc'] = acc_test.item()
            best_result['f1'] = f1_test
            best_result['parity'] = parity
            best_result['equality'] = equality
            best_output = output_org
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
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('our_rep_gcn_seed0.pickle', 'wb') as handle:
    #    pickle.dump(best_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result

#%%
with open('adj_german_a_gat.pickle', 'wb') as handle:
   pickle.dump(best_g, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
import seaborn as sns
from matplotlib.lines import Line2D
orange = 'orange'
blue = 'blue'
red = '#C44E52'
green = '#5AAE61'
sns.set_context("talk")
import matplotlib.pyplot as plt
x = best_x
color = []
for i in range(len(labels)):
    if labels[i] == 0:
        color.append(red)
    elif labels[i] == 1:
        color.append(green)
    elif labels[i] == 2:
        color.append('b')
    elif labels[i] == 3:
        color.append('c')
    elif labels[i] == 4:
        color.append('m')
    elif labels[i] == 5:
        color.append('k')
    else:
        color.append('y')
fig, ax = plt.subplots()
ax.scatter((x[:,-1].detach().numpy()), (x[:,-2].detach().numpy()),c =np.array(color),s = 30, alpha = 0.8)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='label = 0',
           markerfacecolor='red', markersize=10,alpha = 0.8),
    Line2D([0], [0], marker='o', color='w', label='label = 1',
           markerfacecolor='green', markersize=10 ,alpha = 0.8),
]
ax.legend(handles=legend_elements, frameon=False, fontsize = 15)


#%%
import seaborn as sns
from matplotlib.lines import Line2D
orange = 'orange'
blue = 'blue'
red = '#C44E52'
green = '#5AAE61'
sns.set_context("talk")
import matplotlib.pyplot as plt
x = best_x
color = []
for i in range(len(labels)):
    if sens[i] == 0:
        color.append(orange)
    elif sens[i] == 1:
        color.append(blue)
fig, ax = plt.subplots()
ax.scatter((x[:,-1].detach().numpy()), (x[:,-2].detach().numpy()),c =np.array(color),s = 30, alpha = 0.8)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='label = 0',
           markerfacecolor='red', markersize=10,alpha = 0.8),
    Line2D([0], [0], marker='o', color='w', label='label = 1',
           markerfacecolor='green', markersize=10 ,alpha = 0.8),
]
ax.legend(handles=legend_elements, frameon=False, fontsize = 15)
#%%
x = best_x
with open('our_both_sim_fairgnn.pickle', 'wb') as handle:
    pickle.dump(x[:,-2:], handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
sns.set_context("talk")
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
src, dst = g.edges()
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
#plt.savefig("GAT_attentions.pdf")


#%%
import seaborn as sns
sns.set_context("talk")
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx

random.seed(5)
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
sens_to_shape = {
    0: 'o',   # circle
    1: 'v',   # square
}
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
        node_shape=sens_to_shape[s_attr.item()],
        node_size=200,
        edgecolors="black",
        ax=axes
    )

nx.draw_networkx_edges(
    subG,
    pos,
    alpha=0.4,
    width=2,
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

    # Sensitive attribute legend (shape)
    Line2D([0], [0], marker='o', color='black',
           label='Sens 0',
           linestyle='None',
           markerfacecolor='lightgray',
           markersize=15),

    Line2D([0], [0], marker='v', color='black',
           label='Sens 1',
           linestyle='None',
           markerfacecolor='lightgray',
           markersize=15),
]

fig.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.75, 0.5),   # center of figure
    ncol=1, 
    fontsize = fontsize,                     # vertical stack
    frameon=False
)

plt.tight_layout()
plt.savefig("original_graph.pdf")
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
        node_shape=sens_to_shape[s_attr.item()],
        node_size=200,
        edgecolors="black",
        ax=axes[0]
    )

nx.draw_networkx_edges(
    subG,
    pos,
    alpha=0.4,
    width=2,
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
        node_shape=sens_to_shape[s_attr.item()],
        edgecolors="black",
        node_size=200,
        ax=axes[1]
    )

# Edge widths proportional to weight
edges = subG_t.edges(data=True)
weights = [d["w"] for (_, _, d) in edges]

# normalize for visualization
max_w = max(weights) if len(weights) > 0 else 1
scaled_widths = [2 * (w / max_w) for w in weights]
from matplotlib import cm
from matplotlib.colors import Normalize

# Edge weights
edges = list(subG_t.edges(data=True))
weights = [d["w"] for (_, _, d) in edges]

# Normalize weights for color mapping
norm = Normalize(vmin=min(weights), vmax=max(weights))
cmap = cm.YlGnBu

# Scale widths (keep your logic)
max_w = max(weights) if len(weights) > 0 else 1
scaled_widths = [2 * (w / max_w) for w in weights]

# Draw edges with colormap
for (u, v, d), width in zip(edges, scaled_widths):
    w = d["w"]
    axes[1].plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        linewidth=width,
        color=cmap(norm(w)),
        alpha=0.8,
        zorder=1
    )

# Add colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

axes[1].axis("off")

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

    # Sensitive attribute legend (shape)
    Line2D([0], [0], marker='o', color='black',
           label='Sens 0',
           linestyle='None',
           markerfacecolor='lightgray',
           markersize=15),

    Line2D([0], [0], marker='v', color='black',
           label='Sens 1',
           linestyle='None',
           markerfacecolor='lightgray',
           markersize=15),
]

fig.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.35, 0.75),   # center of figure
    ncol=1, 
    fontsize = fontsize,                     # vertical stack
    frameon=False
)

plt.tight_layout()
#plt.savefig("updated_edge_fairness.pdf")
plt.show()

#%%
import statistics
data1 =[0.9457142857142857,0.9628571428571429,0.9657142857142857,0.9657142857142857]
data2 = [0.9396825396825397,0.9570957095709571,0.96,0.9605263157894737]
dp = [0.4,0.3134796238244514,0.2732567651582082,0.4270598608797884]
eq = [0.04666130329847151,0.016659975913287828,0.008284241531664183,0.023504273504273532]
print(statistics.mean(eq),statistics.stdev(eq))

#%%
import statistics
data1 =[0.9514285714285714,0.9514285714285714,0.8857142857142857,0.9485714285714286]
data2 = [0.9453376205787781,0.9453376205787781,0.8019801980198019,0.9268292682926829]
dp = [0.2908829676071056,0.2908829676071056,0.25008980764834593,0.35540576262225754]
eq = [0.004817342432757887,0.004817342432757887,0.05982905982905984,0.03141711229946531]
print(statistics.mean(eq),statistics.stdev(eq))
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
from matplotlib.lines import Line2D
orange = 'orange'
blue = 'blue'
red = '#C44E52'
green = '#5AAE61'
sns.set_context("talk")
import matplotlib.pyplot as plt
x = model.x_t
color = []
for i in range(len(labels)):
    if labels[i] == 0:
        color.append(red)
    elif labels[i] == 1:
        color.append(green)
    elif labels[i] == 2:
        color.append('b')
    elif labels[i] == 3:
        color.append('c')
    elif labels[i] == 4:
        color.append('m')
    elif labels[i] == 5:
        color.append('k')
    else:
        color.append('y')
fig, ax = plt.subplots()
ax.scatter((x[:,-1].detach().numpy()), (x[:,-2].detach().numpy()),c =np.array(color),s = 30, alpha = 0.8)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='label = 0',
           markerfacecolor='red', markersize=10,alpha = 0.8),
    Line2D([0], [0], marker='o', color='w', label='label = 1',
           markerfacecolor='green', markersize=10 ,alpha = 0.8),
]
ax.legend(handles=legend_elements, frameon=False, fontsize = 15)
#np.zeros_like
#ax.bar(np.arange(len(labels)),model.x_t[:,-1].detach().numpy(),color =color)
plt.savefig("features_fair_labels.pdf")
     # %%
file = open("output_gat_cora.pickle",'rb')
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
0.951429, 0.945338, 0.290883, 0.004817

{'acc': 0.94,
 'f1': 0.9329073482428115,
 'parity': 0.29254709254709255,
 'equality': 0.0038503850385038785}

0.954286, 0.93985, 0.306908, 0.001515

{'acc': 0.9542857142857143,
 'f1': 0.9398496240601504,
 'parity': 0.3069078947368421,
 'equality': 0.0015151515151515804}

0.962857, 0.958199, 0.348571, 0.028854
{'acc': 0.9628571428571429,
 'f1': 0.9581993569131833,
 'parity': 0.34857142857142853,
 'equality': 0.028853754940711407}

0.954286	0.944056	0.326471 0.048073

{'acc': 0.9571428571428572,
 'f1': 0.9473684210526315,
 'parity': 0.3205882352941176,
 'equality': 0.048072562358276616}





{'acc': 0.9371428571428572,
 'f1': 0.9285714285714285,
 'parity': 0.4242335002126894,
 'equality': 0.016933638443935917}

array([0.42325825, 0.42325825]), array([0.02991453, 0.02991453])
