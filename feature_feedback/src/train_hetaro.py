#%%
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import time
import argparse
import numpy as np
from sim_graph import get_graph
import torch.nn.functional as F   
import torch.optim as optim

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
#%%
class Args:
    no_cuda = True
    fastmode = False
    epochs = 500
    lr = 1e-3
    lr2 = 1e-2
    weight_decay = 0
    hidden = 18
    dropout = 0
    alpha = 10
    beta = 0
    gama = 1
    model = "GAT"
    dataset = 'sim'
    name = 'p2'
    sens_number = 0.4
    num_hidden = 18
    num_heads = 1
    num_out_heads = 1
    num_layers = 1
    residual = False
    attn_drop = 0
    negative_slope = 0.2
    acc= 0.0
    roc = 0.0
    label_number = 300
    loss = 'weight'
    tau = 0.4
    k = 1
    seed = 3
    decayRate = 1
#%%
# Training settings
args = Args()
#%%
# Parameters
num_nodes = 900
num_high_degree = 100

# Create directed graph
G = nx.DiGraph()

# Define node labels
nodes = [f"Account_{i}" for i in range(num_nodes)]
high_degree_nodes = random.sample(nodes, num_high_degree)
low_degree_nodes = list(set(nodes) - set(high_degree_nodes))

# Add all nodes
G.add_nodes_from(nodes)

# --- Step 1: Make graph weakly connected (via a simple chain) ---
random.shuffle(nodes)
for i in range(len(nodes) - 1):
    G.add_edge(nodes[i], nodes[i + 1])

# --- Step 2: Add more edges for high-degree nodes ---
for node in high_degree_nodes:
    # More outgoing edges
    targets = random.choices(low_degree_nodes, k=random.randint(15, 20))
    for target in targets:
        if target != node and target not in high_degree_nodes:
            G.add_edge(node, target)

    # More incoming edges
    

# --- Step 3: Optionally add a few low-degree edges ---
for node in high_degree_nodes:
    if random.random() < 0.5:
        target = random.choice(high_degree_nodes)
        if target != node:
            G.add_edge(node, target)

# Visualization
node_labels = np.array([1 if node in high_degree_nodes else 0 for node in G.nodes()])
nx.set_node_attributes(G, node_labels, 'y')
correlated_features = 3
non_correlated_features = 0
total_samples = len(node_labels)
alpha = np.concatenate((np.ones(correlated_features),np.zeros(non_correlated_features)))
X = np.zeros((total_samples, correlated_features+non_correlated_features))
for i in range(correlated_features+non_correlated_features):
    np.random.seed(args.seed)
    X[:,i] =  alpha[i]*node_labels + np.random.normal(loc = 0 , scale = 0.4, size = total_samples)
adj = nx.adjacency_matrix(G)
features = torch.FloatTensor(X)
labels = torch.LongTensor(node_labels)
import random
random.seed(args.seed)
label_idx = np.where(labels>=0)[0]
random.shuffle(label_idx)

idx_train = label_idx[:min(int(0.5 * len(label_idx)),args.label_number)]
idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
idx_test = label_idx[int(0.75 * len(label_idx)):]



idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
g = dgl.DGLGraph()
g.from_scipy_sparse_matrix(adj)

# %%
seeds = [args.seed]
accu = []
roc = []
f1 = []
dp = {}
eq = {}
#%%
loss_base = []
acc_val_base = []
acc_test_base = []
#print(args.sens_number)
#args.model = "GCN"
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
        model.optimize(g,features,labels,idx_train)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        model.eval()
        rep,output = model(g, features)
        #output_org = torch.argmax(output, dim=1)
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        output_org = (output.squeeze()>0).type_as(labels)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val].detach().cpu().numpy())
        validation_f1.append(roc_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        roc_test = average_precision_score(labels[idx_test].cpu().numpy(),output_org[idx_test].detach().cpu().numpy())
        
        if best_accu <= acc_val:
        
            
            best_accu = acc_val

            best_result['acc'] = acc_test.item()
            best_result['avpr'] = roc_test
            best_result['f1'] = f1_test
            #best_result['parity'] = parity
            #best_result['equality'] = equality
            best_output = output_org
            best_rep = rep
                

        print("=================================")

        print('Epoch: {:04d}'.format(epoch+1),
            
            'cls: {:.4f}'.format(cls_loss.item()),
            
            'acc_val: {:.4f}'.format(acc_val.item()),
            "avpr_val: {:.4f}".format(roc_val))
        print("Test:",
                "accuracy: {:.4f}".format(acc_test.item()),
                "avpr: {:.4f}".format(roc_test))
        loss_base.append(cls_loss.item())
        acc_val_base.append(acc_val.item())
        acc_test_base.append(acc_test.item())
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    roc.append(best_result['avpr'])
    f1.append(best_result['f1'])
    with open('output_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('rep_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result
# %%
#file = open("rep_gat_cora.pickle",'rb')
#rep = pickle.load(file)
loss_f = []
acc_val_f = []
acc_test_f = []
rep = torch.rand(features.shape[0],args.hidden)
#args.model = "GCN"
torch.autograd.set_detect_anomaly(True)
new_feature = torch.cat((features,(torch.rand((len(features),2)))), 1)
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Feature_generator(nfeat = new_feature.shape[1],rep = rep,x=new_feature, args = args)
    
    
    t_total = time.time()
    best_result = {}
    best_accu = 0
    training_loss = []    
    validation_error = []
    validation_f1 = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(g,labels,idx_train)
        
        cls_loss = model.G_loss
        
        training_loss.append(cls_loss.item())
        model.eval()
        rep,output = model(g, new_feature)
        #output_org = torch.argmax(output, dim=1)
        #out_ = out.squeeze()
        #output = (out_.squeeze()>0).type_as(labels)
        output_org = (output.squeeze()>0).type_as(labels)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val].detach().cpu().numpy())
        validation_f1.append(roc_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        roc_test = average_precision_score(labels[idx_test].cpu().numpy(),output_org[idx_test].detach().cpu().numpy())
             
        
        
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
        loss_f.append(cls_loss.item())
        acc_val_f.append(acc_val.item())
        acc_test_f.append(acc_test.item())
        
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    roc.append(best_result['roc'])
    f1.append(best_result['f1'])
    with open('our_output_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('our_rep_gcn_seed0.pickle', 'wb') as handle:
        pickle.dump(best_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result

# %%
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
#plt.savefig("seed3_hetaro.pdf")



#%%
import matplotlib.pyplot as plt
color = []
for i in range(len(labels)):
    if labels[i] == 0:
        color.append('r')
    else:
        color.append('g')
fig, ax = plt.subplots()
ax.bar(np.arange(len(labels)),model.x[:,-2].detach().numpy(),color =color)
#plt.savefig("seed3_hetaro_feature.pdf")
# %%
hetarophili graph