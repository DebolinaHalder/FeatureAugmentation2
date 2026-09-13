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
from models.featuregeneratormlp import Feature_generator
from models.basemlp import Basemlp
from dgl.data import TexasDataset, CoraGraphDataset, CiteseerGraphDataset, CornellDataset, AmazonCoBuyPhotoDataset

#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    cuda = True
    fastmode = False
    epochs = 150      
    lr = 1e-2
    lr2 = 1e-2
    lr3 = 1e-3
    weight_decay = 0
    hidden = 18
    num_hidden = 18
    dropout = 0
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
    k = 1
    seed = 4
    decayRate = 0.6
    new_features = 2
    num_classes = 2
    temperature = 2
    num_steps = 1
    graph_skip_conn = 0
    graph_beta = 0.4
    alpha_new = 0.1
    delta = 0.5
    topk = 60

#%%
# Training settings
args = Args()

# %%
#data = citegrh.load_citeseer()
from dgl.data import TexasDataset
data = CornellDataset()
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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
        model.optimize(g,features,labels,idx_train)
        
        cls_loss = model.cls_loss
        
        training_loss.append(cls_loss.item())
        model.eval()  
        rep,output = model(g, features)
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

#print(args.sens_number)

#features = torch.cat((features,labels.view(-1,1)), 1)
for j in seeds:
    args.seed = j
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    


    #file = open("best_output_nba.pickle",'rb')
    #labels = pickle.load(file)
    
    # Model and optimizer
    
    model = Basemlp(nfeat = features.shape[1], args = args)
    
    
        
    
    t_total = time.time()
    best_result = {}
    best_accu = 0
    training_loss = []
    validation_error = []
    validation_f1 = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        model.optimize(features,labels,idx_train)
        
        cls_loss = model.G_loss
        
        training_loss.append(cls_loss.item())
        model.eval()  
        rep,output = model(features)
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
        
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    accu.append(best_result['acc'])
    f1.append(best_result['f1'])
    
best_result




# %%
file_output = open("output_t_texas.pickle",'rb')
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
    
    model = Feature_generator(nfeat = new_feature.shape[1],x=new_feature, rep = rep,args = args)
    
    
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
        model.optimize(g,torch.from_numpy(adj.toarray()),new_feature,labels,idx_train,end_epoch,epoch,teacher_output,teacher_rep)
        
        cls_loss = model.G_loss
        
        training_loss.append(cls_loss.item())
        kdloss_f.append(model.dist_loss.item())
        model.eval()
        output = model(g,new_feature)
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
            best_output = output_org
            best_x = model.x_t
            
            

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
#%%
with open('features_cornell_mlp_gt.pickle', 'wb') as handle:
   pickle.dump(x[:,-2:], handle, protocol=pickle.HIGHEST_PROTOCOL)
#np.zeros_like
#ax.bar(np.arange(len(labels)),model.x_t[:,-1].detach().numpy(),color =color)
#plt.savefig("features_GAT_cornell_labels.pdf")
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
