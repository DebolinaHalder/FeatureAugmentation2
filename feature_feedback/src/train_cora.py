#%%
import time
import argparse
import numpy as np
from sim_graph import get_graph
import torch
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
import random
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score, average_precision_score
import networkx as nx
from dgl.data import citation_graph as citegrh
from models.feature_generator import Feature_generator

#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    no_cuda = True
    fastmode = False
    epochs = 300
    lr = 1e-3
    lr2 = 1e-4
    weight_decay = 0
    hidden = 18
    dropout = 0
    alpha = 1e-1
    beta = 0
    gama = 1
    model = "GCN"
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
    k = 0.4
    seed = 1
#%%
# Training settings
args = Args()
torch.manual_seed(args.seed)
# %%
 
data = citegrh.load_cora()
g = data[0]
g_org = g
features_org, labels_org = g.ndata["feat"], g.ndata["label"]
adj_org = g.adjacency_matrix_scipy()
features_org = torch.FloatTensor(np.array(features_org))
labels_org = torch.LongTensor(labels_org)
labels_org = (labels_org > 0).type_as(labels_org)
#%%
label_idx = np.where(labels_org>=0)[0]
random.shuffle(label_idx)
idx_hide = torch.LongTensor(label_idx[:20])
g.remove_nodes(torch.LongTensor(label_idx[:20]))
features, labels = g.ndata["feat"], g.ndata["label"]
adj = g.adjacency_matrix_scipy()
label_idx = np.where(labels>=0)[0]
random.shuffle(label_idx)
idx_train = label_idx[min(500,(int(0.5 * len(label_idx))))]
idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
idx_test = label_idx[int(0.75 * len(label_idx)):]

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(labels)
labels = (labels > 0).type_as(labels)
# %%
result_df = pd.DataFrame(columns=['p','dp','dp_std','eq','eq_std','ndkl','ndkl_std'])
seeds = [0]



#for i in p:
accu = []
roc = []
f1 = []
dp = {}
eq = {}


#args.sens_number = i
#%%
#print(args.sens_number)
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
        
        if best_accu <= acc_val:
        
            
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
    with open('output_gat_cora.pickle', 'wb') as handle:
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('rep_gat_cora.pickle', 'wb') as handle:
        pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
best_result
#%%
data = citegrh.load_cora()
g_org = data[0]
rep,output = model(g_org,features_org)
output_org = (output.squeeze()>0.5).type_as(labels)
print(f1_score(labels_org[idx_hide].cpu().numpy(),output_org[idx_hide]))
print(accuracy(output_org[idx_hide], labels_org[idx_hide]))
print(average_precision_score(labels_org[idx_hide].cpu().numpy(),output_org[idx_hide].detach().cpu().numpy()))
# %%
rep = torch.rand(features.shape[0],args.hidden)
seeds = [0]
args.model = "GCN"
torch.autograd.set_detect_anomaly(True)
new_feature = torch.cat((features,torch.rand((len(features),1))), 1)
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
        output_org = (output.squeeze()>0.5).type_as(labels)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val].detach().cpu().numpy())
        validation_f1.append(roc_val.item())
        #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
        
        
        
        f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
        acc_test = accuracy(output_org[idx_test], labels[idx_test])
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
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    
    
best_result

#%%
data = citegrh.load_cora()
g_org = data[0]

# %%
rep = torch.rand(features_org.shape[0],args.hidden)
model.x = torch.cat((features_org,torch.rand((len(features_org),1))), 1)
for i in range(30):
    f0 = model.adv(rep)
    model.x[:,-1] = f0.view(-1)
    rep, output = model(g_org,model.x)
output_org = (output.squeeze()>0.5).type_as(labels_org)
print(f1_score(labels_org[idx_hide].cpu().numpy(),output_org[idx_hide]))
print(accuracy(output_org[idx_hide], labels_org[idx_hide]))
print(average_precision_score(labels_org[idx_hide].cpu().numpy(),output_org[idx_hide].detach().cpu().numpy()))

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
color =  
import matplotlib.pyplot as plt

