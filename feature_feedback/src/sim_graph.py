#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from generate_graph import *
from scipy import stats
import torch
#%%
method = 'partition'
sizes = [700, 300]
probs = [[0.15, 0.005], [0.005, 0.15]]
nb_classes = 'binary'
seed = 10
label_number = 300
# %%
def get_graph(sizes=sizes, probs=probs, number_class='multi',choice=method,seed = 10,label_number = 300,sens_number = 0.4,p = 0.5):
    g, s = get_graph_prot(sizes=sizes, probs=probs, number_class=nb_classes,
        choice=method)
    print('Assortativity coefficient on the original graph: %0.3f'% nx.attribute_assortativity_coefficient(g, 's'))
    y = np.array(list(s.values()))
    print(np.unique(y))
    prot = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == 1:
            prot[i] = np.random.binomial(1,p,1)
        else:
            prot[i] = np.random.binomial(1,1-p,1)
    nx.set_node_attributes(g, y, 'y')
    correlated_features = 2
    non_correlated_features = 1
    total_samples = len(y)
    alpha = np.concatenate((np.ones(correlated_features),np.zeros(non_correlated_features)))
    X = np.zeros((total_samples, correlated_features+non_correlated_features))
    for i in range(correlated_features+non_correlated_features):
        np.random.seed(seed)
        X[:,i] =  alpha[i]*y + np.random.normal(loc = 0 , scale = 0.4, size = total_samples)
    adj = nx.adjacency_matrix(g)
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.65 * len(label_idx))]
    idx_test = label_idx[int(0.65 * len(label_idx)):]


    sens = torch.FloatTensor(prot)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    total_sens = int(sens_number * len(idx_sens_train))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:total_sens])

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train,g
