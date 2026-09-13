#%%
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from utils import load_data, accuracy,load_pokec, feature_norm
from models.debias import Debias
import pandas as pd
from utils2 import load_german,load_credit,load_bail
import matplotlib.pyplot as plt
import torch.nn as nn
from sim_graph import get_graph
import seaborn as sns




#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    no_cuda = True
    fastmode = False
    epochs = 1000
    lr = 1e-3
    weight_decay = 1e-3
    hidden = 10
    dropout = 0
    alpha = 1e-2
    beta = 3*1e-1
    model = "GCN"
    dataset = 'nba'
    name = 'p2'
    num_hidden = 10
    num_heads = 1
    num_out_heads = 1
    num_layers = 1
    residual = False
    attn_drop = 0
    negative_slope = 0.2
    acc= 0.0
    roc = 0.0
    label_number = 400
    loss = 'ldam'
    sens_number = 1
#%%
# Training settings
args = Args()
name = "sim_debias_all.csv"

#%%
cuda = not args.no_cuda and torch.cuda.is_available()
#%%

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score, average_precision_score


#%%
def get_weights(y,train_idx):
    num_classes, count = np.unique(y[train_idx].cpu().numpy(), return_counts= True)
    weights = np.zeros(len(num_classes))
    total = np.sum(count)
    for i in range(len(num_classes)):
        weights[i] = total/count[i]
    return torch.FloatTensor(weights),count
#%%
# Train model
result_df = pd.DataFrame(columns=['sens_number','accuracy','acc_std','avpr','avpr_std','f1','f1_std','dp','dp_std','eq','eq_std'])
seeds = [0]
sens_number  = [1]

for i in sens_number:
    accu = []
    roc = []
    f1 = []
    dp = []
    eq = []
    args.sens_number = i

    print(args.sens_number)
    for j in seeds:
        args.seed = j
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cuda:
            torch.cuda.manual_seed(args.seed)

        
        
        if args.dataset == 'german':
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

        elif args.dataset == 'credit':
            dataset = 'credit'
            sens_attr = 'Age'
            predict_attr = 'NoDefaultNextMonth'
            label_number = 8000
            sens_number = args.sens_number
            seed = args.seed
            path = "../dataset/credit/"
            test_idx = False
            adj, features, labels, labels_onehot,idx_train, idx_val, idx_test,sens,idx_sens_train = load_credit(dataset,
                                                                                            sens_attr,
                                                                                            predict_attr,
                                                                                            path=path,
                                                                                            label_number=label_number,
                                                                                            sens_number=sens_number,
                                                                                            seed=seed)
        elif args.dataset == 'bail':
            dataset = 'bail'
            sens_attr = "WHITE"
            predict_attr = "RECID"
            label_number = 14000
            sens_number = args.sens_number
            seed = args.seed
            path = "../dataset/bail"
            test_idx = False
            adj, features, labels, labels_onehot,idx_train, idx_val, idx_test,sens,idx_sens_train = load_bail(dataset,
                                                                                            sens_attr,
                                                                                            predict_attr,
                                                                                            path=path,
                                                                                            label_number=label_number,
                                                                                            sens_number=sens_number,
                                                                                            seed=seed)
            
        elif args.dataset == 'sim':
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
        elif args.dataset == "nba":
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
        else:
            if args.dataset == 'pokec_z':
                dataset = 'region_job'
            else:
                dataset = 'region_job_2'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path="../dataset/pokec/"
            test_idx=False
            adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
        
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
        import dgl
        from utils import feature_norm
        G = dgl.DGLGraph()
        G=dgl.from_scipy(adj)
        


        
        
        # Model and optimizer
        weights,count = get_weights(labels,idx_train)
        model = Debias(nfeat = features.shape[1], args = args)
        #model.GNN.weight.data.uniform_(-1, 1)
        
        
        if cuda:
            model.cuda()
            features = features.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            sens = sens.cuda()
            idx_sens_train = idx_sens_train.cuda()
        t_total = time.time()
        best_result = {}
        best_accu = 0
        training_loss = []
        validation_error = []
        validation_f1 = []
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            model.optimize(torch.from_numpy(adj.toarray()).float(),features,labels,idx_train,sens,idx_sens_train)
            
            cls_loss = model.cls_loss
            adv_loss = model.A_loss
            training_loss.append(cls_loss.item())
            model.eval()
            output = model(torch.from_numpy(adj.toarray()).float(), features)
            #output_org = torch.argmax(torch.sigmoid(output), dim=1)
            #out_ = out.squeeze()
            #output = (out_.squeeze()>0).type_as(labels)
            output_org = (output.squeeze()>0).type_as(labels).cpu().numpy()
            acc_val = accuracy(output[idx_val], labels[idx_val])
            roc_val = average_precision_score(labels[idx_val].cpu().numpy(),output_org[idx_val])
            validation_f1.append(roc_val.item())
            #validation_error.append(criterion(output[idx_val],labels[idx_val].float()).item())
            
            
            parity_val, equality_val = fair_metric(output,idx_val)
            f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            roc_test = average_precision_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
            parity,equality = fair_metric(output,idx_test)
            if acc_val > args.acc and roc_val > args.roc:
            
                if best_accu <= roc_val :
                    best_accu = roc_val

                    best_result['acc'] = acc_test.item()
                    best_result['roc'] = roc_test
                    best_result['f1'] = f1_test
                    best_result['parity'] = parity
                    best_result['equality'] = equality
                    best_output = output

                print("=================================")

                print('Epoch: {:04d}'.format(epoch+1),
                    'cls: {:.4f}'.format(cls_loss.item()),
                    'adv: {:.4f}'.format(adv_loss.item()),
                    'acc_val: {:.4f}'.format(acc_val.item()),
                    "roc_val: {:.4f}".format(roc_val),
                    "parity_val: {:.4f}".format(parity_val),
                    "equality: {:.4f}".format(equality_val))
                print("Test:",
                        "accuracy: {:.4f}".format(acc_test.item()),
                        "roc: {:.4f}".format(roc_test),
                        "parity: {:.4f}".format(parity),
                        "equality: {:.4f}".format(equality))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        accu.append(best_result['acc'])
        roc.append(best_result['roc'])
        f1.append(best_result['f1'])
        dp.append(best_result['parity'])
        eq.append(best_result['equality'])
        with open('output_simedge_seed0.pickle', 'wb') as handle:
            pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #result_df = result_df.append({'sens_number':i,'accuracy':np.mean(accu),'acc_std':np.var(accu),'avpr':np.mean(roc),'avpr_std':np.var(roc),'f1':np.mean(f1),'f1_std':np.var(f1),'dp':np.mean(dp),'dp_std':np.var(dp),'eq':np.mean(eq),'eq_std':np.var(eq)}, ignore_index=True)
    result_df.loc[len(result_df)] = {'sens_number':i,'accuracy':np.mean(accu),'acc_std':np.var(accu),'avpr':np.mean(roc),'avpr_std':np.var(roc),'f1':np.mean(f1),'f1_std':np.var(f1),'dp':np.mean(dp),'dp_std':np.var(dp),'eq':np.mean(eq),'eq_std':np.var(eq)}
result_df
# %%
output_simedge_seed0.pickle
1	0.951429	0.0	0.911037	0.0	0.945338	0.0	0.290883	0.0	0.004817	0.0

[[0.35, 0.005], [0.005, 0.35]]
[[0.35, 0.25], [0.25, 0.35]]

output_simfea_seed0.pickle
[[0.35, 0.1], [0.1, 0.35]]
0	1	0.948571	0.0	0.90739	0.0	0.941935	0.0	0.296565	0.0	0.026094


modified ( 0.954285740852356, 0.9543159636062861        0.2646520146520147, 0.012962962962962954)
modifiededge(0.9599999785423279, 0.9599416077767824 0.3074529074529075, 0.0038503850385038785)
