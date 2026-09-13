#%%
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy,load_pokec
from utils2 import load_german,load_credit, load_bail
from models.worst_cross_ldam import BFtS
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sim_graph import get_graph
from sklearn.preprocessing import OneHotEncoder
import pickle
#%%
criterion = nn.BCEWithLogitsLoss()
class Args:
    no_cuda = True
    fastmode = False
    epochs = 100
    lr = 1e-2
    lr2 = 1e-3
    weight_decay = 1e-2
    hidden = 18
    dropout = 0.3
    alpha = 0
    beta = 1
    gama = 10
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
#%%
# Training settings
args = Args()
name = "sim_bfts_18.csv"
#%%
cuda = not args.no_cuda and torch.cuda.is_available()
#%%
def get_weights(y,train_idx):
    num_classes, count = np.unique(y[train_idx].cpu().numpy(), return_counts= True)
    weights = np.zeros(len(num_classes))
    total = np.sum(count)
    for i in range(len(num_classes)):
        weights[i] = total/count[i]
    return torch.FloatTensor(weights),count
#%%
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,average_precision_score

#%%
# Train model
result_df = pd.DataFrame(columns=['sens_number','accuracy','acc_std','avpr','avpr_std','f1','f1_std','dp','dp_std','eq','eq_std'])
seeds = np.arange(1)
sens_number = [1]
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
            idx_sens_train = np.load('german.pkl', allow_pickle=True)
            idx_sens_train = torch.LongTensor(idx_sens_train)
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
            idx_sens_train = np.load('bail.pkl', allow_pickle=True)
            idx_sens_train = torch.LongTensor(idx_sens_train)
        elif args.dataset == 'sim':
            dataset = 'sim'
            method = 'partition'
            sens_attr = "s"
            sizes = [600, 400]
            probs = [[0.08, 0.008], [0.008, 0.08]]
            nb_classes = 'binary'
            seed = args.seed
            label_number = 300
            sens_number = args.sens_number
            p = 0.7
            adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g = get_graph(sizes, probs, nb_classes,method,seed,label_number,sens_number,p)

        
        elif args.dataset == "nba":
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
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
        print(len(idx_test))
        sens[sens>0]=1
        if sens_attr:
            sens[sens>0]=1
        import dgl
        from utils import feature_norm
        G = dgl.DGLGraph()
        G=dgl.from_scipy(adj)
        idx_sens_train = torch.argsort(G.in_degrees(),descending = True)[:int(len(features)*i)]
        


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
        
        # Model and optimizer
        weights,count = get_weights(labels,idx_train)
        weights2, count2 = get_weights(sens, idx_sens_train)
        weights_sens, count_sens = get_weights(sens,idx_sens_train)
        model = BFtS(nfeat = features.shape[1], args = args,weights=weights,class_num_list=count,loss=args.loss,weights_sens = weights_sens, count_sens = count_sens)

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
        best_roc = 0
        training_loss = []
        validation_error = []
        validation_f1 = []
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            model.optimize(G,features,labels,idx_train,sens,idx_sens_train)
            cov = 0
            cls_loss = model.cls_loss
            adv_loss = model.A_loss
            model.eval()
            training_loss.append(cls_loss.item())
            out,rep = model(G, features)
            #out_ = out.squeeze()
            
            output_org = (out.squeeze()>0).type_as(labels)
            #output_org = (out.squeeze()>0).type_as(labels)
            acc_val = accuracy(out[idx_val], labels[idx_val])
            f1_val = f1_score(labels[idx_val].cpu().numpy(),output_org[idx_val])
            roc_val = average_precision_score(labels[idx_val].cpu().numpy(),out[idx_val].detach().cpu().numpy())
            #validation_error.append(criterion(output_org[idx_val],labels[idx_val].unsqueeze(1).float()).item())
            #roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),out[idx_val].detach().cpu().numpy())
            validation_f1.append(roc_val.item())
            #acc_sens = accuracy(s[idx_test], sens[idx_test])
            
            parity_val, equality_val = fair_metric(output_org,idx_val)
            f1_test = f1_score(labels[idx_test].cpu().numpy(),output_org[idx_test])
            acc_test = accuracy(out[idx_test], labels[idx_test])
            roc_test = average_precision_score(labels[idx_test].cpu().numpy(),out[idx_test].detach().cpu().numpy())
            parity,equality = fair_metric(output_org,idx_test)
            if acc_val > args.acc or roc_val > args.roc:
                
                if best_roc <= f1_val:
                    best_roc = f1_val
                    best_result['acc'] = acc_test.item()
                    best_result['roc'] = roc_test
                    best_result['f1'] = f1_test
                    best_result['parity'] = parity
                    best_result['equality'] = equality
                    best_output = output_org
                    best_epoch = epoch
                    best_rep = rep
                if epoch % 20 == 0: 
                    print("=================================")
                    print('Epoch: {:04d}'.format(epoch+1),
                        'cov: {:.4f}'.format(0),
                        'cls: {:.4f}'.format(cls_loss.item()),
                        'adv: {:.4f}'.format(adv_loss.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        "roc_val: {:.4f}".format(roc_val),
                        "parity_val: {:.4f}".format(parity_val),
                        "equality: {:.4f}".format(equality_val))
                    print("Test:",
                            "accuracy: {:.4f}".format(acc_test.item()),
                            "roc: {:.4f}".format(roc_test),
                            #"acc_sens: {:.4f}".format(acc_sens),
                            "parity: {:.4f}".format(parity),
                            "equality: {:.4f}".format(equality))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        accu.append(best_result['acc'])
        roc.append(best_result['roc'])
        f1.append(best_result['f1'])
        dp.append(best_result['parity'])
        eq.append(best_result['equality'])
        with open('bfts_bail_rep.pkl', 'wb') as file:
            pickle.dump(best_rep, file)
        with open('bfts_bail_out.pkl', 'wb') as file:
            pickle.dump(best_output, file)
    result_df.loc[len(result_df)] = {'sens_number':i,'accuracy':np.mean(accu),'acc_std':np.var(accu),'avpr':np.mean(roc),'avpr_std':np.var(roc),'f1':np.mean(f1),'f1_std':np.var(f1),'dp':np.mean(dp),'dp_std':np.var(dp),'eq':np.mean(eq),'eq_std':np.var(eq)}


result_df

#%%
from scipy.stats import pearsonr
sens_pred = model.GNN_sens(G,features)
sens_pred = torch.argmax(sens_pred, dim=1).view(-1,1).float()
pearsonr(labels.numpy(), sens_pred.numpy().reshape(-1))
#%%
pearsonr(labels.numpy(), (sens.numpy()))
#%%
result_df.to_csv(name)
  # %%
year= [,2010,2011]
number = [574,568,679,682,778,857,1005,1102,1091,1355,1347,1389,1586,1620,1622,1696,1961	2360	3211	3861	5084	6443	8859	12400	9103]