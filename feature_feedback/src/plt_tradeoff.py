#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pickle
from utils2 import load_german,load_credit,load_bail
sns.set(font_scale=3)
from sklearn.decomposition import KernelPCA
from matplotlib.lines import Line2D
sns.set_style(style='white')


 # %%
base = pd.read_csv("sim_rank_500.csv")
nifty = pd.read_csv("nifty_sim_rank_600.csv")
fgnn = pd.read_csv("fgnn_sim_rank_500.csv")
fgnn_dp = fgnn['dp']
fgnn_dp_var = fgnn['dp_std']
fgnn_eq = fgnn['eq']
fgnn_eq_var = fgnn['eq_std']
fgnn_ndkl = fgnn['ndkl']
fgnn_bdkl_var = fgnn['ndkl_std']

nifty_dp = nifty['dp']
nifty_dp_var = nifty['dp_std']
nifty_eq = nifty['eq']
nifty_eq_var = nifty['eq_std']
nifty_ndkl = nifty['ndkl']
nifty_bdkl_var = nifty['ndkl_std']

base_dp = base['dp']
base_dp_var = base['dp_std']
base_eq = base['eq']
base_eq_var = base['eq_std']
base_ndkl = base['ndkl']
base_bdkl_var = base['ndkl_std']
x_ids = base['p']

# %%
fig, ax = plt.subplots(1, 2,figsize = (25,8),sharex=True)
ax[0].plot(x_ids,fgnn_dp,color = 'r',label = "DP")
#ax[1].plot(x_ids,base_eq,color = 'g',label = "EQ")
ax[1].plot(x_ids,fgnn_ndkl,color = 'b',label = "NDKL")

#ax[0].plot(x_ids,nifty_dp,color = 'r',label = "DP Nifty",linestyle='dashdot', marker='o')
#ax[1].plot(x_ids,nifty_eq,color = 'g',label = "EQ Nifty",linestyle='dashdot', marker='o')
#ax[2].plot(x_ids,nifty_ndkl,color = 'b',label = "NDKL Nifty",linestyle='dashdot', marker='o')

ax[0].plot(x_ids,base_dp,color = 'r',label = "DP Adversarial",linestyle='dashed', marker='o')
#ax[1].plot(x_ids,fgnn_eq,color = 'g',label = "EQ Adversarial",linestyle='dashed', marker='o')
ax[1].plot(x_ids,base_ndkl,color = 'b',label = "NDKL Adversarial",linestyle='dashed', marker='o')


ax[0].set_xlabel("Top k")
ax[1].set_xlabel("Top k")
#ax[2].set_xlabel("Top k")
ax[0].legend()
ax[1].legend()
#ax[2].legend()
plt.savefig("top_k_ind_sim3_500.pdf",bbox_inches='tight')
# %%
sns.set_context("talk")
fig, ax = plt.subplots(1, 2,figsize = (30,8),sharex=True)
ax[0].plot(x_ids,base_dp,color = 'r',label = "DP")
ax[0].plot(x_ids,base_eq,color = 'g',label = "EQ")
ax[0].plot(x_ids,base_ndkl,color = 'b',label = "NDKL")
ax[1].plot(x_ids,fgnn_dp,color = 'r',label = "DP")
ax[1].plot(x_ids,fgnn_eq,color = 'g',label = "EQ")
ax[1].plot(x_ids,fgnn_ndkl,color = 'b',label = "NDKL")

ax[0].set_ylabel("Base Model")
ax[1].set_ylabel("Adversarial")
ax[0].set_xlabel("Top k")
ax[1].set_xlabel("Top k")
plt.legend()
#plt.savefig("top_k_sim_800.pdf",bbox_inches='tight')
#%%
#%%
acc = [0.715962,0.514085,0.647887,0.713615,0.553991]
dp = [0.106072,0.014884,0.064226,0.068432,0.031439]
eq = [0.081839,0.013333,0.077471,0.071839,0.003908]  

#%%
fig, ax = plt.subplots(1, 2,figsize = (30,8),sharex=True)
ax[0].scatter(acc,1-np.array(dp),s = 300)
ax[1].scatter(acc,1-np.array(eq),s = 300)
ax[0].set_ylabel("1 - Demographic Parity")
ax[1].set_ylabel("1 - Equality of Opportunity")
ax[0].set_xlabel("Accuracy")
ax[1].set_xlabel("Accuracy")
plt.savefig("trade_off.pdf",bbox_inches='tight')
# %%
ranges = np.arange(100,4600,500)
# %%
dp = [0,0,0.11342309261525901,0.13104124965657993,0.10965601536581107,0.10018450490394612,0.08690909075768041,0.07496408515084235,0.06776800341938596]
eq = [0,0,0.041216678058783374,0.06100979192166456,0.052789855072463754,0.05692401181441753,0.04664147430104876,0.07496408515084235,0.044545799678867626]

#%%
fig, ax = plt.subplots(1, 2,figsize = (30,8),sharex=True)
ax[0].scatter(ranges,dp,s = 300)
ax[1].scatter(ranges,eq,s = 300)
ax[0].set_ylabel("Demographic Parity")
ax[1].set_ylabel("Equality of Opportunity")
ax[0].set_xlabel("# of top probability")
ax[1].set_xlabel("# of top probability")
plt.savefig("top_k.pdf",bbox_inches='tight')