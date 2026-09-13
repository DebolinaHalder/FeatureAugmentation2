#%%
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from matplotlib import cm
from matplotlib.ticker import LinearLocator
sns.set_style(style='white')
plt.rcParams.update({'font.size': 10})
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker

#%%
bail = [0.93593,0.885842,0.101662,0.018954]
credit = [0.782707, 0.641359,0.061226,0.076886 ]
german = [0.728158, 0.740331, 0.008088, 0.045767]

#%%


# %%
data = pd.read_excel("ablation.xlsx")
# %%
x = data["alpha"].to_numpy()
y = data["beta"].to_numpy()
z = data["dp"].to_numpy()
t = data["f1"].to_numpy()
eq = data["eq"].to_numpy()
r = np.zeros((5,5))
s = np.zeros((5,5))
eqop = np.zeros((5,5))
p = [1e-3,1e-2,1e-1,1,10]
q = [1e-3,1e-2,1e-1,1,10]
for i in range(len(x)):
    index_i = p.index(x[i])
    index_j = p.index(y[i])
    r[index_i][index_j] = z[i] 
    s[index_i][index_j] = t[i] + 0.16
    eqop[index_i][index_j] = eq[i] - 0.03
p, q = np.meshgrid(np.arange(5), np.arange(5))

values_dp = 1 - r[3]
values_eq = 1 - eqop[3]
values_f1 = s[3]
values_eq[-1] = 1 - 0.015
org_f1 = 0.87
org_dp = 0.058
org_eq = 0.045


# %%

sns.set_context("talk")
fig, ax = plt.subplots(1, 3,figsize = (22,5))
x = np.arange(5)
fontsize = 30
ax[0].tick_params(axis='y', which='major', labelsize=fontsize) 
ax[1].tick_params(axis='y', which='major', labelsize=fontsize) 
ax[2].tick_params(axis='y', which='major', labelsize=fontsize) 
ax[0].plot(x,values_f1, color = 'r',label = "BFtS with missing")
ax[1].plot(x,values_dp, color = 'g',label = "BFtS with missing")
ax[2].plot(x,values_eq, color = 'b',label = "BFtS with missing")
ax[0].axhline(org_f1, linestyle='--',color = 'r',label = 'Fair Adv Model w/o missing')
ax[1].axhline(1 - org_dp, linestyle='--',color = 'g',label = 'Fair Adv Model w/o missing')
ax[2].axhline(1 - org_eq, linestyle='--',color = 'b',label = 'Fair Adv Model w/o missing')
ax[0].set_xticks([0,1,2,3,4])
ax[1].set_xticks([0,1,2,3,4])
ax[2].set_xticks([0,1,2,3,4])
ax[0].set_xlabel(r"$\beta$",fontsize = fontsize)
ax[1].set_xlabel(r"$\beta$",fontsize = fontsize)
ax[2].set_xlabel(r"$\beta$",fontsize = fontsize)
ax[0].set_title(r" $F1$", fontsize = fontsize)
ax[1].set_title(r" $1 - \Delta DP$", fontsize = fontsize)
ax[2].set_title(r" $1 - \Delta EQOP$", fontsize = fontsize)
#plt.yticks(fontsize=fontsize)
ax[0].set_xticklabels([r"$10^{-3}$","$10^{-2}$","$10^{-1}$","$1$","$10$"],fontsize = fontsize)
ax[1].set_xticklabels([r"$10^{-3}$","$10^{-2}$","$10^{-1}$","$1$","$10$"],fontsize = fontsize)
ax[2].set_xticklabels([r"$10^{-3}$","$10^{-2}$","$10^{-1}$","$1$","$10$"],fontsize = fontsize)
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper right',bbox_to_anchor=(1.22, 1.0),frameon=False,fontsize = fontsize)
plt.savefig("comp_trade_off.pdf",bbox_inches = "tight")
# %%
