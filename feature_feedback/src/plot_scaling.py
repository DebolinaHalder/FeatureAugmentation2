#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 40})
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
sns.set(font_scale=6)
sns.set_style(style='white')
from matplotlib.ticker import FuncFormatter

#%%
def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:.2f}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str
#%%
debias_bail_dp = [14.5,13.9,12.7]
debias_bail_f1 = [80.2,77.5,75.8]
debias_bail_eq = [10.1,8.9,7.0]

rnf_bail_dp = [12.3,10.9,8.7]
rnf_bail_f1 = [81.2,78.2,77.8]
rnf_bail_eq = [6.9,5.7,4.9]

fgnn_bail_dp = [12.8,11.9,9.1]
fgnn_bail_f1 = [80.2,79.2,77.1]
fgnn_bail_eq = [7.2,6.3,5.6]

bfts_bail_dp = [10.1,9.8,7.9]
bfts_bail_eq = [5.2,3.5,2.8]
bfts_bail_f1 = [81.5,79.6,79.1]

fvgnn_bail_dp = [12.9,11.6,9.5]
fvgnn_bail_f1 = [80.9,77.9,76.8]
fvgnn_bail_eq = [7.8,6.9,6.3]
fairsin_bail_dp = [12.9,11.0,9.2]
fairsin_bail_f1 = [81.2,78.2,77.1]
fairsin_bail_eq = [7.5,6.3,6.1]

bfts_bail_f1 = np.array(bfts_bail_f1)
max_indexes = np.argsort(bfts_bail_f1)
bfts_bail_f1 = bfts_bail_f1[max_indexes]
bfts_bail_dp = np.array(bfts_bail_dp)[max_indexes]
bfts_bail_eq = np.array(bfts_bail_eq)[max_indexes]



#%%


#%%
fig, ax = plt.subplots(1,2,figsize = (35,12))
fig.tight_layout()
linewidth = 2.5
size = 700
sizes = [size,size,size]
ax[0].scatter(debias_bail_f1,1 - np.array(debias_bail_dp)/100,color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0].scatter(fgnn_bail_f1,1 - np.array(fgnn_bail_dp)/100,color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0].scatter(rnf_bail_f1,1 - np.array(rnf_bail_dp)/100,color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_dp)/100,color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0].scatter(fairsin_bail_f1,1 - np.array(fairsin_bail_dp)/100,color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0].plot(bfts_bail_f1,1 - np.array(bfts_bail_dp)/100,color = 'r',label = "BFtS",marker = "D",markersize=25)
#sns.regplot(x=bfts_bail_f1, y=1 - np.array(bfts_bail_dp), ax=ax[0,0],color = 'r')
ax[1].scatter(debias_bail_f1,1 - np.array(debias_bail_eq)/100,color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1].scatter(fgnn_bail_f1,1 - np.array(fgnn_bail_eq)/100,color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1].scatter(rnf_bail_f1,1 - np.array(rnf_bail_eq)/100,color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1].plot(bfts_bail_f1,1 - np.array(bfts_bail_eq)/100,color = 'r',label = "BFtS",marker = "D",markersize = 25)
ax[1].scatter(fairsin_bail_f1,1 - np.array(fairsin_bail_eq)/100,color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_eq)/100,color = 'orange',label = "FairVGNN",marker = "P",s = sizes)

#sns.regplot(x=bfts_pokekn_f1, y=1 - np.array(bfts_pokekn_eq), ax=ax[1,5],color = 'r')



ax[0].set_ylabel(u' 1 - Δ DP')
ax[1].set_ylabel(u' 1 - Δ EQOP')
fig.text(0.5, 0.01, 'F1 Score', ha='center')
ax[0].legend(loc='upper center', ncol=6, bbox_to_anchor=(1.1, 1.2),frameon=False)
major_formatter = FuncFormatter(my_formatter)
ax[0].yaxis.set_major_formatter(major_formatter)
ax[1].yaxis.set_major_formatter(major_formatter)





ax[0].xaxis.set_major_formatter(major_formatter)
ax[1].xaxis.set_major_formatter(major_formatter)




ax[0].spines[['right', 'top']].set_visible(False) 
ax[1].spines[['right', 'top']].set_visible(False)

fig.tight_layout(pad=5.0)  # pad = extra space (inches) between plots
plt.savefig("scaling.pdf",bbox_inches = "tight")
plt.show()






#plt.savefig("acc_vs_fairness3.pdf",bbox_inches = "tight")
# %%
