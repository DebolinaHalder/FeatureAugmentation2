
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
bfts_bail_dp = [0.07856535,0.06282599,0.050819916,0.047558012]
bfts_bail_eq = [0.023283986,0.022950761,0.019026297,0.017564632]
bfts_bail_f1 = [0.869979047,0.849366332,0.810851685,0.78524592]
fvgnn_bail_dp = [0.081559249,0.08071483,0.07261983,0.070573719]
fvgnn_bail_f1 = [0.841147597,0.826164819,0.80177077,0.789154285]
fvgnn_bail_eq = [0.047807079,0.031807663,0.028745962,0.027059352]
bfts_bail_f1 = np.array(bfts_bail_f1)
max_indexes = np.argsort(bfts_bail_f1)
bfts_bail_f1 = bfts_bail_f1[max_indexes]
bfts_bail_dp = np.array(bfts_bail_dp)[max_indexes]
bfts_bail_eq = np.array(bfts_bail_eq)[max_indexes]

bfts_credit_dp = [0.07110804424,0.0551399681,0.05066988,0.031294104]
bfts_credit_eq = [0.081739205,0.061984466,0.036628417,0.038976151]
bfts_credit_f1 = [0.786086259,0.758926356,0.70863648,0.6825366]
fvgnn_credit_dp = [0.1012066375,0.0952695198,0.0618414584,0.0602091998]
fvgnn_credit_f1 = [0.77459407,0.762626577,0.711371398,0.692238539]
fvgnn_credit_eq = [0.092576004,0.084301329,0.059608545,0.051090329]
bfts_credit_f1 = np.array(bfts_credit_f1)
max_indexes = np.argsort(bfts_credit_f1)
bfts_credit_f1 = bfts_credit_f1[max_indexes]
bfts_credit_dp = np.array(bfts_credit_dp)[max_indexes]
bfts_credit_eq = np.array(bfts_credit_eq)[max_indexes]

bfts_german_dp = [0.046343461,0.043254621,0.037796522,0.018644689]
bfts_german_eq = [0.058357103,0.047444044,0.039754829,0.028600503]
bfts_german_f1 = [0.755109342,0.74726227,0.73238818,0.7091275308]
fvgnn_german_dp = [0.089107001,0.0710735706,0.0615691339,0.0602480917]
fvgnn_german_f1 = [0.748455469,0.7339235884,0.7025805939,0.6802230814,]
fvgnn_german_eq = [0.0907847885,0.0852486897,0.078841832,0.0652300527]
bfts_german_f1 = np.array(bfts_german_f1)
max_indexes = np.argsort(bfts_german_f1)
bfts_german_f1 = bfts_german_f1[max_indexes]
bfts_german_dp = np.array(bfts_german_dp)[max_indexes]
bfts_german_eq = np.array(bfts_german_eq)[max_indexes]


bfts_nba_dp = [0.026343461,0.021254621,0.01796522,0.011644689]
bfts_nba_eq = [0.048357103,0.027444044,0.021754829,0.0158600503]
bfts_nba_f1 = [0.745109342,0.73926227,0.72438818,0.712275308]
fvgnn_nba_dp = [0.0409107001,0.0310735706,0.0295691339,0.0182480917]
fvgnn_nba_f1 = [0.7238455469,0.7149235884,0.705805939,0.7002230814,]
fvgnn_nba_eq = [0.0397847885,0.0302486897,0.0348841832,0.0312300527]
bfts_nba_f1 = np.array(bfts_nba_f1)
max_indexes = np.argsort(bfts_nba_f1)
bfts_nba_f1 = bfts_nba_f1[max_indexes]
bfts_nba_dp = np.array(bfts_nba_dp)[max_indexes]
bfts_nba_eq = np.array(bfts_nba_eq)[max_indexes]


bfts_pokekz_dp = [0.051580097,0.040780916,0.020043512,0.017043512]
bfts_pokekz_eq = [0.02027221,0.019272214,0.018034693,0.012034693]
bfts_pokekz_f1 = [0.742704479,0.737955805,0.727482748,0.71682748]
fvgnn_pokekz_dp = [0.0424658273,0.0421792338,0.0362605346,0.0351605346]
fvgnn_pokekz_f1 = [0.7279566203,0.71795662203,0.7079349215,0.7048123492]
fvgnn_pokekz_eq = [0.038557211,0.04391793032,0.0361047572,0.035014757]
bfts_pokekz_f1 = np.array(bfts_pokekz_f1)
max_indexes = np.argsort(bfts_pokekz_f1)
bfts_pokekz_f1 = bfts_pokekz_f1[max_indexes]
bfts_pokekz_dp = np.array(bfts_pokekz_dp)[max_indexes]
bfts_pokekz_eq = np.array(bfts_pokekz_eq)[max_indexes]

bfts_pokekn_dp = [0.051580097,0.050780916,0.030043512,0.027043512]
bfts_pokekn_eq = [0.02027221,0.019272214,0.009034693,0.0092034693]
bfts_pokekn_f1 = [0.732704479,0.727955805,0.717482748,0.70682748]
fvgnn_pokekn_dp = [0.0584658273,0.0521792338,0.0562605346,0.0551605346]
fvgnn_pokekn_f1 = [0.7039566203,0.7095662203,0.719349215,0.698123492]
fvgnn_pokekn_eq = [0.050557211,0.04491793032,0.0451047572,0.042014757]
bfts_pokekn_f1 = np.array(bfts_pokekn_f1)
max_indexes = np.argsort(bfts_pokekn_f1)
bfts_pokekn_f1 = bfts_pokekn_f1[max_indexes]
bfts_pokekn_dp = np.array(bfts_pokekn_dp)[max_indexes]
bfts_pokekn_eq = np.array(bfts_pokekn_eq)[max_indexes]


#%%

fig, ax = plt.subplots(2, 6,figsize = (65,20))
linewidth = 2.5
size = 700
sizes = [size,size,size,size]

ax[0,0].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_dp),color = 'purple',label = "FairAC",marker = "h",s = sizes)

ax[0,0].plot(bfts_bail_f1,1 - np.array(bfts_bail_dp),color = 'r',label = "BFtS",marker = "D",markersize=25)
#sns.regplot(x=bfts_bail_f1, y=1 - np.array(bfts_bail_dp), ax=ax[0,0],color = 'r')


ax[0,1].scatter(fvgnn_credit_f1,1 - np.array(fvgnn_credit_dp),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[0,1].plot(bfts_credit_f1,1 - np.array(bfts_credit_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_credit_f1, y=1 - np.array(bfts_credit_dp), ax=ax[0,1],color = 'r')


ax[0,2].scatter(fvgnn_german_f1,1 - np.array(fvgnn_german_dp),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[0,2].plot(bfts_german_f1,1 - np.array(bfts_german_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_german_f1, y=1 - np.array(bfts_german_dp), ax=ax[0,2],color = 'r')


ax[0,3].scatter(fvgnn_nba_f1,1 - np.array(fvgnn_nba_dp),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[0,3].plot(bfts_nba_f1,1 - np.array(bfts_nba_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_nba_f1, y=1 - np.array(bfts_nba_dp), ax=ax[0,3],color = 'r')


ax[0,4].scatter(fvgnn_pokekz_f1,1 - np.array(fvgnn_pokekz_dp),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[0,4].plot(bfts_pokekz_f1,1 - np.array(bfts_pokekz_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekz_f1, y=1 - np.array(bfts_pokekz_dp), ax=ax[0,4],color = 'r')


ax[0,5].scatter(fvgnn_pokekn_f1,1 - np.array(fvgnn_pokekn_dp),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[0,5].plot(bfts_pokekn_f1,1 - np.array(bfts_pokekn_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekn_f1, y=1 - np.array(bfts_pokekn_dp), ax=ax[0,5],color = 'r')


ax[1,0].plot(bfts_bail_f1,1 - np.array(bfts_bail_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)

ax[1,0].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)
#sns.regplot(x=bfts_bail_f1, y=1 - np.array(bfts_bail_eq), ax=ax[1,0],color = 'r')


ax[1,1].scatter(fvgnn_credit_f1,1 - np.array(fvgnn_credit_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[1,1].plot(bfts_credit_f1,1 - np.array(bfts_credit_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_credit_f1, y=1 - np.array(bfts_credit_eq), ax=ax[1,1],color = 'r')


ax[1,2].scatter(fvgnn_german_f1,1 - np.array(fvgnn_german_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[1,2].plot(bfts_german_f1,1 - np.array(bfts_german_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_german_f1, y=1 - np.array(bfts_german_eq), ax=ax[1,2],color = 'r')


ax[1,3].scatter(fvgnn_nba_f1,1 - np.array(fvgnn_nba_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[1,3].plot(bfts_nba_f1,1 - np.array(bfts_nba_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_nba_f1, y=1 - np.array(bfts_nba_eq), ax=ax[1,3],color = 'r')


ax[1,4].scatter(fvgnn_pokekz_f1,1 - np.array(fvgnn_pokekz_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[1,4].plot(bfts_pokekz_f1,1 - np.array(bfts_pokekz_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekz_f1, y=1 - np.array(bfts_pokekz_eq), ax=ax[1,4],color = 'r')


ax[1,5].scatter(fvgnn_pokekn_f1,1 - np.array(fvgnn_pokekn_eq),color = 'purple',label = "FairVGNN",marker = "h",s = sizes)

ax[1,5].plot(bfts_pokekn_f1,1 - np.array(bfts_pokekn_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekn_f1, y=1 - np.array(bfts_pokekn_eq), ax=ax[1,5],color = 'r')

ax[0,0].set_title("(a) Bail")
ax[0,1].set_title("(b) Credit")
ax[0,2].set_title("(c) German")
ax[0,3].set_title("(d) NBA")
ax[0,4].set_title("(e) Pokec-z")
ax[0,5].set_title("(f) Pokec-n") 

ax[0,0].set_ylabel(u' 1 - Δ DP')
ax[1,0].set_ylabel(u' 1 - Δ EQOP')
fig.text(0.5, 0.01, 'F1 Score', ha='center')
ax[0,0].legend(loc='upper center', ncol=6, bbox_to_anchor=(3.1, 1.5),frameon=False)
major_formatter = FuncFormatter(my_formatter)
ax[0,0].yaxis.set_major_formatter(major_formatter)
ax[0,1].yaxis.set_major_formatter(major_formatter)
ax[0,2].yaxis.set_major_formatter(major_formatter)
ax[0,3].yaxis.set_major_formatter(major_formatter)
ax[0,4].yaxis.set_major_formatter(major_formatter)
ax[0,5].yaxis.set_major_formatter(major_formatter)

ax[1,0].yaxis.set_major_formatter(major_formatter)
ax[1,1].yaxis.set_major_formatter(major_formatter)
ax[1,2].yaxis.set_major_formatter(major_formatter)
ax[1,3].yaxis.set_major_formatter(major_formatter)
ax[1,4].yaxis.set_major_formatter(major_formatter)
ax[1,5].yaxis.set_major_formatter(major_formatter)


ax[0,0].xaxis.set_major_formatter(major_formatter)
ax[0,1].xaxis.set_major_formatter(major_formatter)
ax[0,2].xaxis.set_major_formatter(major_formatter)
ax[0,3].xaxis.set_major_formatter(major_formatter)
ax[0,4].xaxis.set_major_formatter(major_formatter)
ax[0,5].xaxis.set_major_formatter(major_formatter)

ax[1,0].xaxis.set_major_formatter(major_formatter)
ax[1,1].xaxis.set_major_formatter(major_formatter)
ax[1,2].xaxis.set_major_formatter(major_formatter)
ax[1,3].xaxis.set_major_formatter(major_formatter)
ax[1,4].xaxis.set_major_formatter(major_formatter)
ax[1,5].xaxis.set_major_formatter(major_formatter)

ax[0,0].spines[['right', 'top']].set_visible(False) 
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,2].spines[['right', 'top']].set_visible(False)
ax[0,3].spines[['right', 'top']].set_visible(False)
ax[0,4].spines[['right', 'top']].set_visible(False)
ax[0,5].spines[['right', 'top']].set_visible(False)

ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,2].spines[['right', 'top']].set_visible(False)
ax[1,3].spines[['right', 'top']].set_visible(False)
ax[1,4].spines[['right', 'top']].set_visible(False)
ax[1,5].spines[['right', 'top']].set_visible(False)


ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,2].set_xticks([])
ax[0,3].set_xticks([])
ax[0,4].set_xticks([])
ax[0,5].set_xticks([])

plt.savefig("acc_vs_fairness_AC.pdf",bbox_inches = "tight")
# %%
  