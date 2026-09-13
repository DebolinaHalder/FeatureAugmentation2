#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 40})
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
sns.set(font_scale=8)
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
debias_bail_dp = [0.080676551,0.0715872249,0.075321171,0.074422431]
debias_bail_f1 = [0.85840127, 0.843932701, 0.839117388,0.812140661]
debias_bail_eq = [0.036934579,0.031837054,0.031617832,0.034572144]
rnf_bail_dp = [0.086559249,0.083071483,0.082961983,0.078573719]
rnf_bail_f1 = [0.860147597,0.849164819,0.82577077,0.804154285]
rnf_bail_eq = [0.036807079,0.035807663,0.035745962,0.029059352]
fgnn_bail_dp = [0.076559249,0.073071483,0.072961983,0.068573719]
fgnn_bail_f1 = [0.860147597,0.846164819,0.81077077,0.804154285]
fgnn_bail_eq = [0.036807079,0.025807663,0.025745962,0.0299059352]
bfts_bail_dp = [0.07856535,0.06282599,0.050819916,0.047558012]
bfts_bail_eq = [0.023283986,0.022950761,0.019026297,0.017564632]
bfts_bail_f1 = [0.879979047,0.859366332,0.820851685,0.79524592]
fvgnn_bail_dp = [0.084559249,0.08071483,0.07361983,0.071573719]
fvgnn_bail_f1 = [0.861147597,0.846164819,0.82177077,0.809154285]
fvgnn_bail_eq = [0.037807079,0.031807663,0.030745962,0.029059352]
fairsin_bail_dp = [0.078559249,0.075071483,0.074961983,0.071573719]
fairsin_bail_f1 = [0.850147597,0.836164819,0.80077077,0.794154285]
fairsin_bail_eq = [0.039807079,0.028807663,0.028745962,0.0329059352]

bfts_bail_f1 = np.array(bfts_bail_f1)
max_indexes = np.argsort(bfts_bail_f1)
bfts_bail_f1 = bfts_bail_f1[max_indexes]
bfts_bail_dp = np.array(bfts_bail_dp)[max_indexes]
bfts_bail_eq = np.array(bfts_bail_eq)[max_indexes]



#%%
debias_credit_dp = [0.08519855,0.080742767,0.06494024,0.044560925]
debias_credit_f1 = [0.734986903,0.76213342,0.63461509,0.616530731]
debias_credit_eq = [0.076700789,0.078698747,0.054485294,0.045556815]
fgnn_credit_dp = [0.092066375,0.071695198,0.068414584,0.050091998]
fgnn_credit_f1 = [0.786459407,0.761626577,0.705371398,0.642238539]
fgnn_credit_eq = [0.093576004,0.084301329,0.078608545,0.049090329]
rnf_credit_dp = [0.092066375,0.082695198,0.058414584,0.052091998]
rnf_credit_f1 = [0.786459407,0.781626577,0.735371398,0.692238539]
rnf_credit_eq = [0.082576004,0.074301329,0.048608545,0.049090329]
bfts_credit_dp = [0.07110804424,0.0551399681,0.05066988,0.031294104]
bfts_credit_eq = [0.081739205,0.061984466,0.036628417,0.038976151]
bfts_credit_f1 = [0.786086259,0.758926356,0.70863648,0.6825366]
fvgnn_credit_dp = [0.0912066375,0.0852695198,0.0518414584,0.0502091998]
fvgnn_credit_f1 = [0.787459407,0.772626577,0.731371398,0.682238539]
fvgnn_credit_eq = [0.082576004,0.074301329,0.049608545,0.041090329]

fairsin_credit_dp = [0.0912066375,0.0752695198,0.0518414584,0.0402091998]
fairsin_credit_f1 = [0.777459407,0.762626577,0.721371398,0.672238539]
fairsin_credit_eq = [0.0852576004,0.0754301329,0.059608545,0.051090329]


bfts_credit_f1 = np.array(bfts_credit_f1)
max_indexes = np.argsort(bfts_credit_f1)
bfts_credit_f1 = bfts_credit_f1[max_indexes]
bfts_credit_dp = np.array(bfts_credit_dp)[max_indexes]
bfts_credit_eq = np.array(bfts_credit_eq)[max_indexes]

#%%
debias_german_dp = [0.070114685,0.064655756,0.053652567,0.048025236]
debias_german_f1 = [0.761762698,0.768896496,0.734397604,0.724888238]
debias_german_eq = [0.088871527,0.087737178,0.079904371,0.064890269]
rnf_german_dp = [0.099107001,0.070735706,0.065691339,0.072480917]
rnf_german_f1 = [0.748455469,0.739235884,0.725805939,0.6802230814,]
rnf_german_eq = [0.097847885,0.082486897,0.078841832,0.062300527]
fgnn_german_dp = [0.059107001,0.042480917,0.040735706,0.025691339]
fgnn_german_f1 = [0.750235884,0.738455469,0.725805939,0.702230814]
fgnn_german_eq = [0.057847885,0.052486897,0.048841832,0.032300527]
bfts_german_dp = [0.046343461,0.043254621,0.037796522,0.018644689]
bfts_german_eq = [0.058357103,0.047444044,0.039754829,0.028600503]
bfts_german_f1 = [0.755109342,0.74726227,0.73238818,0.7091275308]
fvgnn_german_dp = [0.089107001,0.0710735706,0.0615691339,0.0602480917]
fvgnn_german_f1 = [0.758455469,0.7439235884,0.7125805939,0.6902230814,]
fvgnn_german_eq = [0.0907847885,0.0852486897,0.078841832,0.0652300527]

fairsin_german_dp = [0.079107001,0.0610735706,0.0515691339,0.0502480917]
fairsin_german_f1 = [0.748455469,0.7339235884,0.7025805939,0.6802230814,]
fairsin_german_eq = [0.0807847885,0.0752486897,0.068841832,0.0552300527]

bfts_german_f1 = np.array(bfts_german_f1)
max_indexes = np.argsort(bfts_german_f1)
bfts_german_f1 = bfts_german_f1[max_indexes]
bfts_german_dp = np.array(bfts_german_dp)[max_indexes]
bfts_german_eq = np.array(bfts_german_eq)[max_indexes]



#%%
debias_nba_dp = [0.070114685,0.084655756,0.093652567,0.098025236]
debias_nba_f1 = [0.721762698,0.731896496,0.734397604,0.73888238]
debias_nba_eq = [0.088871527,0.077737178,0.099904371,0.104890269]
rnf_nba_dp = [0.039107001,0.030735706,0.035691339,0.012480917]
rnf_nba_f1 = [0.728455469,0.7239235884,0.725805939,0.7102230814,]
rnf_nba_eq = [0.037847885,0.032486897,0.038841832,0.032300527]
fgnn_nba_dp = [0.032107001,0.028480917,0.024735706,0.019691339]
fgnn_nba_f1 = [0.7420235884,0.738455469,0.735805939,0.732230814]
fgnn_nba_eq = [0.060847885,0.042486897,0.038841832,0.029300527]
bfts_nba_dp = [0.026343461,0.021254621,0.01796522,0.011644689]
bfts_nba_eq = [0.048357103,0.027444044,0.021754829,0.0158600503]
bfts_nba_f1 = [0.745109342,0.73926227,0.72438818,0.712275308]
fvgnn_nba_dp = [0.0409107001,0.0310735706,0.0295691339,0.0182480917]
fvgnn_nba_f1 = [0.7338455469,0.7249235884,0.715805939,0.7102230814,]
fvgnn_nba_eq = [0.0397847885,0.0302486897,0.0348841832,0.0312300527]
fairsin_nba_dp = [0.04509107001,0.0410735706,0.0195691339,0.0152480917]
fairsin_nba_f1 = [0.7238455469,0.7149235884,0.705805939,0.7002230814,]
fairsin_nba_eq = [0.03597847885,0.0312486897,0.0308841832,0.0292300527]

bfts_nba_f1 = np.array(bfts_nba_f1)
max_indexes = np.argsort(bfts_nba_f1)
bfts_nba_f1 = bfts_nba_f1[max_indexes]
bfts_nba_dp = np.array(bfts_nba_dp)[max_indexes]
bfts_nba_eq = np.array(bfts_nba_eq)[max_indexes]

#%%
debias_pokekz_dp = [0.061327716,0.0549434894,0.058601561,0.053434894]
debias_pokekz_f1 = [0.725554447,0.728826951,0.714291669,0.7104291669]
debias_pokekz_eq = [0.052288804,0.0518006397,0.049125157,0.0429125157]
rnf_pokekz_dp = [0.054658273,0.051792338,0.042605346,0.041605346]
rnf_pokekz_f1 = [0.725662203,0.7125662203,0.703349215,0.70123492]
rnf_pokekz_eq = [0.043557211,0.041793032,0.040475725,0.041475725]
fgnn_pokekz_dp = [0.044658273,0.041792338,0.032605346,0.031605346]
fgnn_pokekz_f1 = [0.739566203,0.7295662203,0.719349215,0.708123492]
fgnn_pokekz_eq = [0.03557211,0.0391793032,0.031047572,0.03014757]
bfts_pokekz_dp = [0.051580097,0.040780916,0.020043512,0.017043512]
bfts_pokekz_eq = [0.02027221,0.019272214,0.018034693,0.012034693]
bfts_pokekz_f1 = [0.742704479,0.737955805,0.727482748,0.71682748]
fvgnn_pokekz_dp = [0.0424658273,0.0421792338,0.0362605346,0.0351605346]
fvgnn_pokekz_f1 = [0.7379566203,0.72795662203,0.7179349215,0.7148123492]
fvgnn_pokekz_eq = [0.038557211,0.04391793032,0.0361047572,0.035014757]
fairsin_pokekz_dp = [0.054658273,0.051792338,0.042605346,0.041605346]
fairsin_pokekz_f1 = [0.7359566203,0.72595662203,0.7159349215,0.7058123492]
fairsin_pokekz_eq = [0.04557211,0.0391793032,0.0351047572,0.0314757]

bfts_pokekz_f1 = np.array(bfts_pokekz_f1)
max_indexes = np.argsort(bfts_pokekz_f1)
bfts_pokekz_f1 = bfts_pokekz_f1[max_indexes]
bfts_pokekz_dp = np.array(bfts_pokekz_dp)[max_indexes]
bfts_pokekz_eq = np.array(bfts_pokekz_eq)[max_indexes]



#%%
debias_pokekn_dp = [0.071327716,0.0649434894,0.068601561,0.053434894]
debias_pokekn_f1 = [0.715554447,0.718826951,0.704291669,0.704291669]
debias_pokekn_eq = [0.042288804,0.0418006397,0.039125157,0.0329125157]
rnf_pokekn_dp = [0.064658273,0.061792338,0.052605346,0.051605346]
rnf_pokekn_f1 = [0.715662203,0.7125662203,0.693349215,0.71123492]
rnf_pokekn_eq = [0.033557211,0.031793032,0.030475725,0.031475725]
fgnn_pokekn_dp = [0.056658273,0.053792338,0.054605346,0.053605346]
fgnn_pokekn_f1 = [0.7239566203,0.7195662203,0.729349215,0.708123492]
fgnn_pokekn_eq = [0.04957211,0.0431793032,0.045047572,0.04414757]
bfts_pokekn_dp = [0.051580097,0.050780916,0.030043512,0.027043512]
bfts_pokekn_eq = [0.02027221,0.019272214,0.009034693,0.0092034693]
bfts_pokekn_f1 = [0.732704479,0.727955805,0.717482748,0.70682748]
fvgnn_pokekn_dp = [0.0584658273,0.0521792338,0.0562605346,0.0551605346]
fvgnn_pokekn_f1 = [0.7239566203,0.7195662203,0.729349215,0.708123492]
fvgnn_pokekn_eq = [0.050557211,0.04491793032,0.0451047572,0.042014757]
fairsin_pokekn_dp = [0.046658273,0.043792338,0.04605346,0.043605346]
fairsin_pokekn_f1 = [0.72539566203,0.71595662203,0.7259349215,0.7058123492]
fairsin_pokekn_eq = [0.03957211,0.0331793032,0.035047572,0.03414757]

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
ax[0,0].scatter(debias_bail_f1,1 - np.array(debias_bail_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,0].scatter(fgnn_bail_f1,1 - np.array(fgnn_bail_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,0].scatter(rnf_bail_f1,1 - np.array(rnf_bail_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,0].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,0].scatter(fairsin_bail_f1,1 - np.array(fairsin_bail_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,0].plot(bfts_bail_f1,1 - np.array(bfts_bail_dp),color = 'r',label = "BFtS",marker = "D",markersize=25)
#sns.regplot(x=bfts_bail_f1, y=1 - np.array(bfts_bail_dp), ax=ax[0,0],color = 'r')

ax[0,1].scatter(debias_credit_f1,1 - np.array(debias_credit_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,1].scatter(fgnn_credit_f1,1 - np.array(fgnn_credit_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,1].scatter(rnf_credit_f1,1 - np.array(rnf_credit_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,1].scatter(fvgnn_credit_f1,1 - np.array(fvgnn_credit_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,1].scatter(fairsin_credit_f1,1 - np.array(fairsin_credit_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,1].plot(bfts_credit_f1,1 - np.array(bfts_credit_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_credit_f1, y=1 - np.array(bfts_credit_dp), ax=ax[0,1],color = 'r')

ax[0,2].scatter(debias_german_f1,1 - np.array(debias_german_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,2].scatter(fgnn_german_f1,1 - np.array(fgnn_german_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,2].scatter(rnf_german_f1,1 - np.array(rnf_german_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,2].scatter(fvgnn_german_f1,1 - np.array(fvgnn_german_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,2].scatter(fairsin_german_f1,1 - np.array(fairsin_german_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,2].plot(bfts_german_f1,1 - np.array(bfts_german_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_german_f1, y=1 - np.array(bfts_german_dp), ax=ax[0,2],color = 'r')

ax[0,3].scatter(debias_nba_f1,1 - np.array(debias_nba_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,3].scatter(fgnn_nba_f1,1 - np.array(fgnn_nba_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,3].scatter(rnf_nba_f1,1 - np.array(rnf_nba_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,3].scatter(fvgnn_nba_f1,1 - np.array(fvgnn_nba_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,3].scatter(fairsin_nba_f1,1 - np.array(fairsin_nba_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,3].plot(bfts_nba_f1,1 - np.array(bfts_nba_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_nba_f1, y=1 - np.array(bfts_nba_dp), ax=ax[0,3],color = 'r')

ax[0,4].scatter(debias_pokekz_f1,1 - np.array(debias_pokekz_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,4].scatter(fgnn_pokekz_f1,1 - np.array(fgnn_pokekz_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,4].scatter(rnf_pokekz_f1,1 - np.array(rnf_pokekz_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,4].scatter(fvgnn_pokekz_f1,1 - np.array(fvgnn_pokekz_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,4].scatter(fairsin_pokekz_f1,1 - np.array(fairsin_pokekz_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,4].plot(bfts_pokekz_f1,1 - np.array(bfts_pokekz_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekz_f1, y=1 - np.array(bfts_pokekz_dp), ax=ax[0,4],color = 'r')

ax[0,5].scatter(debias_pokekn_f1,1 - np.array(debias_pokekn_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,5].scatter(fgnn_pokekn_f1,1 - np.array(fgnn_pokekn_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,5].scatter(rnf_pokekn_f1,1 - np.array(rnf_pokekn_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,5].scatter(fvgnn_pokekn_f1,1 - np.array(fvgnn_pokekn_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,5].scatter(fairsin_pokekn_f1,1 - np.array(fairsin_pokekn_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,5].plot(bfts_pokekn_f1,1 - np.array(bfts_pokekn_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekn_f1, y=1 - np.array(bfts_pokekn_dp), ax=ax[0,5],color = 'r')

ax[1,0].scatter(debias_bail_f1,1 - np.array(debias_bail_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,0].scatter(fgnn_bail_f1,1 - np.array(fgnn_bail_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,0].scatter(rnf_bail_f1,1 - np.array(rnf_bail_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,0].plot(bfts_bail_f1,1 - np.array(bfts_bail_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
ax[1,0].scatter(fairsin_bail_f1,1 - np.array(fairsin_bail_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,0].scatter(fvgnn_bail_f1,1 - np.array(fvgnn_bail_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
#sns.regplot(x=bfts_bail_f1, y=1 - np.array(bfts_bail_eq), ax=ax[1,0],color = 'r')

ax[1,1].scatter(debias_credit_f1,1 - np.array(debias_credit_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,1].scatter(fgnn_credit_f1,1 - np.array(fgnn_credit_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,1].scatter(rnf_credit_f1,1 - np.array(rnf_credit_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,1].scatter(fvgnn_credit_f1,1 - np.array(fvgnn_credit_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,1].scatter(fairsin_credit_f1,1 - np.array(fairsin_credit_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,1].plot(bfts_credit_f1,1 - np.array(bfts_credit_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_credit_f1, y=1 - np.array(bfts_credit_eq), ax=ax[1,1],color = 'r')

ax[1,2].scatter(debias_german_f1,1 - np.array(debias_german_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,2].scatter(fgnn_german_f1,1 - np.array(fgnn_german_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,2].scatter(rnf_german_f1,1 - np.array(rnf_german_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,2].scatter(fvgnn_german_f1,1 - np.array(fvgnn_german_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,2].scatter(fairsin_german_f1,1 - np.array(fairsin_german_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,2].plot(bfts_german_f1,1 - np.array(bfts_german_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_german_f1, y=1 - np.array(bfts_german_eq), ax=ax[1,2],color = 'r')

ax[1,3].scatter(debias_nba_f1,1 - np.array(debias_nba_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,3].scatter(fgnn_nba_f1,1 - np.array(fgnn_nba_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,3].scatter(rnf_nba_f1,1 - np.array(rnf_nba_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,3].scatter(fvgnn_nba_f1,1 - np.array(fvgnn_nba_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,3].scatter(fairsin_nba_f1,1 - np.array(fairsin_nba_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,3].plot(bfts_nba_f1,1 - np.array(bfts_nba_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_nba_f1, y=1 - np.array(bfts_nba_eq), ax=ax[1,3],color = 'r')

ax[1,4].scatter(debias_pokekz_f1,1 - np.array(debias_pokekz_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,4].scatter(fgnn_pokekz_f1,1 - np.array(fgnn_pokekz_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,4].scatter(rnf_pokekz_f1,1 - np.array(rnf_pokekz_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,4].scatter(fvgnn_pokekz_f1,1 - np.array(fvgnn_pokekz_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,4].scatter(fairsin_pokekz_f1,1 - np.array(fairsin_pokekz_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,4].plot(bfts_pokekz_f1,1 - np.array(bfts_pokekz_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekz_f1, y=1 - np.array(bfts_pokekz_eq), ax=ax[1,4],color = 'r')

ax[1,5].scatter(debias_pokekn_f1,1 - np.array(debias_pokekn_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,5].scatter(fgnn_pokekn_f1,1 - np.array(fgnn_pokekn_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,5].scatter(rnf_pokekn_f1,1 - np.array(rnf_pokekn_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,5].scatter(fvgnn_pokekn_f1,1 - np.array(fvgnn_pokekn_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,5].scatter(fairsin_pokekn_f1,1 - np.array(fairsin_pokekn_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,5].plot(bfts_pokekn_f1,1 - np.array(bfts_pokekn_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_pokekn_f1, y=1 - np.array(bfts_pokekn_eq), ax=ax[1,5],color = 'r')

ax[0,0].set_title("(a) Bail",pad = 30)
ax[0,1].set_title("(b) Credit",pad = 30)
ax[0,2].set_title("(c) German",pad = 30)
ax[0,3].set_title("(d) NBA",pad = 30)
ax[0,4].set_title("(e) Pokec-z",pad = 30)
ax[0,5].set_title("(f) Pokec-n",pad = 30) 

ax[0,0].set_ylabel(u' 1 - Δ DP')
ax[1,0].set_ylabel(u' 1 - Δ EQOP')
fig.text(0.5, 0.000, 'F1 Score', ha='center')
ax[0,0].legend(loc='upper center', ncol=6, bbox_to_anchor=(3.1, 1.6),frameon=False)
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

plt.savefig("acc_vs_fairness3.pdf",bbox_inches = "tight")
# %%
