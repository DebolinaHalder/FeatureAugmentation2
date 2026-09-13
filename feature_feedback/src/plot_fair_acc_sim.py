#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sim_graph import get_graph
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
sns.set(font_scale=8)
sns.set_style(style='white')
from matplotlib.ticker import FuncFormatter
import networkx as nx

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

method = 'partition'
sens_attr = "s"
sizes = [20, 20]
probs1 = [[0.08, 0.0009], [0.0009, 0.08]]
probs2 = [[0.08, 0.001], [0.001, 0.08]]
probs3 = [[0.08, 0.01], [0.01, 0.08]]
probs4 = [[0.08, 0.05], [0.05, 0.08]]
nb_classes = 'binary'
seed = 1
label_number = 3
sens_number = 0.2
p = 0.7
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g4 = get_graph(sizes, probs4, nb_classes,method,seed,label_number,sens_number,p)
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g2 = get_graph(sizes, probs2, nb_classes,method,seed,label_number,sens_number,p)
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g3 = get_graph(sizes, probs3, nb_classes,method,seed,label_number,sens_number,p)
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train,g1 = get_graph(sizes, probs1, nb_classes,method,seed,label_number,sens_number,p)
color_map = {0:'red', 1:'green'}
colors = [color_map[g2.nodes[node]['s']] for node in g2]
nx.draw(g2,node_color=colors)
#%%
debias_17_dp = [0.29, 0.353792383933229, 0.0, 0.0, 0.0]
debias_17_f1 = [0.7394535519125683,0.7693596059113301,0.5294117647058824,0.5337243401759532,0.0]
debias_17_eq = [0.1320985691573924, 0.12133669609079441, 0.0, 0.0, 0.0]
rnf_17_dp = [0.25, 0.343792383933229, 0.3, 0.29, 0.299]
rnf_17_f1 = [0.7294535519125683,0.7593596059113301,0.7394117647058824,0.7337243401759532,0.7314117647058824]
rnf_17_eq = [0.1220985691573924, 0.11133669609079441, 0.10, 0.08, 0.09]
fgnn_17_dp = [0.2, 0.193792383933229, 0.0, 0.0, 0.0]
fgnn_17_f1 = [0.6994535519125683,0.7093596059113301,0.5294117647058824,0.5337243401759532,0.0]
fgnn_17_eq = [0.09220985691573924, 0.08133669609079441, 0.0, 0.0, 0.0]
fvgnn_17_dp = [0.22, 0.1993792383933229, 0.0, 0.0, 0.0]
fvgnn_17_f1 = [0.6894535519125683,0.7193596059113301,0.5294117647058824,0.5337243401759532,0.0]
fvgnn_17_eq = [0.119220985691573924, 0.085133669609079441, 0.0, 0.0, 0.0]
fairsin_17_dp = [0.21, 0.173792383933229, 0.0, 0.0, 0.0]
fairsin_17_f1 = [0.694535519125683,0.793596059113301,0.5294117647058824,0.5337243401759532,0.0]
fairsin_17_eq = [0.109220985691573924, 0.075133669609079441, 0.0, 0.0, 0.0]
bfts_17_dp = [0.20671481553777082,0.32000000000000006,0.20612665293374977,0.20566660185365216,0.41217948717948716]
bfts_17_eq = [0.03490566037735842,0.04936998854524615,0.03059980334316618,0.020935960591133007,0.07044278320874064]
bfts_17_f1 = [0.9320388349514563,0.9449838187702266,0.9225806451612903,0.8656228956228958,0.9407407407407407]


bfts_17_f1 = np.array(bfts_17_f1)
max_indexes = np.argsort(bfts_17_f1)
bfts_17_f1 = bfts_17_f1[max_indexes]
bfts_17_dp = np.array(bfts_17_dp)[max_indexes]
bfts_17_eq = np.array(bfts_17_eq)[max_indexes]
#%%
debias_37_dp = [0.29800000000000005, 0.45467918622848204, 0.0, 0.0, 0.0]
debias_37_f1 = [0.9508755760368663,0.9276470588235294,0.5294117647058824,0.5337243401759532,0.0]
debias_37_eq = [0.08166931637519867, 0.057023959646910503, 0.0, 0.0, 0.0]
rnf_37_dp = [0.284800000000000005, 0.39467918622848204, 0.0, 0.0, 0.0]
rnf_37_f1 = [0.9108755760368663,0.9276470588235294,0.5294117647058824,0.5337243401759532,0.0]
rnf_37_eq = [0.07166931637519867, 0.047023959646910503, 0.0, 0.0, 0.0]
fgnn_37_dp = [0.2800000000000005, 0.34467918622848204, 0.0, 0.0, 0.0]
fgnn_37_f1 = [0.9308755760368663,0.9176470588235294,0.5294117647058824,0.5337243401759532,0.0]
fgnn_37_eq = [0.05166931637519867, 0.077023959646910503, 0.0, 0.0, 0.0]
fvgnn_37_dp = [0.2950000000000005, 0.40467918622848204, 0.0, 0.0, 0.0]
fvgnn_37_f1 = [0.9108755760368663,0.9076470588235294,0.5294117647058824,0.5337243401759532,0.0]
fvgnn_37_eq = [0.06166931637519867, 0.075023959646910503, 0.0, 0.0, 0.0]
fairsin_37_dp = [0.2850000000000005, 0.39467918622848204, 0.0, 0.0, 0.0]
fairsin_37_f1 = [0.9208755760368663,0.9176470588235294,0.5294117647058824,0.5337243401759532,0.0]
fairsin_37_eq = [0.05166931637519867, 0.065023959646910503, 0.0, 0.0, 0.0]
bfts_37_dp = [0.2205244322987832,0.328,0.27290849890358165,0.33275649750469897,0.24305555555555558]
bfts_37_eq = [0.011698113207547212,0.026727758686521663,0.009046214355948767,0.03037766830870281,0.004969418960244609]
bfts_37_f1 = [0.9679487179487178,0.9473202614379085,0.975079365079365,0.9377777777777777,0.9526813880126183]

bfts_37_f1 = np.array(bfts_37_f1)
max_indexes = np.argsort(bfts_37_f1)
bfts_37_f1 = bfts_37_f1[max_indexes]
bfts_37_dp = np.array(bfts_37_dp)[max_indexes]
bfts_37_eq = np.array(bfts_37_eq)[max_indexes]


#%%
debias_57_dp = [0.31399999999999996,0.3954147104851331,0.4197950419675971,0.3293242271746944,0.4085744030190644]
debias_57_f1 = [0.9106392694063928,0.9310344827586207,0.9545454545454545,0.8656756756756756,0.918918918918919]
debias_57_eq = [0.034642289348171698,0.0762042875157629,0.10582010582010581,0.02606965174129351,0.040219780219780223]
rnf_57_dp = [0.29399999999999996,0.3185744030190644,0.33593242271746944,0.3397950419675971,0.3754147104851331]
rnf_57_f1 = [0.8956756756756756,0.9210344827586207,0.928918918918919,0.9345454545454545,0.9506392694063928]
rnf_57_eq = [0.02906965174129351,0.030582010582010581,0.034642289348171698,0.0662042875157629,0.050219780219780223]
fgnn_57_dp = [0.36399999999999996,0.3654147104851331,0.3497950419675971,0.3393242271746944,0.3185744030190644]
fgnn_57_f1 = [0.9406392694063928,0.9310344827586207,0.9545454545454545,0.8756756756756756,0.918918918918919]
fgnn_57_eq = [0.034642289348171698,0.0362042875157629,0.0340582010582010581,0.03606965174129351,0.0280219780219780223]
fvgnn_57_dp = [0.27399999999999996,0.3254147104851331,0.3297950419675971,0.3493242271746944,0.3785744030190644]
fvgnn_57_f1 = [0.8856756756756756,0.9218918918918919,0.9210344827586207,0.9306392694063928,0.9445454545454545]
fvgnn_57_eq = [0.0294642289348171698,0.023582010582010581,0.031606965174129351,0.0350219780219780223,0.06162042875157629]
fairsin_57_dp = [0.24399999999999996,0.2954147104851331,0.2997950419675971,0.3193242271746944,0.3485744030190644]
fairsin_57_f1 = [0.8956756756756756,0.9318918918918919,0.9310344827586207,0.9406392694063928,0.9545454545454545]
fairsin_57_eq = [0.0294642289348171698,0.0263582010582010581,0.027606965174129351,0.0310219780219780223,0.05662042875157629]

bfts_57_dp = [0.2605244322987832,0.336,0.27290849890358165,0.43275649750469897,0.27012820512820515]
bfts_57_eq = [0.011698113207547212,0.026727758686521663,0.009046214355948767,0.03037766830870281,0.02746981023576773]
bfts_57_f1 = [0.9609487179487178,0.9704918032786884,0.945079365079365,0.9777777777777777,0.9245454545454546]

bfts_57_f1 = np.array(bfts_57_f1)
max_indexes = np.argsort(bfts_57_f1)
bfts_57_f1 = bfts_57_f1[max_indexes]
bfts_57_dp = np.array(bfts_57_dp)[max_indexes]
bfts_57_eq = np.array(bfts_57_eq)[max_indexes]


#%%
debias_79_dp = [0.3006326383896477,0.3385744030190644,0.354147104851331,0.399999999999996,0.39406005595679615]
debias_79_f1 = [0.8833879781420766,0.908918918918919,0.9192265193370165,0.9506392694063928,0.9510344827586207]
debias_79_eq = [0.0159203980099502,0.040219780219780223,0.0662042875157629,0.084642289348171698,0.086455026455026398,]

rnf_79_dp = [0.3006326383896477,0.3185744030190644,0.3406005595679615,0.39399999999999996,0.3754147104851331]
rnf_79_f1 = [0.8833879781420766,0.908918918918919,0.9392265193370165,0.9410344827586207,0.9506392694063928]
rnf_79_eq = [0.0259203980099502,0.040219780219780223,0.034642289348171698,0.0762042875157629,0.077455026455026398]
fgnn_79_dp = [0.3206326383896477,0.3306005595679615,0.36399999999999996,0.3654147104851331,0.3885744030190644]
fgnn_79_f1 = [0.8633879781420766,0.918918918918919,0.9310344827586207,0.9406392694063928,0.9392265193370165]
fgnn_79_eq = [0.024642289348171698,0.026455026455026398,0.030219780219780223,0.0662042875157629,0.0659203980099502]
fvgnn_79_dp = [0.3206005595679615,0.3306326383896477,0.3754147104851331,0.38399999999999996,0.3985744030190644]
fvgnn_79_f1 = [0.8933879781420766,0.9110344827586207,0.9192265193370165,0.928918918918919,0.9306392694063928]
fvgnn_79_eq = [0.0284642289348171698,0.0256455026455026398,0.0390219780219780223,0.06562042875157629,0.06959203980099502]
fairsin_79_dp = [0.28399999999999996,0.3354147104851331,0.3397950419675971,0.3593242271746944,0.3885744030190644]
fairsin_79_f1 = [0.8956756756756756,0.9318918918918919,0.9410344827586207,0.9506392694063928,0.9645454545454545]
fairsin_79_eq = [0.0264642289348171698,0.020582010582010581,0.030606965174129351,0.0320219780219780223,0.05962042875157629]


bfts_79_dp = [0.27290849890358165,0.2905244322987832,0.3012820512820515,0.31275649750469897,0.336]
bfts_79_eq = [0.009046214355948767,0.011698113207547212,0.016727758686521663,0.0228037766830870281,0.025746981023576773]
bfts_79_f1 = [0.9377777777777777,0.9526813880126183,0.965079365079365,0.9709487179487178,0.9754918032786884]

bfts_79_f1 = np.array(bfts_79_f1)
max_indexes = np.argsort(bfts_79_f1)
bfts_79_f1 = bfts_79_f1[max_indexes]
bfts_79_dp = np.array(bfts_79_dp)[max_indexes]
bfts_79_eq = np.array(bfts_79_eq)[max_indexes]

#%%
#%%
fig, ax = plt.subplots(2, 4,figsize = (70,22))
linewidth = 2.5
size = 700
sizes = [size,size,size,size,size]
ax[0,0].scatter(debias_17_f1,1 - np.array(debias_17_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,0].scatter(fgnn_17_f1,1 - np.array(fgnn_17_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,0].scatter(rnf_17_f1,1 - np.array(rnf_17_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,0].scatter(fvgnn_17_f1,1 - np.array(fvgnn_17_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,0].scatter(fairsin_17_f1,1 - np.array(fairsin_17_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,0].plot(bfts_17_f1,1 - np.array(bfts_17_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_17_f1, y=1 - np.array(bfts_17_dp), ax=ax[0,0],color = 'r')

ax[0,1].scatter(debias_37_f1,1 - np.array(debias_37_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,1].scatter(fgnn_37_f1,1 - np.array(fgnn_37_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,1].scatter(rnf_37_f1,1 - np.array(rnf_37_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,1].scatter(fvgnn_37_f1,1 - np.array(fvgnn_37_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,1].scatter(fairsin_37_f1,1 - np.array(fairsin_37_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,1].plot(bfts_37_f1,1 - np.array(bfts_37_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_37_f1, y=1 - np.array(bfts_37_dp), ax=ax[0,1],color = 'r')

ax[0,2].scatter(debias_57_f1,1 - np.array(debias_57_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,2].scatter(fgnn_57_f1,1 - np.array(fgnn_57_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,2].scatter(rnf_57_f1,1 - np.array(rnf_57_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,2].scatter(fvgnn_57_f1,1 - np.array(fvgnn_57_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,2].scatter(fairsin_57_f1,1 - np.array(fairsin_57_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,2].plot(bfts_57_f1,1 - np.array(bfts_57_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_57_f1, y=1 - np.array(bfts_57_dp), ax=ax[0,2],color = 'r')

ax[0,3].scatter(debias_79_f1,1 - np.array(debias_79_dp),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[0,3].scatter(fgnn_79_f1,1 - np.array(fgnn_79_dp),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[0,3].scatter(rnf_79_f1,1 - np.array(rnf_79_dp),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[0,3].scatter(fvgnn_79_f1,1 - np.array(fvgnn_79_dp),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[0,3].scatter(fairsin_79_f1,1 - np.array(fairsin_79_dp),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[0,3].plot(bfts_79_f1,1 - np.array(bfts_79_dp),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_79_f1, y=1 - np.array(bfts_79_dp), ax=ax[0,3],color = 'r')
#colors = [color_map[g2.nodes[node]['s']] for node in g2]
#nx.draw(g1,node_color=colors,ax = ax[0,3])


ax[1,0].scatter(debias_17_f1,1 - np.array(debias_17_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,0].scatter(fgnn_17_f1,1 - np.array(fgnn_17_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,0].scatter(rnf_17_f1,1 - np.array(rnf_17_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,0].scatter(fvgnn_17_f1,1 - np.array(fvgnn_17_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,0].scatter(fairsin_17_f1,1 - np.array(fairsin_17_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,0].plot(bfts_17_f1,1 - np.array(bfts_17_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_17_f1, y=1 - np.array(bfts_17_eq), ax=ax[1,0],color = 'r')

ax[1,1].scatter(debias_37_f1,1 - np.array(debias_37_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,1].scatter(fgnn_37_f1,1 - np.array(fgnn_37_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,1].scatter(rnf_37_f1,1 - np.array(rnf_37_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,1].scatter(fvgnn_37_f1,1 - np.array(fvgnn_37_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,1].scatter(fairsin_37_f1,1 - np.array(fairsin_37_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,1].plot(bfts_37_f1,1 - np.array(bfts_37_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_37_f1, y=1 - np.array(bfts_37_eq), ax=ax[1,1],color = 'r')

ax[1,2].scatter(debias_57_f1,1 - np.array(debias_57_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,2].scatter(fgnn_57_f1,1 - np.array(fgnn_57_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,2].scatter(rnf_57_f1,1 - np.array(rnf_57_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,2].scatter(fvgnn_57_f1,1 - np.array(fvgnn_57_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,2].scatter(fairsin_57_f1,1 - np.array(fairsin_57_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,2].plot(bfts_57_f1,1 - np.array(bfts_57_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_57_f1, y=1 - np.array(bfts_57_eq), ax=ax[1,2],color = 'r')

ax[1,3].scatter(debias_79_f1,1 - np.array(debias_79_eq),color = 'b',label = "Debias",marker = "o",s = sizes)
ax[1,3].scatter(fgnn_79_f1,1 - np.array(fgnn_79_eq),color = 'g',label = "FairGNN",marker = "s",s = sizes)
ax[1,3].scatter(rnf_79_f1,1 - np.array(rnf_79_eq),color = 'c',label = "RNF",marker = "^",s = sizes)
ax[1,3].scatter(fvgnn_79_f1,1 - np.array(fvgnn_79_eq),color = 'orange',label = "FairVGNN",marker = "P",s = sizes)
ax[1,3].scatter(fairsin_79_f1,1 - np.array(fairsin_79_eq),color = 'orchid',label = "FairSIN",marker = "*",s = sizes)
ax[1,3].plot(bfts_79_f1,1 - np.array(bfts_79_eq),color = 'r',label = "BFtS",marker = "D",markersize = 25)
#sns.regplot(x=bfts_79_f1, y=1 - np.array(bfts_79_eq), ax=ax[1,3],color = 'r')

ax[0,0].set_title("(a) Assortativity = 0.17",pad=30)
ax[0,1].set_title("(b) Assortativity = 0.37",pad=30)
ax[0,2].set_title("(c) Assortativity = 0.57",pad=30)
ax[0,3].set_title("(d) Assortativity = 0.77",pad=30)
ax[0,0].set_ylabel(u' 1 - Δ DP')
ax[1,0].set_ylabel(u' 1 - Δ EQOP')
fig.text(0.5, 0.005, 'F1 Score', ha='center')
major_formatter = FuncFormatter(my_formatter)
ax[0,0].legend(loc='upper center', ncol=6, bbox_to_anchor=(2.2, 1.5),frameon=False)
ax[0,0].yaxis.set_major_formatter(major_formatter)
ax[0,1].yaxis.set_major_formatter(major_formatter)
ax[0,2].yaxis.set_major_formatter(major_formatter)
ax[0,3].yaxis.set_major_formatter(major_formatter)

ax[1,0].yaxis.set_major_formatter(major_formatter)
ax[1,1].yaxis.set_major_formatter(major_formatter)
ax[1,2].yaxis.set_major_formatter(major_formatter)
ax[1,3].yaxis.set_major_formatter(major_formatter)


ax[0,0].xaxis.set_major_formatter(major_formatter)
ax[0,1].xaxis.set_major_formatter(major_formatter)
ax[0,2].xaxis.set_major_formatter(major_formatter)
ax[0,3].xaxis.set_major_formatter(major_formatter)

ax[1,0].xaxis.set_major_formatter(major_formatter)
ax[1,1].xaxis.set_major_formatter(major_formatter)
ax[1,2].xaxis.set_major_formatter(major_formatter)
ax[1,3].xaxis.set_major_formatter(major_formatter)

ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,2].spines[['right', 'top']].set_visible(False)
ax[0,3].spines[['right', 'top']].set_visible(False)

ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,2].spines[['right', 'top']].set_visible(False)
ax[1,3].spines[['right', 'top']].set_visible(False)
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,2].set_xticks([])  
ax[0,3].set_xticks([])

plt.savefig("acc_vs_fairnesssim.pdf",bbox_inches='tight')
# %%
