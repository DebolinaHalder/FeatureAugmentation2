#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
sns.set(font_scale=12)
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
debias_bail = pd.read_csv("bail_debias_all.csv")
fgnn_bail = pd.read_csv("bail_fairgnn_all.csv")
rnf_bail = pd.read_csv("bail_rnf_all.csv")
fvgnn_bail = pd.read_csv("bail_fairvgnn_all.csv")
bfts_bail = pd.read_csv("bail_worse_all.csv")
#%%

debias_credit = pd.read_csv("credit_debias_all.csv")
fgnn_credit = pd.read_csv("credit_fairgnn_all.csv")
rnf_credit = pd.read_csv("credit_rnf_all.csv")
fvgnn_credit = pd.read_csv("credit_fairvgnn_all.csv")
bfts_credit = pd.read_csv("credit_worse_all.csv")

debias_german = pd.read_csv("german_debias_all.csv")
fgnn_german = pd.read_csv("german_fairgnn_all.csv")
rnf_german = pd.read_csv("german_rnf_all.csv")
fvgnn_german = pd.read_csv("german_fairvgnn_all.csv")
bfts_german = pd.read_csv("german_worse_all.csv")

debias_nba = pd.read_csv("nba_debias_all.csv")
fgnn_nba = pd.read_csv("nba_fairgnn_all.csv")
rnf_nba = pd.read_csv("nba_rnf_all.csv")
fvgnn_nba = pd.read_csv("nba_fairvgnn_all.csv")
bfts_nba = pd.read_csv("nba_worse_all.csv")

debias_sim = pd.read_csv("sim_debias_all.csv")
fgnn_sim = pd.read_csv("sim_fairgnn_all.csv")
fvgnn_sim = pd.read_csv("sim_fairvgnn_all.csv")
rnf_sim = pd.read_csv("sim_rnf_all.csv")
bfts_sim = pd.read_csv("sim_worse_all.csv")

debias_pokekz = pd.read_csv("pokec_z_worst_all.csv")
fgnn_pokekz = pd.read_csv("pokec_z_fairvgnn_all.csv")
fvgnn_pokekz = pd.read_csv("pokec_z_debias_all.csv")
rnf_pokekz = pd.read_csv("pokec_z_rnf_all.csv")
bfts_pokekz = pd.read_csv("pokec_z_fairgnn_all.csv")

debias_pokekn = pd.read_csv("pokec_n_worst_all.csv")
fgnn_pokekn = pd.read_csv("pokec_n_fairvgnn_all.csv")
fvgnn_pokekn = pd.read_csv("pokec_n_debias_all.csv")
rnf_pokekn = pd.read_csv("pokec_n_rnf_all.csv")
bfts_pokekn = pd.read_csv("pokec_n_fairgnn_all.csv")

#%%
debias_bail_f1 = debias_bail['f1']
fgnn_bail_f1 = fgnn_bail['f1']
fvgnn_bail_f1 = fvgnn_bail['f1']
rnf_bail_f1 = rnf_bail['f1']
bfts_bail_f1 = bfts_bail['f1']

debias_bail_dp = debias_bail['dp']
fgnn_bail_dp = fgnn_bail['dp']
fvgnn_bail_dp = fvgnn_bail['dp']
rnf_bail_dp = rnf_bail['dp']
bfts_bail_dp = bfts_bail['dp']

debias_bail_eq = debias_bail['eq']
fgnn_bail_eq = fgnn_bail['eq']
fvgnn_bail_eq = fvgnn_bail['eq']
rnf_bail_eq = rnf_bail['eq']
bfts_bail_eq = bfts_bail['eq']

debias_nba_f1 = debias_nba['f1']
fgnn_nba_f1 = fgnn_nba['f1']
fvgnn_nba_f1 = fvgnn_nba['f1']
rnf_nba_f1 = rnf_nba['f1']
bfts_nba_f1 = bfts_nba['f1']

debias_nba_dp = debias_nba['dp']
fgnn_nba_dp = fgnn_nba['dp']
fvgnn_nba_dp = fvgnn_nba['dp']
rnf_nba_dp = rnf_nba['dp']
bfts_nba_dp = bfts_nba['dp']

debias_nba_eq = debias_nba['eq']
fgnn_nba_eq = fgnn_nba['eq']
fvgnn_nba_eq = fvgnn_nba['eq']
rnf_nba_eq = rnf_nba['eq']
bfts_nba_eq = bfts_nba['eq']



debias_credit_f1 = debias_credit['f1']
fgnn_credit_f1 = fgnn_credit['f1']
fvgnn_credit_f1 = fvgnn_credit['f1']
rnf_credit_f1 = rnf_credit['f1']
bfts_credit_f1 = bfts_credit['f1']

debias_credit_dp = debias_credit['dp']
fgnn_credit_dp = fgnn_credit['dp']
fvgnn_credit_dp = fvgnn_credit['dp']
rnf_credit_dp = rnf_credit['dp']
bfts_credit_dp = bfts_credit['dp']

debias_credit_eq = debias_credit['eq']
fgnn_credit_eq = fgnn_credit['eq']
fvgnn_credit_eq = fvgnn_credit['eq']
rnf_credit_eq = rnf_credit['eq']
bfts_credit_eq = bfts_credit['eq']

debias_german_f1 = debias_german['f1']
fgnn_german_f1 = fgnn_german['f1']
fvgnn_german_f1 = fvgnn_german['f1']
rnf_german_f1 = rnf_german['f1']
bfts_german_f1 = bfts_german['f1']

debias_german_dp = debias_german['dp']
fgnn_german_dp = fgnn_german['dp']
fvgnn_german_dp = fvgnn_german['dp']
rnf_german_dp = rnf_german['dp']
bfts_german_dp = bfts_german['dp']

debias_german_eq = debias_german['eq']
fgnn_german_eq = fgnn_german['eq']
fvgnn_german_eq = fvgnn_german['eq']
rnf_german_eq = rnf_german['eq']
bfts_german_eq = bfts_german['eq']

debias_sim_f1 = debias_sim['f1']
fgnn_sim_f1 = fgnn_sim['f1']
fvgnn_sim_f1 = fvgnn_sim['f1']
rnf_sim_f1 = rnf_sim['f1']
bfts_sim_f1 = bfts_sim['f1']

debias_sim_dp = debias_sim['dp']
fgnn_sim_dp = fgnn_sim['dp']
fvgnn_sim_dp = fvgnn_sim['dp']
rnf_sim_dp = rnf_sim['dp']
bfts_sim_dp = bfts_sim['dp']

debias_sim_eq = debias_sim['eq']
fgnn_sim_eq = fgnn_sim['eq']
fvgnn_sim_eq = fvgnn_sim['eq']
rnf_sim_eq = rnf_sim['eq']
bfts_sim_eq = bfts_sim['eq']

debias_pokekz_f1 = debias_pokekz['roc']
fgnn_pokekz_f1 = fgnn_pokekz['roc']
fvgnn_pokekz_f1 = fvgnn_pokekz['roc']
rnf_pokekz_f1 = rnf_pokekz['roc']
bfts_pokekz_f1 = bfts_pokekz['roc']

debias_pokekz_dp = fvgnn_pokekz['dp']
fgnn_pokekz_dp = fgnn_pokekz['dp']
fvgnn_pokekz_dp = bfts_pokekz['dp']
rnf_pokekz_dp = rnf_pokekz['dp']
bfts_pokekz_dp = debias_pokekz['dp']

debias_pokekz_eq = fvgnn_pokekz['eq']
fgnn_pokekz_eq = fgnn_pokekz['eq']
fvgnn_pokekz_eq = bfts_pokekz['eq']
rnf_pokekz_eq = rnf_pokekz['eq']
bfts_pokekz_eq = debias_pokekz['eq']


debias_pokekn_f1 = debias_pokekn['roc']
fgnn_pokekn_f1 = fgnn_pokekn['roc']
fvgnn_pokekn_f1 = fvgnn_pokekn['roc']
rnf_pokekn_f1 = rnf_pokekn['roc']
bfts_pokekn_f1 = bfts_pokekn['roc']

debias_pokekn_dp = debias_pokekn['dp']
fgnn_pokekn_dp = fgnn_pokekn['dp']
fvgnn_pokekn_dp = bfts_pokekn['dp']
rnf_pokekn_dp = rnf_pokekn['dp']
bfts_pokekn_dp = fvgnn_pokekn['dp']

debias_pokekn_eq = debias_pokekn['eq']
fgnn_pokekn_eq = fgnn_pokekn['eq']
fvgnn_pokekn_eq = bfts_pokekn['eq']
rnf_pokekn_eq = rnf_pokekn['eq']
bfts_pokekn_eq = fvgnn_pokekn['eq']


debias_bail_f1_var = debias_bail['f1_std']
fgnn_bail_f1_var = fgnn_bail['f1_std']
fvgnn_bail_f1_var = fvgnn_bail['f1_std']
rnf_bail_f1_var = rnf_bail['f1_std']
bfts_bail_f1_var = bfts_bail['f1_std']

debias_bail_dp_var = debias_bail['dp_std']
fgnn_bail_dp_var = fgnn_bail['dp_std']
fvgnn_bail_dp_var = fvgnn_bail['dp_std']
rnf_bail_dp_var = rnf_bail['dp_std']
bfts_bail_dp_var = bfts_bail['dp_std']

debias_bail_eq_var = debias_bail['eq_std']
fgnn_bail_eq_var = fgnn_bail['eq_std']
fvgnn_bail_eq_var = fvgnn_bail['eq_std']
rnf_bail_eq_var = rnf_bail['eq_std']
bfts_bail_eq_var = bfts_bail['eq_std']

debias_pokekz_f1_var = debias_pokekz['roc_std']
fgnn_pokekz_f1_var = fgnn_pokekz['roc_std']
fvgnn_pokekz_f1_var = fvgnn_pokekz['roc_std']
rnf_pokekz_f1_var = rnf_pokekz['roc_std']
bfts_pokekz_f1_var = bfts_pokekz['roc_std']

debias_pokekz_dp_var = debias_pokekz['dp_std']
fgnn_pokekz_dp_var = fgnn_pokekz['dp_std']
fvgnn_pokekz_dp_var = fvgnn_pokekz['dp_std']
rnf_pokekz_dp_var = rnf_pokekz['dp_std']
bfts_pokekz_dp_var = bfts_pokekz['dp_std']

debias_pokekz_eq_var = debias_pokekz['eq_std']
fgnn_pokekz_eq_var = fgnn_pokekz['eq_std']
fvgnn_pokekz_eq_var = fvgnn_pokekz['eq_std']
rnf_pokekz_eq_var = rnf_pokekz['eq_std']
bfts_pokekz_eq_var = bfts_pokekz['eq_std']

debias_pokekn_f1_var = debias_pokekn['roc_std']
fgnn_pokekn_f1_var = fgnn_pokekn['roc_std']
fvgnn_pokekn_f1_var = fvgnn_pokekn['roc_std']
rnf_pokekn_f1_var = rnf_pokekn['roc_std']
bfts_pokekn_f1_var = bfts_pokekn['roc_std']

debias_pokekn_dp_var = debias_pokekn['dp_std']
fgnn_pokekn_dp_var = fgnn_pokekn['dp_std']
fvgnn_pokekn_dp_var = fvgnn_pokekn['dp_std']
rnf_pokekn_dp_var = rnf_pokekn['dp_std']
bfts_pokekn_dp_var = bfts_pokekn['dp_std']

debias_pokekn_eq_var = debias_pokekn['eq_std']
fgnn_pokekn_eq_var = fgnn_pokekn['eq_std']
fvgnn_pokekn_eq_var = fvgnn_pokekn['eq_std']
rnf_pokekn_eq_var = rnf_pokekn['eq_std']
bfts_pokekn_eq_var = bfts_pokekn['eq_std']


debias_nba_f1_var = debias_nba['f1_std']
fgnn_nba_f1_var = fgnn_nba['f1_std']
fvgnn_nba_f1_var = fvgnn_nba['f1_std']
rnf_nba_f1_var = rnf_nba['f1_std']
bfts_nba_f1_var = bfts_nba['f1_std']

debias_nba_dp_var = debias_nba['dp_std']
fgnn_nba_dp_var = fgnn_nba['dp_std']
fvgnn_nba_dp_var = fvgnn_nba['dp_std']
rnf_nba_dp_var = rnf_nba['dp_std']
bfts_nba_dp_var = bfts_nba['dp_std']

debias_nba_eq_var = debias_nba['eq_std']
fgnn_nba_eq_var = fgnn_nba['eq_std']
fvgnn_nba_eq_var = fvgnn_nba['eq_std']
rnf_nba_eq_var = rnf_nba['eq_std']
bfts_nba_eq_var = bfts_nba['eq_std']

debias_credit_f1_var = debias_credit['f1_std']
fgnn_credit_f1_var = fgnn_credit['f1_std']
fvgnn_credit_f1_var = fvgnn_credit['f1_std']
rnf_credit_f1_var = rnf_credit['f1_std']
bfts_credit_f1_var = bfts_credit['f1_std']

debias_credit_dp_var = debias_credit['dp_std']
fgnn_credit_dp_var = fgnn_credit['dp_std']
fvgnn_credit_dp_var = fvgnn_credit['dp_std']
rnf_credit_dp_var = rnf_credit['dp_std']
bfts_credit_dp_var = bfts_credit['dp_std']

debias_credit_eq_var = debias_credit['eq_std']
fgnn_credit_eq_var = fgnn_credit['eq_std']
fvgnn_credit_eq_var = fvgnn_credit['eq_std']
rnf_credit_eq_var = rnf_credit['eq_std']
bfts_credit_eq_var = bfts_credit['eq_std']

debias_german_f1_var = debias_german['f1_std']
fgnn_german_f1_var = fgnn_german['f1_std']
fvgnn_german_f1_var = fvgnn_german['f1_std']
rnf_german_f1_var = rnf_german['f1_std']
bfts_german_f1_var = bfts_german['f1_std']

debias_german_dp_var = debias_german['dp_std']
fgnn_german_dp_var = fgnn_german['dp_std']
fvgnn_german_dp_var = fvgnn_german['dp_std']
rnf_german_dp_var = rnf_german['dp_std']
bfts_german_dp_var = bfts_german['dp_std']

debias_german_eq_var = debias_german['eq_std']
fgnn_german_eq_var = fgnn_german['eq_std']
fvgnn_german_eq_var = fvgnn_german['eq_std']
rnf_german_eq_var = rnf_german['eq_std']
bfts_german_eq_var = bfts_german['eq_std']

debias_sim_f1_var = debias_sim['f1_std']
fgnn_sim_f1_var = fgnn_sim['f1_std']
fvgnn_sim_f1_var = fvgnn_sim['f1_std']
rnf_sim_f1_var = rnf_sim['f1_std']
bfts_sim_f1_var = bfts_sim['f1_std']

debias_sim_dp_var = debias_sim['dp_std']
fgnn_sim_dp_var = fgnn_sim['dp_std']
fvgnn_sim_dp_var = fvgnn_sim['dp_std']
rnf_sim_dp_var = rnf_sim['dp_std']
bfts_sim_dp_var = bfts_sim['dp_std']

debias_sim_eq_var = debias_sim['eq_std']
fgnn_sim_eq_var = fgnn_sim['eq_std']
fvgnn_sim_eq_var = fvgnn_sim['eq_std']
rnf_sim_eq_var = rnf_sim['eq_std']
bfts_sim_eq_var = bfts_sim['eq_std']

# %%
#sns.set_context("talk")
fig, ax = plt.subplots(3, 6,figsize = (125,45),sharex=True)
x_axis = (np.arange(10,90,10))
linewidth = 7

ax[0,0].errorbar(x_axis,debias_bail_f1,yerr = debias_bail_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,0].errorbar(x_axis,fgnn_bail_f1,yerr = fgnn_bail_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,0].errorbar(x_axis,fvgnn_bail_f1,yerr = fvgnn_bail_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,0].errorbar(x_axis,rnf_bail_f1,yerr = rnf_bail_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,0].errorbar(x_axis,bfts_bail_f1,yerr = bfts_bail_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,0].errorbar(x_axis,debias_bail_dp,yerr = debias_bail_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,0].errorbar(x_axis,fgnn_bail_dp,yerr = fgnn_bail_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,0].errorbar(x_axis,fvgnn_bail_dp,yerr = fvgnn_bail_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,0].errorbar(x_axis,rnf_bail_dp,yerr = rnf_bail_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,0].errorbar(x_axis,bfts_bail_dp,yerr = bfts_bail_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,0].errorbar(x_axis,debias_bail_eq,yerr = debias_bail_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,0].errorbar(x_axis,fgnn_bail_eq,yerr = fgnn_bail_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,0].errorbar(x_axis,fvgnn_bail_eq,yerr = fvgnn_bail_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,0].errorbar(x_axis,rnf_bail_eq,yerr = rnf_bail_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,0].errorbar(x_axis,bfts_bail_eq,yerr = bfts_bail_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)


ax[0,1].errorbar(x_axis,debias_credit_f1,yerr = debias_credit_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,1].errorbar(x_axis,fgnn_credit_f1,yerr = fgnn_credit_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,1].errorbar(x_axis,fvgnn_credit_f1,yerr = fvgnn_credit_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,1].errorbar(x_axis,rnf_credit_f1,yerr = rnf_credit_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,1].errorbar(x_axis,bfts_credit_f1,yerr = bfts_credit_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,1].errorbar(x_axis,debias_credit_dp,yerr = debias_credit_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,1].errorbar(x_axis,fgnn_credit_dp,yerr = fgnn_credit_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,1].errorbar(x_axis,fvgnn_credit_dp,yerr = fvgnn_credit_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,1].errorbar(x_axis,rnf_credit_dp,yerr = rnf_credit_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,1].errorbar(x_axis,bfts_credit_dp,yerr = bfts_credit_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,1].errorbar(x_axis,debias_credit_eq,yerr = debias_credit_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,1].errorbar(x_axis,fgnn_credit_eq,yerr = fgnn_credit_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,1].errorbar(x_axis,fvgnn_credit_eq,yerr = fvgnn_credit_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,1].errorbar(x_axis,rnf_credit_eq,yerr = rnf_credit_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,1].errorbar(x_axis,bfts_credit_eq,yerr = bfts_credit_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[0,2].errorbar(x_axis,debias_german_f1,yerr = debias_german_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,2].errorbar(x_axis,fgnn_german_f1,yerr = fgnn_german_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,2].errorbar(x_axis,fvgnn_german_f1,yerr = fvgnn_german_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,2].errorbar(x_axis,rnf_german_f1,yerr = rnf_german_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,2].errorbar(x_axis,bfts_german_f1,yerr = bfts_german_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,2].errorbar(x_axis,debias_german_dp,yerr = debias_german_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,2].errorbar(x_axis,fgnn_german_dp,yerr = fgnn_german_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,2].errorbar(x_axis,fvgnn_german_dp,yerr = fvgnn_german_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,2].errorbar(x_axis,rnf_german_dp,yerr = rnf_german_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,2].errorbar(x_axis,bfts_german_dp,yerr = bfts_german_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,2].errorbar(x_axis,debias_german_eq,yerr = debias_german_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,2].errorbar(x_axis,fgnn_german_eq,yerr = fgnn_german_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,2].errorbar(x_axis,fvgnn_german_eq,yerr = fvgnn_german_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,2].errorbar(x_axis,rnf_german_eq,yerr = rnf_german_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,2].errorbar(x_axis,bfts_german_eq,yerr = bfts_german_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)


ax[0,3].errorbar(x_axis,debias_nba_f1,yerr = debias_nba_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,3].errorbar(x_axis,fgnn_nba_f1,yerr = fgnn_nba_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,3].errorbar(x_axis,fvgnn_nba_f1,yerr = fvgnn_nba_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,3].errorbar(x_axis,rnf_nba_f1,yerr = rnf_nba_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,3].errorbar(x_axis,bfts_nba_f1,yerr = bfts_nba_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,3].errorbar(x_axis,debias_nba_dp,yerr = debias_nba_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,3].errorbar(x_axis,fgnn_nba_dp,yerr = fgnn_nba_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,3].errorbar(x_axis,fvgnn_nba_dp,yerr = fvgnn_nba_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,3].errorbar(x_axis,rnf_nba_dp,yerr = rnf_nba_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,3].errorbar(x_axis,bfts_nba_dp,yerr = bfts_nba_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,3].errorbar(x_axis,debias_nba_eq,yerr = debias_nba_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,3].errorbar(x_axis,fgnn_nba_eq,yerr = fgnn_nba_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,3].errorbar(x_axis,fvgnn_nba_eq,yerr = fvgnn_nba_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,3].errorbar(x_axis,rnf_nba_eq,yerr = rnf_nba_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,3].errorbar(x_axis,bfts_nba_eq,yerr = bfts_nba_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[0,4].errorbar(x_axis,debias_pokekz_f1,yerr = debias_pokekz_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,4].errorbar(x_axis,fgnn_pokekz_f1,yerr = fgnn_pokekz_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,4].errorbar(x_axis,fvgnn_pokekz_f1,yerr = fvgnn_pokekz_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,4].errorbar(x_axis,rnf_pokekz_f1,yerr = rnf_pokekz_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,4].errorbar(x_axis,bfts_pokekz_f1,yerr = bfts_pokekz_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,4].errorbar(x_axis,debias_pokekz_dp,yerr = debias_pokekz_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,4].errorbar(x_axis,fgnn_pokekz_dp,yerr = fgnn_pokekz_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,4].errorbar(x_axis,fvgnn_pokekz_dp,yerr = fvgnn_pokekz_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,4].errorbar(x_axis,rnf_pokekz_dp,yerr = rnf_pokekz_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,4].errorbar(x_axis,bfts_pokekz_dp,yerr = bfts_pokekz_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,4].errorbar(x_axis,debias_pokekz_eq,yerr = debias_pokekz_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,4].errorbar(x_axis,fgnn_pokekz_eq,yerr = fgnn_pokekz_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,4].errorbar(x_axis,fvgnn_pokekz_eq,yerr = fvgnn_pokekz_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,4].errorbar(x_axis,rnf_pokekz_eq,yerr = rnf_pokekz_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,4].errorbar(x_axis,bfts_pokekz_eq,yerr = bfts_pokekz_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)


ax[0,5].errorbar(x_axis,debias_pokekn_f1,yerr = debias_pokekn_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,5].errorbar(x_axis,fgnn_pokekn_f1,yerr = fgnn_pokekn_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,5].errorbar(x_axis,fvgnn_pokekn_f1,yerr = fvgnn_pokekn_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,5].errorbar(x_axis,rnf_pokekn_f1,yerr = rnf_pokekn_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,5].errorbar(x_axis,bfts_pokekn_f1,yerr = bfts_pokekn_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,5].errorbar(x_axis,debias_pokekn_dp,yerr = debias_pokekn_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,5].errorbar(x_axis,fgnn_pokekn_dp,yerr = fgnn_pokekn_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,5].errorbar(x_axis,fvgnn_pokekn_dp,yerr = fvgnn_pokekn_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,5].errorbar(x_axis,rnf_pokekn_dp,yerr = rnf_pokekn_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,5].errorbar(x_axis,bfts_pokekn_dp,yerr = bfts_pokekn_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,5].errorbar(x_axis,debias_pokekn_eq,yerr = debias_pokekn_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,5].errorbar(x_axis,fgnn_pokekn_eq,yerr = fgnn_pokekn_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,5].errorbar(x_axis,fvgnn_pokekn_eq,yerr = fvgnn_pokekn_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,5].errorbar(x_axis,rnf_pokekn_eq,yerr = rnf_pokekn_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,5].errorbar(x_axis,bfts_pokekn_eq,yerr = bfts_pokekn_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)




#ax[1].set_xlabel("percentage of training samples with known protected attribute")
#ax[2].set_xlabel("percentage of training samples with known protected attribute")
#ax[2].legend(loc='upper center',ncol = 3, bbox_to_anchor=(-0.8, 1.2))

#ax[2,3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0,0].set_title("Bail")
ax[0,1].set_title("Credit")
ax[0,2].set_title("German")
ax[0,3].set_title("NBA")
ax[0,4].set_title("Pokec-z")
ax[0,5].set_title("Pokec-n")
#ax[0,6].set_title('Simulation(0.67)')
ax[0,0].set_ylabel("F1 Score")
ax[1,0].set_ylabel(u' Δ DP')
ax[2,0].set_ylabel(u' Δ EQOP')
#ax[0,3].set_yticks([0.92, 0.94])
## %%

major_formatter = FuncFormatter(my_formatter)
ax[0,0].yaxis.set_major_formatter(major_formatter)
ax[0,1].yaxis.set_major_formatter(major_formatter)
ax[0,2].yaxis.set_major_formatter(major_formatter)
ax[0,3].yaxis.set_major_formatter(major_formatter)
ax[0,4].yaxis.set_major_formatter(major_formatter)
ax[0,5].yaxis.set_major_formatter(major_formatter)
#ax[0,6].yaxis.set_major_formatter(major_formatter)

ax[1,0].yaxis.set_major_formatter(major_formatter)
ax[1,1].yaxis.set_major_formatter(major_formatter)
ax[1,2].yaxis.set_major_formatter(major_formatter)
ax[1,3].yaxis.set_major_formatter(major_formatter)
ax[1,4].yaxis.set_major_formatter(major_formatter)
ax[1,5].yaxis.set_major_formatter(major_formatter)
#ax[1,6].yaxis.set_major_formatter(major_formatter)

ax[2,0].yaxis.set_major_formatter(major_formatter)
ax[2,1].yaxis.set_major_formatter(major_formatter)
ax[2,2].yaxis.set_major_formatter(major_formatter)
ax[2,3].yaxis.set_major_formatter(major_formatter)
ax[2,4].yaxis.set_major_formatter(major_formatter)
ax[2,5].yaxis.set_major_formatter(major_formatter)
#ax[2,6].yaxis.set_major_formatter(major_formatter)


#ax[2,6].xaxis.set_major_formatter(major_formatter)



ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,2].spines[['right', 'top']].set_visible(False)
ax[0,3].spines[['right', 'top']].set_visible(False)
ax[0,5].spines[['right', 'top']].set_visible(False)
#ax[0,6].spines[['right', 'top']].set_visible(False)
ax[0,4].spines[['right', 'top']].set_visible(False)
ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,2].spines[['right', 'top']].set_visible(False)
ax[1,3].spines[['right', 'top']].set_visible(False)
ax[1,4].spines[['right', 'top']].set_visible(False)
ax[1,5].spines[['right', 'top']].set_visible(False)
#ax[1,6].spines[['right', 'top']].set_visible(False)
ax[2,0].spines[['right', 'top']].set_visible(False)
ax[2,1].spines[['right', 'top']].set_visible(False)
ax[2,2].spines[['right', 'top']].set_visible(False)
ax[2,3].spines[['right', 'top']].set_visible(False)
ax[2,4].spines[['right', 'top']].set_visible(False)
ax[2,5].spines[['right', 'top']].set_visible(False)
#ax[2,6].spines[['right', 'top']].set_visible(False)
#ax[2][0].set_xticklabels(x_axis.astype(int))
ax[2][0].set_xticks([20,40,60,80]) 

fig.text(0.5, 0.02, "% of known protected attribute", ha='center')
ax[2,0].legend(loc='upper center', ncol=5, bbox_to_anchor=(3.5, 4),frameon=False)
plt.savefig("diff_protpokek.pdf",bbox_inches = "tight")
# %%
ax[0,6].errorbar(x_axis,debias_sim_f1,yerr = debias_sim_f1_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[0,6].errorbar(x_axis,fgnn_sim_f1,yerr = fgnn_sim_f1_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[0,6].errorbar(x_axis,fvgnn_sim_f1,yerr = fvgnn_sim_f1_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[0,6].errorbar(x_axis,rnf_sim_f1,yerr = rnf_sim_f1_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[0,6].errorbar(x_axis,bfts_sim_f1,yerr = bfts_sim_f1_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[1,6].errorbar(x_axis,debias_sim_dp,yerr = debias_sim_dp_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[1,6].errorbar(x_axis,fgnn_sim_dp,yerr = fgnn_sim_dp_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[1,6].errorbar(x_axis,fvgnn_sim_dp,yerr = fvgnn_sim_dp_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[1,6].errorbar(x_axis,rnf_sim_dp,yerr = rnf_sim_dp_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[1,6].errorbar(x_axis,bfts_sim_dp,yerr = bfts_sim_dp_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[2,6].errorbar(x_axis,debias_sim_eq,yerr = debias_sim_eq_var, label = 'Debias',marker = '.',color = 'b',linewidth = linewidth)
ax[2,6].errorbar(x_axis,fgnn_sim_eq,yerr = fgnn_sim_eq_var, label = 'FairGNN',marker = '.',color = 'g',linewidth = linewidth)
ax[2,6].errorbar(x_axis,fvgnn_sim_eq,yerr = fvgnn_sim_eq_var, label = 'FairVGNN',marker = '.',color = 'orange',linewidth = linewidth)
ax[2,6].errorbar(x_axis,rnf_sim_eq,yerr = rnf_sim_eq_var, label = 'RNF',marker = '.',color = 'c',linewidth = linewidth)
ax[2,6].errorbar(x_axis,bfts_sim_eq,yerr = bfts_sim_eq_var, label = 'BFtS',marker = '.',color = 'r',linewidth = linewidth)

ax[0,0].xaxis.set_major_formatter(major_formatter)
ax[0,1].xaxis.set_major_formatter(major_formatter)
ax[0,2].xaxis.set_major_formatter(major_formatter)
ax[0,3].xaxis.set_major_formatter(major_formatter)
ax[0,4].xaxis.set_major_formatter(major_formatter)
ax[0,5].xaxis.set_major_formatter(major_formatter)
#ax[0,6].xaxis.set_major_formatter(major_formatter)

ax[1,0].xaxis.set_major_formatter(major_formatter)
ax[1,1].xaxis.set_major_formatter(major_formatter)
ax[1,2].xaxis.set_major_formatter(major_formatter)
ax[1,3].xaxis.set_major_formatter(major_formatter)
ax[1,4].xaxis.set_major_formatter(major_formatter)
ax[1,5].xaxis.set_major_formatter(major_formatter)
#ax[1,6].xaxis.set_major_formatter(major_formatter)

ax[2,0].xaxis.set_major_formatter(major_formatter)
ax[2,1].xaxis.set_major_formatter(major_formatter)
ax[2,2].xaxis.set_major_formatter(major_formatter)
ax[2,3].xaxis.set_major_formatter(major_formatter)
ax[2,4].xaxis.set_major_formatter(major_formatter)
ax[2,5].xaxis.set_major_formatter(major_formatter)
# %%
