
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 4})
import seaborn as sns
sns.set(font_scale=3)
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, NullFormatter
sns.set_style(style='white')
from matplotlib.pyplot import cm


#%%
german = [-0.072147911,-0.053156162, -0.043156162,-0.0251117289,-0.011117289,0.075492697]
bail = [-0.010920554,-0.009311582, -0.008311582,0.025638199, 0.045638199, 0.077326037]
credit = [-0.004642332,-0.0093765081, -0.013765081,0.002051919,0.006051919, 0.038283324]
nba = [-0.157657652,-0.1062089, -0.08462089,0.0092771458, 0.011771458,0.016993261 ]
pokek_z = [0.0014674055,0.00727382,0.01527382,0.02304957, 0.0384957,0.034224096]
pokek_n = [0.00255578,0.0043007489394781,0.01263007489394781,0.021753262,0.039753262,0.034224096]


german_bfts = [-0.0508620710654506,0.0235226434402719,0.0635226434402719,0.07214791053658207,0.09214791053658207,0.075492697]
bail_bfts = [0.17166963499816,0.190021541, 0.210021541,0.229747797,0.2306398239788153, 0.077326037]
credit_bfts = [0.0932653663397614,0.0873594152717873,0.0823594152717873,0.0623594152717873,0.0543442446149904, 0.038283324]
nba_bfts = [0.0832824310525589,0.0632824310525589,0.0432824310525589,0.0232824310525589,0.0149036638638652,0.016993261 ]
pokek_z_bfts = [0.0211674055,0.02327382,0.02527382,0.0384957, 0.04584957,0.034224096]
pokek_n_bfts = [0.02255578,0.0243007489394781,0.0263007489394781,0.036753262,0.044753262,0.034224096]


german_rndm = [-0.012147911,0.0156162, 0.019156162,0.0251117289,0.031117289,0.075492697]
bail_rndm = [0.010920554,0.0311582, 0.0351582,0.038638199, 0.05638199, 0.077326037]
credit_rndm = [-0.00342332,-0.006765081, 0.00165081,0.0091919,0.017051919, 0.038283324]
nba_rndm = [-0.07657652,-0.0262089, 0.0062089,0.0102771458, 0.012771458,0.016993261]
pokek_z_rndm = [0.009674055,0.0127382,0.01527382,0.02804957, 0.0394957,0.034224096]
pokek_n_rndm = [0.0055578,0.0103007489394781,0.01263007489394781,0.021753262,0.039753262,0.034224096]


german_bfts_rndm = [0.0108620710654506,0.025226434402719,0.065226434402719,0.07914791053658207,0.07614791053658207,0.075492697]
bail_bfts_rndm = [0.1166963499816,0.10021541, 0.090021541,0.119747797,0.1006398239788153, 0.077326037]
credit_bfts_rndm = [0.0632653663397614,0.0573594152717873,0.0523594152717873,0.0423594152717873,0.0343442446149904, 0.038283324]
nba_bfts_rndm = [0.0532824310525589,0.0432824310525589,0.0332824310525589,0.0132824310525589,0.0159036638638652,0.016993261]
pokek_z_bfts_rndm = [0.0201674055,0.02327382,0.02527382,0.034957, 0.03584957,0.034224096]
pokek_n_bfts_rndm = [0.021255578,0.0253007489394781,0.02693007489394781,0.03753262,0.039753262,0.034224096]


#%%
matrix = [german,bail,credit,pokek_z,nba,pokek_n]
matrix_bfts = [german_bfts,bail_bfts,credit_bfts,pokek_z_bfts,(nba_bfts),pokek_n_bfts]
matrix_rndm = [german_rndm,bail_rndm,credit_rndm,pokek_z_rndm,nba_rndm,pokek_n_rndm]
matrix_rndm_bfts = [german_bfts_rndm,bail_bfts_rndm,credit_bfts_rndm,pokek_z_bfts_rndm,(nba_bfts_rndm),pokek_n_bfts_rndm]
datasets = ['German','Bail','Credit','Pokec-z','NBA','Pokec-n']
ranges = ['20','30','40','50','60','100']

#%%
sns.set(font_scale=4)

fig, ax = plt.subplots(1,4,figsize = (36,8),sharex=True)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=10, as_cmap=True)
sns.heatmap(matrix_bfts, cmap=rdgn, center=0.00, fmt ='.0%', 
            linewidths=1.3, linecolor='black', cbar=True, ax=ax[2],cbar_ax=cbar_ax,vmin=-0.25, vmax=0.25)
sns.heatmap(matrix_rndm_bfts, cmap=rdgn, center=0.00, fmt ='.0%', 
            linewidths=1.3, linecolor='black', cbar=False, ax=ax[3],cbar_ax=cbar_ax,vmin=-0.25, vmax=0.25)
sns.heatmap(matrix, cmap=rdgn, center=0.00, fmt ='.0%', 
            linewidths=1.3, linecolor='black',cbar=False, ax=ax[0],vmin=-0.25, vmax=0.25)
sns.heatmap(matrix_rndm, cmap=rdgn, center=0.00, fmt ='.0%', 
            linewidths=1.3, linecolor='black',cbar=False, ax=ax[1],vmin=-0.25, vmax=0.25)
ax[0].set_yticks([.5, 1.5, 2.5,3.5,4.5,5.5])
ax[0].set_xticks([.5, 1.5, 2.5,3.5,4.5,5.5])
ax[1].set_xticks([.5, 1.5, 2.5,3.5,4.5,5.5])
ax[2].set_xticks([.5, 1.5, 2.5,3.5,4.5,5.5])
ax[3].set_xticks([.5, 1.5, 2.5,3.5,4.5,5.5])
ax[0].set_xticklabels(ranges,rotation=45, ha="right", rotation_mode="anchor")
ax[1].set_xticklabels(ranges,rotation=45, ha="right", rotation_mode="anchor")
ax[2].set_xticklabels(ranges,rotation=45, ha="right", rotation_mode="anchor")
ax[3].set_xticklabels(ranges,rotation=45, ha="right", rotation_mode="anchor")
ax[0].set_yticklabels(datasets,rotation=45, ha="right", rotation_mode="anchor")
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])
fig.text(0.5, -0.1, '% of available sensitive information', ha='center')
ax[0].set_title("(a) Independent, Adv", pad=15)
ax[1].set_title("(b) Independent, MCAR", pad=15)
ax[2].set_title("(c) BFtS, Adv", pad=15)
ax[3].set_title("(d) BFtS, MCAR", pad=15)
fig.text(0.55, 1,' Bias in data represented as correlation between node labels and sensitive attribute ',ha = 'center')
plt.savefig("heatmap_correlation.pdf",bbox_inches='tight')
# %%
