#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_curly_brace(fig, x0, x1, y, height=0.02, lw=3, color='black'):
    """
    Draw a horizontal curly brace between x0 and x1 at height y (figure coords)
    """
    mid = (x0 + x1) / 2

    verts = [
        (x0, y),
        (x0, y + height),
        (mid, y + height),
        (mid, y),

        (mid, y),
        (mid, y - height),
        (x1, y - height),
        (x1, y)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,

        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO
    ]

    path = Path(verts, codes)
    patch = patches.PathPatch(
        path, fill=False, lw=lw, color=color,
        transform=fig.transFigure
    )
    fig.patches.append(patch)


plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2
})

def no_leading_zero(x, pos):
    s = f"{x:.2f}"
    if s.startswith("0"):
        s = s[1:]
    elif s.startswith("-0"):
        s = "-" + s[2:]
    return s

# ======================
# COLORS (UPDATED)
# ======================
color_map = {
    "Vanilla": "#000000",
    "Teacher": "magenta",
    "Feature": "blue",
    "Adjacency": "green",
    "Both": "cyan",
    "wo":'r'
}

# ======================
# DATA
# ======================
data = [
    ["Simulation","Vanilla","Vanilla",96.58,42.42,4.21],
    ["NBA","Vanilla","Vanilla",72.06,9.09,22.13],
    ["German","Vanilla","Vanilla",74.06,9.91,7.18],
    ["Simulation","FairGNN","Teacher",95.57,31.82,2.08],
    ["Simulation","FairGNN","Feature",96.00,34.34,2.38],
    ["Simulation","FairGNN","Adjacency",95.42,31.68,2.52],
    ["Simulation","FairGNN","Both",95.36,31.25,2.05],
    ["Simulation","FairGNN","wo",95.67,40.82,3.08],
    ["Simulation","FairVGNN","Teacher",95.81,30.01,1.31],
    ["Simulation","FairVGNN","Feature",95.96,32.25,1.98],
    ["Simulation","FairVGNN","Adjacency",94.35,30.11,1.26],
    ["Simulation","FairVGNN","Both",95.59,40.31,3.02],
    ["Simulation","FairVGNN","wo",95.67,40.82,3.08],
    ["Simulation","NIFTY","Teacher",94.93,29.18,2.52],
    ["Simulation","NIFTY","Feature",95.12,30.51,3.01],
    ["Simulation","NIFTY","Adjacency",93.84,28.98,2.29],
    ["Simulation","NIFTY","Both",94.87,29.20,2.49],
    ["Simulation","NIFTY","wo",95.69,41.31,3.62],
    ["NBA","FairGNN","Teacher",71.01,3.04,16.01],
    ["NBA","FairGNN","Feature",72.15,2.94,17.15],
    ["NBA","FairGNN","Adjacency",65.42,7.48,20.11],
    ["NBA","FairGNN","Both",69.15,7.02,20.01],
    ["NBA","FairGNN","wo",71.06,8.19,22.16],
    ["NBA","FairVGNN","Teacher",70.12,4.56,18.01],
    ["NBA","FairVGNN","Feature",69.52,3.12,16.05],
    ["NBA","FairVGNN","Adjacency",65.02,5.68,20.01],
    ["NBA","FairVGNN","Both",69.02,4.98,19.01],
    ["NBA","FairVGNN","wo",71.16,8.01,21.16],
    ["NBA","NIFTY","Teacher",71.05,2.61,14.02],
    ["NBA","NIFTY","Feature",69.53,2.01,14.15],
    ["NBA","NIFTY","Adjacency",64.18,5.08,20.01],
    ["NBA","NIFTY","Both",69.01,4.01,17.02],
    ["NBA","NIFTY","wo",69.06,6.19,20.11],
    ["German","FairGNN","Teacher",71.21,2.35,0.93],
    ["German","FairGNN","Feature",72.41,2.57,3.18],
    ["German","FairGNN","Adjacency",70.10,1.98,0.81],
    ["German","FairGNN","Both",70.24,2.51,1.02],
    ["German","FairGNN","wo",72.06,7.91,5.18],
    ["German","FairVGNN","Teacher",72.01,4.12,2.10],
    ["German","FairVGNN","Feature",72.95,4.91,3.05],
    ["German","FairVGNN","Adjacency",70.15,2.01,0.01],
    ["German","FairVGNN","Both",71.59,4.59,1.21],
    ["German","FairVGNN","wo",72.16,6.21,5.10],
    ["German","NIFTY","Teacher",72.05,2.69,1.01],
    ["German","NIFTY","Feature",72.81,3.01,5.05],
    ["German","NIFTY","Adjacency",70.11,2.01,0.98],
    ["German","NIFTY","Both",70.08,2.91,1.58],
    ["German","NIFTY","wo",71.96,6.81,5.32],

]

df = pd.DataFrame(data, columns=["Dataset","Method","Variant","Accuracy","DP","EQOP"])

df["DP_score"] = 1 - df["DP"] / 100.0
df["EQOP_score"] = 1 - df["EQOP"] / 100.0

datasets = ["Simulation","NBA","German"]
methods = ["FairGNN","FairVGNN","NIFTY"]

marker_map = {"Vanilla":"X","Teacher":"o","Feature":"s","Adjacency":"^","Both":"D","wo":"P"}

# ======================
# FIGURE
# ======================
sns.set_context("talk", font_scale=2.5)
fig, axes = plt.subplots(3, 6, figsize=(85, 44))
f_size = 120

for i, dataset in enumerate(datasets):

    vanilla = df[(df.Dataset==dataset)&(df.Method=="Vanilla")].iloc[0]

    for j, method in enumerate(methods):
        for k, metric in enumerate(["DP_score","EQOP_score"]):

            col = j if k==0 else j+3
            ax = axes[i, col]

            # Vanilla
            ax.scatter( vanilla[metric], vanilla["Accuracy"]/100,
                       marker='X', s=4500,
                       color=color_map["Vanilla"],
                       edgecolor='black', linewidth=1.8)

            # Method points
            sub = df[(df.Dataset==dataset)&(df.Method==method)]
            for _, r in sub.iterrows():
                ax.scatter(r[metric], r["Accuracy"]/100, 
                           marker=marker_map[r["Variant"]],
                           s=4500,
                           color=color_map[r["Variant"]],
                           edgecolor='black', linewidth=1.5, alpha=0.5)

            if ax.is_first_col():
                ax.set_ylabel(dataset, fontsize=f_size)

            ax.grid(False)
            ax.tick_params(labelsize=f_size-10)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_formatter(FuncFormatter(no_leading_zero))
            ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

# Titles
titles = ["FairGNN","FairVGNN","NIFTY"]*2
for i in range(6):
    axes[0,i].set_title(titles[i], fontsize=f_size)

# Legend
legend_elements = [
    Line2D([0],[0],marker='X',linestyle='None',label='Vanilla',markerfacecolor=color_map["Vanilla"],markeredgecolor='black',markersize=60),
    Line2D([0],[0],marker='o',linestyle='None',label='Teacher',markerfacecolor=color_map["Teacher"],markeredgecolor='black',markersize=60),
    Line2D([0],[0],marker='s',linestyle='None',label='GCN (Feat)',markerfacecolor=color_map["Feature"],markeredgecolor='black',markersize=60),
    Line2D([0],[0],marker='^',linestyle='None',label='GCN (Adj)',markerfacecolor=color_map["Adjacency"],markeredgecolor='black',markersize=60),
    Line2D([0],[0],marker='D',linestyle='None',label='GCN (Both)',markerfacecolor=color_map["Both"],markeredgecolor='black',markersize=60),
    Line2D([0],[0],marker='P',linestyle='None',label='GCN (w/o)',markerfacecolor=color_map["wo"],markeredgecolor='black',markersize=60)
]
# ======================
# GROUP LINES (CLEAN VERSION)
# ======================

# Get exact subplot boundaries (so lines align perfectly)
left_dp  = axes[0,0].get_position().x0 - 0.05
right_dp = axes[0,2].get_position().x1

left_eq  = axes[0,3].get_position().x0 + 0.05
right_eq = axes[0,5].get_position().x1 + 0.07

# Vertical position (just above subplots)
y = axes[0,0].get_position().y1 - 0.82

# --- DP line ---
fig.lines.append(
    plt.Line2D([left_dp, right_dp], [y, y],
               transform=fig.transFigure,
               lw=4, color='black')
)

fig.text(
    (left_dp + right_dp)/2,
    y - 0.05,
    "1 - DP",
    ha='center',
    va='bottom',
    fontsize=f_size
)

# --- EQOP line ---
fig.lines.append(
    plt.Line2D([left_eq, right_eq], [y, y],
               transform=fig.transFigure,
               lw=4, color='black')
)

fig.text(
    (left_eq + right_eq)/2,
    y - 0.05,
    "1 - EQOP",
    ha='center',
    va='bottom',
    fontsize=f_size
)
fig.legend(handles=legend_elements, loc='upper center',
           ncol=3, frameon=False, bbox_to_anchor=(0.5,1.1),
           fontsize=f_size)

fig.supxlabel("Fairness (↑)", fontsize=f_size, y=-0.052)
fig.supylabel("Accuracy (↑)", fontsize=f_size, x=-0.001)

plt.tight_layout(rect=[0,0,1,0.93])
#plt.savefig("f_acc_plot.pdf", dpi=300, bbox_inches='tight')
plt.show()
# %%
