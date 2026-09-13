#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example: loss values for 20 epochs
loss1 = [17.43,16.54,8.29, 6.95, 2.67,2.59,2.43,1.95 ,1.66, 1.54,1.21,0.82 ,0.69, 0.67,0.64, 0.64, 0.63 , 0.61, 0.59,0.55, 0.52]

loss2 = [10.22, 9.87, 6.54, 4.21 ,1.27, 1.22,1.20,1.01 , 0.93, 0.85,0.81,0.80 , 0.79,0.78,0.77,0.77 , 0.76, 0.75,0.75,0.73 , 0.73 ]

loss3 = [13.5, 12.1,10.9,8.5 , 6.69 , 6.21, 6.01,5.59, 4.65, 4.01, 3.59,1.95 , 1.21 ,1.01,0.98, 0.85, 0.69, 0.51,0.44,0.42 , 0.42]

# X-axis: epochs
epochs = np.arange(0, len(loss1))

# Plot
sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 3))
ax.semilogy(epochs, loss1, label=r'$\mathcal{L}_{class}$', linewidth=2)
ax.semilogy(epochs, loss2, label=r'$-\mathcal{L}_{bias}$', linewidth=2)
ax.semilogy(epochs, loss3, label=r'$\mathcal{L}_{imp}$', linewidth=2)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Value (log)")
ax.set_title("Training Loss Curves")
ax.set_xticklabels(['0','25', '50', '75', '100','125','150','200','225','250'])  
ax.legend()
ax.spines[['right', 'top']].set_visible(False) 
plt.savefig("convergence.pdf",bbox_inches = "tight")
plt.show()
# %%
