import matplotlib
from matplotlib import pyplot as plt


walk_lengths = [50, 100, 200, 500, 1000]
barbell_mse = [0.591, 0.297, 0.092, 0.012, 0.007]
barbell_mse_std = [0.082, 0.050, 0.024, 0.002, 0.001]
clique_mse = [0.066, 0.049, 0.037, 0.024, 0.023]
clique_mse_std = [0.017, 0.007, 0.008, 0.004, 0.006]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].errorbar(walk_lengths, clique_mse, yerr=clique_mse_std)
ax[0].axhline(0.86, color='r', linestyle='--')
ax[0].axhline(0.08, color='b', linestyle='--')
ax[0].axhline(0.03, color='y', linestyle='--')
ax[0].set_xlim(30, 1020)
ax[0].set_ylim(0.017 / 1.1, 0.86 * 1.1)
ax[0].set_xlabel('Walk length')
ax[0].set_ylabel('MSE')
ax[0].set_title('Clique (over-smoothing)')
ax[0].legend(['SAGE', 'NSD', 'BuNN', 'RWNN (Ours)'], bbox_to_anchor=(1.0, 0.92))
ax[0].set_yscale('log')
ax[0].set_yticks([2e-2, 1e-1, 5e-1])
ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax[1].errorbar(walk_lengths, barbell_mse, yerr=barbell_mse_std)
ax[1].axhline(0.90, color='r', linestyle='--')
ax[1].axhline(1.09, color='b', linestyle='--')
ax[1].axhline(0.01, color='y', linestyle='--')
ax[1].set_xlim(30, 1020)
ax[1].set_ylim(0.006 / 1.1, 1.09 * 1.1)
ax[1].set_xlabel('Walk length')
ax[1].set_yscale('log')
ax[1].set_title('Barbell (over-squashing)')
ax[1].legend(['SAGE', 'NSD', 'BuNN', 'RWNN (Ours)'], bbox_to_anchor=(1.0, 0.92))
ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.tight_layout()
plt.savefig('barbell_clique.pdf')
plt.show()
plt.close('all')
