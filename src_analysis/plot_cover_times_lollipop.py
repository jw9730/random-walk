from matplotlib import pyplot as plt


n_vertices = [9, 12, 15, 18]

uniform_cover_time = [145.9, 346.8, 632.9, 1123.8]
uniform_cover_time_std = [16.5, 30.0, 47.8, 67.6]
uniform_no_backtrack_cover_time = [43.94, 80.33, 132.5, 190.0]
uniform_no_backtrack_cover_time_std = [8.20, 4.76, 10.8, 25.5]
mdlr_cover_time = [55.04, 117.23, 165.7, 266.0]
mdlr_cover_time_std = [3.31, 5.21, 8.0, 41.5]
mdlr_no_backtrack_cover_time = [24.03, 37.13, 49.4, 66.1]
mdlr_no_backtrack_cover_time_std = [1.26, 2.03, 1.0, 4.0]
node2vec_cover_time = [85.92, 155.54, 290.6, 446.0]
node2vec_cover_time_std = [5.72, 15.73, 51.4, 49.7]

uniform_edge_cover_time = [628.4, 1330.8, 2727.2, 4900.6]
uniform_edge_cover_time_std = [113.7, 176.7, 202.6, 620.5]
uniform_no_backtrack_edge_cover_time = [171.60, 304.00, 572.0, 845.6]
uniform_no_backtrack_edge_cover_time_std = [27.83, 30.41, 43.4, 95.9]
mdlr_edge_cover_time = [256.20, 495.40, 1036.2, 1506.8]
mdlr_edge_cover_time_std = [22.53, 22.39, 158.1, 233.9]
mdlr_no_backtrack_edge_cover_time = [162.40, 328.60, 550.6, 859.8]
mdlr_no_backtrack_edge_cover_time_std = [30.96, 17.34, 26.9, 1.6]
node2vec_edge_cover_time = [325.60, 718.80, 1444.8, 2683.8]
node2vec_edge_cover_time_std = [23.05, 206.06, 528.4, 847.0]

# plot cover time and edge cover time (with error bars) in two separate plots with same y-axis
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].errorbar(n_vertices, uniform_edge_cover_time, yerr=uniform_edge_cover_time_std, label='Uniform')
ax[0].errorbar(n_vertices, uniform_no_backtrack_edge_cover_time, yerr=uniform_no_backtrack_edge_cover_time_std, label='Uniform + NB')
ax[0].errorbar(n_vertices, node2vec_edge_cover_time, yerr=node2vec_edge_cover_time_std, label='node2vec')
ax[0].errorbar(n_vertices, mdlr_edge_cover_time, yerr=mdlr_edge_cover_time_std, label='MDLR (Ours)')
ax[0].errorbar(n_vertices, mdlr_no_backtrack_edge_cover_time, yerr=mdlr_no_backtrack_edge_cover_time_std, label='MDLR + NB (Ours)')
ax[0].set_xlabel('Number of vertices')
ax[0].set_ylabel('Random walk steps')
ax[0].legend()
ax[0].set_title('Edge cover time C_E(G)')

ax[1].errorbar(n_vertices, uniform_cover_time, yerr=uniform_cover_time_std, label='Uniform')
ax[1].errorbar(n_vertices, uniform_no_backtrack_cover_time, yerr=uniform_no_backtrack_cover_time_std, label='Uniform + NB')
ax[1].errorbar(n_vertices, node2vec_cover_time, yerr=node2vec_cover_time_std, label='node2vec')
ax[1].errorbar(n_vertices, mdlr_cover_time, yerr=mdlr_cover_time_std, label='MDLR (Ours)')
ax[1].errorbar(n_vertices, mdlr_no_backtrack_cover_time, yerr=mdlr_no_backtrack_cover_time_std, label='MDLR + NB (Ours)')
ax[1].set_xlabel('Number of vertices')
ax[1].legend()
ax[1].set_title('Vertex cover time C_V(G)')

# save as a pdf
plt.tight_layout()
plt.savefig('cover_time.pdf')
plt.show()
plt.close('all')
