import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functions import *
from functools import partial
import parameters as param

########################################################################################################################
print('setup ')
########################################################################################################################
distribution, p, n_it, n_it_giveng, type_walk, delta, a = param.distribution, param.p, param.n_it, param.n_it_giveng, param.type_walk, param.delta, param.a
params_filename_it = 'RMF'+distribution+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng))+ type_walk + '_'.join([x + str(y) for x, y in [('p', p), ('delta', delta), ('a', a)]])
plot_basins, plot_walks = True, True
########################################################################################################################
print('more detailed walk plots of two parameter values ')
########################################################################################################################
K_list = [2, 4]
f_basins, ax_basins = plt.subplots(ncols=4, nrows=len(K_list),  figsize=(15, 3*len(K_list)), sharey=True)
f_walks, ax_walks = plt.subplots(ncols=5, nrows=len(K_list),  figsize=(18, 3*len(K_list)))
for landscapei, K in enumerate(K_list):
	L = 32//K
	p = 0.24/((K-1)*L)
	landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
	########################################################################################################################
	print('set up landscape')
	########################################################################################################################
	if plot_walks or plot_basins:
		network_reduced = landscape_function(K=K, L=L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
		peak_heights = [node_vs_fitness[peak] for peak in peaks]
	########################################################################################################################
	print('basin plots')
	########################################################################################################################
	if plot_basins:
		ax_basins[landscapei,0].set_title('K='+str(K) + ',  L='+str(L))
		make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax_basins[landscapei], L, K, node_vs_fitness, type_walk=type_walk, params={'delta': delta, 'p': p})
		[ax_basins[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(4)]
	########################################################################################################################
	print('paths - by fitness')
	########################################################################################################################
	if plot_walks:
		print('number peaks', len(peaks))
		walks_plots_sRMF(landscape_function, ax_walks[landscapei], L, K, p, n_it_giveng)
		ax_walks[landscapei,0].set_title('K='+str(K) + ',  L='+str(L))
		plot_rya(ax_walks[landscapei,-1], network_reduced, K, L, p, n_it)
		[ax_walks[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(5)]
###
if plot_walks:
	f_walks.tight_layout()
	f_walks.savefig('./plots/walks_fitness'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
if plot_basins:
	f_basins.tight_layout()
	f_basins.savefig('./plots/basins'+params_filename_it+'.png', bbox_inches='tight', dpi=200)

