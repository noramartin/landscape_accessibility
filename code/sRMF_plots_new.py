import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functions import *
from functools import partial
import parameters as param


#######################################################################################################################
print('initialise ')
########################################################################################################################
distribution, p, L, K, n_it, n_it_giveng, type_walk, delta, a = param.distribution, param.p, param.L, param.K, param.n_it, param.n_it_giveng, param.type_walk, param.delta, param.a
params_filename_it = 'RMF'+distribution+str(L)+'_'+str(K)+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng)) + type_walk + '_'.join([x + str(y) for x, y in [('p', p), ('delta', delta), ('a', a)]])
landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)

########################################################################################################################
print('paths ')
########################################################################################################################
f, ax = plt.subplots(ncols=4, figsize=(12, 2.2))
walks_plots_sRMF(landscape_function, ax, L, K, p, n_it_giveng)
[ax[i].annotate('ABCDEFGHIJ'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/peaks_in_shells'+params_filename_it+'_slides.png', bbox_inches='tight', dpi=200)
########################################################################################################################
print('set up landscape')
########################################################################################################################
network_reduced = landscape_function(K=K, L=L)
node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
peak_heights = [node_vs_fitness[peak] for peak in peaks]
########################################################################################################################
print('RYA')
########################################################################################################################
f, ax = plt.subplots(figsize=(5, 2.5))
plot_rya(ax, network_reduced, K, L, p, n_it)
f.tight_layout()
f.savefig('./plots/rya'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
########################################################################################################################
print('basins')
########################################################################################################################
f, ax = plt.subplots(ncols=4, figsize=(12, 2.7), sharey=True)
make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax, L, K, node_vs_fitness, params={'delta': delta, 'p': p}, print_stats=True)
###
[ax[i].annotate('ABCDEFGHIJ'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/basin_analysis'+params_filename_it+'.png', bbox_inches='tight', dpi=300)
########################################################################################################################
print('basins analysis - cut-off version')
########################################################################################################################
sRMF_landscape = landscape_function(K=K, L=L)
node_vs_fitness = nx.get_node_attributes(sRMF_landscape, 'fitness')
cutoff = np.percentile([f for f in node_vs_fitness.values()], 93)

f, ax = plt.subplots(ncols=5, nrows=4, figsize=(15, 10))#, sharey=True)
for axis in ax[:, :4].flat:
    if axis is not ax[0, 0]:
        axis.sharey(ax[0, 0])
        axis.sharex(ax[0, 0])
for i, type_network in enumerate(['full sRMF', 'thresholded sRMF', 'thresholded sRMF\n(largest comp.)', 'thresholded sRMF\nwith neighbours\n(largest comp.)']):
    if 'full' in type_network:
        network = sRMF_landscape.copy()
    elif 'threshold' in type_network and not 'neighbours' in type_network and 'largest' not in type_network:
        network = reduce_Papkou_network(sRMF_landscape.copy(), threshold=cutoff, add_neighbours=False, largest_comp=False)
    elif 'threshold' in type_network and not 'neighbours' in type_network and 'largest' in type_network:
        network = reduce_Papkou_network(sRMF_landscape.copy(), threshold=cutoff, add_neighbours=False, largest_comp=True)
    elif 'threshold' in type_network and 'neighbours' in type_network:
        network = reduce_Papkou_network(sRMF_landscape.copy(), threshold=cutoff, add_neighbours=True, largest_comp=True)
    node_vs_fitness = nx.get_node_attributes(network, 'fitness')
    network_reduced_directed = make_directed_graph(node_vs_fitness, network)
    peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
    peak_heights = [node_vs_fitness[peak] for peak in peaks]
    peak_vs_basin = make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax[i], L, K, node_vs_fitness, params={'delta': delta, 'p': p}, print_stats=True, theory=False, full_graph=False)
    #####
    dist_to_peak = [np.mean([Hamming_dist(g, p) for g in peak_vs_basin[p]]) for p in peaks]
    ax[i, -1].scatter(peak_heights, dist_to_peak,  c='r', s = 2, alpha=0.7)
    ax[i, -1].set_xlabel('peak height')
    ax[i, -1].set_ylabel('mean Hamming dist.\npeak to basin genotypes')
    ax[i, -1].set_ylim(-0.03, max(dist_to_peak) + 1.5)
    #####
    ax[i, 0].set_title(type_network, fontsize=10)
    if i > 0:
        ax[i,0].annotate('threshold = '+ str(round(cutoff, 2)) + '\n'+f"{len(network):.2e}"+   ' genotypes', xy=(0.99, 0.01), xycoords='axes fraction', ha='right', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax[i,0].annotate(f"{len(network):.2e}"+   ' genotypes', xy=(0.99, 0.01), xycoords='axes fraction', ha='right', va='bottom', fontsize=9, fontweight='bold')        
    [ax[i,j].annotate(str(i + 1) +'ABCDEFGHIJ'[j], xy=(-0.03, 1.05), xycoords='axes fraction', fontsize=11, fontweight='bold') for j in range(5)]
f.tight_layout()
f.savefig('./plots/basin_analysis_cutoff'+params_filename_it+'.png', bbox_inches='tight', dpi=300)

