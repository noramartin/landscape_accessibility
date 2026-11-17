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
L, K, n_it, n_it_giveng, type_walk, delta, a = param.L, param.K, param.n_it, param.n_it_giveng, param.type_walk, param.delta, param.a
params_filename_it = 'generalRMF'+str(L)+'_'+str(K)+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng)) + type_walk 
plot_basins, plot_walks = True, True
########################################################################################################################
print('general RMF')
########################################################################################################################
dist_list = [('spikes', 0.01), ('uniform', 1), ('exponential', 0.3), ('pareto', 2.2), ('spike_random_height', 0.01)]
dist_vs_label = {'spikes': 'sRMF', 'uniform': 'uniform', 'exponential': 'exponential', 'pareto': 'Lomax', 'spike_random_height': 'variable-height\nspikes'}
if plot_basins or plot_walks:
    f_basins, ax_basins = plt.subplots(ncols=4, nrows=len(dist_list),  figsize=(12, 2.4*len(dist_list)), sharey=True)
    f_walks, ax_walks = plt.subplots(ncols=6, nrows=len(dist_list),  figsize=(21, 2.5*len(dist_list)))
    for landscapei, (distribution, param_dist) in enumerate(dist_list):
        ########################################################################################################################
        print('set up landscape')
        ########################################################################################################################
        landscape_function = partial(RMF_landscape, p=param_dist, distribution=distribution, two_cones=False, delta=delta, a=a)
        network_reduced = landscape_function(K, L)
        node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
        network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
        peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
        peak_heights = [node_vs_fitness[peak] for peak in peaks]
        ########################################################################################################################
        print('basin plots')
        ########################################################################################################################
        if plot_basins:
            ax_basins[landscapei,0].set_title(dist_vs_label[distribution], fontsize=11)
            make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax_basins[landscapei], L, K, node_vs_fitness, theory=False, type_walk=type_walk)
            [ax_basins[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(-0.03, 1.05), xycoords='axes fraction', fontsize=12, fontweight='bold') for i in range(4)]
        ########################################################################################################################
        print('paths - by fitness')
        ########################################################################################################################
        if plot_walks:
            ax_walks[landscapei,0].set_title(dist_vs_label[distribution], fontsize=11)
            walks_plots_sRMF(landscape_function, ax_walks[landscapei], L, K, n_it_giveng=n_it_giveng, plot_theory=False)
            ###
            plot_rya(ax_walks[landscapei,-2], network_reduced, K, L, param_dist, n_it, plot_theory=False)
            [ax_walks[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(-0.03, 1.05), xycoords='axes fraction', fontsize=12, fontweight='bold') for i in range(6)]
            ###
            ref_seq = '0'*L
            shell_list, fitness_list = zip(*[(Hamming_dist(g, ref_seq), f) for g, f in node_vs_fitness.items()])
            ax_walks[landscapei,-1].scatter(shell_list, fitness_list, c='r', marker='x', linewidths=0.5, s=5, label='all')
            shell_list, fitness_list = zip(*[(Hamming_dist(g, ref_seq), node_vs_fitness[g]) for g in peaks])
            ax_walks[landscapei,-1].scatter(shell_list, fitness_list, c='purple', marker='x', linewidths=0.5, s=5, label='peaks')
            ax_walks[landscapei,-1].set_xlabel(r'shell $d$')
            ax_walks[landscapei,-1].set_ylabel('fitness')
            ax_walks[landscapei,-1].legend(bbox_to_anchor=(1.04, -0.05), loc="lower left", borderaxespad=0, fontsize=7)
            del landscape_function
    ###
    if plot_walks:
        f_walks.tight_layout()
        f_walks.savefig('./plots/walks_fitness'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
    if plot_basins:
        f_basins.tight_layout()
        f_basins.savefig('./plots/basins'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
########################################################################################################################
print('two-cone')
########################################################################################################################
L, K = 14, 2 ## important to have binary K and even L for cone geometry
distribution, p = 'spikes', param.p
params_filename_it = 'twoconeRMF'+distribution+str(L)+'_'+str(K)+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng))
if plot_basins or plot_walks:
    f_basins, ax_basins = plt.subplots(ncols=4,  figsize=(15, 3), sharey=True)
    f_walks, ax_walks = plt.subplots(ncols=6,  figsize=(23, 2.7))
    ########################################################################################################################
    print('set up landscape')
    ########################################################################################################################
    landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=True, delta=delta, a=a)
    network_reduced = landscape_function(K, L)
    node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
    network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
    ####
    peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
    peak_heights = [node_vs_fitness[peak] for peak in peaks]
    ########################################################################################################################
    print('basin plots')
    ########################################################################################################################
    if plot_basins:
        make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax_basins, L, K, node_vs_fitness, theory=False, type_walk=type_walk)
        [ax_basins[i].annotate('ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(4)]
    ########################################################################################################################
    print('paths - by fitness')
    ########################################################################################################################
    if plot_walks:
        walks_plots_sRMF(landscape_function, ax_walks, L, K, n_it_giveng=n_it_giveng,  plot_theory=False)
        ###
        plot_rya(ax_walks[-2], network_reduced, K, L, p, n_it, plot_theory=False, antipode_plot=False)
        [ax_walks[i].annotate('ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(6)]
        ###
        ref_seq = '0'*L
        shell_list, fitness_list = zip(*[(Hamming_dist(g, ref_seq), f) for g, f in node_vs_fitness.items()])
        ax_walks[-1].scatter(shell_list, fitness_list, c='r', marker='x', linewidths=0.5, s=5, label='all')
        shell_list, fitness_list = zip(*[(Hamming_dist(g, ref_seq), node_vs_fitness[g]) for g in peaks])
        ax_walks[-1].scatter(shell_list, fitness_list, c='purple', marker='x', linewidths=0.5, s=5, label='peaks')
        ax_walks[-1].set_xlabel(r'shell $d$')
        ax_walks[-1].set_ylabel('fitness')
        ax_walks[-1].legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, fontsize=7)
        del landscape_function
    ###
    if plot_walks:
        f_walks.tight_layout()
        f_walks.savefig('./plots/walks_fitness'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
    if plot_basins:
        f_basins.tight_layout()
        f_basins.savefig('./plots/basins'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
