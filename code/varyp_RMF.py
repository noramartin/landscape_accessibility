import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functions import *
from functools import partial
from math import comb
import parameters as param

########################################################################################################################
print('setup ')
########################################################################################################################
distribution, L, K, n_it, n_it_giveng, type_walk, delta, a = param.distribution, param.L, param.K, param.n_it, param.n_it_giveng, param.type_walk, param.delta, param.a
params_filename_it = 'RMF'+distribution+str(L)+'_'+str(K)+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng)) + type_walk + '_'.join([x + str(y) for x, y in [('delta', delta), ('a', a)]])
plot_basins, plot_walks = True, True

########################################################################################################################
print('general plots for varying p ')
########################################################################################################################
p_list = np.linspace(0, 1, num=40)
pindex_vs_peak_count_list, pindex_vs_toppercent_list1, pindex_vs_toppercent_list2 = {i: [] for i in range(len(p_list))}, {i: [] for i in range(len(p_list))}, {i: [] for i in range(len(p_list))}
cutoffs_nav = (10, 20)
for pindex, p in enumerate(p_list):
	print('p', p)
	landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
	for rep in range(10):
		network_reduced = landscape_function(K, L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		nodelist = list(network_reduced_directed.nodes())
		####
		peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get, reverse=True)
		peak_heights = [node_vs_fitness[peak] for peak in peaks]
		assert all(a >= b for a, b in zip(peak_heights, peak_heights[1:]))
		pindex_vs_peak_count_list[pindex].append(len(peaks))
		if len(peaks) >= 20:
			peaks_set = set(peaks)
			peaks_reached = [run_evolution(nodelist[np.random.randint(len(nodelist))], peaks_set, network_reduced_directed, node_vs_fitness, type_walk=type_walk, record_walk=False) for i in range(n_it)]
			fitness_cutoff =  top_x_percent_cutoff(peak_heights, cutoffs_nav[0])
			pindex_vs_toppercent_list1[pindex].append(len([p for p in peaks_reached if node_vs_fitness[p] >= fitness_cutoff])/len(peaks_reached))
			fitness_cutoff = top_x_percent_cutoff(peak_heights, cutoffs_nav[1])
			pindex_vs_toppercent_list2[pindex].append(len([p for p in peaks_reached if node_vs_fitness[p] >= fitness_cutoff])/len(peaks_reached))
		else:
			pindex_vs_toppercent_list1[pindex].append(np.nan)
			pindex_vs_toppercent_list2[pindex].append(np.nan)
####
peak_count_list = [np.mean(pindex_vs_peak_count_list[pindex]) for pindex, p in enumerate(p_list)]
peak_count_list_std = [np.std(pindex_vs_peak_count_list[pindex]) for pindex, p in enumerate(p_list)]
toppercent_list1 = [np.mean(pindex_vs_toppercent_list1[pindex]) for pindex, p in enumerate(p_list)]
toppercent_list_std1 = [np.std(pindex_vs_toppercent_list1[pindex]) for pindex, p in enumerate(p_list)]
toppercent_list2 = [np.mean(pindex_vs_toppercent_list2[pindex]) for pindex, p in enumerate(p_list)]
toppercent_list_std2 = [np.std(pindex_vs_toppercent_list2[pindex]) for pindex, p in enumerate(p_list)]
### HoC baseline
toppercent_HoC1, toppercent_HoC2 = [], []
for rep in range(10):
	network_HoC = HoC(network_reduced, new_fitness = True)
	node_vs_fitness_HoC = nx.get_node_attributes(network_HoC, 'fitness')
	nodelist = list(network_HoC.nodes())
	network_reduced_directed_HoC = make_directed_graph(node_vs_fitness_HoC, network_HoC)
	peaks_HoC = sorted([n for n in node_vs_fitness_HoC if network_reduced_directed_HoC.out_degree(n) == 0], key=node_vs_fitness_HoC.get, reverse=True)
	peak_heights_HoC = [node_vs_fitness_HoC[peak] for peak in peaks_HoC]
	peaks_reached_HoC = [run_evolution(nodelist[np.random.randint(len(nodelist))], set(peaks_HoC), network_reduced_directed_HoC, node_vs_fitness_HoC, type_walk=type_walk, record_walk=False) for i in range(n_it)]
	fitness_cutoff = top_x_percent_cutoff(peak_heights_HoC, cutoffs_nav[0])
	toppercent_HoC1.append(len([p for p in peaks_reached_HoC if node_vs_fitness_HoC[p] >= fitness_cutoff])/len(peaks_reached_HoC))
	fitness_cutoff = top_x_percent_cutoff(peak_heights_HoC, cutoffs_nav[1])
	toppercent_HoC2.append(len([p for p in peaks_reached_HoC if node_vs_fitness_HoC[p] >= fitness_cutoff])/len(peaks_reached_HoC))
###
f, ax = plt.subplots(ncols=3, figsize=(12, 2.6))
ax[0].errorbar(p_list, peak_count_list, yerr=peak_count_list_std,  c='r', ms=3.5, zorder=2, ls='', marker='o')
peaks_theory = [no_peaks_theory(p, L, K) for p in p_list ]
ax[0].plot(p_list, peaks_theory, c='g', label='# peaks', zorder=-1)
ax[0].plot(p_list, [K**L*p for p in p_list], c='grey', label='# spikes', zorder=-1)
#ax[0].plot(p_list, [K**L/((K-1)*L + 1) for p in p_list], c='k', label='HoC', ls=':', zorder=-1)
ax[0].set_ylim(0, 1.2*max(peaks_theory+peak_count_list))
ax[0].set_ylabel('# peaks')
##
for i, (list_to_save, list_to_save_std, HoC_value) in enumerate([(toppercent_list1, toppercent_list_std1, toppercent_HoC1), (toppercent_list2, toppercent_list_std2, toppercent_HoC2)]):
	ax[i+1].errorbar(p_list, list_to_save, yerr=list_to_save_std,  c='r', ms=3.5, zorder=2, ls='', marker='o')
	ax[i+1].plot([0, 1], [np.mean(HoC_value),]*2, c='k', ls=':', zorder=-1)
	ax[i+1].axhspan(np.mean(HoC_value)-np.std(HoC_value), np.mean(HoC_value)+np.std(HoC_value), color='grey', alpha=0.2, zorder=-2)
	ax[i+1].set_ylabel('fraction walks reaching\ntop-'+str(cutoffs_nav[i])+'% peaks')
####
for i in range(3):
	ax[i].set_xlabel(r'spike density $p$')
	ax[i].set_xlim(-0.01, 1.02)
[ax[i].annotate('ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(3)]
f.tight_layout()
f.savefig('./plots/varypoverview_'+params_filename_it+'.png', bbox_inches='tight', dpi=200)

########################################################################################################################
print('more detailed walk plots of three parameter values ')
########################################################################################################################
p_list = [ 0.01, 0.2, 0.5]
f_basins, ax_basins = plt.subplots(ncols=4, nrows=len(p_list),  figsize=(15, 3*len(p_list)), sharey=True)
f_walks, ax_walks = plt.subplots(ncols=5, nrows=len(p_list),  figsize=(18, 3*len(p_list)))
for landscapei, p in enumerate(p_list):
	landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
	########################################################################################################################
	print('set up landscape')
	########################################################################################################################
	if plot_walks or plot_basins:
		network_reduced = landscape_function(K, L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
		peak_heights = [node_vs_fitness[peak] for peak in peaks]
	########################################################################################################################
	print('basin plots')
	########################################################################################################################
	if plot_basins:
		make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax_basins[landscapei], L, K, node_vs_fitness, type_walk=type_walk, theory=True, params={'delta': delta, 'p': p})
		ax_basins[landscapei,0].set_title('p='+str(p))
		[ax_basins[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(4)]
	########################################################################################################################
	print('paths - by fitness')
	########################################################################################################################
	if plot_walks:
		ax_walks[landscapei,0].set_title('p='+str(p))
		walks_plots_sRMF(landscape_function, ax_walks[landscapei], L, K, p=p, n_it_giveng=n_it_giveng, plot_theory=False)
		d_list =  np.arange(L+1)
		ax_walks[landscapei, 0].plot(d_list, [comb(L, d) * (K-1)**d for d in d_list], c='b', zorder=-2)
		ax_walks[landscapei, 1].plot(d_list, [no_peaks_theory_shell(d, p, L, K) for d in d_list], c='b', zorder=-2)
		ax_walks[landscapei, 2].plot(d_list, [no_peaks_theory_shell(d, p, L, K)/(comb(L, d) * (K-1)**d) for d in d_list], c='b', zorder=-2)

		plot_rya(ax_walks[landscapei,-1], network_reduced, K, L, p, n_it, plot_theory=False)
		[ax_walks[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(5)]
if plot_walks:
	f_walks.tight_layout()
	f_walks.savefig('./plots/walks_fitness'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
if plot_basins:
	f_basins.tight_layout()
	f_basins.savefig('./plots/basins'+params_filename_it+'.png', bbox_inches='tight', dpi=200)

