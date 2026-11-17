import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functions import *
from functools import partial
import parameters as param
########################################################################################################################
print('initialise ')
########################################################################################################################
distribution, p, L, K, n_it, n_it_giveng, delta, a = param.distribution, param.p, param.L, param.K, param.n_it, param.n_it_giveng, param.delta, param.a
type_walk = 'neidhart'
params_filename_it = 'RMF'+distribution+str(L)+'_'+str(K)+'_'+str(int(np.log10(n_it)))+'_'+str(int(n_it_giveng)) + type_walk+ '_'.join([x + str(y) for x, y in [('p', p), ('delta', delta), ('a', a)]])

########################################################################################################################
print('paths - true adaptive walks')
########################################################################################################################
def successful_next_step_fitness_diff(d, L, K, p, delta):
	ratio = (p*d*(1+delta) + (K-2)*p*d*delta + (K-1)*p*(L-d)*(delta-1))/((1-p)*d)
	return (1+ratio)**(-1)		

f, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), sharey='row')
for colindex, delta in enumerate([1.2, 1.5, 1.8]):
	landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
	shell_vs_list_trapping_prob = {d: [] for d in range(L + 1)}
	for networkindex in range(100):
		shell_vs_list_trapping_prob_thislandscape = {d: [] for d in range(L + 1)}
		network_reduced = landscape_function(K, L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		node_vs_additive_fitness= nx.get_node_attributes(network_reduced, 'additive_fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		for g, fit in node_vs_fitness.items():
			d = Hamming_dist([int(x) for x in g], [0,]*L) 
			if network_reduced_directed.out_degree(g) > 0:
				prob_exit = prob_exit_uphill_no_trapping(g, network_reduced_directed, n_it=n_it_giveng, node_vs_additive_fitness=node_vs_additive_fitness, shell_change_threshold=0.8, type_walk=type_walk , g_vs_f=node_vs_fitness)
				shell_vs_list_trapping_prob_thislandscape[d].append(1-prob_exit)
		for d in range(1, L+1):
			if shell_vs_list_trapping_prob_thislandscape[d]:
				shell_vs_list_trapping_prob[d].append(np.mean(shell_vs_list_trapping_prob_thislandscape[d]))
	data_trapping_prob = [np.mean(shell_vs_list_trapping_prob[d]) if len(shell_vs_list_trapping_prob[d]) else np.nan for d in range(L+1)]
	data_trapping_prob_std = [np.std(shell_vs_list_trapping_prob[d]) if len(shell_vs_list_trapping_prob[d]) else np.nan for d in range(L+1)]

	ax[0, colindex].errorbar(np.arange(L+1), data_trapping_prob, yerr=data_trapping_prob_std, c='r', fmt='o', markersize=4, elinewidth=1, capsize=2)
	#####theory
	xaxis_theory = np.arange(2, L+1)
	peak_transition_prob_theory = [1-successful_next_step_fitness_diff(d, L, K, p, delta) for d in xaxis_theory]
	ax[0, colindex].plot(xaxis_theory, peak_transition_prob_theory, c='b')
	###
	ax[0, colindex].set_xlabel('shell d')
	ax[0, colindex].set_ylabel(r'$P_{\text{trapping}} (d)$')
	ax[0, colindex].set_yscale('log')
	ax[0, colindex].set_title(r'$\delta =$'+str(delta))
	###
	network_reduced = landscape_function(K, L)
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
	####
	plot_rya(ax[1, colindex], network_reduced, K, L, p=p, n_it=n_it, plot_theory=False, antipode_plot=True, type_walk=type_walk, legend=False)
	peaks_per_shell = [no_peaks_theory_shell(d, p, L, K) for d in range(0, L+1)]
	peak_percentile_given_d0 = [sum(peaks_per_shell[:d0])/sum(peaks_per_shell) for d0 in xaxis_theory] 
	next_step_function = partial(successful_next_step_fitness_diff, delta=delta)
	theory_antipode = [prob_success(d0, L, L, K, p, next_step_function = next_step_function) for d0 in xaxis_theory]
	theory_full = [sum([prob_success(d0, start, L, K, p, next_step_function = next_step_function)*comb(L, start)*(K-1)**start for start in range(0, L+1)])/K**L for d0 in xaxis_theory]
	ax[1, colindex].plot(100*np.array([0,] + peak_percentile_given_d0 + [1,]), 100*np.array([0,] + theory_full+ [1,]), c='b', label='analytic-\nrandom initialisation', drawstyle='steps-post', zorder=0, alpha=0.8)
	ax[1, colindex].plot(100*np.array([0,] + peak_percentile_given_d0+ [1,]), 100*np.array([0,] + theory_antipode+ [1,]), c='b', ls=':', label='analytic-\nfrom antipode', drawstyle='steps-post', zorder=0, alpha=0.8)
l2 = ax[1, -1].legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, fontsize=7)
[ax[j, i].annotate('ABCDEFG'[j]+'123456'[i], xy=(0.02, 1.05), xycoords='axes fraction', fontsize=12, fontweight='bold') for i in range(3) for j in range(2)]
###
f.tight_layout()
f.savefig('./plots/true_adaptive_walks'+params_filename_it+'.png', bbox_inches='tight', dpi=250)

########################################################################################################################
print('basins - true adaptive walks')
########################################################################################################################
f, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 7.5), sharey=True)#sharey='col')
for colindex, delta in enumerate([1.2, 1.5, 1.8]):
	landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
	network_reduced = landscape_function(K, L)
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
	peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
	peak_heights = [node_vs_fitness[peak] for peak in peaks]
	ax[colindex,0].set_title(r'$\delta =$'+str(delta))
	make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax[colindex], L, K, node_vs_fitness, type_walk=type_walk, params={'delta': delta, 'p': p})
	[ax[colindex,i].annotate(str(colindex + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/basin_analysis_true_adaptive_walks'+params_filename_it+'.png', bbox_inches='tight', dpi=300)

########################################################################################################################
print('paths - greedy and two-step adaptive walks')
########################################################################################################################
delta = param.delta
from scipy.optimize import brentq
p_low = 0.01
delta = 2.5
no_peaks_fixedparam = partial(no_peaks_theory, K=K, L=L)
p_high = round(brentq(lambda p: no_peaks_fixedparam(p) - no_peaks_fixedparam(p_low), p_low + 0.01, 1), 2)
###
runtype_vs_title = {'greedy': 'greedy adaptive walk', '2greedy': 'double-mut.\ngreedy adaptive walk', 'neidhart': 'true adaptive walk', '2neidhart': 'double-mut.\ntrue adaptive walk'}
cmap = plt.get_cmap('tab20')
runtype_vs_colour = {'neidhart': cmap(4),'greedy': cmap(8), '2neidhart': cmap(5), '2greedy':   cmap(9)}

f2, ax2 = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), sharey='col')
for pindex, p in enumerate([p_low, p_high]):
	for deltaindex, delta in enumerate([1.5, 2.5]):
		rowindex = deltaindex + pindex * 2
		landscape_function = partial(RMF_landscape, p=p, distribution=distribution, two_cones=False, delta=delta, a=a)
		network_reduced = landscape_function(K, L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		node_vs_additive_fitness = nx.get_node_attributes(network_reduced, 'additive_fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)	
		peaks_single = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
		peaks_double = [n for n, f in node_vs_fitness.items() if f > max([node_vs_fitness[g] for g in all_neighbours_rad_two(network_reduced, n)])]
		node_vs_spike = nx.get_node_attributes(network_reduced, 'spike')
		nodelist = [n for n in node_vs_fitness]
		network_HoC = HoC(network_reduced, new_fitness = True, min_fit = min(list(node_vs_fitness.values())), max_fit = max(list(node_vs_fitness.values())))
		node_vs_fitness_HoC = nx.get_node_attributes(network_HoC, 'fitness')
		network_HoC_directed = make_directed_graph(node_vs_fitness_HoC, network_HoC)	
		peaks_single_HoC = [n for n in node_vs_fitness_HoC if network_HoC_directed.out_degree(n) == 0]
		peaks_double_HoC = [n for n, f in node_vs_fitness_HoC.items() if f > max([node_vs_fitness_HoC[g] for g in all_neighbours_rad_two(network_HoC, n)])]
		for walktypeindex, run_type in enumerate(['neidhart','greedy', '2neidhart', '2greedy']):
			if run_type.startswith('2'):
				peaks, peaks_HoC = deepcopy(peaks_double), deepcopy(peaks_double_HoC)
			else:
				peaks, peaks_HoC = deepcopy(peaks_single), deepcopy(peaks_single_HoC)
			if len(peaks) >= 5:
				###
				shell_vs_list_trapping_prob = {d: [] for d in range(L + 1)}
				for networkindex in range(100):
					shell_vs_list_trapping_prob_thislandscape = {d: [] for d in range(L + 1)}
					network_reduced_trapping = landscape_function(K, L)
					node_vs_fitness_trapping = nx.get_node_attributes(network_reduced_trapping, 'fitness')
					node_vs_additive_fitness_trapping= nx.get_node_attributes(network_reduced_trapping, 'additive_fitness')
					network_reduced_directed_trapping = make_directed_graph(node_vs_fitness_trapping, network_reduced_trapping)
					if run_type.startswith('2'):
						peaks_trapping = [n for n, f in node_vs_fitness_trapping.items() if network_reduced_directed_trapping.out_degree(n) == 0 and f > max([node_vs_fitness_trapping[g] for g in all_neighbours_rad_two(network_reduced_trapping, n)])]
					else:
						peaks_trapping = [n for n in node_vs_fitness_trapping if network_reduced_directed_trapping.out_degree(n) == 0]
					for g, fit in node_vs_fitness_trapping.items():
						d = Hamming_dist([int(x) for x in g], [0,]*L) 
						if not is_peak(g, network_reduced_directed_trapping, peaks=peaks_trapping):
							prob_exit = prob_exit_uphill_no_trapping(g, network_reduced_directed_trapping, n_it=n_it_giveng, node_vs_additive_fitness=node_vs_additive_fitness_trapping, shell_change_threshold=0.8, type_walk=run_type , g_vs_f=node_vs_fitness_trapping, undirected_network=network_reduced_trapping, peaks=peaks_trapping)
							shell_vs_list_trapping_prob_thislandscape[d].append(1-prob_exit)
					for d in range(1, L+1):
						if shell_vs_list_trapping_prob_thislandscape[d]:
							shell_vs_list_trapping_prob[d].append(np.mean(shell_vs_list_trapping_prob_thislandscape[d]))
				data_trapping_prob = [np.mean(shell_vs_list_trapping_prob[d]) if len(shell_vs_list_trapping_prob[d]) else np.nan for d in range(L+1)]
				data_trapping_prob_std = [np.std(shell_vs_list_trapping_prob[d]) if len(shell_vs_list_trapping_prob[d]) else np.nan for d in range(L+1)]
				ax2[rowindex, 0].errorbar(np.arange(L+1), data_trapping_prob, yerr=data_trapping_prob_std, c=runtype_vs_colour[run_type], fmt='o', markersize=4, elinewidth=1, capsize=2)
				ax2[rowindex, 0].set_xlabel('shell d')
				ax2[rowindex, 0].set_ylabel(r'$P_{\text{trapping}} (d)$')
				ax2[rowindex, 0].set_yscale('log')
				###
				x_data, y_data = get_accessibility_plot(network_reduced_directed, K=K, initialisation='', n_it=n_it, type_walk=run_type, peaks=peaks)
				x_data_Hoc, y_data_HoC = get_accessibility_plot(network_HoC_directed, K=K, initialisation='', n_it=n_it, type_walk=run_type, peaks=peaks_HoC)
				ax2[rowindex, 1].plot(x_data_Hoc, y_data_HoC, color=runtype_vs_colour[run_type], ls=':', drawstyle='steps-post', zorder=-5, alpha=0.8)
				ax2[rowindex, 1].plot(x_data, y_data, color=runtype_vs_colour[run_type], ls='-', label=runtype_vs_title[run_type], drawstyle='steps-post', zorder=0, alpha=0.8)
				ax2[rowindex, 1].set_xlabel('top-x% peaks')
				ax2[rowindex, 1].set_ylabel('x% adaptive walks')
				ax2[rowindex, 1].plot([0, 100], [0, 100], c='grey', zorder=-5, alpha=0.8)
				############
				walk_l = []
				for it in range(n_it):
					n_initial = nodelist[np.random.choice(len(nodelist))]
					nfin, walk = run_evolution(n_initial, peaks, network_reduced_directed, node_vs_fitness, type_walk=run_type, record_walk=True, undirected_network=network_reduced)
					if node_vs_spike[nfin]:
						walk_l.append(len([s for s in walk if not node_vs_spike[s]]))
					else:
						walk_l.append(np.inf)
					assert nfin == '0'*L or node_vs_spike[nfin]
					assert walk[-1] == nfin
				sorted_lengths = np.arange(0, 5)
				offset = (walktypeindex - 1.5) * 0.06   # small dodge so identical curves stay visible
				ax2[rowindex, 2].plot(sorted_lengths + offset, [len([l2 for l2 in walk_l if l2 <= l])/len(walk_l) for l in sorted_lengths], color=runtype_vs_colour[run_type])
				ax2[rowindex, 2].set_xlabel('# steps until spike, '+r'$l$')
				ax2[rowindex, 2].set_ylabel('fraction adaptive\nwalks at spike\nby step '+r'$l$')
			ax2[rowindex, 0].set_title('sRMF with p='+str(p)+r', $\delta=$'+str(delta)+',\n# peaks='+str(len(peaks_single))+',\n# peaks with double-mut.='+str(len(peaks_double)), fontsize=11)
[ax2[j, i].annotate('ABCDEFG'[j]+'123456'[i], xy=(-0.15, 1.05), xycoords='axes fraction', fontsize=11, fontweight='bold') for i in range(3) for j in range(4)]

###
handles, labels = ax2[1, 1].get_legend_handles_labels()
l2 = ax2[0, 2].legend(handles, labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=9)
###
f2.tight_layout()
f2.savefig('./plots/greedy_double_adaptive_walks'+params_filename_it+'.png', bbox_inches='tight', dpi=250)




