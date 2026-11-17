import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product
import random
from collections import Counter
from math import comb
import numpy as np
from matplotlib.lines import Line2D

def draw_network(network_reduced, ax, f, linewidths=0.6, node_size = 8, nodelist_per_shell=[]):
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
	peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	L_sketch = len([n for n in network_reduced.nodes()][0])
	if len(nodelist_per_shell) == 0:
		nodelist_per_shell = [[n for n in network_reduced.nodes() if Hamming_dist(n, '0'*L_sketch) == d] for d in range(0, L_sketch + 1)]
	pos = nx.shell_layout(network_reduced, nlist=nodelist_per_shell)
	nodes= nx.draw_networkx_nodes(network_reduced, node_size=node_size, nodelist= network_reduced.nodes(), ax=ax, pos=pos, edgecolors=None, node_color = [node_vs_fitness[n] for n in network_reduced.nodes()])
	nx.draw_networkx_edges(network_reduced, nodelist= network_reduced.nodes(), ax=ax, pos=pos, edge_color='grey', width=linewidths)
	nx.draw_networkx_nodes(network_reduced, node_size=node_size*7, nodelist= peaks, ax=ax, pos=pos, edgecolors='r', node_color = 'none', linewidths=linewidths*2)
	cb=f.colorbar(nodes, ax=ax)
	cb.set_label('fitness')

def all_neighbours_rad_two(undirected_network, n_initial):
	neighbours_one = [n for n in undirected_network.neighbors(n_initial)] 
	neighbours_full = neighbours_one + [n2 for n in neighbours_one for n2 in undirected_network.neighbors(n)]
	return list(set([n for n in neighbours_full if n != n_initial])) 

    
def run_evolution(n_initial, peaks, directed_network, g_vs_f, type_walk='uphill', record_walk=False, undirected_network=None):
	if record_walk:
		walk = [n_initial]
	if type_walk.startswith('2') and not undirected_network:
		undirected_network = directed_network.to_undirected()
	while n_initial not in peaks:
		assert directed_network.out_degree(n_initial) > 0 or type_walk.startswith('2')
		if not type_walk.startswith('2'):
			neighbours = [n for n in directed_network.successors(n_initial)]
			assert min([g_vs_f[n] for n in neighbours]) >= g_vs_f[n_initial] #successors in directed network
		elif type_walk.startswith('2'):
			f_current = g_vs_f[n_initial]
			neighbours = [n for n in all_neighbours_rad_two(undirected_network, n_initial) if g_vs_f[n] >= f_current]
		if type_walk == 'uphill':
			n_initial = random.choice(neighbours)
		elif 'greedy' in type_walk:
			n_initial = max(neighbours, key=g_vs_f.get)	
		elif 'neidhart' in type_walk:
			p_list = [g_vs_f[n] - g_vs_f[n_initial] for n in neighbours]
			n_initial = np.random.choice(neighbours, p=np.divide(p_list, np.sum(p_list)))
		else:
			raise RuntimeError(type_walk, 'not implemented')			
		if record_walk:
			walk.append(n_initial)
	if record_walk:
		return n_initial, walk
	return n_initial

def is_peak(g, directed_network, peaks=None):
	if directed_network.out_degree(g) == 0:
		if not peaks:
			return True
		elif g in peaks:
			return True
	return False

def prob_exit_uphill_no_trapping(n_initial, directed_network, n_it, node_vs_additive_fitness, g_vs_f= {}, type_walk='uphill', shell_change_threshold = 0.5, undirected_network=False, peaks=None):
	count_all, count_exit = 0, 0
	for rep in range(n_it):
		g = deepcopy(n_initial)
		additive_fitness_init = node_vs_additive_fitness[g]
		assert abs(node_vs_additive_fitness[g] - additive_fitness_init) < shell_change_threshold and not is_peak(g, directed_network, peaks=peaks)
		if type_walk.startswith('2') and not undirected_network:
			undirected_network = directed_network.to_undirected()
		while node_vs_additive_fitness[g] - additive_fitness_init < shell_change_threshold and not is_peak(g, directed_network, peaks=peaks):
			if not type_walk.startswith('2'):
				neighbours = [n for n in directed_network.successors(g)]
			elif type_walk.startswith('2'):
				f_current = g_vs_f[g]
				neighbours = [n for n in all_neighbours_rad_two(undirected_network, g) if g_vs_f[n] >= f_current]
			if type_walk == 'uphill':
				g = random.choice(neighbours)	
			elif 'neidhart' in type_walk:
				p_list = [g_vs_f[n] - g_vs_f[g] for n in neighbours]
				g = np.random.choice(neighbours, p=np.divide(p_list, np.sum(p_list)))
			elif 'greedy' in type_walk:
				g = max(neighbours, key=g_vs_f.get)	
			else:
				raise RuntimeError(type_walk, 'not implemented')
		if not is_peak(g, directed_network, peaks=peaks) and node_vs_additive_fitness[g] >= additive_fitness_init + shell_change_threshold:
			count_exit += 1
		count_all += 1
	return count_exit/count_all

def no_peaks_theory_shell(d, p, L, K):
	if d > 0:
		return comb(L, d)*(K-1)**d*p*(1-p)**d*sum([1/(x+1)*comb((K-2)*d, x)*p**x*(1-p)**((K-2)*d-x) for x in range((K-2)*d+1)])
	else:
		return (1-p)**(L*(K-1)+1) + p

def no_peaks_theory(p, L, K):
	peaks_per_shell = [no_peaks_theory_shell(d, p, L, K) for d in range(0, L+1)]
	return sum(peaks_per_shell)

def walks_plots_sRMF(landscape_function, ax, L, K, p=0.01, n_it_giveng=10, plot_theory=True, type_walk='uphill'):
	shell_change_threshold = 0.5
	shell_vs_no_genotypes, shell_vs_no_peaks, shell_vs_exit_uphill_no_trapping = {d: [] for d in range(L + 1)}, {d: [] for d in range(L + 1)}, {d: [] for d in range(L + 1)}
	for rep in range(100):
		network_reduced = landscape_function(K=K, L=L)
		node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
		node_vs_additive_fitness = nx.get_node_attributes(network_reduced, 'additive_fitness')
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		max_add = max(node_vs_additive_fitness.values())
		####
		peaks = sorted([n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0], key=node_vs_fitness.get)
		peak_heights = [node_vs_fitness[peak] for peak in peaks]
		for d in range(0, L+1):
			shell_vs_no_genotypes[d].append(0)
			shell_vs_no_peaks[d].append(0)
		shell_vs_exit_uphill_no_trapping_thislandscape = {d: [] for d in range(L + 1)}
		for g, fit in node_vs_fitness.items():
			d = Hamming_dist([int(x) for x in g], [0,]*L) 
			shell_vs_no_genotypes[d][-1] += 1
			if network_reduced_directed.out_degree(g) == 0:
				shell_vs_no_peaks[d][-1] += 1
			elif node_vs_additive_fitness[g] < max_add - 0.00000001:
				assert network_reduced_directed.out_degree(g) + network_reduced_directed.in_degree(g) == (K-1)*L
				prob_exist = prob_exit_uphill_no_trapping(g, network_reduced_directed, n_it=n_it_giveng, node_vs_additive_fitness=node_vs_additive_fitness, shell_change_threshold=shell_change_threshold, g_vs_f=node_vs_fitness, type_walk=type_walk)
				shell_vs_exit_uphill_no_trapping_thislandscape[d].append(prob_exist)
		for d in range(L+1):
			if len(shell_vs_exit_uphill_no_trapping_thislandscape[d]):
				shell_vs_exit_uphill_no_trapping[d].append(np.mean(shell_vs_exit_uphill_no_trapping_thislandscape[d]))
	data_peak_numbers = [np.mean(shell_vs_no_peaks[d]) for d in range(L+1)]
	data_no_genotypes = [np.mean(shell_vs_no_genotypes[d]) for d in range(L+1)]
	data_peak_densities = [np.mean(np.divide(shell_vs_no_peaks[d], shell_vs_no_genotypes[d])) for d in range(L+1)]
	data_exit_uphill_no_trapping = [np.mean(1-np.array(shell_vs_exit_uphill_no_trapping[d])) if len(shell_vs_exit_uphill_no_trapping[d]) else np.nan for d in range(L+1)]
	data_peak_numbers_std = [np.std(shell_vs_no_peaks[d]) for d in range(L+1)]
	data_no_genotypes_std = [np.std(shell_vs_no_genotypes[d]) for d in range(L+1)]
	data_peak_densities_std = [np.std(np.divide(shell_vs_no_peaks[d], shell_vs_no_genotypes[d])) for d in range(L+1)]
	data_exit_uphill_no_trapping_std = [np.std(1 - np.array(shell_vs_exit_uphill_no_trapping[d])) if len(shell_vs_exit_uphill_no_trapping[d]) else np.nan for d in range(L+1)]

	d_list =  np.arange(L+1)
	xaxis = [x for x in d_list]
	ax[0].errorbar(xaxis, data_no_genotypes, yerr=data_no_genotypes_std, c='r', fmt='o', markersize=4, elinewidth=1, capsize=2)
	ax[0].plot(xaxis, [comb(L, d) * (K-1)**d for d in d_list], c='b', zorder=-1)
	ax[0].set_xlabel(r'shell $d$')
	ax[0].set_ylabel('# genotypes')
	###
	ax[1].errorbar(xaxis, data_peak_numbers, yerr=data_peak_numbers_std, c='r', fmt='o', markersize=4, elinewidth=1, capsize=2)
	if plot_theory:
		ax[1].plot(xaxis, [no_peaks_theory_shell(d, p, L, K) for d in d_list], c='b', zorder=-2)
	ax[1].set_xlabel(r'shell $d$')
	ax[1].set_ylabel('# peaks')
	###
	ax[2].errorbar(xaxis, np.divide(data_peak_numbers, data_no_genotypes), yerr=np.divide(data_peak_numbers_std, data_no_genotypes), c='r', fmt='o', markersize=4, elinewidth=1, capsize=2)
	if plot_theory:
		ax[2].plot(xaxis, [no_peaks_theory_shell(d, p, L, K)/(comb(L, d) * (K-1)**d) for d in d_list], c='b', zorder=-2)
	ax[2].set_xlabel(r'shell $d$')
	ax[2].set_ylabel('# peaks/# genotypes')
	ax[2].set_ylim(0, 1)
	###### prob that next "additively non-neutral" step is fitness-increasing-non-peak
	ax[3].errorbar(xaxis, np.array(data_exit_uphill_no_trapping), np.array(data_exit_uphill_no_trapping_std), c='r', fmt='o', markersize=4, elinewidth=1, capsize=2)
	if plot_theory:
		xaxis_theory = np.arange(2, L+1)
		ax[3].plot([x for x in xaxis_theory], [1-successful_next_step(d, L, K, p) for d in xaxis_theory], c='b', zorder=-2)
	ax[3].set_xlabel(r'shell $d$')
	ax[3].set_ylabel(r'$P_{\text{trapping}} (d)$')
	ax[3].set_yscale('log')



def plot_rya(ax, network_reduced, K, L, p=0.01, n_it=10**4, plot_theory=True, antipode_plot=True, type_walk='uphill', legend=True):
	for initialisation in ['', 'antipode']:
		if initialisation == '' or antipode_plot:
			x_data, y_data = get_accessibility_plot(network_reduced, K=K, initialisation=initialisation, n_it=n_it, type_walk=type_walk)
			ax.plot(x_data, y_data, color='r', label={'': 'simulation -\nrandom initialisation', 'antipode': 'simulation -\nfrom antipode'}[initialisation],
				ls={'': '-', 'antipode': ':'}[initialisation], drawstyle='steps-post', zorder=0, alpha=0.8)
	if plot_theory:
		d0_list = np.arange(2, L)
		peaks_per_shell = [no_peaks_theory_shell(d, p, L, K) for d in range(0, L+1)]
		peak_percentile_given_d0 = [sum(peaks_per_shell[:d0])/sum(peaks_per_shell) for d0 in d0_list] 
		theory_antipode_simplified = [prob_success(d0, L, L, K, p) for d0 in d0_list]
		theory_full = [sum([prob_success(d0, start, L, K, p)*comb(L, start)*(K-1)**start for start in range(0, L+1)])/K**L for d0 in d0_list]
		ax.plot(100*np.array([0,] + peak_percentile_given_d0 + [1,]), 100*np.array([0,] + theory_full+ [1,]), c='b', label='analytic-\nrandom initialisation', drawstyle='steps-post', zorder=0, alpha=0.8)
		ax.plot(100*np.array([0,] + peak_percentile_given_d0+ [1,]), 100*np.array([0,] + theory_antipode_simplified+ [1,]), c='b', ls=':', label='analytic-\nfrom antipode', drawstyle='steps-post', zorder=0, alpha=0.8)
	#HoC context
	network_HoC = HoC(network_reduced, new_fitness = True)
	x_data_Hoc, y_data_HoC = get_accessibility_plot(network_HoC, K=K, n_it=n_it, initialisation='', type_walk=type_walk)
	ax.plot(x_data_Hoc, y_data_HoC, color='k', ls=':', label='simulation -\nHouse-of-Cards', drawstyle='steps-post', zorder=-5, alpha=0.8)
	ax.set_xlabel('top-x% peaks')
	ax.set_ylabel('x% adaptive walks')
	ax.plot([0, 100], [0, 100], c='grey', zorder=-5, alpha=0.8)
	if legend:
		l2 = ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, fontsize=7)

def make_basins_plots(network_reduced_directed, peaks, peak_heights, n_it, ax, L, K, node_vs_fitness, params={}, type_walk='uphill', theory=True, print_stats=False, full_graph=True):
	min_count = 5
	peak_vs_basin, g_vs_peaks = get_basins(network_reduced_directed, peaks)		
	basin_sizes_norm = [len(peak_vs_basin[peak])/len(network_reduced_directed.nodes()) for peak in peaks]
	if full_graph:
		assert len(network_reduced_directed.nodes()) == K**L
	nodelist_nonpeaks = [n for n in network_reduced_directed.nodes() if network_reduced_directed.out_degree(n) > 0]
	peak_reached_prob_if_starting_in_basin, peak_reached, nonpeak_reached = probabilistic_picture_basins(peaks, network_reduced_directed, g_vs_peaks, n_it*10, type_walk=type_walk, min_count=min_count, extra_genos_to_track=nodelist_nonpeaks)
	####
	ax[0].scatter(peak_heights, basin_sizes_norm, c='r', s=5)
	ax[0].set_xlabel('peak height')
	ax[0].set_ylabel("P(start in s's basin)")
	ax[0].set_yscale('log')
	###
	ax[1].scatter(peak_heights, peak_reached_prob_if_starting_in_basin, c='r', s=5)
	ax[1].set_xlabel('peak height')
	ax[1].set_ylabel("P(reach s|\nstart in s's basin)")
	ax[1].set_yscale('log')
	ax[1].set_xlim(min(peak_heights) - 1, max(peak_heights)+1)
	####
	ax[2].scatter(peak_heights, peak_reached, c='r', s=5)
	ax[2].set_xlabel('peak height')
	ax[2].set_ylabel("P(reach s)")
	ax[2].set_yscale('log')
	#### prob of random non-peaks to be reached
	ax[3].scatter([node_vs_fitness[n] for n in nodelist_nonpeaks], nonpeak_reached, c='r', s=5)
	ax[3].set_xlabel('fitness of\n'+r'non-peak $g$')
	ax[3].set_ylabel("P(pass g)")
	ax[3].set_yscale('log')
	####
	if print_stats:
		print('global optimum reached', peak_reached[np.argmax(peak_heights)])
		print('min and max P(reach s)', np.nanmin(peak_reached), np.nanmax(peak_reached), 'min by cutoff', 5/n_it)
	####
	if theory:
		delta, p = params['delta'], params['p']
		d_to_plot = [0]+[d for d in range(2, L+1)]
		peak_heights_at_d_theory = [1 + L - d - p * delta + delta if d > 0 else 1 + L - d - p * delta for d in d_to_plot]
		geno_heights_at_d_theory = [1 + L - d - p * delta for d in d_to_plot]
		basin_sizes_theory = [(d/(K-1) + 1) *((K-1)/K)**d if d > 0 else 1 for d in d_to_plot]
		prob_reached_basin_singlespiked_theory_density = [1/(comb(L, d)*(K-1)**d) for d in d_to_plot]
		prob_reached_basin_singlespiked_theory = [(K / (K - 1)) ** d_s * K**(-1*L) * sum([(comb(L - d_s, d_g - d_s) * (K - 1) ** (d_g - d_s)) / comb(d_g, d_s) for d_g in range(d_s, L + 1)]) for d_s in d_to_plot]
		ax[0].plot(peak_heights_at_d_theory, basin_sizes_theory, c='b', ls=':')
		ax[1].plot(peak_heights_at_d_theory, prob_reached_basin_singlespiked_theory, c='b', ls=':', scaley=False)
		#plot_without_autoscale(ax[1], peak_heights_at_d_theory, prob_reached_basin_singlespiked_theory, c='b', ls=':')
		plot_without_autoscale(ax[2], peak_heights_at_d_theory, prob_reached_basin_singlespiked_theory_density, c='cyan', ls=':')
		######
		d_to_plot_genos = [d for d in range(0, L+1)]
		geno_heights_at_d_theory = [1 + L - d - p * delta for d in d_to_plot_genos]
		prob_reached_basin_singlespiked_theory_density_genos = [1/(comb(L, d)*(K-1)**d) for d in d_to_plot_genos]
		plot_without_autoscale(ax[3], geno_heights_at_d_theory, prob_reached_basin_singlespiked_theory_density_genos, c='cyan', ls=':')
	for a in ax.flat:
		a.tick_params(axis='y', labelleft=True)
	return peak_vs_basin

def reduce_Papkou_network(pnetwork, threshold=-0.508, add_neighbours=True, largest_comp=True):
	network_reduced = pnetwork.copy()
	node_vs_fitness = nx.get_node_attributes(pnetwork, 'fitness')
	nodes_to_remove = []
	for n in pnetwork.nodes():
		if add_neighbours:
			if max([node_vs_fitness[n2] for n2 in pnetwork.neighbors(n)] + [node_vs_fitness[n],]) < threshold:
				nodes_to_remove.append(n)
		elif not add_neighbours:
			if node_vs_fitness[n] < threshold:
				nodes_to_remove.append(n)
	network_reduced.remove_nodes_from(nodes_to_remove)
	if largest_comp: 
		components = [c for c in nx.connected_components(network_reduced)]
		print('number of components', len(components))
		nodes = max(components, key=len)
		return network_reduced.subgraph(nodes).copy()
	else:
		return network_reduced.copy()

def plot_without_autoscale(ax, x, y, **kwargs):
	# generated by Claude
	line = Line2D(x, y, **kwargs)
	ax.add_artist(line)  # sets transform to ax.transData and clips to the axes patch
	return line

def make_directed_graph(NC_vs_f, NC_network, return_loopcount = False): #10**(-6)
	NC_directed = nx.DiGraph()
	NC_directed.add_nodes_from([n for n in NC_network.nodes(data=False)])
	assert len(NC_directed.edges()) == 0
	nx.set_node_attributes(NC_directed, NC_vs_f, 'fitness')
	acyclic_graph, loops = 1, 0
	for NC1, NC2 in NC_network.edges():
		if  NC_vs_f[NC1] > NC_vs_f[NC2]:
			NC_directed.add_edge(NC2, NC1)
		elif NC_vs_f[NC2] > NC_vs_f[NC1]:
			NC_directed.add_edge(NC1, NC2)
		else:
			NC_directed.add_edge(NC1, NC2)
			NC_directed.add_edge(NC2, NC1)
			acyclic_graph = 0
			if return_loopcount:
				loops += 1
	nx.set_node_attributes(NC_directed, acyclic_graph, 'acyclic_graph')
	check_directed_network(NC_vs_f, NC_directed)
	if return_loopcount:
		print('# loops', loops)
		return NC_directed, loops
	return NC_directed

def Hamming_dist(p1, p2):
	assert len(p1) == len(p2)
	return len([i for i in range(len(p1)) if p1[i] != p2[i]])

def check_directed_network(NC_vs_f, NC_network):
	for n1, n2 in NC_network.edges():
		assert NC_vs_f[n1] <= NC_vs_f[n2]

def neighbours_seq(seq, alphabet=['A', 'G', 'C', 'T']):
	for c in seq:
		assert c in alphabet
	return [''.join([str(c) if i != pos else new_c for i, c in enumerate(seq)]) for pos in range(len(seq)) for new_c in alphabet if seq[pos] != new_c]

def HoC(pnetwork, new_fitness = False, min_fit = 0, max_fit = 1):
	g_vs_fitness = 	nx.get_node_attributes(pnetwork, name='fitness')
	fitness_list, g_list = deepcopy(list(g_vs_fitness.values())), deepcopy(list(g_vs_fitness.keys()))
	if new_fitness:
		fitness_list = list(np.random.uniform(min_fit, max_fit, size=len(g_list)))
	network_new = nx.Graph()
	network_new.add_nodes_from(pnetwork.nodes(data=False))
	network_new.add_edges_from(pnetwork.edges())
	np.random.shuffle(fitness_list)
	g_vs_fitness_shuffled = {n: f for n, f in zip(g_list, fitness_list)}
	nx.set_node_attributes(network_new, g_vs_fitness_shuffled, name='fitness')
	return network_new	

def add_spikes_above_d0_until_x_peaks(pnetwork, n_peaks, d0, delta=1.5, a=0.1):
	assert a < 0.2
	L = [len(n) for n in pnetwork.nodes()][0]
	node_vs_fitness = {n: 1 + L - Hamming_dist(n, '0'*L) + np.random.uniform(-1*a, a) for n in pnetwork.nodes()}
	peaks = ['0'*L,]
	spikes = set()
	while len(peaks) < n_peaks:
		print('number peaks', len(peaks), 'target', n_peaks)
		new_spike = np.random.choice([n for n, f in node_vs_fitness.items() if Hamming_dist(n, '0'*L) <= d0 and n not in spikes])
		node_vs_fitness[new_spike] +=  delta
		spikes.add(new_spike)
		network_reduced_directed = make_directed_graph(node_vs_fitness, pnetwork)
		peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
		if len(peaks) < n_peaks and len([n for n, f in node_vs_fitness.items() if Hamming_dist(n, '0'*L) <= d0 and n not in spikes]) == 0:
			raise RuntimeError('did not reach enough peaks')
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')	
	for n, f in node_vs_fitness.items():
		base = 1 + L - Hamming_dist(n, '0'*L)
		assert min(abs(base - f), abs(base + delta - f)) <= a * 1.01  # no double spikes
	return pnetwork

def bottleneck_middle_region(pnetwork, n_peaks, delta, a):
	node_vs_fitness = nx.get_node_attributes(pnetwork, name='fitness')
	L = [len(n) for n in pnetwork.nodes()][0]
	for n in node_vs_fitness:
		d = Hamming_dist(n, '1'*L)
		if d + 3 < L and '0' in n[d + 3:]:
			pnetwork.remove_node(n)
	for n in pnetwork:
		d = Hamming_dist(n, '0'*L)
		assert d == 0 or len([n2 for n2 in pnetwork.neighbors(n) if Hamming_dist(n2, '0'*L) == d - 1]) > 0 
		assert d == L or len([n2 for n2 in pnetwork.neighbors(n) if Hamming_dist(n2, '0'*L) == d + 1]) > 0 
	pnetwork = add_spikes_above_d0_until_x_peaks(pnetwork, n_peaks, d0=L + 1, delta=delta, a=a)
	return pnetwork

def turn_RMF_landscape_flat_nonpeaks(pnetwork):
	node_vs_fitness = nx.get_node_attributes(pnetwork, name='fitness')
	network_reduced_directed = make_directed_graph(node_vs_fitness, pnetwork)
	peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	for n, f in node_vs_fitness.items():
		if n not in peaks:
			node_vs_fitness[n] = 1
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')
	return pnetwork

def successful_next_step(d, L, K, p):
	ratio = p*(K-1)*L/((1-p)*d)
	return (1+ratio)**(-1)


def get_accessibility_plot(network_reduced, K, n_it, initialisation='', type_walk='uphill', return_peaks=False, peaks=[]):
	if initialisation == 'antipode':
		L = [len(n) for n in network_reduced.nodes()][0]
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	node_list = [n for n in network_reduced.nodes()]
	if not nx.is_directed(network_reduced):
		network_reduced_directed = make_directed_graph(node_vs_fitness, network_reduced)
		undirected_network = network_reduced.copy()
	else:
		network_reduced_directed = network_reduced.copy()
		undirected_network = network_reduced_directed.to_undirected()
	if len(peaks) == 0:
		peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	peak_heights = [node_vs_fitness[p] for p in peaks]
	fitness_reached_list = []
	peaks_set = set(peaks)
	for iterations in range(n_it):
		if initialisation == '':
			n_initial = node_list[np.random.randint(len(node_list))] 
		elif initialisation == 'antipode':
			n_initial = ''.join([np.random.choice([str(i) for i in range(1, K)]) for j in range(L)])
		else:
			n_initial = np.random.choice(initialisation)
		final_node = run_evolution(n_initial, peaks_set, network_reduced_directed, node_vs_fitness, type_walk=type_walk, record_walk=False, undirected_network=undirected_network)
		fitness_reached_list.append(node_vs_fitness[final_node])
	percentile_list_sorted = sorted([return_percentile_from_top_conservative(f, peak_heights) for f in fitness_reached_list]) 
	percentile_pop = sorted([return_percentile_from_bottom_inclusive(f, percentile_list_sorted) for f in percentile_list_sorted])
	if not return_peaks:
		return 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ])
	elif return_peaks:
		return 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ]), peaks

def return_percentile_from_top_conservative(f, f_list):
	#conservative here means that in the list [9, 8, 8, 8, 5, 5, 5, 5, 4, 3], 8 would be the top-40% and 5 the top-80%
	return len([p for p in f_list if p >= f - 1e-7])/len(f_list) 

def return_percentile_from_bottom_inclusive(f, f_list):
	#conservative here means that in the list [9, 8, 8, 8, 5, 5, 5, 5, 4, 3], 5 would be 60%
	return len([p for p in f_list if p <= f + 1e-7])/len(f_list) 

def get_basins(pnetwork_directed, peaks):
	if len(nx.get_node_attributes(pnetwork_directed, 'acyclic_graph').values()) and min([v for v in nx.get_node_attributes(pnetwork_directed, 'acyclic_graph').values()]) == 1:
		peak_vs_basin = {p: [p]+[s for s in nx.ancestors(pnetwork_directed, p)] for p in peaks}
	else:
		peak_vs_basin = {}
		for p in peaks:
			shortest_paths = nx.shortest_path_length(pnetwork_directed, target=p)
			assert p in shortest_paths
			peak_vs_basin[p] = list(set([g for g in shortest_paths if shortest_paths[g]] + [p]))
	g_vs_peaks = {n: [] for n in pnetwork_directed.nodes()}
	for p, basin in peak_vs_basin.items():
		for g in basin:
			g_vs_peaks[g].append(p)	
	return peak_vs_basin, g_vs_peaks

def RMF_landscape(K, L, p, distribution='spikes', two_cones=False, delta=1.5, a=0.05):
	assert K <= 10 #because assuming one seq position == one char
	node_vs_fitness, node_vs_additive_fitness, pnetwork, node_vs_spike = {}, {}, nx.Graph(), {}
	for seq_tuple in product(list(range(K)), repeat=L):
		d = sum([1 if c != 0 else 0 for i, c in enumerate(seq_tuple)])
		if not two_cones:
			additive_comp = 1 + L  - d
		elif two_cones:
			if d <= L//2:
				additive_comp = L-d-L//2 + 1
			else:
				additive_comp = - 1*  (L-d-L//2) + 1
		if distribution == 'exponential':
			random_comp = np.random.exponential(scale=p)
		elif distribution == 'pareto':
			random_comp = np.random.pareto(a=p) # lomax with shape 2.2
		elif distribution == 'uniform':
			random_comp = np.random.uniform(-1*p, p)
		elif distribution == 'spikes':
			if random.uniform(0, 1) < p:
				random_comp = float(np.random.uniform(-1*a, a)) + delta - delta * p
				node_vs_spike[''.join([str(c) for c in seq_tuple])] = 1
			else:
				random_comp = float(np.random.uniform(-1*a, a)) - delta * p
				node_vs_spike[''.join([str(c) for c in seq_tuple])] = 0
		elif distribution == 'spike_random_height':
			random_comp = float(np.random.uniform(-1*a, a)) + random.choices([np.random.exponential(scale=4), 0], weights=[p, 1-p])[0] 
		else:
			raise RuntimeError('distribution not known', distribution)
		node_vs_fitness[''.join([str(c) for c in seq_tuple])] = additive_comp + random_comp
		node_vs_additive_fitness[''.join([str(c) for c in seq_tuple])] = additive_comp 
		pnetwork.add_node(''.join([str(c) for c in seq_tuple]))
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')
	nx.set_node_attributes(pnetwork, node_vs_additive_fitness, name='additive_fitness')
	if node_vs_spike:
		nx.set_node_attributes(pnetwork, node_vs_spike, name='spike')
	for i, n1 in enumerate(pnetwork.nodes()):
		for n2 in neighbours_seq(n1, alphabet=[str(c) for c in range(K)]):
			pnetwork.add_edge(n1, n2)
	assert len(pnetwork.edges()) == len(pnetwork.nodes()) * L * (K-1) //2
	for n in pnetwork.nodes():
		assert len([n2 for n2 in pnetwork.neighbors(n)]) == L * (K-1)
		assert len([n2 for n2 in pnetwork.neighbors(n) if not Hamming_dist(n2, n) == 1]) == 0
	return pnetwork

def probabilistic_picture_basins(peaks, pnetwork_directed, g_vs_peaks, n_it, min_count=5, type_walk='uphill', extra_genos_to_track = []):
	peak_vs_reached, peak_vs_reached_if_starting_in_basin_norm = {peak: 0 for peak in peaks}, {peak: 0 for peak in peaks}
	prob_reached_extra_genos_to_track = {g: 0 for g in extra_genos_to_track}
	node_vs_fitness = nx.get_node_attributes(pnetwork_directed, 'fitness')
	peaks_set = set(peaks)
	node_list = list(pnetwork_directed.nodes())
	for i in range(n_it):
		n_initial = node_list[np.random.randint(len(node_list))]
		for p_possible in g_vs_peaks[n_initial]:
			peak_vs_reached_if_starting_in_basin_norm[p_possible] += 1
		final_node, walk = run_evolution(n_initial, peaks_set, pnetwork_directed, node_vs_fitness, type_walk=type_walk, record_walk=True)
		peak_vs_reached[final_node] += 1
		for g in set(walk):
			try:
				prob_reached_extra_genos_to_track[g] += 1
			except KeyError:
				pass
	peak_reached_prob_if_starting_in_basin = [peak_vs_reached[p]/peak_vs_reached_if_starting_in_basin_norm[p] if peak_vs_reached[p] >= min_count else np.nan for p in peaks]
	peak_reached = [peak_vs_reached[p]/n_it if peak_vs_reached[p] >= min_count else np.nan for p in peaks]
	if len(extra_genos_to_track):
		prob_reached_extra_genos_to_track = [prob_reached_extra_genos_to_track[p]/n_it if prob_reached_extra_genos_to_track[p] >= min_count else np.nan for p in extra_genos_to_track]
		return peak_reached_prob_if_starting_in_basin, peak_reached, prob_reached_extra_genos_to_track
	else:
		return peak_reached_prob_if_starting_in_basin, peak_reached

def prob_success(d0, dinit, L, K, p, next_step_function = successful_next_step):
	if dinit >= d0 - 1:
		return np.prod([next_step_function(d, L, K, p) for d in range(d0-1, dinit + 1)])
	else:
		return 1



def top_x_percent_cutoff(peak_heights, x):
	# Claude cole
	n_top = (x * len(peak_heights)) // 100   # integer arithmetic avoids float rounding
	assert n_top >= 1
	return sorted(peak_heights, reverse=True)[n_top - 1]

if __name__ == "__main__":
	import random
	l = [9, 8, 8, 8, 5, 5, 5, 5, 4, 3]
	for i in range(2):
		print(l)
		assert abs(return_percentile_from_top_conservative(9, l) - 0.1) < 0.0001
		assert abs(return_percentile_from_top_conservative(8, l) - 0.4) < 0.0001
		assert abs(return_percentile_from_top_conservative(5, l) - 0.8) < 0.0001
		assert abs(return_percentile_from_top_conservative(4, l) - 0.9) < 0.0001
		assert abs(return_percentile_from_top_conservative(3, l) - 1.0) < 0.0001
		random.shuffle(l)
	print('finished percentile test')

