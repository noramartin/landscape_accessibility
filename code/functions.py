#! /usr/bin/env python

#     functions.py

# Additional code used by `rmf_plots.py`

# Written by Nora Martin
# Last updated 20260413

import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product
import random

def draw_network(network_reduced, ax, f, linewidths=0.6, node_size = 8):
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = directed_graph_from_discrete_dist(node_vs_fitness, network_reduced)
	peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	shell_vs_nodes = {}
	L_sketch = len([n for n in network_reduced.nodes()][0])
	pos = nx.shell_layout(network_reduced, nlist=[[n for n in network_reduced.nodes() if Hamming_dist(n, '0'*L_sketch) == d] for d in range(0, L_sketch + 1)])
	nodes= nx.draw_networkx_nodes(network_reduced, node_size=node_size, nodelist= network_reduced.nodes(), ax=ax, pos=pos, edgecolors=None, node_color = [node_vs_fitness[n] for n in network_reduced.nodes()])
	nx.draw_networkx_edges(network_reduced, nodelist= network_reduced.nodes(), ax=ax, pos=pos, edge_color='grey', width=linewidths)
	nx.draw_networkx_nodes(network_reduced, node_size=node_size*2, nodelist= peaks, ax=ax, pos=pos, edgecolors='r', node_color = 'none', linewidths=1.5)
	cb=f.colorbar(nodes, ax=ax)
	cb.set_label('fitness')

def run_evolution(n_initial, peaks, directed_network):
	while n_initial not in peaks:
		neighbours = [n for n in directed_network.successors(n_initial)]
		n_initial = deepcopy(neighbours[random.choice(np.arange(len(neighbours)))])	
	return n_initial

def directed_graph_from_discrete_dist(node_vs_fitness, network_undirected, print_neutral_mut = True):
	network_directed = nx.DiGraph()
	network_directed.add_nodes_from([n for n in network_undirected.nodes(data=False)])
	assert len(network_directed.edges()) == 0
	nx.set_node_attributes(network_directed, node_vs_fitness, 'fitness')
	neutral = 0
	for NC1, NC2 in network_undirected.edges():
		if  node_vs_fitness[NC1] > node_vs_fitness[NC2]:
			network_directed.add_edge(NC2, NC1)
		elif node_vs_fitness[NC2] > node_vs_fitness[NC1]:
			network_directed.add_edge(NC1, NC2)
		elif node_vs_fitness[NC2] == node_vs_fitness[NC1]:
			network_directed.add_edge(NC1, NC2)
			network_directed.add_edge(NC2, NC1)
			neutral += 1
	if neutral > 0:
		print('network has', neutral, 'neutral mutations')
	check_directed_network(node_vs_fitness, network_directed)
	return network_directed

def check_directed_network(NC_vs_f, NC_network):
	for n1, n2 in NC_network.edges():
		assert NC_vs_f[n1] <= NC_vs_f[n2]

def Hamming_dist(p1, p2):
	assert len(p1) == len(p2)
	return len([i for i in range(len(p1)) if p1[i] != p2[i]])


def neighbours_seq(seq, alphabet=['A', 'G', 'C', 'T']):
	for c in seq:
		assert c in alphabet
	return [''.join([str(c) if i != pos else new_c for i, c in enumerate(seq)]) for pos in range(len(seq)) for new_c in alphabet if seq[pos] != new_c]

def HoC(pnetwork, new_fitness = False):
	g_vs_fitness = nx.get_node_attributes(pnetwork, name='fitness')
	fitness_list, g_list = deepcopy(list(g_vs_fitness.values())), deepcopy(list(g_vs_fitness.keys()))
	if new_fitness:
		fitness_list = [np.random.uniform(0, 1) for i in range(len(g_list))]
	network_new = nx.Graph()
	network_new.add_nodes_from(pnetwork.nodes(data=False))
	network_new.add_edges_from(pnetwork.edges())
	random.shuffle(fitness_list)
	g_vs_fitness_shuffled = {n: f for n, f in zip(g_list, fitness_list)}
	nx.set_node_attributes(network_new, g_vs_fitness_shuffled, name='fitness')
	return network_new	

def simpler_RMF_landscape_single_coeff(L, p, delta=1.5):
	node_vs_fitness, pnetwork = {}, nx.Graph()
	for seq_tuple in product(list(range(2)), repeat=L):
		node_vs_fitness[''.join([str(c) for c in seq_tuple])] = L - Hamming_dist(seq_tuple, [0,]*L) + random.choices([delta, 0], weights=[p, 1-p])[0]
		pnetwork.add_node(''.join([str(c) for c in seq_tuple]))
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')
	for i, n1 in enumerate(pnetwork.nodes()):
		for n2 in neighbours_seq(n1, alphabet=[str(c) for c in range(2)]):
			pnetwork.add_edge(n1, n2)
	assert len(pnetwork.edges()) == len(pnetwork.nodes()) * L //2
	for n in pnetwork.nodes():
		assert len([n2 for n2 in pnetwork.neighbors(n)]) == L
		assert len([n2 for n2 in pnetwork.neighbors(n) if not Hamming_dist(n2, n) == 1]) == 0
	return pnetwork

def add_spikes_above_d0_until_x_peaks(pnetwork, n_peaks, d0, delta=1.5):
	L = [len(n) for n in pnetwork.nodes()][0]
	node_vs_fitness = {n: L - Hamming_dist(n, '0'*L) for n in pnetwork.nodes()}
	node_vs_fitness_smooth = {n: f for n, f in node_vs_fitness.items()}
	nodes_d0 = [n for n in pnetwork.nodes() if Hamming_dist(n, '0'*L) <= d0]
	peaks = ['0'*L,]
	while len(peaks) < n_peaks:
		print('number peaks', len(peaks), 'target', n_peaks)
		try:
			potential_spikes = [n for n in nodes_d0 if abs(node_vs_fitness[n] - node_vs_fitness_smooth[n]) < 0.01]
			new_spike = deepcopy(potential_spikes[random.choice(np.arange(len(potential_spikes)))])
		except IndexError: # no more genotypes to choose; start from scratch
			assert len(potential_spikes) == 0
			node_vs_fitness = {n: f for n, f in node_vs_fitness_smooth.items()}
		node_vs_fitness[new_spike] = node_vs_fitness_smooth[new_spike] +  delta
		network_reduced_directed = directed_graph_from_discrete_dist(node_vs_fitness, pnetwork)
		peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')	
	return pnetwork

def bottleneck_middle_region(pnetwork, n_peaks):
	node_vs_fitness = nx.get_node_attributes(pnetwork, name='fitness')
	L = [len(n) for n in pnetwork.nodes()][0]
	for n in node_vs_fitness:
		d = Hamming_dist(n, '1'*L)
		if d + 3 < L and '0' in n[d + 3:]:
			pnetwork.remove_node(n)
	for p in pnetwork:
		d = Hamming_dist(n, '0'*L)
		assert d == 0 or len([n2 for n2 in pnetwork.neighbors(n) if Hamming_dist(n2, '0'*L) == d - 1]) > 0 
		assert d == L or len([n2 for n2 in pnetwork.neighbors(n) if Hamming_dist(n2, '0'*L) == d + 1]) > 0 
	pnetwork = add_spikes_above_d0_until_x_peaks(pnetwork, n_peaks, d0=L + 1, delta=1.5)
	return pnetwork

def turn_RMF_landscape_flat_nonpeaks(pnetwork):
	node_vs_fitness = nx.get_node_attributes(pnetwork, name='fitness')
	network_reduced_directed = directed_graph_from_discrete_dist(node_vs_fitness, pnetwork)
	peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	for n, f in node_vs_fitness.items():
		if n not in peaks:
			node_vs_fitness[n] = 1
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')
	return pnetwork

def get_accessibility_plot(network_reduced, n_it, initialisation='', return_peaks=False):
	assert initialisation in ['', 'antipode']
	if initialisation == 'antipode':
		L = [len(n) for n in network_reduced.nodes()][0]
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = directed_graph_from_discrete_dist(node_vs_fitness, network_reduced)
	peaks = [n for n in node_vs_fitness if network_reduced_directed.out_degree(n) == 0]
	print('number peaks', len(peaks))
	peak_heights = [node_vs_fitness[p] for p in peaks]
	fitness_reached_list = []
	for iterations in range(n_it):
		if initialisation == '':
			nodes = [n for n in network_reduced.nodes()]
			n_initial = deepcopy(nodes[random.choice(np.arange(len(nodes)))])
		elif initialisation == 'antipode':
			n_initial = '1'*L
		else:
			raise RuntimeError('initialisation not implemented')
		final_node = run_evolution(n_initial, peaks, network_reduced_directed)
		fitness_reached_list.append(node_vs_fitness[final_node])
	percentile_list_sorted, percentile_pop = process_fitness_to_rank_plots(fitness_reached_list, peak_heights)
	if not return_peaks:
		return 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ])
	elif return_peaks:
		return 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ]), peaks

def process_fitness_to_rank_plots(fitness_reached_list, peak_heights):
	percentile_list_sorted = sorted([return_percentile_from_top_conservative(f, peak_heights) for f in fitness_reached_list]) 
	percentile_list_sorted_nodupl = []
	for f in percentile_list_sorted:
		if len([f2 for f2 in percentile_list_sorted_nodupl if abs(f2 - f) < 0.1/len(peak_heights)]) == 0:
			percentile_list_sorted_nodupl.append(f)
	percentile_pop = sorted([return_percentile_from_bottom_inclusive(f, percentile_list_sorted, min_spacing = 0.1/len(peak_heights)) for f in percentile_list_sorted_nodupl])
	return percentile_list_sorted_nodupl, percentile_pop

def return_percentile_from_top_conservative(f, f_list):
	#conservative here means that in the list [9, 8, 8, 8, 5, 5, 5, 5, 4, 3], 8 would be the top-40% and 5 the top-80%
	return len([p for p in f_list if p >= f - 0.0001])/len(f_list)

def return_percentile_from_bottom_inclusive(f, f_list, min_spacing = 0.00001):
	return len([p for p in f_list if p <= f + min_spacing])/len(f_list) 

def Papkou_to_network(df_papkou, seq_column='SV', fitness_column='m', alphabet=['A', 'G', 'C', 'T']):
	pnetwork = nx.Graph()
	node_vs_fitness = {}
	for rowi, row in df_papkou.iterrows():
		if not np.isnan(float(row[fitness_column])):
			seq = row[seq_column]
			pnetwork.add_node(seq)
			node_vs_fitness[seq] = float(row[fitness_column])
	for i, n1 in enumerate(pnetwork.nodes()):
		for n2 in neighbours_seq(n1, alphabet=alphabet):
			if n2 in pnetwork.nodes():
				pnetwork.add_edge(n1, n2)
	nx.set_node_attributes(pnetwork, node_vs_fitness, name='fitness')
	return pnetwork

def reduce_Papkou_network(pnetwork, threshold=-0.508):
	network_reduced = pnetwork.copy()
	node_vs_fitness = nx.get_node_attributes(pnetwork, 'fitness')
	nodes_to_remove = []
	for n in pnetwork.nodes():
		if max([node_vs_fitness[n2] for n2 in pnetwork.neighbors(n)] + [node_vs_fitness[n],]) < threshold:
			nodes_to_remove.append(n)
	network_reduced.remove_nodes_from(nodes_to_remove)
	components = [c for c in nx.connected_components(network_reduced)]
	print('number of components', len(components))
	nodes = max(components, key=len)
	return network_reduced.subgraph(nodes).copy()
	

if __name__ == "__main__":
	########################################################################################################################
	import random
	l = [9, 8, 8, 8, 5, 5, 5, 5, 4, 3]
	for i in range(5):
		print(l)
		assert abs(return_percentile_from_top_conservative(9, l) - 0.1) < 0.0001
		assert abs(return_percentile_from_top_conservative(8, l) - 0.4) < 0.0001
		assert abs(return_percentile_from_top_conservative(5, l) - 0.8) < 0.0001
		assert abs(return_percentile_from_top_conservative(4, l) - 0.9) < 0.0001
		assert abs(return_percentile_from_top_conservative(3, l) - 1.0) < 0.0001
		random.shuffle(l)
	print('finished percentile test')
	########################################################################################################################
	fitness_reached_list = [9, 9, 9, 8, 8, 5, 3, 3, 3, 3]
	random.shuffle(fitness_reached_list)
	percentile_list_sorted, percentile_pop = process_fitness_to_rank_plots(fitness_reached_list, peak_heights = l)
	percentile_list_sorted_test, percentile_pop_test = [0.1, 0.4, 0.8, 1.0], [0.3, 0.5, 0.6, 1.0]
	for i in range(len(percentile_list_sorted)):
		assert abs(percentile_list_sorted[i] - percentile_list_sorted_test[i]) < 0.0001
		assert abs(percentile_pop[i] - percentile_pop_test[i]) < 0.0001
	assert len(percentile_list_sorted) == len(percentile_list_sorted_test)
	########################################################################################################################
	fitness_reached_list = [9, 5, 5, 5, 3, 3, 3, 3, 5]
	random.shuffle(fitness_reached_list)
	percentile_list_sorted, percentile_pop = process_fitness_to_rank_plots(fitness_reached_list, peak_heights = l)
	percentile_list_sorted_test, percentile_pop_test = [0.1, 0.8, 1.0], [1/9, 5/9, 1.0]
	for i in range(len(percentile_list_sorted)):
		assert abs(percentile_list_sorted[i] - percentile_list_sorted_test[i]) < 0.0001
		assert abs(percentile_pop[i] - percentile_pop_test[i]) < 0.0001
	assert len(percentile_list_sorted) == len(percentile_list_sorted_test)
	########################################################################################################################
	fitness_reached_list = [5, 5, 5, 3, 4, 4, 4, 3, 3, 3, 5]
	l = [9, 8, 8, 8, 5, 5, 5, 5, 4, 3, 3]
	random.shuffle(fitness_reached_list)
	percentile_list_sorted, percentile_pop = process_fitness_to_rank_plots(fitness_reached_list, peak_heights = l)
	percentile_list_sorted_test, percentile_pop_test = [8/11, 9/11, 1.0], [4/11, 7/11, 1.0]
	for i in range(len(percentile_list_sorted)):
		assert abs(percentile_list_sorted[i] - percentile_list_sorted_test[i]) < 0.0001
		assert abs(percentile_pop[i] - percentile_pop_test[i]) < 0.0001
	assert len(percentile_list_sorted) == len(percentile_list_sorted_test)
	########################################################################################################################
	fitness_reached_list = [5, 5, 4, 4, 4, 4, 9, 9, 9, 5, 5]
	random.shuffle(fitness_reached_list)
	percentile_list_sorted, percentile_pop = process_fitness_to_rank_plots(fitness_reached_list, peak_heights = l)
	percentile_list_sorted_test, percentile_pop_test = [1/11, 8/11, 9/11], [3/11, 7/11, 1.0]
	for i in range(len(percentile_list_sorted)):
		assert abs(percentile_list_sorted[i] - percentile_list_sorted_test[i]) < 0.0001
		assert abs(percentile_pop[i] - percentile_pop_test[i]) < 0.0001
	assert len(percentile_list_sorted) == len(percentile_list_sorted_test)
	########################################################################################################################
	import matplotlib
	matplotlib.use('Agg') 
	import matplotlib.pyplot as plt
	f, ax = plt.subplots()
	ax.plot(percentile_list_sorted_test, percentile_pop_test, ls=':', c='r', drawstyle='steps-post')
	ax.plot(percentile_list_sorted_test, percentile_pop_test, ls='--', c='b', zorder=-5)
	ax.set_xlabel('top x% peaks')
	ax.set_ylabel('x% walkers')
	ax.legend()
	f.tight_layout()
	f.savefig('./plots/test_steps.png', dpi=200, bbox_inches='tight')

