
import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
from math import comb
import parameters as param
from functools import partial

n_it = param.n_it

########################################################################################################################
print('plot small network as example')
########################################################################################################################
L_sketch = 6
K = 3
network_reduced = RMF_landscape(K, L_sketch, p=0.05, delta=1.8, distribution='spikes', two_cones=False, a=param.a)
f, ax = plt.subplots(figsize=(3.8, 3))
draw_network(network_reduced, ax, f, node_size=0.3, linewidths=0.2)
f.tight_layout()
f.savefig('./plots/network_sketch.png', bbox_inches='tight', dpi=500)
########################################################################################################################
print('Papkou landscape - small plot for schematic')
########################################################################################################################
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


	
df_papkou = pd.read_csv('./papkou/fitness_data_wt.csv')
pnetwork = Papkou_to_network(df_papkou)
print('papkou data, total number of genotypes with fitness info', len(pnetwork.nodes()))
network_reduced = reduce_Papkou_network(pnetwork, threshold=-0.508)
print('papkou data, number of genotypes in subgraph', len(network_reduced.nodes()))
f, ax = plt.subplots(ncols=2, figsize=(4, 2))
x_data, y_data = get_accessibility_plot(network_reduced, n_it=n_it, K=4)
ax[0].plot(x_data, y_data, color='r', drawstyle='steps-post')
ax[0].set_title('folA sublandscape 1')
####
network_reduced = RMF_landscape(K=param.K, L=param.L, p=param.p, a=param.a, delta=param.delta, two_cones=False, distribution='spikes')
x_data, y_data = get_accessibility_plot(network_reduced, n_it=n_it, K=2)
ax[1].plot(x_data, y_data, color='r', drawstyle='steps-post')
ax[1].set_title('sRMF landscape')
for i in range(2):
	ax[i].set_xlabel('top-x% peaks')
	ax[i].set_ylabel('x% walkers')
	ax[i].plot([0, 100], [0, 100], c='grey')
	ax[i].set_xticks([0, 50, 100])
	ax[i].set_yticks([0, 50, 100])
f.tight_layout()
f.savefig('./plots/papkou_sRMF'+str(int(np.log10(n_it)))+'walk'+'.png', bbox_inches='tight', dpi=300)

########################################################################################################################
print('plot RYA - contrast with null models')
########################################################################################################################
L_null, K = 15, 2
d0 = 5 #5
params_filename_it_null = 'L'+str(L_null) + 'it' + str(int(np.log10(n_it))) + 'uphill'
landscape_function = partial(RMF_landscape, distribution=param.distribution, two_cones=False, delta=param.delta, a=param.a)
f, ax = plt.subplots(ncols=3, nrows=2, figsize=(9.5, 5))
for networktype in ['RMF', 'RMF_spikes_top']:
	if networktype == 'RMF':
		network_reduced = landscape_function(K, L_null, p=0.01)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = add_spikes_above_d0_until_x_peaks(network_RMF, n_peaks=n_peaks_RMF, d0=d0, delta=param.delta, a=param.a) 
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True, K=K)
	if networktype == 'RMF':
		n_peaks_RMF = len(peaks)
	ax[1, 0].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 0].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], 
					color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_spikes_top': (0, (1, 4))}[networktype])
	ax[0, 0].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], 
					color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_spikes_top': (0, (1, 4))}[networktype])
ax[1, 0].set_xlabel('top-x% peaks')
ax[1, 0].set_ylabel('x% adaptive walks')
ax[1, 0].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 0].set_title('A2) null model:\n'+r'peaks at $d\leq $'+str(d0))
ax[0, 0].set_xlabel('shell d')
ax[0, 0].set_ylabel('number genotypes (+) and\nnumber peaks (x)')
ax[0, 0].set_yscale('log')
ax[0, 0].set_title('A1) null model:\n'+r'peaks at $d\leq $'+str(d0))
##########
for networktype in ['RMF', 'RMF_bottleneck']:
	if networktype == 'RMF':
		network_reduced = landscape_function(K, L_null, p=0.002)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = bottleneck_middle_region(network_RMF, n_peaks=n_peaks_RMF, delta=param.delta, a=param.a)
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True, K=K)
	if networktype == 'RMF':
		n_peaks_RMF = len(peaks)
	ax[1, 1].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 1].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_bottleneck': (0, (1, 4))}[networktype])
	ax[0, 1].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_bottleneck': (0, (1, 4))}[networktype])
ax[1, 1].set_xlabel('top-x% peaks')
ax[1, 1].set_ylabel('x% adaptive walks')
ax[1, 1].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 1].set_title('B2) null model:\nsmaller low-fitness region')
ax[0, 1].set_xlabel('shell d')
ax[0, 1].set_ylabel('number genotypes (+) and\nnumber peaks (x)')
ax[0, 1].set_yscale('log')
ax[0, 1].set_title('B1) null model:\nsmaller low-fitness region')
########
for networktype in ['RMF', 'RMF_flatbasis']:
	if networktype == 'RMF':
		network_reduced = landscape_function(K, L_null, p=0.01)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = turn_RMF_landscape_flat_nonpeaks(network_RMF)
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True, K=K)
	ax[1, 2].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 2].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_flatbasis': (0, (1, 4))}[networktype])
	ax[0, 2].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_flatbasis': (0, (1, 4))}[networktype])
ax[1, 2].set_xlabel('top-x% peaks')
ax[1, 2].set_ylabel('x% adaptive walks')
ax[1, 2].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 2].set_title('C2) null model:\nno guiding gradient')
ax[0, 2].set_xlabel('shell d')
ax[0, 2].set_ylabel('number genotypes (+) and\nnumber peaks (x)')
ax[0, 2].set_yscale('log')
ax[0, 2].set_title('C1) null model:\nno guiding gradient')
f.tight_layout()
f.savefig('./plots/2prmf'+params_filename_it_null+'nullmodels'+'.png', bbox_inches='tight', dpi=200)

#############
f, ax = plt.subplots(ncols=4, figsize=(16, 3.3))
L_sketch = 9
for i in range(4):
	if i == 0:
		network_reduced = landscape_function(K, L_sketch, p=0.02, delta=1.5, a=param.a)
		network_RMF = network_reduced.copy()
		n_peaks_RMF = len(get_accessibility_plot(network_reduced, n_it=2, return_peaks=True, K=K)[-1])
		ax[i].set_title('reference:\nsRMF landscape')
	if i == 1:
		network_reduced = add_spikes_above_d0_until_x_peaks(network_RMF.copy(), n_peaks=n_peaks_RMF, d0=3, delta=1.5, a=param.a)
		ax[i].set_title('null model:\n'+r'peaks at $d\leq 3$')
	if i == 2:
		network_reduced = bottleneck_middle_region(network_RMF.copy(), n_peaks=n_peaks_RMF, delta=1.5, a=param.a)
		ax[i].set_title('null model:\nsmaller low-fitness region')
	if i == 3:
		network_reduced = turn_RMF_landscape_flat_nonpeaks(network_RMF.copy())
		ax[i].set_title('null model:\nno guiding gradient')
	draw_network(network_reduced, ax[i], f, linewidths=0.4, node_size = 5)
[ax[i].annotate('0ABCDEFGHIJ'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/network_sketch_nullmodels.png', bbox_inches='tight', dpi=300)
