#! /usr/bin/env python

#     rmf_plots.py

# Generates figures 1A, 1C, 2, 3, S1, and S2 of the paper

# Written by Nora Martin
# Last updated 20260413 by Kye Hunter

import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
#from final_functions import *
from math import comb
from scipy.stats import binom
import os

if not os.path.exists('./plots'):
    os.mkdir('./plots')

########################################################################################################################
print('plots for L=15, p=0.01')
########################################################################################################################
L, p = 15, 0.01
n_it = 10**4
params_filename = 'L'+str(L)+'_100p'+str(int(100*p)) 
params_filename_it = params_filename + 'it' + str(int(np.log10(n_it))) + 'uphill'

########################################################################################################################
print('plot number of genotypes and peaks per shell')
########################################################################################################################
data_peak_numbers, data_peak_densities, data_peak_transition_prob, data_no_genotypes = {}, {}, {}, {}
for iteration in range(10**2):
	network_reduced = simpler_RMF_landscape_single_coeff(L, p)
	node_vs_fitness = nx.get_node_attributes(network_reduced, 'fitness')
	network_reduced_directed = directed_graph_from_discrete_dist(node_vs_fitness, network_reduced)
	####
	shell_vs_no_genotypes, shell_vs_no_peaks, shell_vs_list_peak_transition_prob = {d: 0 for d in range(L + 1)}, {d: 0 for d in range(L + 1)}, {d: [] for d in range(L + 1)}
	for g, fit in node_vs_fitness.items():
		d = Hamming_dist([int(x) for x in g], [0,]*L) 
		assert abs(L-d - fit) < 0.01 or abs(L-d + 1.5 - fit) < 0.01
		shell_vs_no_genotypes[d] += 1
		if network_reduced_directed.out_degree(g) == 0:
			shell_vs_no_peaks[d] += 1
		else:
			shell_vs_list_peak_transition_prob[d].append(len([p for p in network_reduced_directed.successors(g) if network_reduced_directed.out_degree(p) == 0])/network_reduced_directed.out_degree(g))
	data_peak_numbers[iteration] = [shell_vs_no_peaks[d] for d in range(L+1)]
	data_no_genotypes[iteration] = [shell_vs_no_genotypes[d] for d in range(L+1)]
	data_peak_densities[iteration] = [shell_vs_no_peaks[d]/shell_vs_no_genotypes[d] for d in range(L+1)]
	data_peak_transition_prob[iteration] = [np.mean(shell_vs_list_peak_transition_prob[d]) if len(shell_vs_list_peak_transition_prob[d]) else np.nan for d in range(L+1)]
	for d in range(L+1):
		if len(shell_vs_list_peak_transition_prob[d]) == 0:
			assert d == 0 or d == L
			if d == 0:
				assert Hamming_dist(max(node_vs_fitness, key=node_vs_fitness.get), ''.join(['0']*L)) == 0
			if d == L:
				assert len([f for f in node_vs_fitness.values() if f < node_vs_fitness[''.join(['1']*L)]- 0.1]) > 0

f, ax = plt.subplots(ncols=4, figsize=(11, 2.2))
xaxis = np.arange(L+1)
mean_no_geno_shell = [np.mean([data[d] for data in data_no_genotypes.values()]) for d in xaxis]
std_no_geno_shell = [np.std([data[d] for data in data_no_genotypes.values()]) for d in xaxis]
ax[0].errorbar(xaxis, mean_no_geno_shell, yerr=std_no_geno_shell, c='r', fmt='o', markersize=4, linestyle='none')
ax[0].plot(xaxis, [comb(L, d) for d in xaxis], c='b')
ax[0].set_xlabel('shell d')
ax[0].set_ylabel('# genotypes')
###
expected_no_peaks_per_shell = [comb(L, d)*p*(1-p)**d if d > 0 else (1-p)**(L+1) + p for d in xaxis]
mean_no_peaks_shell = [np.mean([data[d] for data in data_peak_numbers.values()]) for d in xaxis]
std_no_peaks_shell = [np.std([data[d] for data in data_peak_numbers.values()]) for d in xaxis]
ax[1].errorbar(xaxis, mean_no_peaks_shell, yerr=std_no_peaks_shell, c='r', fmt='o', markersize=4, linestyle='none')
ax[1].plot(xaxis, expected_no_peaks_per_shell, c='b')
ax[1].set_xlabel('shell d')
ax[1].set_ylabel('# peaks')
##
expected_peak_density_per_shell = [p*(1-p)**d if d > 0 else p + (1-p)**(L+1) for d in xaxis]
mean_peak_density_shell = [np.mean([data[d] for data in data_peak_densities.values()]) for d in xaxis]
std_peak_density_shell = [np.std([data[d] for data in data_peak_densities.values()]) for d in xaxis]
ax[2].errorbar(xaxis, mean_peak_density_shell, yerr=std_peak_density_shell, c='r', fmt='o', markersize=4, linestyle='none')
ax[2].plot(xaxis, expected_peak_density_per_shell, c='b')
ax[2].set_xlabel('shell d')
ax[2].set_ylabel('# peaks/# genotypes')
###
xaxis = np.arange(L+1)
xaxis_theory = np.arange(2, L+1)
mean_peak_transition_shell = [np.mean([data[d] for data in data_peak_transition_prob.values() if not np.isnan(data[d])]) for d in xaxis]
std_peak_transition_shell = [np.std([data[d] for data in data_peak_transition_prob.values() if not np.isnan(data[d])]) for d in xaxis]
ax[3].errorbar(xaxis, mean_peak_transition_shell, yerr=std_peak_transition_shell, c='r', fmt='o', markersize=4, linestyle='none')
ax[3].plot(xaxis_theory, [sum([binom.pmf(j, L-d, p) * (d*p + j)/(d + j) for j in range(0, L-d + 1)]) for d in xaxis_theory], c='b')
ax[3].plot(xaxis, [p,]*len(xaxis), c='grey', ls=':')
ax[3].set_xlabel('shell d')
ax[3].set_ylabel(r'peak transition prob. $P_{\text{pt}}$')
ax[3].set_yscale('log')
[ax[i].annotate('ABCDEFGHIJ'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/peaks_in_shells'+params_filename+'.png', bbox_inches='tight', dpi=200)

####
########################################################################################################################
print('plot RYA with prediction')
########################################################################################################################
def peak_transition(d, L, p):
	return sum([binom.pmf(j, L-d - 1, p) * (d*p + j)/(d + j) for j in range(0, L-d)])

def backward_peak_transition(d, L, p):
	return sum([binom.pmf(j, L-d - 1, p) * j/(d + j) for j in range(0, L-d)])

def prob_no_peak_transition(start, end, L, p):
	if start > end:
		return np.prod([1-peak_transition(d1, L, p) for d1 in range(end + 1, start + 1)]) * (1-backward_peak_transition(end, L, p))
	elif start == end:
		return (1-backward_peak_transition(start, L, p))
	else:
		return 1

d0_list = np.arange(2, L)
peaks_per_shell = [p + (1-p)**(L+1)] + [comb(L, d)*p*(1-p)**d for d in range(1, L+1)] 
peak_percentile_given_d0 = [sum(peaks_per_shell[1:d0])/sum(peaks_per_shell[1:]) for d0 in d0_list] # neglect the peaks at zero
theory_antipode_simplified = [(1-backward_peak_transition(d0 -1, L, p))*(1-peak_transition(d0, L, p))**(L-d0 + 1) for d0 in d0_list]
theory_full = [sum([prob_no_peak_transition(start, end=d0 -1, L=L, p=p)*comb(L, start) for start in range(0, L+1)])/2**L for d0 in d0_list]
#######
network_reduced = simpler_RMF_landscape_single_coeff(L, p)
f, ax = plt.subplots(figsize=(5, 2.5))
for initialisation in ['', 'antipode']:
	x_data, y_data = get_accessibility_plot(network_reduced, initialisation=initialisation, n_it=n_it)
	ax.plot(x_data, y_data, color='r', label={'': 'simulation -\nrandom initialisation', 'antipode': 'simulation -\nfrom antipode'}[initialisation],
		ls={'': '-', 'antipode': ':'}[initialisation], drawstyle='steps-post', zorder=0, alpha=0.8)
	if initialisation == '':
		ax.plot(100*np.array([0,] + peak_percentile_given_d0 + [1,]), 100*np.array([0,] + theory_full+ [1,]), c='b', label='analytic-\nrandom initialisation', drawstyle='steps-post', zorder=0, alpha=0.8)
	else:
		ax.plot(100*np.array([0,] + peak_percentile_given_d0+ [1,]), 100*np.array([0,] + theory_antipode_simplified+ [1,]), c='b', ls=':', label='simplified analytic-\nfrom antipode', drawstyle='steps-post', zorder=0, alpha=0.8)
#HoC context
network_HoC = HoC(network_reduced, new_fitness = True)
x_data_Hoc, y_data_HoC = get_accessibility_plot(network_HoC, n_it, initialisation='')
ax.plot(x_data_Hoc, y_data_HoC, color='k', ls=':', label='simulation -\nHouse-of-Cards', drawstyle='steps-post', zorder=-5, alpha=0.8)
###
ax.set_xlabel('top x% peaks')
ax.set_ylabel('x% adaptive walks')
ax.plot([0, 100], [0, 100], c='grey', zorder=-5, alpha=0.8)
l2 = plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, fontsize=7)
f.tight_layout()
f.savefig('./plots/2prmf_rya'+params_filename_it+'.png', bbox_inches='tight', dpi=200)

########################################################################################################################
print('plot small network as example')
########################################################################################################################
L_sketch = 8
network_reduced = simpler_RMF_landscape_single_coeff(L=L_sketch, p=0.05, delta=1.8)
f, ax = plt.subplots(figsize=(3.8, 3))
draw_network(network_reduced, ax, f)
f.tight_layout()
f.savefig('./plots/network_sketch.png', bbox_inches='tight', dpi=300)

########################################################################################################################
print('Papkou landscape - small plot for schematic')
########################################################################################################################
df_papkou = pd.read_csv('./papkou_2023_folA.csv')
pnetwork = Papkou_to_network(df_papkou)
print('papkou data, total number of genotypes with fitness info', len(pnetwork.nodes()))
network_reduced = reduce_Papkou_network(pnetwork, threshold=-0.508)
print('papkou data, number of genotypes in subgraph', len(network_reduced.nodes()))
f, ax = plt.subplots(ncols=2, figsize=(4, 2))
x_data, y_data = get_accessibility_plot(network_reduced, n_it=n_it)
ax[0].plot(x_data, y_data, color='r', drawstyle='steps-post')
ax[0].set_title('folA sublandscape')
####
network_reduced = simpler_RMF_landscape_single_coeff(L, p)
x_data, y_data = get_accessibility_plot(network_reduced, n_it=n_it)
ax[1].plot(x_data, y_data, color='r', drawstyle='steps-post')
ax[1].set_title('sRMF landscape')
for i in range(2):
	ax[i].set_xlabel('top x% peaks')
	ax[i].set_ylabel('x% walkers')
	ax[i].plot([0, 100], [0, 100], c='grey')
	ax[i].set_xticks([0, 50, 100])
	ax[i].set_yticks([0, 50, 100])
f.tight_layout()
f.savefig('./plots/papkou_sRMF'+params_filename_it+'walk'+'.png', bbox_inches='tight', dpi=100)

########################################################################################################################
print('plot RYA - contrast with null models')
########################################################################################################################
L_null = 15
params_filename_null = 'L'+str(L_null)+'_100p'+str(int(100*p)) 
params_filename_it_null = params_filename_null + 'it' + str(int(np.log10(n_it))) + 'uphill'
f, ax = plt.subplots(ncols=3, nrows=2, figsize=(9.5, 5))
for networktype in ['RMF', 'RMF_spikes_top']:
	if networktype == 'RMF':
		network_reduced = simpler_RMF_landscape_single_coeff(L_null, p=0.01)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = add_spikes_above_d0_until_x_peaks(network_RMF, n_peaks=n_peaks_RMF, d0=5, delta=1.5) #d0=4
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True)
	if networktype == 'RMF':
		n_peaks_RMF = len(peaks)
	ax[1, 0].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 0].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], 
					color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_spikes_top': (0, (1, 4))}[networktype])
	ax[0, 0].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], 
					color={'RMF': 'r', 'RMF_spikes_top': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_spikes_top': (0, (1, 4))}[networktype])
ax[1, 0].set_xlabel('top x% peaks')
ax[1, 0].set_ylabel('x% adaptive walks')
ax[1, 0].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 0].set_title('A2) null model:\n'+r'peaks only at $d\leq 5$')
ax[0, 0].set_xlabel('shell d')
ax[0, 0].set_ylabel('number nodes (+)/\nnumber peaks (x)')
ax[0, 0].set_yscale('log')
ax[0, 0].set_title('A1) null model:\n'+r'peaks at $d\leq 5$')
##########
for networktype in ['RMF', 'RMF_bottleneck']:
	if networktype == 'RMF':
		network_reduced = simpler_RMF_landscape_single_coeff(L_null, p=0.001)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = bottleneck_middle_region(network_RMF, n_peaks=n_peaks_RMF)
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True)
	if networktype == 'RMF':
		n_peaks_RMF = len(peaks)
	ax[1, 1].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 1].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_bottleneck': (0, (1, 4))}[networktype])
	ax[0, 1].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_bottleneck': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_bottleneck': (0, (1, 4))}[networktype])
ax[1, 1].set_xlabel('top x% peaks')
ax[1, 1].set_ylabel('x% adaptive walks')
ax[1, 1].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 1].set_title('B2) null model:\nsmaller low-fitness-region')
ax[0, 1].set_xlabel('shell d')
ax[0, 1].set_ylabel('number nodes (+)/\nnumber peaks (x)')
ax[0, 1].set_yscale('log')
ax[0, 1].set_title('B1) null model:\nsmaller low-fitness-region')
########
for networktype in ['RMF', 'RMF_flatbasis']:
	if networktype == 'RMF':
		network_reduced = simpler_RMF_landscape_single_coeff(L_null, p=0.01)
		network_RMF = network_reduced.copy()
	else:
		network_reduced = turn_RMF_landscape_flat_nonpeaks(network_RMF)
	x_plot, y_plot, peaks = get_accessibility_plot(network_reduced, n_it=n_it, return_peaks=True)
	ax[1, 2].plot(x_plot, y_plot, color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], drawstyle='steps-post', alpha=0.8)
	ax[0, 2].plot(np.arange(0, L_null + 1), [len([n for n in network_reduced.nodes() if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], marker='+', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_flatbasis': (0, (1, 4))}[networktype])
	ax[0, 2].plot(np.arange(0, L_null + 1), [len([n for n in peaks if Hamming_dist('0'*L_null, n) == d]) for d in range(L_null + 1)], color={'RMF': 'r', 'RMF_flatbasis': 'grey'}[networktype], marker='x', alpha=0.5, ls={'RMF': (0, (1, 5)), 'RMF_flatbasis': (0, (1, 4))}[networktype])
ax[1, 2].set_xlabel('top x% peaks')
ax[1, 2].set_ylabel('x% adaptive walks')
ax[1, 2].plot([0, 100], [0, 100], c='k', ls=':', zorder=-3)
ax[1, 2].set_title('C2) null model:\nno guiding gradient')
ax[0, 2].set_xlabel('shell d')
ax[0, 2].set_ylabel('number nodes (+)/\nnumber peaks (x)')
ax[0, 2].set_yscale('log')
ax[0, 2].set_title('C1) null model:\nno guiding gradient')
f.tight_layout()
f.savefig('./plots/2prmf'+params_filename_it_null+'nullmodels'+'.png', bbox_inches='tight', dpi=200)

#############
f, ax = plt.subplots(ncols=4, figsize=(16, 3.3))
L_sketch = 9
for i in range(4):
	if i == 0:
		network_reduced = simpler_RMF_landscape_single_coeff(L_sketch, p=0.02)
		network_RMF = network_reduced.copy()
		n_peaks_RMF = len(get_accessibility_plot(network_reduced, n_it=2, return_peaks=True)[-1])
		ax[i].set_title('reference:\nsRMF landscape')
	if i == 1:
		network_reduced = add_spikes_above_d0_until_x_peaks(network_RMF.copy(), n_peaks=n_peaks_RMF, d0=2, delta=1.5)
		ax[i].set_title('null model:\n'+r'peaks at $d\leq 2$')
	if i == 2:
		network_reduced = bottleneck_middle_region(network_RMF.copy(), n_peaks=n_peaks_RMF)
		ax[i].set_title('null model:\nsmaller low-fitness-region')
	if i == 3:
		network_reduced = turn_RMF_landscape_flat_nonpeaks(network_RMF.copy())
		ax[i].set_title('null model:\nno guiding gradient')
	draw_network(network_reduced, ax[i], f, linewidths=0.4, node_size = 5)
[ax[i].annotate('0ABCDEFGHIJ'[i], xy=(0.05, 1.05), xycoords='axes fraction', fontsize=17, fontweight='bold') for i in range(4)]
f.tight_layout()
f.savefig('./plots/network_sketch_nullmodels.png', bbox_inches='tight', dpi=300)

