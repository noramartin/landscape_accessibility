import networkx as nx
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functions import *
import matplotlib.ticker as mticker
from varyL_functions import *
import parameters as param
########################################################################################################################
print('setup ')
########################################################################################################################
distribution, K, n_it_giveng, delta, a = param.distribution, param.K, param.n_it_giveng, param.delta, param.a
number_samples_shell = 10**3
n_it = 10**5
params_filename_it = 'sRMFvaryL' +'K'+str(K)+ 'it' + str(int(np.log10(n_it))) +distribution+ 'itshell' + str(int(np.log10(number_samples_shell))) +'itg' + str(int(np.log10(n_it_giveng))) + 'delta'+str(delta) + 'a' + str(a)

########################################################################################################################
print('basins ')
########################################################################################################################
L_list = [10, 20]
f_basins, ax_basins = plt.subplots(ncols=2, nrows=len(L_list),  figsize=(8, 2.5*len(L_list)))
for landscapei, L in enumerate(L_list):
	p = 0.24/((K-1)*L)
	peak_vs_basin_size_lower, peak_vs_reached, peak_vs_fitness = basin_approx(L, K, number_samples_shell, n_it, params={'p': p, 'delta': delta, 'a': a})
	ax_basins[landscapei,0].set_title('L=' +str(L))
	ax_basins[landscapei,1].set_title('L=' +str(L))
	####
	peaks = [peak for peak in peak_vs_basin_size_lower]
	peak_heights = [peak_vs_fitness[peak] for peak in peaks]
	basin_sizes_norm = [peak_vs_basin_size_lower[peak] for peak in peaks]
	ax_basins[landscapei,0].scatter(peak_heights, basin_sizes_norm, c='r', s=5)
	ax_basins[landscapei,0].set_xlabel('peak height')
	ax_basins[landscapei,0].set_ylabel("P(start in s's basin)\n(lower bound counting\ndirect paths only)")
	###
	peak_reached = [peak_vs_reached[peak] for peak in peaks]
	ax_basins[landscapei,1].scatter(peak_heights, peak_reached, c='r', s=5)
	ax_basins[landscapei,1].set_xlabel('peak height')
	ax_basins[landscapei,1].set_ylabel("P(reach s)")
	ax_basins[landscapei,1].set_yscale('log')
	######
	d_to_plot = sorted([Hamming_dist(peak, '0'*len(peak)) for peak in peaks])
	peak_heights_at_d_theory = [L-d + 1 - delta*p + delta if d > 0 else L+ 1 - delta*p for d in d_to_plot]
	basin_sizes_theory = [(d/(K-1) + 1) *((K-1)/K)**d if d > 0 else 1  for d in d_to_plot]
	prob_reached_basin_singlespiked_theory_dilut = [1/(comb(L, d)*(K-1)**d) if d > 0 else 1 for d in d_to_plot]
	ax_basins[landscapei,0].plot(peak_heights_at_d_theory, basin_sizes_theory, c='b', ls=':')
	ax_basins[landscapei,1].plot(peak_heights_at_d_theory, prob_reached_basin_singlespiked_theory_dilut, c='cyan', ls=':')

	[ax_basins[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(2)]

f_basins.tight_layout()
f_basins.savefig('./plots/varyL_basins'+params_filename_it+'.png', bbox_inches='tight', dpi=200)

########################################################################################################################
print('more detailed walk plots of three parameter values: paths - by fitness')
########################################################################################################################
L_list = [10, 20, 30]
f_walks, ax_walks = plt.subplots(ncols=5, nrows=len(L_list),  figsize=(18, 3*len(L_list)))
for landscapei, L in enumerate(L_list):
	p = 0.24/((K-1)*L)
	########################################################################################################################
	print('paths - by fitness')
	########################################################################################################################
	peak_heights_reached, peak_heights_reached_antipode, peak_heights_reference, shell_vs_exit_prob = simulate_landscape_on_the_fly(L, K, number_samples_shell, n_it, n_it_giveng, type_landscape = 'sRMF', params={'p': p, 'delta': delta, 'a': a})
	shells_with_data = sorted([d for d in shell_vs_exit_prob])
	ax_walks[landscapei,0].set_title('L=' + str(L))
	###
	xaxis = [int(d) for d in np.arange(L+1)]
	ax_walks[landscapei,0].plot(xaxis, [comb(L, d)*(K-1)**d for d in xaxis], c='b')
	ax_walks[landscapei,0].set_xlabel(r'shell $d$')
	ax_walks[landscapei,0].set_ylabel('# genotypes')
	###
	expected_no_peaks_per_shell = [no_peaks_theory_shell(d, p, L, K) for d in xaxis]
	ax_walks[landscapei,1].plot(xaxis, expected_no_peaks_per_shell, c='b', zorder=-2)
	ax_walks[landscapei,1].set_xlabel(r'shell $d$')
	ax_walks[landscapei,1].set_ylabel('# peaks')
	##
	expected_frac_peaks_per_shell = [no_peaks_theory_shell(d, p, L, K)/(comb(L, d)*(K-1)**d) for d in xaxis]
	ax_walks[landscapei,2].plot(xaxis, expected_frac_peaks_per_shell, c='b', zorder=-2)
	ax_walks[landscapei,2].set_xlabel(r'shell $d$')
	ax_walks[landscapei,2].set_ylabel('# peaks/# genotypes')
	###
	xaxis_theory = np.arange(2, L+1)
	### prob that next "additively non-neutral" step is fitness-increasing-non-peak
	ax_walks[landscapei,3].plot(xaxis_theory, [1-successful_next_step(d, L, K, p) for d in xaxis_theory], c='b', zorder=-2)
	ax_walks[landscapei,3].scatter(shells_with_data, [1-np.mean(shell_vs_exit_prob[d]) for d in shells_with_data], color='r', s=3)
	ax_walks[landscapei,3].set_xlabel(r'shell $d$')
	ax_walks[landscapei,3].set_ylabel(r'$P_{\text{trapping}} (d)$')
	###
	for initialisation in ['', 'antipode']:
		if initialisation == '':
			percentile_list_sorted = sorted([return_percentile_from_top_conservative(f, peak_heights_reference) for f in peak_heights_reached]) 
			percentile_pop = sorted([return_percentile_from_bottom_inclusive(f, percentile_list_sorted) for f in percentile_list_sorted])
			x_data, y_data = 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ])
		elif initialisation == 'antipode':
			percentile_list_sorted = sorted([return_percentile_from_top_conservative(f, peak_heights_reference) for f in peak_heights_reached_antipode]) 
			percentile_pop = sorted([return_percentile_from_bottom_inclusive(f, percentile_list_sorted) for f in percentile_list_sorted])
			x_data, y_data = 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ])
		ax_walks[landscapei,-1].plot(x_data, y_data, color='r', label={'': 'simulation -\nrandom initialisation', 'antipode': 'simulation -\nfrom antipode'}[initialisation],
			ls={'': '-', 'antipode': ':'}[initialisation], drawstyle='steps-post', zorder=0, alpha=0.8)
	d0_list = np.arange(2, L)
	peaks_per_shell = [no_peaks_theory_shell(d, p, L, K) for d in range(0, L+1)]
	peak_percentile_given_d0 = [sum(peaks_per_shell[:d0])/sum(peaks_per_shell) for d0 in d0_list] 
	theory_antipode_simplified = [prob_success(d0, L, L, K, p) for d0 in d0_list]
	theory_full = [sum([prob_success(d0, start, L, K, p)*comb(L, start)*(K-1)**start for start in range(0, L+1)])/K**L for d0 in d0_list]
	ax_walks[landscapei,-1].plot(100*np.array([0,] + peak_percentile_given_d0 + [1,]), 100*np.array([0,] + theory_full+ [1,]), c='b', label='analytic-\nrandom initialisation', drawstyle='steps-post', zorder=0, alpha=0.8)
	ax_walks[landscapei,-1].plot(100*np.array([0,] + peak_percentile_given_d0 + [1,]), 100*np.array([0,] + theory_antipode_simplified+ [1,]), c='b', label='analytic-\nantipode', drawstyle='steps-post', zorder=0, alpha=0.8, ls=':')
	ax_walks[landscapei,-1].set_xlabel('top x% peaks')
	ax_walks[landscapei,-1].set_ylabel('x% adaptive walks')
	#HoC context
	peak_heights_reached, peak_heights_reached_antipode, peak_heights_reference = simulate_landscape_on_the_fly(L, K, number_samples_shell, n_it, n_it_giveng, type_landscape = 'HoC', shell_exit=False)
	percentile_list_sorted = sorted([return_percentile_from_top_conservative(f, peak_heights_reference) for f in peak_heights_reached]) 
	percentile_pop = sorted([return_percentile_from_bottom_inclusive(f, percentile_list_sorted) for f in percentile_list_sorted])
	x_data, y_data = 100*np.array([0,] + percentile_list_sorted + [1, ]), 100*np.array([0,] + percentile_pop + [1, ])
	ax_walks[landscapei,-1].plot(x_data, y_data, c='k', label='HoC-\nrandom initialisation', drawstyle='steps-post', zorder=0, alpha=0.8, ls=':')
	ax_walks[landscapei,-1].plot([0, 100], [0, 100], c='k', zorder=-5)
	######
	l2 = ax_walks[landscapei,-1].legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, fontsize=7)
	[ax_walks[landscapei,i].annotate(str(landscapei + 1) +'ABCDEFGHIJ'[i], xy=(0.03, 1.05), xycoords='axes fraction', fontsize=13, fontweight='bold') for i in range(5)]
	[ax_walks[landscapei,i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e')) for i in range(2)]
f_walks.tight_layout()
f_walks.savefig('./plots/varyL_walks_fitness'+params_filename_it+'.png', bbox_inches='tight', dpi=200)
