#! /usr/bin/env python

#     fola_ruggedness_inference.py

# Written by Kye Hunter
# Last updated 20261021

import itertools, sys
import scipy, matplotlib, uncertainties, uncertainties.unumpy
import rugged_maps

matplotlib.rcParams.update({ 
    'font.size':  14,
    'font.family' : 'serif',
    'font.sans-serif': 'cm',
    'mathtext.default': 'it',
    'mathtext.fontset': 'cm',
    'figure.figsize': (16,5)
})

def holling_3(Xs, a, b, c):
    return [a * x ** 2 / (a * b * x ** 2 + 1) + c for x in Xs]

def fit_and_plot(cutoff, axes):
    file_base = 'fitnesses_own_calculation_ref_GCCGATCTC_rep_'
    num_reps = 6
    if cutoff == 'cutoff_DE':
        log_file = 'fola_ruggedness_inference_cutoff_DE_log.txt'
        output_file = 'fola_ruggedness_inference_cutoff_DE.pdf'
        plot_title = '$folA$ sublandscape\nSecond codon ∈ {D, E}'
    elif cutoff == 'cutoff_CDE':
        log_file = 'fola_ruggedness_inference_cutoff_CDE_log.txt'
        output_file = 'fola_ruggedness_inference_cutoff_CDE.pdf'
        plot_title = '$folA$ sublandscape\nSecond codon ∈ {C, D, E}'
    elif cutoff == 'full':
        log_file = 'fola_ruggedness_inference_full_log.txt'
        output_file = 'fola_ruggedness_inference_full.pdf'
        plot_title = '$folA$ full landscape\n'
    else:
        raise RuntimeError
    sys.stdout = open(log_file, 'w')
    landscape_list = [[
        '_'.join(str(val) for val in comb)
        for comb in itertools.combinations(range(num_reps), length)
    ] for length in range(1, num_reps + 1)]
    num_peaks = [0] * num_reps
    for ilength, length in enumerate(landscape_list):
        temp_num_peaks = [0] * len(length)
        for icomb, comb in enumerate(length):
            graph = rugged_maps.make_graph(
                graph_type = f'point_mutations,{file_base}{comb}.csv',
            )[0]
            print(f'Landscape of replicates {comb}')
            graph.calc_peaks(check_plateaus = True)
            if cutoff == 'cutoff_DE':
                temp_num_peaks[icomb] = len(tuple(
                    inode for inode in graph.peaks 
                    if rugged_maps.genetic_code_1[graph[inode].sequence[3:6]] 
                    in ('D', 'E')
                ))
            elif cutoff == 'cutoff_CDE':
                temp_num_peaks[icomb] = len(tuple(
                    inode for inode in graph.peaks 
                    if rugged_maps.genetic_code_1[graph[inode].sequence[3:6]] 
                    in ('C', 'D', 'E')
                ))
            elif cutoff == 'full':
                temp_num_peaks[icomb] = len(graph.peaks)
            print(f'Number of peaks: {temp_num_peaks[icomb]}')
        num_peaks[ilength] = temp_num_peaks
    num_peaks_flat = [val for length in num_peaks for val in length]
    num_peaks_mean = [sum(length) / len(length) for length in num_peaks]
    uncertainty_flat = [
        1 / len(landscape.split('_')) ** 0.5 
        for length in landscape_list for landscape in length
    ]
    uncertainty_mean = [
        1 / len(length[0].split('_')) ** 0.5 for length in landscape_list
    ]

    result1, var1, info1, _, _ = scipy.optimize.curve_fit(
        holling_3, uncertainty_mean, num_peaks_mean, 
        # Bounds for a positive-sloped function
        bounds = ([0, 0, 0], [float('inf'), float('inf'), float('inf')]), 
        full_output = True
    )
    squared_residuals1 = sum(info1['fvec'] ** 2)
    print('Fit 1:')
    print((result1, var1, info1))
    print(f'Sum of squared residuals = {squared_residuals1}')
    Xs_fit = [val / 100 for val in range(101)]
    Ys_1_expected = holling_3(Xs_fit, *result1)
    print(f'Predicted peaks at 0 noise: {Ys_1_expected[0]}')
    Ys_1_std = [
        val.std_dev for val 
        in holling_3(Xs_fit, *uncertainties.correlated_values(result1, var1))
    ]
    Ys_1_95_below = [
        expected - std_dev * 1.959963984540 
        for expected, std_dev in zip(Ys_1_expected, Ys_1_std)
    ]
    Ys_1_95_above = [
        expected + std_dev * 1.959963984540
        for expected, std_dev in zip(Ys_1_expected, Ys_1_std)
    ]

    result2, var2, info2, _, _ = scipy.optimize.curve_fit(
        holling_3, uncertainty_mean, num_peaks_mean, 
        # Bounds for a negative sloped function
        #bounds = ([-float('inf'), -float('inf'), 0],[0, 0, float('inf')]), 
        # A guess based on the data
        p0 = [num_peaks_flat[0], 0, num_peaks_flat[-1]],
        full_output = True
    )
    squared_residuals2 = sum(info2['fvec'] ** 2)
    print('Fit 2:')
    print((result2, var2, info2))
    print(f'Sum of squared residuals = {squared_residuals2}')
    Ys_2_expected = holling_3(Xs_fit, *result2)
    print(f'Predicted peaks at 0 noise: {Ys_2_expected[0]}')
    Ys_2_std = [
        val.std_dev for val 
        in holling_3(Xs_fit, *uncertainties.correlated_values(result2, var2))
    ]
    Ys_2_95_below = [
        expected - std_dev * 1.959963984540
        for expected, std_dev in zip(Ys_2_expected, Ys_2_std)
    ]
    Ys_2_95_above = [
        expected + std_dev * 1.959963984540
        for expected, std_dev in zip(Ys_2_expected, Ys_2_std)
    ]

    # In these cases we know the first fit is always good
    #result = result1 if r_squared1 < r_squared2 else result2

    axes.scatter(
        uncertainty_flat, num_peaks_flat, 
        color = 'blue', label = 'Individual landscapes'
    )
    axes.scatter(
        uncertainty_mean, num_peaks_mean, 
        color = 'orange', label = 'Mean at uncertainty level'
    )
    axes.plot(Xs_fit, Ys_1_expected, color = 'red', label = 'Prediction')
    axes.plot(
        Xs_fit, Ys_1_95_below, 
        color = 'red', linestyle = ':', label = '95% confidence'
    )
    axes.plot(
        Xs_fit, Ys_1_95_above, 
        color = 'red', linestyle = ':'
    )
    #axes.plot(Xs_fit, Ys_2_expected, color = 'violet', label = 'Unrestricted fit')
    #axes.plot(Xs_fit, Ys_2_95_below, color = 'violet', linestyle = ':', label = '95% confidence')
    #axes.plot(Xs_fit, Ys_2_95_above, color = 'violet', linestyle = ':')
    axes.set_title(plot_title)
    axes.set_xlabel('Relative uncertainty')
    axes.set_ylabel('Number of peaks')
    #axes.legend()

def main():
    fig, the_axes = matplotlib.pyplot.subplots(1,3)
    fit_and_plot('full', the_axes[0])
    fit_and_plot('cutoff_DE', the_axes[1])
    the_axes[1].legend()
    fit_and_plot('cutoff_CDE', the_axes[2])
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('fola_ruggedness_inference.pdf')

if __name__ == '__main__':
    main()


