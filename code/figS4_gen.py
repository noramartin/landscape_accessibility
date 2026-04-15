#! /usr/bin/env python

#     figS4_gen.py

# Generates Figure S4 for the paper

# Written by Kye Hunter
# Last updated 20260415

import math, os, json
import matplotlib.pyplot, matplotlib.font_manager, matplotlib.ticker
import rugged_maps, geometric_mean

matplotlib.rcParams.update({
    'font.size':  10,
    'font.family' : 'serif',
    'font.sans-serif': 'cm',
    'mathtext.default': 'it',
    'mathtext.fontset': 'cm',
})

pt = 1.0 / 72.27 # in to pt
a4_width = 210 / 25.4 # mm to in
fig_width = (a4_width - (40 / 25.4)) / 3 #* pt
phi = (1 + 5 ** 0.5) / 2

L = 15
p = 0.01
num_walks = 1000000

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

row_titles = {
    'sRMF_15_0.01.not_real' : 'sRMF ($L=15$, $p=0.01$)',
    'papkou_2023_folA.csv' : '$folA$ (full)', 
    'papkou_2023_folA_cutoff.not_real' : '$folA$ (functional)',
    'wu_2016_1.csv' : 'GB1',
    'kuo_2020_SD_FL_arti.csv' : 'SD-arti',
    'poelwijk_2019_FP.tsv' : 'FP',
    'domingo_2018.tsv' : 'tRNA',
    'moulana_2022_RBD_ACE2.tsv' : 'RBD-ACE2',
    'house_of_cards.not_real' : 'HoC ($L=15$, $A=2$)'
}

def str_to_tuple(string, dtype : type = int):
    return tuple(dtype(val) for val in string.strip('()[]{} \n\t').split(', '))

def plot_row(row, axes, **keyword_args):
    input_file = keyword_args.pop('input_file')
    input_base = '.'.join(input_file.split('.')[:-1])
    basins_file = input_base + '_basins.tsv'
    log_file = input_base + '_log_figS4.txt'
    walks_file = input_base + '_walks.json'
    keyword_args.setdefault(
        'graph_type', f'point_mutations,{input_file}'
    )

    graph_args = keyword_args.copy()
    graph_args.pop('evolution_type')
    rugged_maps.random.seed(rugged_maps.the_seed)

    graph = rugged_maps.make_graph(**graph_args)[0]
    graph.walk_analyzer = rugged_maps.walk_analysis.WalkAnalyzer(
        graph, 'HoC'
    )
    
    # Column 1:
    graph.open_log(filename = log_file)
    graph.calc_peaks(check_plateaus = True)
    if 'permute' in keyword_args.keys() or isinstance(
        graph.walk_analyzer, rugged_maps.walk_analysis.HoCAnalyzer
    ):
        graph.walk_analyzer.reinit(auto = False)

    low_fitness_genotypes = [
        node.fitness for inode, node in enumerate(graph) 
        if not graph.walk_analyzer.check_high_fitness(inode)
    ]
    high_fitness_genotypes = [ 
        node.fitness for inode, node in enumerate(graph) 
        if graph.walk_analyzer.check_high_fitness(inode)
    ]
    low_fitness_peaks = [ 
        graph[inode].fitness for inode in graph.peaks 
        if not graph.walk_analyzer.check_high_fitness(inode)
    ]
    high_fitness_peaks = [
        graph[inode].fitness for inode in graph.peaks 
        if graph.walk_analyzer.check_high_fitness(inode)
    ] 
    low_fitness_nonpeaks = [
        node.fitness for inode, node in enumerate(graph)
        if not graph.walk_analyzer.check_high_fitness(inode)
        and not node.is_peak
    ]
    high_fitness_nonpeaks = [
        node.fitness for inode, node in enumerate(graph)
        if graph.walk_analyzer.check_high_fitness(inode)
        and not node.is_peak
    ]

    if input_base == 'sRMF_15_0.01':
        bins = [-0.25] + [val * 0.5 - 0.25 for val in range(L * 2 + 1)]
    else:
        bins = 50

    key = f'{alphabet[row]}1'
    axis = axes[key]
    axis.hist(
        [
            low_fitness_peaks, high_fitness_peaks,
            low_fitness_nonpeaks, high_fitness_nonpeaks
        ],
        bins = bins, color = ['cyan', 'blue', 'peru', 'red'],
        histtype = 'bar', stacked = True,
        label = [
            'Low-fitness peaks', 'High-fitness peaks',
            'Low-fitness non-peaks', 'High-fitness non-peaks'
        ]
    )
    axis.set_xlabel('Fitness')
    axis.set_ylabel('Counts')
    axis.set_yscale('log')
    axis.text(
        0.02, 1.02, key, transform = axis.transAxes, 
        ha = 'left', va = 'bottom', color = 'black', 
        weight = 'bold', size = 'large'
    )

    ihigh_fitness_genotypes = [ 
        inode for inode, node in enumerate(graph) 
        if graph.walk_analyzer.check_high_fitness(inode)
    ]
    geno_frac = len(ihigh_fitness_genotypes) / len(graph)
    low_fit_high_fit_geno = min(
        ihigh_fitness_genotypes, key = lambda x: graph[x].fitness
    )
    low_fit_high_fit_geno_rank = 1 - sorted(
        range(len(graph)), key = lambda x: graph[x].fitness
    ).index(low_fit_high_fit_geno) / len(graph)
    ihigh_fitness_peaks = [
        inode for inode in graph.peaks 
        if graph.walk_analyzer.check_high_fitness(inode)
    ]
    peak_frac = len(high_fitness_peaks) / len(graph.peaks)
    low_fit_high_fit_peak = min(
        ihigh_fitness_peaks, key = lambda x: graph[x].fitness
    )
    low_fit_high_fit_peak_rank = 1 - sorted(
        graph.peaks, key = lambda x: graph[x].fitness
    ).index(low_fit_high_fit_peak) / len(graph.peaks)
    graph.write_log(
        f'Genotypes: {len(graph)}\n'
        f'Low-fitness genotypes: {len(low_fitness_genotypes)}\n'
        f'High-fitness genotypes: {len(high_fitness_genotypes)}\n'
        f'Fraction high-fitness genotypes: {geno_frac}\n'
        'Rank of lowest-fitness high-fitness genotype:'
            f' {low_fit_high_fit_geno_rank}\n'
        f'Peaks: {len(graph.peaks)}\n'
        f'Low-fitness peaks: {len(low_fitness_peaks)}\n'
        f'High-fitness peaks: {len(high_fitness_peaks)}\n'
        f'Fraction high-fitness peaks: {peak_frac}\n'
        'Rank of lowest-fitness high-fitness peak:'
            f' {low_fit_high_fit_peak_rank}\n'
    )

    # Column 2
    graph.calc_transition_matrix(
        evolution_type = 'random', check_plateaus = True
    )
    for inode, node in enumerate(graph):
        node.peak_trans_prob = len([
            jnode for jnode in node.neighbors if graph[jnode].is_peak
            and not graph.walk_analyzer.check_high_fitness(jnode)
        ]) / graph.transition_matrix_csc[:, inode].nnz 
    peak_trans_prob = [
        node.peak_trans_prob for inode, node in enumerate(graph) 
        if not node.is_peak 
        and not graph.walk_analyzer.check_high_fitness(inode)
    ]
    mean_low_fit_trans_prob = sum(peak_trans_prob) / len(peak_trans_prob)
    
    key = f'{alphabet[row]}2'
    axis2 = axes[key]
    counts, _, _ = axis2.hist(
        peak_trans_prob, bins = 50, color = 'peru', 
        label = 'Peak-transition probabilities\nin low-fitness region'
    )
    axis2.vlines(
        mean_low_fit_trans_prob, ymin = 0, ymax = max(counts) * 1.02, 
        color = ['black'], linestyle = ':', 
        label = 'Mean peak-transition\nprobability'
    )
    axis2.set_xlabel('Transition probability')
    axis2.set_ylabel('Counts')
    axis2.text(
        0.02, 1.02, key, transform = axis2.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )
    
    # Column 3
    
    if os.path.isfile(walks_file):
        walk_data = json.load(open(walks_file, 'r'))
        # This line is mainly just to make sure the stats are logged every time
        walk_data[1] |= graph.walk_analyzer.analyze_walks(walk_data[1]['walk'])
    else:
        walk_data = graph.random_walkers(
            num_walkers = num_walks, trans_only_low_fit = True
        )
        json.dump(walk_data, open(walks_file, 'w'))
    
    good_walk_lengths = [
        val for val in walk_data[1]['steps_to_high_fitness'] if val != -1
    ]
    bad_walk_lengths = [
        len(walk[1]) - 1 for walk in walk_data[1]['walk'] if not walk[1][-1][4]
    ]
    filtered_prob = [
        val for val in walk_data[1]['max_transition_prob'] if val != -1
    ]
    mean_max_transition_prob = (
        sum(filtered_prob) / len(filtered_prob)
        if len(filtered_prob) != 0 else 1
    )
    graph.write_log(
        'Average peak transition probability in low-fitness region:'
            f' {mean_low_fit_trans_prob}\n'
        'Average maximum peak transition probability see in each walk:'
            f' {mean_max_transition_prob}\n'
    )
    max_length = max(max(good_walk_lengths), max(bad_walk_lengths))
    hist_bins = [-0.5] + [val + 0.5 for val in range(max_length + 1)]
    key = f'{alphabet[row]}3'
    axis = axes[key]
    axis.hist(
        [good_walk_lengths, bad_walk_lengths],
        bins = hist_bins,
        color = ['red', 'cyan'],
        histtype='bar', stacked=True, label = [
            'Steps to reach\nhigh-fitness region', 
            'Steps to reach\na low-fitness peak'
        ]
    )
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.set_xlabel('Steps')
    axis.set_ylabel("Counts")
    axis.text(
        0.02, 1.02, key, transform = axis.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )

    filtered_steps2 = [
        stat if stat != -1 else (len(walk[1]) - 1)
        for walk, stat 
        in zip(walk_data[1]['walk'], walk_data[1]['steps_to_high_fitness'])
    ]

    filtered_geom_means = [
        val for val in walk_data[1]['geom_mean_trans_prob'] if val != -1
    ]
    arith_walk_geom_step_mean_prob = (
        sum(filtered_geom_means) 
        / len(filtered_geom_means)
    )
    axis2.vlines(
        1 - arith_walk_geom_step_mean_prob, ymin = 0, ymax = max(counts) * 1.02, 
        color = ['black'], linestyle = '--', 
        label = '$P_{pt,rep}$, as in eq. 10'
    )
    walk_length_dist = [
        good_walk_lengths.count(length) / len(good_walk_lengths)
        for length in range(max(good_walk_lengths) + 1)
    ]
    least_simple_lower_bound = sum( 
        prob_len * arith_walk_geom_step_mean_prob ** length 
        for length, prob_len in enumerate(walk_length_dist)
    )

    walk_peak_trans_prob_by_length = [[
        1 - geom_mean for geom_mean, length2 in zip(
            walk_data[1]['geom_mean_trans_prob'], filtered_steps2
        ) if geom_mean != -1 and length2 == length
    ] for length in range(1, max_length)]
    mean_geom_means_by_length = [
        sum(val) / len(val) for val in walk_peak_trans_prob_by_length
    ]
    very_accurate_estimate = sum(
        prob_len * (1 - len_mean_geom_mean) ** length 
        for (length, prob_len), len_mean_geom_mean 
        in zip(enumerate(walk_length_dist), mean_geom_means_by_length)
    )

    graph.write_log(
        'Arithmetic mean by walk of geometric mean by step of (1- transition'
            f' probability) (p_rep): {arith_walk_geom_step_mean_prob}\n'
        f'Total walk lengths used: {len(good_walk_lengths)}\n'
        f'P(l): {walk_length_dist}\n'
        f'sum(P(l) * (p_rep) ** l): {least_simple_lower_bound}\n'
        f'sum(P(l) * (p_rep_l) ** l): {very_accurate_estimate}\n'
    )

    # Take some averages over steps:
    # Number of steps starting at a low-fitness non-peak
    norm = sum(
        not step[4] and not graph[step[0]].is_peak 
        for walk in walk_data[1]['walk'] for step in walk[1][:-1]
    )
    # P(step to low-fitness peak | low-fitness non-peak)
    low_fit_to_low_peak = sum(
        not step1[4] and not graph[step1[0]].is_peak 
        and not step2[4] and graph[step2[0]].is_peak 
        for walk in walk_data[1]['walk'] 
        for step1, step2 in zip(walk[1][:-1], walk[1][1:])
    ) / norm
    # P(step to high-fitness genotype | low-fitness non-peak)
    low_fit_to_high_fit = sum(
        not step1[4] and not graph[step1[0]].is_peak and step2[4] 
        for walk in walk_data[1]['walk'] 
        for step1, step2 in zip(walk[1][:-1], walk[1][1:])
    ) / norm
    graph.write_log(
        'Probability of a step going from low fitness to a low-fitness peak:'
            f' {low_fit_to_low_peak}\n'
        'Probability of a step going from low fitness to high fitness:'
            f' {low_fit_to_high_fit}\n'
        'Ratio of going to high fitness / going to low-fitness peak:'
            f' {low_fit_to_high_fit / low_fit_to_low_peak}\n'
    )

    graph.close_log()
    
    # Column 4

    if input_base == 'sRMF_15_0.01':
        permuted_result = 0.0
    else:
        # Make the permuted landscape and run the walks just to get one number
        rugged_maps.random.seed(rugged_maps.the_seed)
        permuted_input_base = input_base + '_permuted'
        permuted_log_file = permuted_input_base + '_log_figS4.txt'
        permuted_walks_file = permuted_input_base + '_walks.json'
        permuted_graph_args = graph_args | {'permute' : 'after_cutoff'}
        permuted_graph = rugged_maps.make_graph(**permuted_graph_args)[0]
        graph.walk_analyzer = rugged_maps.walk_analysis.WalkAnalyzer(
            graph, 'HoC'
        )
        permuted_graph.open_log(filename = permuted_log_file)
        permuted_graph.calc_peaks(check_plateaus = True)
        permuted_graph.walk_analyzer.peak_cutoff = low_fit_high_fit_peak_rank
        permuted_graph.walk_analyzer.reinit(auto = False)
        permuted_graph.calc_transition_matrix(
            evolution_type = 'random', check_plateaus = True
        )
        if os.path.isfile(permuted_walks_file):
            permuted_walk_data = json.load(open(permuted_walks_file, 'r'))
            permuted_walk_data[1] |= permuted_graph.walk_analyzer.analyze_walks(
                permuted_walk_data[1]['walk']
            )
        else:
            permuted_walk_data = permuted_graph.random_walkers(
                num_walkers = num_walks,
                trans_only_low_fit = True
            )
            json.dump(permuted_walk_data, open(permuted_walks_file, 'w'))
        permuted_result = sum(
            permuted_walk_data[1]['ends_at_high_fitness']
        ) / len(permuted_walk_data[1]['ends_at_high_fitness'])
        permuted_graph.close_log()

    key = f'{alphabet[row]}4'
    axis = axes[key]
    axis.bar(
        ['1', '2', '3', '4', '5'],
        [
            100 * low_fit_high_fit_peak_rank, 
            100 * len(good_walk_lengths) 
            / (len(good_walk_lengths) + len(bad_walk_lengths)),
            100 * least_simple_lower_bound,
            100 * very_accurate_estimate,
            100 * permuted_result
        ], 
        color = ['blue', 'red', 'lightcoral', 'green', 'gray'],
        label = [
            'Fraction of high-fitness peaks', 
            'Fraction of walkers reaching\nhigh-fitness peaks (simulation)', 
            'Fraction of walkers reaching\nhigh-fitness peaks (estimated\nfrom eq. 10)',
            'sum$(P(l)(1-P_{pt,l})^l)$',
            'Fraction of walkers reaching\nhigh-fitness peaks (permuted\nlandscape, simulation)'
        ], 
    )
    axis.set_ylabel('Percentage')
    axis.set_ylim([0, 100])
    axis.tick_params('x', top = False, bottom = False, labelbottom = False)
    axis.text(
        0.02, 1.02, key, transform = axis.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )

def main():
    args_array = [
        {
            'input_file' : 'sRMF_15_0.01.not_real', 
            'graph_type' : f'hamming,{L},2', 
            'fitness_distribution' : f'noras_rmf,1,0,1.5,{1-p},{p}'
        },
        {'input_file' : 'papkou_2023_folA.csv'},
        {
            'input_file' : 'papkou_2023_folA_cutoff.not_real', 
            'graph_type' : 'point_mutations,papkou_2023_folA.csv', 
            'fitness_cutoff' : -0.508, 'cutoff_type' : 'low_fitness_neighbors'
        },
        {'input_file' : 'wu_2016_1.csv'},
        {'input_file' : 'kuo_2020_SD_FL_arti.csv'},
        {'input_file' : 'poelwijk_2019_FP.tsv'},
        {'input_file' : 'domingo_2018.tsv'},
        {'input_file' : 'moulana_2022_RBD_ACE2.tsv'},
    ]
    args_array = [val | {
        'evolution_type' : f'walkers,{num_walks}',
    } for val in args_array]
    
    fig, fig_axes = matplotlib.pyplot.subplot_mosaic(                
        [                                                                        
            [
                f'{alphabet[irow]}1', f'{alphabet[irow]}2', 
                f'{alphabet[irow]}3', f'{alphabet[irow]}4'
            ] 
            for irow in range(len(args_array))
        ] + [['l1', 'l2', 'l3', 'l4']], 
        figsize = (fig_width * phi * 4, fig_width * len(args_array) + 1), 
        layout = 'tight'
    ) 

    for iarg, arg in enumerate(args_array):
        artists = plot_row(iarg, fig_axes, **arg)
        # Title for the row
        num_rows = len(args_array)
        fig.text(
            0.5, 1 - iarg / (num_rows + 1) - iarg / num_rows * 0.005, 
            row_titles[arg['input_file']], 
            ha = 'center', va = 'top', color = 'black',
            size = 'large'
        )
    
    # The legends
    fig_axes['l1'].legend(
        *fig_axes['A1'].get_legend_handles_labels(), loc = 'center'
    )
    fig_axes['l1'].set_axis_off()
    fig_axes['l2'].legend(
        *fig_axes['A2'].get_legend_handles_labels(), loc = 'center'
    )
    fig_axes['l2'].set_axis_off()
    fig_axes['l3'].legend(
        *fig_axes['A3'].get_legend_handles_labels(), loc = 'center'
    )
    fig_axes['l3'].set_axis_off()
    fig_axes['l4'].legend(
        *fig_axes['A4'].get_legend_handles_labels(), loc = 'center'
    )
    fig_axes['l4'].set_axis_off()
    
    fig.savefig(f'figS4.pdf')
    #matplotlib.pyplot.show()
    matplotlib.pyplot.close()

if __name__ == '__main__':
    main()

