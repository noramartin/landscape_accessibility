#! /usr/bin/env python

#     fig5_gen.py

# Generates Figure 5 for the paper

# Written by Kye Hunter
# Last updated 20261104

import math, os, json, random
import matplotlib.pyplot, matplotlib.font_manager
import rugged_maps

matplotlib.rcParams.update({
    'font.size':  12,
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
    'sRMF_15_0.01.not_real' : 'sRMF',
    'papkou_2023_folA.csv' : 'folA',
    'papkou_2023_folA_cutoff.not_real' : 'folA sublandscape 1',
    'papkou_2023_folA_high_fitness.not_real' : 'folA sublandscape 2',
    'wu_2016_1.csv' : 'GB1',
    'kuo_2020_SD_FL_arti.csv' : 'SD-arti',
    'poelwijk_2019_FP.tsv' : 'FP',
    'domingo_2018.tsv' : 'tRNA',
    'moulana_2022_RBD_ACE2.tsv' : 'RBD-ACE2',
    'house_of_cards.not_real' : 'HoC',
    'house_of_cards_L8_A4.not_real' : 'HoC'
}

def str_to_tuple(string, dtype : type = int):
    return tuple(dtype(val) for val in string.strip('()[]{} \n\t').split(', '))

def count_nodes_to_cross_fitness(graph, fitnesses : list):
    num_crossings = [[fitness, 0] for fitness in fitnesses]
    for node in graph:
        fitness1 = node.fitness
        for icrossing in range(len(num_crossings)):
            for ineighbor in node.neighbors:
                fitness2 = graph[ineighbor].fitness
                if fitness2 < num_crossings[icrossing][0] <= fitness1:
                    num_crossings[icrossing][1] += 1
                    break
    return num_crossings

def plot_row(row, axes, **keyword_args):
    input_file = keyword_args.pop('input_file')
    input_base = '.'.join(input_file.split('.')[:-1])
    basins_file = input_base + '_basins.tsv'
    log_file = input_base + '_log_fig5.txt'
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
    
    key1 = f'{alphabet[row]}1'
    axis1 = axes[key1]
    key2 = f'{alphabet[row]}2'
    axis2 = axes[key2]
    key3 = f'{alphabet[row]}3'
    axis3 = axes[key3]
    key4 = f'{alphabet[row]}4'
    axis4 = axes[key4]

    if input_base == 'sRMF_15_0.01':
        global L, p
        d_to_plot = [0] + [d for d in range(2, L+1)]
        peak_heights_at_d = [L - d + 1.5 if d > 0 else L for d in d_to_plot]
        basin_sizes = [(2 ** (L - d) * (d + 1)) / 2 ** L  for d in d_to_plot]
        prob_from_basin = [
            1 / (2 ** (L - d)) / d * sum(
                math.comb(L - d, j - d + 1) 
                / math.comb(j, d) for j in range(d, L)
            ) if d > 1 else 1 for d in d_to_plot]
        prob_reached = [s * p for s, p in zip(basin_sizes, prob_from_basin)]
        point_size = 0.5
    else:
        point_size = 0.5

    # Column 1:
    graph.open_log(filename = log_file)
    graph.calc_peaks(check_plateaus = True)
    if 'permute' in keyword_args.keys() or isinstance(
        graph.walk_analyzer, rugged_maps.walk_analysis.HoCAnalyzer
    ):
        graph.walk_analyzer.reinit(auto = False)

    graph.calc_transition_matrix(
        evolution_type = 'random', check_plateaus = True
    )
    graph.calc_basins_slow(file_name = basins_file)
    basin_sizes = [(
        graph.node_lookup(line.strip().split('\t')[0]).fitness, 
        len(str_to_tuple(line.strip().split('\t')[1])) / len(graph),
        graph.inode_lookup(line.strip().split('\t')[0]) 
    ) for line in open(basins_file)]
    
    axis1.scatter(*list(zip(*basin_sizes))[:2], color = 'red', s = point_size)
    axis1.set_xlabel('Fitness')
    axis1.set_ylabel("P(start in $s$'s basin)")
    axis1.set_yscale('log')
    axis1.tick_params(which = 'minor', left = True)
    axis1.text(
        0.00, 1.005, key1, transform = axis1.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )
    
    # Column 2:
    for inode in graph.peaks:
        graph[inode].walkers_starting_in_basin = 0
        graph[inode].walkers_ending_at_peak = 0
    for node in graph:
        node.walkers_starting_at_node = 0
    
    if os.path.isfile(walks_file):
        walk_data = json.load(open(walks_file, 'r'))
        walk_data[1] |= graph.walk_analyzer.analyze_walks(walk_data[1]['walk'])
    else:
        walk_data = graph.random_walkers(
            num_walkers = num_walks, trans_only_low_fit = True
        )
        json.dump(walk_data, open(walks_file, 'w'))
    graph.close_log()
    
    for walk in walk_data[1]['walk']:
        graph[walk[0]].walkers_ending_at_peak += 1
        graph[walk[1][0][0]].walkers_starting_at_node += 1
    for line in open(basins_file):
        peak = graph.node_lookup(line.strip().split('\t')[0])
        peak.walkers_starting_in_basin = sum(
            graph[inode].walkers_starting_at_node 
            for inode in str_to_tuple(line.strip().split('\t')[1])
        )
       
    basin_to_peak_prob = [(
        graph[ipeak].fitness, 
        graph[ipeak].walkers_ending_at_peak 
        / graph[ipeak].walkers_starting_in_basin,
        ipeak
    ) for ipeak in graph.peaks]
    
    axis2.scatter(
        *list(zip(*basin_to_peak_prob))[:2], color = 'red', s = 0.5
    )
    axis2.set_xlabel('Fitness')
    axis2.set_ylabel("P(reach $s$\n|start in $s$'s basin)")
    axis2.set_yscale('log')
    axis2.tick_params(which = 'minor', left = True)
    axis2.text(
        0.00, 1.005, key2, transform = axis2.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )
    
    # Column 3:
    peak_reach_prob = [
        (b[0], b[1] * p[1]) for b, p in zip(
            sorted(basin_sizes, key = lambda x: x[2]), 
            sorted(basin_to_peak_prob, key = lambda x: x[2]))
    ]
    
    axis3.scatter(*list(zip(*peak_reach_prob)), color = 'red', s = 0.5)
    axis3.set_xlabel('Fitness')
    axis3.set_ylabel('P(reach $s$)')
    axis3.set_yscale('log')
    axis3.tick_params(which = 'minor', left = True)
    axis3.text(
        0.00, 1.005, key3, transform = axis3.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )    

    # Column 4: 
    fitness_min = min(node.fitness for node in graph)
    fitness_max = max(node.fitness for node in graph)
    num_crossings = count_nodes_to_cross_fitness(graph, [
        fitness_min + i * (fitness_max - fitness_min) / 100 for i in range(101)
    ])
    num_crossings = [[
        val[0], 
        1 / val[1] if val[1] != 0 else float('nan'),
    ] for val in num_crossings]
    num_crossings = [[*val, 0, 0] for val in num_crossings]

    for node in graph:
        node.walkers_passing_through_node = 0
    for walk in walk_data[1]['walk']:
        # For graphs with neutral mutations, remove nodes visited more than once
        for inode in set(step[0] for step in walk[1]):
            graph[inode].walkers_passing_through_node += 1
        for crossing in num_crossings:
            # Count how many walks start higher in fitness than this crossing
            if walk[1][0][1] > crossing[0]:
                crossing[2] += 1
            # And count how many end lower than this crossing
            if walk[1][-1][1] < crossing[0]:
                crossing[3] += 1
    prob_seen_in_walk_non_peak = [
        (node.fitness, node.walkers_passing_through_node / num_walks)
        for node in graph if not node.is_peak
    ]
    prob_seen_in_walk_peak = [(
        graph[inode].fitness, 
        graph[inode].walkers_passing_through_node / num_walks
    ) for inode in graph.peaks]

    num_crossings = [[
        val[0], val[1],
        # The corrected value which multiplies by the fraction of walks that 
        # cross this fitness
        val[1] * (num_walks - val[2] - val[3]) / num_walks
    ] for val in num_crossings]
    num_crossings = list(zip(*num_crossings))

    x_range = axis3.get_xlim()
    prob_seen_in_walk_non_peak_in_range = [
        val for val in prob_seen_in_walk_non_peak 
        if x_range[0] < val[0] < x_range[1]
    ]
    axis4.set_yscale('log')
    axis4.scatter(
        *list(zip(*random.sample(prob_seen_in_walk_non_peak_in_range, k=2000))), 
        color = 'grey', s = point_size, #alpha = 0.3,
        label = 'Non-peak genotypes\n(2000 random samples)'
    )
    axis4.scatter(
        *list(zip(*prob_seen_in_walk_peak)), 
        color = 'red', s = point_size, #alpha = 0.3,
        label = 'Peaks'
    )
    # Don't let these lines mess with the ylims
    original_ylim = axis4.get_ylim()
    axis4.plot(
        num_crossings[0], num_crossings[1], color = 'black', linewidth = 0.75,
        label = 'Estimate of walker\ndilution'
    )
    axis4.plot(
        num_crossings[0], num_crossings[2], color = 'darkviolet', linewidth = 0.75,
        label = 'Estimate of walker\ndilution for walks \npassing a given fitness'
    )
    axis4.set_ylim(original_ylim)

    axis4.set_xlabel('Fitness')
    axis4.set_xlim(axis3.get_xlim())
    axis4.set_ylabel('P(pass $g$)')
    axis4.tick_params(which = 'minor', left = True)
    axis4.text(
        0.00, 1.005, key4, transform = axis4.transAxes, 
        ha = 'left', va = 'bottom', color = 'black',
        weight = 'bold', size = 'large'
    )    

def main():
    args_array = [
        {
            'input_file' : 'house_of_cards_L8_A4.not_real', 
            'graph_type' : f'hamming,8,4', 
            'fitness_distribution' : 'random'
        },
        {'input_file' : 'papkou_2023_folA.csv'},
        {
            'input_file' : 'papkou_2023_folA_cutoff.not_real', 
            'graph_type' : 'point_mutations,papkou_2023_folA.csv', 
            'fitness_cutoff' : -0.508, 'cutoff_type' : 'low_fitness_neighbors'
        },
        {
            'input_file' : 'papkou_2023_folA_high_fitness.not_real', 
            'graph_type' : 'point_mutations,papkou_2023_folA.csv', 
            'fitness_cutoff' : -0.508, 'cutoff_type' : 'low_fitness'
        },
        {'input_file' : 'wu_2016_1.csv'},
        {'input_file' : 'kuo_2020_SD_FL_arti.csv'},
        {'input_file' : 'poelwijk_2019_FP.tsv'}, 
        {'input_file' : 'domingo_2018.tsv'},
        {'input_file' : 'moulana_2022_RBD_ACE2.tsv'},
    ]
    args_array = [
        val | {'evolution_type' : f'walkers,{num_walks}'} for val in args_array
    ]
    
    fig, fig_axes = matplotlib.pyplot.subplot_mosaic(                
        [                                                                        
            [
                f'{alphabet[irow]}1', f'{alphabet[irow]}2', 
                f'{alphabet[irow]}3', f'{alphabet[irow]}4',
                f'{alphabet[irow]}l'
            ] for irow in range(len(args_array))
        ], figsize = (fig_width * phi * 5, fig_width * len(args_array)), 
        layout = 'tight'#, sharey = 'row'
    ) 
    num_rows = len(args_array)
    
    for iarg, arg in enumerate(args_array):
        plot_row(iarg, fig_axes, **arg)
        fig_axes[f'{alphabet[iarg]}l'].set_axis_off()
        fig.text(
            0, 1 - 0.5 / num_rows - iarg / (num_rows) + iarg / num_rows * 0.03,
            row_titles[arg['input_file']],
            ha = 'left', va = 'center', color = 'black',
            rotation = 'vertical', weight = 'bold'
        )
    fig_axes['Al'].legend(
        *fig_axes['A4'].get_legend_handles_labels(), loc = 'center',
        markerscale = 3
    )
    fig.tight_layout(pad = 2, h_pad = 0.1, w_pad = 0.1)

    # Now the widest ylims of all the plots, but not the legends
    all_ylims = list(zip(*[
        fig_axes[f'{alphabet[iarg]}{col}'].get_ylim() 
        for iarg, arg in enumerate(args_array)
        for col in range(1,5)
    ]))
    all_ylims_new = (min(all_ylims[0]), max(all_ylims[1]))
    for iarg, arg in enumerate(args_array):
        fig_axes[f'{alphabet[iarg]}1'].set_ylim(all_ylims_new)
        fig_axes[f'{alphabet[iarg]}2'].set_ylim(all_ylims_new)
        fig_axes[f'{alphabet[iarg]}3'].set_ylim(all_ylims_new)
        fig_axes[f'{alphabet[iarg]}4'].set_ylim(all_ylims_new)
    fig.savefig(f'fig5.pdf')

    #matplotlib.pyplot.show()
    matplotlib.pyplot.close()

if __name__ == '__main__':                                                       
    main()

