#! /usr/bin/env python

#     figS3_gen.py

# Generates Figure S3 for the paper

# Written by Kye Hunter
# Last updated 20260407

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
    log_file = input_base + '_log_figS3.txt'
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

    graph.calc_transition_matrix(
        evolution_type = 'random', check_plateaus = True
    )
    for inode, node in enumerate(graph):
        node.peak_trans_prob = len([
            jnode for jnode in node.neighbors if graph[jnode].is_peak
            and not graph.walk_analyzer.check_high_fitness(jnode)
        ]) / graph.transition_matrix_csc[:, inode].nnz 

    peak_trans_prob = [
        node.peak_trans_prob for node in graph 
        if not node.is_peak 
        and not graph.walk_analyzer.check_high_fitness(inode)
    ]
    mean_low_fit_trans_prob = sum(peak_trans_prob) / len(peak_trans_prob)

    if os.path.isfile(walks_file):
        walk_data = json.load(open(walks_file, 'r'))
        # This line is mainly just to make sure the stats are logged every time
        walk_data[1] |= graph.walk_analyzer.analyze_walks(walk_data[1]['walk'])
    else:
        walk_data = graph.random_walkers(
            num_walkers = num_walks, trans_only_low_fit = True
        )
        json.dump(walk_data, open(walks_file, 'w'))    

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

    filtered_steps2 = [
        stat if stat != -1 else (len(walk[1]) - 1)
        for walk, stat 
        in zip(walk_data[1]['walk'], walk_data[1]['steps_to_high_fitness'])
    ]
    max_length = max(filtered_steps2)

    filtered_geom_means = [
        val for val in walk_data[1]['geom_mean_trans_prob'] if val != -1
    ]
    arith_walk_geom_step_mean_prob = ( 
        sum(filtered_geom_means)
        / len(filtered_geom_means)
    )

    # A list of lists of transition probabilities, one for each walk length
    walk_peak_trans_prob_by_length = [[
        1 - geom_mean for geom_mean, length2 in zip(
            walk_data[1]['geom_mean_trans_prob'], filtered_steps2
        ) if geom_mean != -1 and length2 == length 
    ] for length in range(1, max_length)]

    key = f'{alphabet[row]}1'
    axis = axes[key]

    violin = axis.violinplot(                                                    
        walk_peak_trans_prob_by_length, showmeans = False, showmedians = False,  
        showextrema = False, widths = 0.75, points = 400                         
    )                                                                            
    for val in violin['bodies']:                                                 
        val.set_facecolor('black')                                               
        val.set_alpha(1)                                                         
    axis.set_xlabel('Steps to reach high-fitness region or low-fitness peak')       
    axis.set_ylabel('$1-GM[1-P_{pt}]$ during walk')                                      
    axis.set_title(row_titles[input_file])
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    graph.close_log()


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
                f'{alphabet[irow]}1', f'{alphabet[irow + 1]}1', 
            ] 
            for irow in range(0, len(args_array), 2)
        ],
        figsize = (fig_width * phi * 3, fig_width * len(args_array) / 2), 
        layout = 'tight'
    ) 

    for iarg, arg in enumerate(args_array):
        artists = plot_row(iarg, fig_axes, **arg)
        # Title for the row
        num_rows = len(args_array)
    
    # Prevent an empty plot on the bottom row
    if len(args_array) % 2:
        fig_axes[f'{alphabet[len(args_array)]}1'].set_axis_off()

    fig.savefig(f'figS3.pdf')
    #matplotlib.pyplot.show()
    matplotlib.pyplot.close()

if __name__ == '__main__':
    main()

