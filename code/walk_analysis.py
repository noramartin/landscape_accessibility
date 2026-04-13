#! /usr/bin/env python

#     walk_analysis.py

# Base class for analyzing walks on fitness landscapes, and subclasses for 
# specific landscapes

# Written by Kye Hunter
# Last updated 20260408

import geometric_mean

class WalkAnalyzer:
    def __new__(cls, graph, landscape_type):
        match landscape_type.strip().lower():
            case 'rmf':
                #return RMFAnalyzer(graph)
                analyzer = object.__new__(RMFAnalyzer)
            case 'fola':
                analyzer = object.__new__(folAAnalyzer)
            case 'tetr':
                analyzer = object.__new__(TetRAnalyzer)
            case 'gb1':
                analyzer = object.__new__(GB1Analyzer)
            case 'pard3-pare3':
                analyzer = object.__new__(ParD3ParE3Analyzer)
            case 'rbd-ace2':
                analyzer = object.__new__(RBDACE2Analyzer)
            case 'sd-arti':
                analyzer = object.__new__(SDartiAnalyzer)
            case 'fp':
                analyzer = object.__new__(FPAnalyzer)
            case 'trna':
                analyzer = object.__new__(tRNAAnalyzer)
            case 'hoc':
                analyzer = object.__new__(HoCAnalyzer)
            case 'fig4':
                analyzer = object.__new__(Fig4Analyzer)
            case _:
                raise ValueError(
                    f'Unrecognized landscape type "{landscape_type}"'
                )
        return analyzer
    def __init__(self, graph):
        self.type = 'default'
        self.graph = graph
        self.reference = []
        self.boundary = 0
        self.get_sequence = lambda inode: self.graph.get_sequence(inode)

    def check_high_fitness(self, inode) -> bool:
        return sum(
            seq in ref for seq, ref  
            in zip(self.get_sequence(inode), self.reference)
        ) >= self.boundary

    def log_step(self, inode):
        prob_random_walk_trans_to_peak = (len([
            neighbor for neighbor in self.graph[inode].neighbors
            if self.graph[neighbor].is_peak
        ]) / (z if (
            z := self.graph.transition_matrix_csc[:, inode].nnz
        ) != 0 else 1) if not self.graph[inode].is_peak else 1.0)
        is_high_fitness : bool = sum(
            seq in ref for seq, ref  
            in zip(self.get_sequence(inode), self.reference)
        ) >= self.boundary
        return [
            inode, 
            self.graph[inode].fitness, 
            self.get_sequence(inode),
            prob_random_walk_trans_to_peak,
            is_high_fitness
        ]

    def relog(self, walks):
        '''Recalculate data logged at each step

        This allows for reanalyzing the walks with different high-fitness 
        regions, e.g., `analyze_walks(relog(walks))`.
        '''
        for walk in walks:
            for step in walk[1]:
                step[4] = self.check_high_fitness(step[0])
        return walks

    def _analyze_walk(self, walk):
        ends_at_high_fitness = walk[1][-1][4]
        steps_to_high_fitness = next((
            istep for istep, step in enumerate(walk[1]) if step[4]
        ), -1) if ends_at_high_fitness else -1 
        if steps_to_high_fitness >= 0: #ends_at_high_fitness:
            last_low_fitness = next((
                istep for istep, step in enumerate(walk[1][::-1]) if not step[4]
            ), len(walk[1]))
            stays_high_fitness = (
                steps_to_high_fitness + last_low_fitness == len(walk[1])
            )
            # This uses the _last_ entry into the high-fitness region, rather 
            # than the first
            #if not stays_high_fitness:
            #    steps_to_high_fitness = len(walk[1]) - last_low_fitness
            #    #print(walk)
        else:
            stays_high_fitness = -1
        #if not (steps_to_high_fitness in (-1, 0)):
        # Include walks that don't reach high fitness:
        if steps_to_high_fitness != 0:
            # For walks that end with low fitness, `steps_to_high_fitness` is 
            # conveniently -1, which gives the desired result, but is a somewhat
            # fragile design
            max_transition_prob = max((
                step[3] for step in walk[1][:steps_to_high_fitness]
            ), default = 1) # It shouldn't be possible to get this default
        else:
            max_transition_prob = -1
        return (
            ends_at_high_fitness, steps_to_high_fitness, stays_high_fitness,
            max_transition_prob
        )

    def analyze_walks(self, walks):
        walk_stats = list(zip(*[self._analyze_walk(walk) for walk in walks]))
        mean_ends_at_high_fitness = sum(walk_stats[0]) / len(walk_stats[0])
        filtered_steps = [val for val in walk_stats[1] if val != -1]
        mean_steps_to_high_fitness = (
            sum(filtered_steps) / len(filtered_steps) 
            if len(filtered_steps) != 0 else 1
        )
        filtered_steps2 = [
            stat if stat != -1 else (len(walk[1]) - 1) 
            for walk, stat in zip(walks, walk_stats[1])
        ]
        steps_to_high_fitness_or_peak_90th = sorted(filtered_steps2)[
            int(len(filtered_steps2) * 0.9)
        ]
        filtered_stays_high = [val for val in walk_stats[2] if val != -1]
        mean_stays_high_fitness = (
            sum(filtered_stays_high) / len(filtered_stays_high) 
            if len(filtered_stays_high) != 0 else 1
        )
        filtered_prob = [val for val in walk_stats[3] if val != -1]
        mean_max_transition_prob = (
            sum(filtered_prob) / len(filtered_prob) 
            if len(filtered_prob) != 0 else 1
        )
        high_fitness_genotypes = sum(sum(
            seq in ref for seq, ref  
            in zip(self.get_sequence(inode), self.reference)
        ) >= self.boundary for inode in range(len(self.graph)))
        high_fitness_peaks = sum(sum(
            seq in ref for seq, ref  
            in zip(self.get_sequence(inode), self.reference)
        ) >= self.boundary for inode in self.graph.peaks)
        self.graph.write_log(
            f'\nWalk analysis of {len(walks)} walks with unkown(!) type:\n'
            'Reference sequence:                                           '
                f'{self.reference}\n'
            'Distance cutoff:                                              '
                f'{self.boundary}\n'
            'Fraction of walks that end at high fitness:                   '
                f'{mean_ends_at_high_fitness}\n'
            'Average steps to reach high fitness:                          '
                f'{mean_steps_to_high_fitness}\n'
            '90th percentile of walk length to reach high fitness or peak: '
                f'{steps_to_high_fitness_or_peak_90th}\n'
            'Probability of never leaving the high-fitness region:         '
                f'{mean_stays_high_fitness}\n'
            'Average maximum probability of stepping to a peak:            '
                f'{mean_max_transition_prob}\n'
            'Number of genotypes with high-fitness:                        '
                f'{high_fitness_genotypes}\n'
            'Number of peaks with high-fitness:                            '
                f'{high_fitness_peaks}\n'
        )
        return {
            'ends_at_high_fitness' : walk_stats[0],
            'steps_to_high_fitness' : walk_stats[1], 
            'stays_high_fitness' : walk_stats[2],
            'max_transition_prob' : walk_stats[3]
        }

class RMFAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'RMF'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['0' for _ in range(self.graph.dtype[1])]
        self.boundary = 2 * self.graph.dtype[1] // 3

class folAAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'folA'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['', 'DE', '']
        #self.reference = ['', 'CDE', '']
        self.boundary = 1
        self.get_sequence = lambda inode: self.graph[inode].amino_sequence

class TetRAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'TetR'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['T', 'A', 'T', 'C', 'G' 'A' 'T' 'A']
        self.boundary = 4

class GB1Analyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'GB1'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['', '', 'G', 'A']
        self.boundary = 1

class ParD3ParE3Analyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'ParD3-ParE3'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['', 'W', '', '']
        self.boundary = 1

class RBDACE2Analyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'RBD-ACE2'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = [
            '', '', '', '', '0', '', '', '1', '', '', '', '', '1', '1', '0'
        ]
        self.boundary = 5

class SDartiAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'SD-arti'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['AG', 'GA', 'AG', 'GA', 'GA', 'GA', 'GA', 'GA', 'GU']
        self.boundary = 6

class FPAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'FP'):
        super().__init__(graph)
        self.type = landscape_type
        self._reference = [
            '', '1', '', '', '1', '', '', '', '1', '1', '', '', ''
        ]
        self.reference = [
            self.graph.alphabet_map[base] if base != '' else '' 
            for base in self._reference
        ]
        self.boundary = 2

class tRNAAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'tRNA'):
        super().__init__(graph)
        self.type = landscape_type
        self.reference = ['G', 'C', 'G', 'C', 'A', '', 'CA', 'A', 'A', 'C']
        self.boundary = 4

class HoCAnalyzer(WalkAnalyzer):
    def __init__(self, graph, landscape_type = 'HoC'):
        self.type = landscape_type
        self.graph = graph
        self.peak_cutoff = 0.14
        self.boundary = 0.9 # Dummy value, don't use!
        self.get_sequence = lambda inode: self.graph.get_sequence(inode)

    def check_high_fitness(self, inode) -> bool:
        # This is for a cutoff at the lowest-fitness high-fitness peak
        #return self.graph[inode].fitness >= self.boundary
        # This is for a cutoff at the highest-fitness low-fitness peak
        return self.graph[inode].fitness > self.boundary

    def reinit(self, auto = True):
        sorted_peaks = sorted(
            self.graph.peaks, key = lambda x: self.graph[x].fitness
        )
        if auto:
            peak_accessibilities = [
                self.graph[peak].population for peak in sorted_peaks
            ] 
            access_cdf = peak_accessibilities.copy()      
            for ipeak in range(len(access_cdf) - 1, 0, -1):         
                access_cdf[ipeak - 1] += access_cdf[ipeak]
            trade_off_value = access_cdf.copy()
            num_peaks = len(trade_off_value)
            for ipeak in range(len(trade_off_value)):         
                trade_off_value[ipeak] += ipeak / num_peaks
            ibest_peak = max(
                range(num_peaks), key = lambda x: trade_off_value[x]
            )
            self.peak_cutoff = 1 - ibest_peak / num_peaks
        # When there are several peaks with equal fitness around the boudary, we
        # could use all of them, or none of them. Here, we want none of them, so
        # we need to find the first peak with fitness greater than the fitness 
        # of the peak at the cutoff we have. This can be done with 
        # `list.reverse().index(cutoff)` (psuedocode)
        if self.peak_cutoff == 1:
            self.boundary = self.graph[sorted_peaks[0]].fitness
        else:
            boundary_wrong_side_index = int(
                (1 - self.peak_cutoff) * len(sorted_peaks)
            )
            if boundary_wrong_side_index == (
                (1 - self.peak_cutoff) * len(sorted_peaks)
            ):
                # Guarantee that the index is actually on the wrong side
                boundary_wrong_side_index -= 1
            boundary_wrong_side = self.graph[
                sorted_peaks[boundary_wrong_side_index]
            ].fitness
            sorted_peak_fitnesses = [
                self.graph[inode].fitness for inode in sorted_peaks
            ]
            boundary_index = (
                len(sorted_peaks) 
                - sorted_peak_fitnesses[::-1].index(boundary_wrong_side)
            )
            # Rather than picking the lowest fitness of the high-fitness peaks,
            # pick the highest fitness of the low-fitness peaks, but with `>` 
            # rather than `>=`
            self.boundary = self.graph[sorted_peaks[boundary_index - 1]].fitness

    def log_step(self, inode, trans_only_low_fit = False):
        # These are both left for `relog`, since we may want to set a cutoff 
        # only after knowing the distriubtion of walk endpoints, and it is more 
        # efficient to calculate the peak-transition probablities once per node 
        # rather than once per step (and they may also depennd on the cutoff).
        prob_random_walk_trans_to_peak = None
        is_high_fitness = None
        return [
            inode, 
            self.graph[inode].fitness, 
            self.get_sequence(inode),
            prob_random_walk_trans_to_peak,
            is_high_fitness
        ]

    def relog(self, walks, trans_only_low_fit = False):
        if trans_only_low_fit:
            for inode, node in enumerate(self.graph):
                node.peak_trans_prob = (len([
                    neighbor for neighbor in self.graph[inode].neighbors
                    if self.graph[neighbor].is_peak 
                    and not self.check_high_fitness(neighbor)
                ]) / (z if (
                    z := self.graph.transition_matrix_csc[:, inode].nnz
                ) != 0 else 1) if not self.graph[inode].is_peak else 1.0)
        else:
            for inode, node in enumerate(self.graph):
                node.peak_trans_prob = (len([
                    neighbor for neighbor in self.graph[inode].neighbors
                    if self.graph[neighbor].is_peak 
                ]) / (z if (
                    z := self.graph.transition_matrix_csc[:, inode].nnz
                ) != 0 else 1) if not self.graph[inode].is_peak else 1.0)
            
        for walk in walks:
            for step in walk[1]:
                step[3] = self.graph[step[0]].peak_trans_prob
                step[4] = self.check_high_fitness(step[0])
        return walks

    def _analyze_walk(self, walk):
        ends_at_high_fitness = walk[1][-1][4]
        steps_to_high_fitness = next((
            istep for istep, step in enumerate(walk[1]) if step[4]
        ), -1) if ends_at_high_fitness else -1 
        if steps_to_high_fitness >= 0: #ends_at_high_fitness:
            last_low_fitness = next((
                istep for istep, step in enumerate(walk[1][::-1]) if not step[4]
            ), len(walk[1]))
            # With a fitness cutoff, the walker should never step from high 
            # fitness to low fitness anyway
            stays_high_fitness = (
                steps_to_high_fitness + last_low_fitness == len(walk[1])
            )
            if not stays_high_fitness:
                steps_to_high_fitness = len(walk[1]) - last_low_fitness
                #print(walk)
        else:
            stays_high_fitness = -1
        #if not (steps_to_high_fitness in (-1, 0)):
        # Include walks that don't reach high fitness, but not walks that start 
        # on a low-fitness peak:
        if steps_to_high_fitness != 0 and len(walk[1]) != 1:
            # For walks that end with low fitness, `steps_to_high_fitness` is 
            # conveniently -1, which gives the desired result, but is a somewhat
            # fragile design
            max_transition_prob = max((
                step[3] for step in walk[1][:steps_to_high_fitness]
            ), default = 1) # It shouldn't be possible to get this default
            geom_mean_trans_prob = geometric_mean.geometric_mean_safe([
                1 - step[3] for step in walk[1][:steps_to_high_fitness]
            ])
        else:
            max_transition_prob = -1
            geom_mean_trans_prob = -1
        return (
            ends_at_high_fitness, steps_to_high_fitness, stays_high_fitness,
            max_transition_prob, geom_mean_trans_prob
        )

    def analyze_walks(self, walks):
        walk_stats = list(zip(*[self._analyze_walk(walk) for walk in walks]))
        mean_ends_at_high_fitness = sum(walk_stats[0]) / len(walk_stats[0])
        filtered_steps = [val for val in walk_stats[1] if val != -1]
        mean_steps_to_high_fitness = (
            sum(filtered_steps) / len(filtered_steps) 
            if len(filtered_steps) != 0 else 1
        )
        steps_to_high_fitness_50th = sorted(filtered_steps)[
            int(len(filtered_steps) * 0.5)
        ]
        filtered_steps2 = [
            stat if stat != -1 else (len(walk[1]) - 1) 
            for walk, stat in zip(walks, walk_stats[1])]
        steps_to_high_fitness_or_peak_90th = sorted(filtered_steps2)[
            int(len(filtered_steps2) * 0.9)
        ]
        filtered_steps3 = [
            len(walk[1]) - 1 
            for walk, stat in zip(walks, walk_stats[1]) if stat == -1
        ]
        steps_to_low_fitness_peak_50th = sorted(filtered_steps3)[
            int(len(filtered_steps3) * 0.5)
        ]
        filtered_stays_high = [val for val in walk_stats[2] if val != -1]
        mean_stays_high_fitness = (
            sum(filtered_stays_high) / len(filtered_stays_high) 
            if len(filtered_stays_high) != 0 else 1
        )
        filtered_prob = [val for val in walk_stats[3] if val != -1]
        mean_max_transition_prob = (
            sum(filtered_prob) / len(filtered_prob) 
            if len(filtered_prob) != 0 else 1
        )
        filtered_geom_prob = [val for val in walk_stats[4] if val != -1]
        mean_geom_mean_transition_prob = (
            sum(filtered_geom_prob) / len(filtered_geom_prob) 
            if len(filtered_geom_prob) != 0 else 1
        )
        high_fitness_genotypes = sum(
            node.fitness > self.boundary for node in self.graph
        )
        high_fitness_peaks = sum(
            self.graph[inode].fitness > self.boundary 
            for inode in self.graph.peaks
        )
        self.graph.write_log(
            f'\nWalk analysis of {len(walks)} walks with unkown(!) type:\n'
            'Target peak fraction:                                         '
                f'{self.peak_cutoff}\n'
            'Fitness cutoff:                                               '
                f'{self.boundary}\n'
            'Fraction of walks that end at high fitness:                   '
                f'{mean_ends_at_high_fitness}\n'
            'Average steps to reach high fitness:                          '
                f'{mean_steps_to_high_fitness}\n'
            '50th percentile of walk length to reach high fitness: '
                f'{steps_to_high_fitness_50th}\n'
            '50th percentile of walk length to reach low-fitness peak: '
                f'{steps_to_low_fitness_peak_50th}\n'
            '90th percentile of walk length to reach high fitness or peak: '
                f'{steps_to_high_fitness_or_peak_90th}\n'
            'Probability of never leaving the high-fitness region:         '
                f'{mean_stays_high_fitness}\n'
            'Average maximum probability of stepping to a peak:            '
                f'{mean_max_transition_prob}\n'
            'Average geometric mean of probability of stepping to a peak:  '
                f'{mean_geom_mean_transition_prob}\n'
            'Number of genotypes with high-fitness:                        '
                f'{high_fitness_genotypes}\n'
            'Number of peaks with high-fitness:                            '
                f'{high_fitness_peaks}\n'
        )
        return {
            'ends_at_high_fitness' : walk_stats[0],
            'steps_to_high_fitness' : walk_stats[1], 
            'stays_high_fitness' : walk_stats[2],
            'max_transition_prob' : walk_stats[3],
            'geom_mean_trans_prob' : walk_stats[4]
        }

