#! /usr/bin/env python

#     rugged_maps.py

# Written by Kye Hunter
# Last updated 20260302

# Classes for modeling fitness landscapes, and simulating evolution on them, and
# a `main` which handles running simulations

import copy, random, math, time, sys, types, io, warnings, os
import pyreadr, numpy, scipy, matplotlib.pyplot
import walk_analysis, sparse_list

usage_str = (
    'Usage: rugged_maps.py '
    '[graph_type=<parameters>] '
    '[evolution_type=<exact|walkers,<int>>] '
    '[evolution_algo=<all|random|kimura|greedy|basin|none>] '
    '[fitness_distribution=<parameters>] '
    '[fitness_cutoff=<float>|eta_cutoff=<float>] '
    '[cutoff_type=<<low|high>_fitness[_neighbors]|random>] '
    '[permute=<true|false|aa27|after_cutoff|matrix|peaks>] '
    '[population_distribution=<uniform|lowest>]'
)

matplotlib.rcParams.update({                                                     
    'font.size':  22,                                                         
    'font.family' : 'serif',                                                
    'mathtext.default': 'it',                                                    
    'mathtext.fontset': 'cm',                                                    
})

#the_seed = 3034128487
#the_seed = 3034128488
the_seed = 3034128489
random.seed(the_seed)

dna_list = ['A', 'C', 'G', 'T']
dna_set = set(dna_list)
rna_list = ['A', 'C', 'G', 'U']
rna_set = set(rna_list)
base_to_int = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3, 'U' : 3}
amino_list_1 = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 
    'S', 'T', 'V', 'W', 'Y', '*'
]
amino_set_1 = set(amino_list_1)
amino_to_int_1 = {
    'A' : 0, 'C' : 1, 'D' : 2, 'E' : 3, 'F' : 4, 'G' : 5, 'H' : 6, 'I' : 7, 
    'K' : 8, 'L' : 9, 'M' : 10, 'N' : 11, 'P' : 12, 'Q' : 13, 'R' : 14, 
    'S' : 15, 'T' : 16, 'V' : 17, 'W' : 18, 'Y' : 19, '*' : 20
}
amino_list_3 = [
    'Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu', 'Met',
    'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr', 'STOP'
]
amino_3_to_1 = {
    'Ala' : 'A', 'Cys' : 'C', 'Asp' : 'D', 'Glu' : 'E', 'Phe' : 'F', 
    'Gly' : 'G', 'His' : 'H', 'Ile' : 'I', 'Lys' : 'K', 'Leu' : 'L', 
    'Met' : 'M', 'Asn' : 'N', 'Pro' : 'P', 'Gln' : 'Q', 'Arg' : 'R', 
    'Ser' : 'S', 'Thr' : 'T', 'Val' : 'V', 'Trp' : 'W', 'Tyr' : 'Y', 
    'STOP' : '*'
}
genetic_code_1 = {
    'AAA' : 'K', 'AAC' : 'N', 'AAG' : 'K', 'AAT' : 'N', 
    'ACA' : 'Y', 'ACC' : 'Y', 'ACG' : 'Y', 'ACT' : 'Y', 
    'AGA' : 'R', 'AGC' : 'S', 'AGG' : 'R', 'AGT' : 'S', 
    'ATA' : 'I', 'ATC' : 'I', 'ATG' : 'M', 'ATT' : 'I', 
    'CAA' : 'Q', 'CAC' : 'H', 'CAG' : 'G', 'CAT' : 'H', 
    'CCA' : 'P', 'CCC' : 'P', 'CCG' : 'P', 'CCT' : 'P', 
    'CGA' : 'R', 'CGC' : 'R', 'CGG' : 'R', 'CGT' : 'R',
    'CTA' : 'L', 'CTC' : 'L', 'CTG' : 'L', 'CTT' : 'L', 
    'GAA' : 'E', 'GAC' : 'D', 'GAG' : 'E', 'GAT' : 'D', 
    'GCA' : 'A', 'GCC' : 'A', 'GCG' : 'A', 'GCT' : 'A', 
    'GGA' : 'E', 'GGC' : 'D', 'GGG' : 'E', 'GGT' : 'D', 
    'GTA' : 'V', 'GTC' : 'V', 'GTG' : 'V', 'GTT' : 'V', 
    'TAA' : '*', 'TAC' : 'Y', 'TAG' : '*', 'TAT' : 'Y', 
    'TCA' : 'S', 'TCC' : 'S', 'TCG' : 'S', 'TCT' : 'S', 
    'TGA' : '*', 'TGC' : 'C', 'TGG' : 'W', 'TGT' : 'C', 
    'TTA' : 'L', 'TTC' : 'F', 'TTG' : 'L', 'TTT' : 'F'
}
genetic_code_3 = {
    'AAA' : 'Lys', 'AAC' : 'Asn', 'AAG' : 'Lys', 'AAT' : 'Asn', 
    'ACA' : 'Thr', 'ACC' : 'Thr', 'ACG' : 'Thr', 'ACT' : 'Thr', 
    'AGA' : 'Arg', 'AGC' : 'Ser', 'AGG' : 'Arg', 'AGT' : 'Ser', 
    'ATA' : 'Ile', 'ATC' : 'Ile', 'ATG' : 'Met', 'ATT' : 'Ile', 
    'CAA' : 'Gln', 'CAC' : 'His', 'CAG' : 'Gln', 'CAT' : 'His', 
    'CCA' : 'Pro', 'CCC' : 'Pro', 'CCG' : 'Pro', 'CCT' : 'Pro', 
    'CGA' : 'Arg', 'CGC' : 'Arg', 'CGG' : 'Arg', 'CGT' : 'Arg',
    'CTA' : 'Leu', 'CTC' : 'Leu', 'CTG' : 'Leu', 'CTT' : 'Leu', 
    'GAA' : 'Glu', 'GAC' : 'Asp', 'GAG' : 'Glu', 'GAT' : 'Asp', 
    'GCA' : 'Ala', 'GCC' : 'Ala', 'GCG' : 'Ala', 'GCT' : 'Ala', 
    'GGA' : 'Glu', 'GGC' : 'Asp', 'GGG' : 'Glu', 'GGT' : 'Asp', 
    'GTA' : 'Val', 'GTC' : 'Val', 'GTG' : 'Val', 'GTT' : 'Val', 
    'TAA' : 'STOP', 'TAC' : 'Tyr', 'TAG' : 'STOP', 'TAT' : 'Tyr', 
    'TCA' : 'Ser', 'TCC' : 'Ser', 'TCG' : 'Ser', 'TCT' : 'Ser', 
    'TGA' : 'STOP', 'TGC' : 'Cys', 'TGG' : 'Trp', 'TGT' : 'Cys', 
    'TTA' : 'Leu', 'TTC' : 'Phe', 'TTG' : 'Leu', 'TTT' : 'Phe'
}
base_n_digits = (
    r'''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%'''
    r'''&'()*+,./:;<=>?@[\]^_`{|}~'''
)
base_n_digits_list = list(base_n_digits)
base_n_digits_set = set(base_n_digits)
base_n_digits_to_int = {
    '0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, 
    '8' : 8, '9' : 9, 'a' : 10, 'b' : 11, 'c' : 12, 'd' : 13, 'e' : 14, 
    'f' : 15, 'g' : 16, 'h' : 17, 'i' : 18, 'j' : 19, 'k' : 20, 'l' : 21, 
    'm' : 22, 'n' : 23, 'o' : 24, 'p' : 25, 'q' : 26, 'r' : 27, 's' : 28, 
    't' : 29, 'u' : 30, 'v' : 31, 'w' : 32, 'x' : 33, 'y' : 34, 'z' : 35, 
    'A' : 36, 'B' : 37, 'C' : 38, 'D' : 39, 'E' : 40, 'F' : 41, 'G' : 42, 
    'H' : 43, 'I' : 44, 'J' : 45, 'K' : 46, 'L' : 47, 'M' : 48, 'N' : 49, 
    'O' : 50, 'P' : 51, 'Q' : 52, 'R' : 53, 'S' : 54, 'T' : 55, 'U' : 56, 
    'V' : 57, 'W' : 58, 'X' : 59, 'Y' : 60, 'Z' : 61, '!' : 62, '"' : 63, 
    '#' : 64, '$' : 65, '%' : 66, '&' : 67, "'" : 68, '(' : 69, ')' : 70, 
    '*' : 71, '+' : 72, ',' : 73, '.' : 74, '/' : 75, ':' : 76, ';' : 77, 
    '<' : 78, '=' : 79, '>' : 80, '?' : 81, '@' : 82, '[' : 83, '\\' : 84, 
    ']' : 85, '^' : 86, '_' : 87, '`' : 88, '{' : 89, '|' : 90, '}' : 91, 
    '~' : 92
}

class GraphNode:
    '''A single node in an abstract graph.
    
    Stores information about the node's identity, weight, and (unweighted) 
    connectivity.
    '''
    def __init__(
        self,
        sequence : str = None,
        amino_sequence : str = None,
        fitness : float or types.NoneType = None,
        neighbors :  list[int] = None
    ) -> 'GraphNode':
        self.sequence = sequence if sequence is not None else ''
        self.amino_sequence = (
            amino_sequence if amino_sequence is not None else ''
        )
        self.fitness = fitness
        self.neighbors = neighbors if neighbors is not None else []
        # Used by Graph.markov_equilibrium, etc.
        self.population : float = None
        # Set by Graph.calc_peaks
        self.is_peak : bool = None
        self.in_plateau : int = -1
    def __str__(self) -> str:
        return (
            f"GraphNode(sequence = '{self.sequence}', "
            f"amino_sequence = '{self.amino_sequence}', "
            f"fitness = {self.fitness}, neighbors = {self.neighbors})"
        )
    def __repr__(self) -> str:
        return (
            f'GraphNode({self.sequence}, {self.amino_sequence}, {self.fitness},'
            f' {self.neighbors})'
        )
    def check_connected(self, node_id : int) -> bool:
        return node_id in self.neighbors
    def check_neighbor(self, node_id : int) -> int:
        if self.isconnected(node_id):
            bond = self.neighbor.index(node_id)
        else:
            raise ValueError(f'Node {node_id} is not connected to this node')
        return bond

class Graph:
    '''A group of GraphNodes.
    
    Implimented with the assumption that each node is a genotype of a specific 
    length.

    Features:
        Calculate edges of common graphs, esp. Hamming graphs and subgraphs
        Support for sequences of DNA, RNA, amino acids, or arbitrary strings
        Modify graphs by applying a variety of cutoffs, or randomizations
        Detect peaks, including peaks within local networks of neutral mutations 
        Detect pits
        Detect diconnected subgraphs
        Calculate transition probabilities based on a variety of algorithms
        Simulate evolution with a fast tree-trimming algorithm or random walkers
        Calculate the basins of peaks, with fast and low-RAM algorithms
        Logging
        CSV output
    '''
    def __init__(
        self,
        nodes : list[GraphNode] or int = None,
    ) -> 'Graph':
        self.nodes = nodes if nodes is not None else [GraphNode('')]
        if type(self.nodes) == int:
            self.nodes = [GraphNode('') for _ in range(self.nodes)]
        self.walk_analyzer : walk_analysis.WalkAnalyzer =  None
        # These are calculated in `calc_peaks`
        self.peaks : list = None
        self.pits : list = None
        self.plateaus : list[dict] = None
        # These are calculated in `calc_transition_matrix`
        self.transition_matrix : scipy.sparse.coo_array = None
        self.transition_matrix_csr : scipy.sparse.csr_array = None
        self.transition_matrix_csc : scipy.sparse.csc_array = None

        # Sort out the sequence inputs to see if edges can be calculated with 
        # point mutations
        self.remapped_sequences = False
        self.alphabet = set(
            base for node in self.nodes for base in node.sequence
        )
        self.amino_alphabet = set(
            base for node in self.nodes for base in node.amino_sequence
        )
        if self.alphabet != set() and (
            self.alphabet.issubset(dna_set) or self.alphabet.issubset(rna_set)
        ):
            # To make calculating edges easier, there's a table of all the nodes
            # One entry for every possible sequence, a -1 means that it is not
            # present in the graph
            # In cases where the graph is only a sparse subgraph of a Hamming 
            # graph, the lookup table is a subclassed `dict`
            self.dtype = [GraphNode, len(self.nodes[0].sequence), 4]
            if len(self.nodes) > ((4 ** self.dtype[1]) // 2):
                self.lookup_table = [-1] * (4 ** self.dtype[1])
            else:
                self.lookup_table = sparse_list.SparseList(default = -1)
            for inode, node in enumerate(self.nodes):
                self.lookup_table[sequence_to_int(node.sequence)] = inode
            # Some helper functions
            def inode_lookup(sequence) -> GraphNode:
                return self.lookup_table[sequence_to_int(sequence)]
            self.inode_lookup = inode_lookup
            def node_lookup(sequence) -> GraphNode:
                return self[self.lookup_table[sequence_to_int(sequence)]]
            self.node_lookup = node_lookup
            def get_sequence(inode) -> str:
                return self[inode].sequence
            self.get_sequence = get_sequence
        elif self.alphabet == set() and self.amino_alphabet.issubset(
            amino_set_1
        ):
            self.dtype = [GraphNode, len(self.nodes[0].amino_sequence), 21]
            if len(self.nodes) > ((21 ** self.dtype[1]) // 2):
                self.lookup_table = [-1] * (21 ** self.dtype[1])
            else:
                self.lookup_table = sparse_list.SparseList(default = -1)
            for inode, node in enumerate(self.nodes):
                self.lookup_table[
                    amino_sequence_to_int(node.amino_sequence)
                ] = inode
            def inode_lookup(sequence) -> GraphNode:
                return self.lookup_table[amino_sequence_to_int(sequence)]
            self.inode_lookup = inode_lookup
            def node_lookup(sequence) -> GraphNode:
                return self[self.lookup_table[amino_sequence_to_int(sequence)]]
            self.node_lookup = node_lookup
            def get_sequence(inode) -> str:
                return self[inode].amino_sequence
            self.get_sequence = get_sequence
        elif self.alphabet != set() and self.alphabet.issubset(
            base_n_digits_set
        ):
            self.dtype = [
                GraphNode, len(self.nodes[0].sequence), len(self.alphabet)
            ]
            if max(
                base_n_digits_to_int[val] for val in self.alphabet
            ) > (len(self.alphabet) - 1):
                # In this case, to prevent the lookup table from exploding, 
                # etc., the sequences will be remapped to a minimal alphabet (at
                # the scope of the whole sequence, not each locus independently)
                new_alphabet = set(base_n_digits[:len(self.alphabet)])
                self.remapped_sequences = True
                warnings.warn(
                    f'Remapping sequences: {sorted(self.alphabet)}'
                    f' -> {sorted(new_alphabet)}',
                    RuntimeWarning, stacklevel = 2
                )
                # It's not important how the sequences are mapped, but these 
                # sets are sorted to make sure that the mapping is the same 
                # every time
                self.alphabet_map = {
                    old : new for old, new 
                    in zip(sorted(self.alphabet), sorted(new_alphabet))
                }
                for node in self.nodes:
                    node.sequence = ''.join(
                        self.alphabet_map[char] for char in node.sequence
                    )
                self.alphabet = new_alphabet
                self.alphabet_map_reverse = {
                    new : old for old, new 
                    in zip(sorted(self.alphabet), sorted(new_alphabet))
                }
                self.dtype[2] = len(self.alphabet)
            if len(self.nodes) > ((self.dtype[2] ** self.dtype[1]) // 2):
                self.lookup_table = [-1] * (self.dtype[2] ** self.dtype[1])
            else:
                self.lookup_table = sparse_list.SparseList(default = -1)
            for inode, node in enumerate(self.nodes):
                self.lookup_table[
                    base_n_sequence_to_int(node.sequence, base = self.dtype[2])
                ] = inode
            def inode_lookup(sequence) -> GraphNode:
                return self.lookup_table[base_n_sequence_to_int(
                    sequence, base = self.dtype[2]
                )]
            self.inode_lookup = inode_lookup
            def node_lookup(sequence) -> GraphNode:
                return self[self.lookup_table[base_n_sequence_to_int(
                    sequence, base = self.dtype[2]
                )]]
            self.node_lookup = node_lookup
            def get_sequence(inode) -> str:
                return self[inode].sequence
            self.get_sequence = get_sequence

        # This is calculated in `find_subgraphs`
        self.subgraphs : list[list[int]] = None
        # Set by `open_log`, unset by `close_log`
        self._log_handle : io.TextIOWrapper = None 
        # This is pretty cursed, but it seems to work??? `write_log` is handled 
        # by `open_log`, but this funny looking lambda makes it so that 
        # `write_log` can initialize itself
        self.write_log : types.FunctionType = lambda x: (
            self.open_log(), self.write_log(x)
        )
        # Also set by `open_log`
        self.start_time : float = None
    def __len__(self) -> int:
        return len(self.nodes)
    def __getitem__(
        self,
        index : int or slice or tuple or list
    ) -> list[GraphNode] or list[int]:
        if isinstance(index, (int, numpy.integer, slice)):
            return self.nodes[index]
        elif isinstance(index, (tuple, list)):
            return [self.nodes[val] for val in index]
        else:
            raise IndexError(
                f'Index "{index}" with type "{type(index)}" is not a valid '
                f'index of this {type(self)}'
            )
    def __setitem__(
        self,
        index : int or list,
        node : GraphNode or list[GraphNode]
    ):
        self.nodes[index] = node
    def __iter__(self):
        for node in self.nodes:
            yield node
    def __repr__(self) -> str:
        return (
            f'Graph([\n  {',\n  '.join([f'{node}' for node in self.nodes])}\n])'
        )
    def write_csv(self, filename) -> None:
        open(filename, 'w').write('\n'.join([
            f'{self.get_sequence(inode)},{node.fitness}' 
            for inode, node in enumerate(self)
        ]))

    def permute_fitnesses(self, group : bool or str = True) -> None:
        '''Randomize the fitness of nodes in certain groups of the graph.

        Valid values for `group` are `False`, `True`, `'aa27'`, ...
        '''
        if group in (True, 'after_cutoff'):
            fitnesses = [node.fitness for node in self]
            random.shuffle(fitnesses)
            for node, fitness in zip(self, fitnesses):
                node.fitness = fitness
        elif group == 'aa27':
            inodes_by_aa27 = [[] for _ in amino_list_1]
            for inode, nodes in enumerate(self):
                inodes_by_aa27[
                    amino_to_int_1[node.amino_sequence[1]]
                ].append(inode)
            for amino in inodes_by_aa27:
                fitnesses = [self[inode].fitness for inode in amino]
                random.shuffle(fitnesses)
                for inode, fitness in zip(amino, fitnesses):
                    self[inode].fitness = fitness
        elif group == False:
            pass
        else:
            raise ValueError(
            f'{group=} is not a valid argument in Graph.permute_fitnesses'
        )
        
    def remove_nodes(
        self, cutoff : float, method : str = 'low_fitness'
    ) -> None:
        '''Get rid of some nodes, using a specified method.

        This function assumes that neighbors have already been calculated. 
        Because of the change in node indicies, it resets the existing graph 
        object, removing the transition matrix, etc.
        '''
        the_real_dtype = self.dtype
        the_real_walk_analyzer = self.walk_analyzer
        if method == 'low_fitness':
            self.__init__([node for node in self if node.fitness > cutoff])
        elif method == 'low_fitness_neighbors':
            self.__init__([
                node for node in self if node.fitness > cutoff or True in [
                    self[neighbor].fitness > cutoff 
                    for neighbor in node.neighbors
                ]
            ])
        elif method == 'high_fitness':
            self.__init__([node for node in self if node.fitness < cutoff])
        elif method == 'high_fitness_neighbors':
            self.__init__([
                node for node in self if node.fitness < cutoff or True in [
                    self[neighbor].fitness < cutoff 
                    for neighbor in node.neighbors
                ]
            ])
        elif method == 'random':
            #self.__init__([node for node in self if random.random() > cutoff])
            self.__init__([
                self[index] for index in random.sample(
                    range(len(self)), k = int((1 - cutoff) * len(self))
                )
            ])
        self.dtype = the_real_dtype
        self.walk_analyzer = the_real_walk_analyzer

    def calc_edges(self, graph_type : str = 'point_mutations', *args):
        match graph_type:
            case 'point_mutations':
                self._calc_edges_point_mutations()
            case 'hamming':
                self._calc_edges_hamming(dim = args[0], alphabet = args[1])
            case 'full':
                self._calc_edges_full(num_vertices = args[0])
            case 'random' | 'erdos_renyi':
                self._calc_edges_random(
                    num_vertices = args[0], edge_prob = args[1]
                )
            case 'regular' | 'random_regular':
                ...
                self._calc_edges_random_regular(
                    num_vertices = args[0], degree = args[0]
                 )
            case 'power_law' | 'scale_free' | 'barabasi' | 'barasabi_albert':
                ...
                self._calc_edges_power_law(degree = args[0], power = args[1])
            case _:
                raise NotImplementedError(
                    f'Graph type {graph_type} has not been implemented'
                )

    # (timing: 3 seconds for 260000 nodes)
    def _calc_edges_point_mutations(self) -> None:
        '''Calculate edges in a graph based on point mutations of the sequences.

        This is basically the same as `_calc_edges_hamming`, but the nodes are 
        not interchangable, so the IDs of neighbors are found through 
        `self.lookup_table`.

        The function has three sections for DNA/RNA sequences, amino acid 
        sequences, and arbitrary base-n sequences.
        '''
        if self.alphabet != set() and (
            self.alphabet.issubset(dna_set) or self.alphabet.issubset(rna_set)
        ):
            for node in self:
                node.neighbors = []
                sequence_value = sequence_to_int(node.sequence)
                # Calculate every possible point mutation, then find its index 
                # in the lookup table
                for ibase in range(self.dtype[1]):
                    digit = self.dtype[2] ** ibase
                    temp_value = (
                        sequence_value 
                        -  digit * base_to_int[node.sequence[ibase]]
                    )
                    mutant_bases = self.alphabet - set(node.sequence[ibase])
                    for base in mutant_bases:
                        neighbor = temp_value + digit * base_to_int[base]
                        # If there's a node present for the mutant sequence, add
                        # it  as a neighbor; unfortunately the easiest thing to 
                        # do is repeat the calculation for the neighbor as well
                        if self.lookup_table[neighbor] != -1:
                            node.neighbors.append(self.lookup_table[neighbor])
        elif (
            self.alphabet == set() and self.amino_alphabet.issubset(amino_set_1)
        ):
            for node in self:
                node.neighbors = []
                sequence_value = amino_sequence_to_int(node.amino_sequence)
                # Calculate every possible point mutation, then find its index 
                # in the lookup table
                for ibase in range(self.dtype[1]):
                    digit = self.dtype[2] ** ibase
                    temp_value = (
                        sequence_value 
                        -  digit * amino_to_int_1[node.amino_sequence[ibase]]
                    )
                    mutant_bases = self.amino_alphabet - set(
                        node.amino_sequence[ibase]
                    )
                    for base in mutant_bases:
                        neighbor = temp_value + digit * amino_to_int_1[base]
                        # If there's a node present for the mutant sequence, add
                        # it as a neighbor; unfortunately the easiest thing to 
                        # do is repeat the calculation for the neighbor as well
                        if self.lookup_table[neighbor] != -1:
                            node.neighbors.append(self.lookup_table[neighbor])
        else: # self.alphabet.issubset(base_n_digits_set)
            for node in self:
                node.neighbors = []
                sequence_value = base_n_sequence_to_int(
                    node.sequence, base = self.dtype[2]
                )
                # Calculate every possible point mutation, then find its index 
                # in the lookup table
                for ibase in range(self.dtype[1]):
                    digit = self.dtype[2] ** ibase
                    temp_value = (
                        sequence_value 
                        -  digit * base_n_digits_to_int[node.sequence[ibase]]
                    )
                    mutant_bases = self.alphabet - set(node.sequence[ibase])
                    for base in mutant_bases:
                        neighbor = (
                            temp_value + digit * base_n_digits_to_int[base]
                        )
                        # If there's a node present for the mutant sequence, add
                        # it as a neighbor; unfortunately the easiest thing to 
                        # do is repeat the calculation for the neighbor as well
                        if self.lookup_table[neighbor] != -1:
                            node.neighbors.append(self.lookup_table[neighbor])

    def _calc_edges_hamming(self, dim : int, alphabet : int) -> None:
        '''Calculate edges in a Hamming graph.

        Hamming graphs are abstractions of hypercube graphs for an alphabet size
        greater than 2, and are abstractions of rook's graphs for dimensions 
        larger than 2. If each vertex has a sequence of characters of length 
        `dim`, then each vertex is connected to vertices which have the same 
        sequence except for one character.
        '''
        assert dim >= 0
        assert alphabet >= 1
        num_vertices = alphabet ** dim
        if len(self) != num_vertices:
            self.nodes = [GraphNode() for _ in range(num_vertices)]
        else:
            for node in self:
                node.neighbors = []
        base_q_vertex = [0] * (dim + 1)
        base_q_vertex[0] = -1
        # Pregenerate some values for later
        base_q_digit_values = [[0] * (alphabet - 1) for _ in range(dim)]
        temp_val1 = 1
        temp_val2 = temp_val1
        for iplace in range(dim):
            for idigit in range(alphabet - 1):
                base_q_digit_values[iplace][idigit] = temp_val2
                temp_val2 += temp_val1
            temp_val1 *= alphabet
        # Iterate over every vertex
        for ivertex in range(num_vertices):
            base_n_increment_in_place(base_q_vertex, base = alphabet)
            self[ivertex].sequence = ''.join(
                base_n_digits[digit] for digit in base_q_vertex[:-1]
            )[::-1]
            # Iterate over all `d` of the base-`q` digits of `ivertex`
            for iplace in range(dim):
                digit_value = base_q_vertex[iplace]
                # Iterate over values of the digit greater than the current 
                # value
                for digit_increase in range(alphabet - digit_value - 1):
                    neighbor = (
                        ivertex + base_q_digit_values[iplace][digit_increase]
                    )
                    self[ivertex].neighbors.append(neighbor)
                    self[neighbor].neighbors.append(ivertex)
            
    def _calc_edges_full(self, num_vertices : int) -> None:
        '''Calculate edges in a full graph

        A full graph has all nodes connected with each other; there is only one 
        peak (except in the case of neutral neighbors).
        '''
        assert num_vertices >= 0
        if len(self) != num_vertices:
            self.nodes = [GraphNode() for _ in range(num_vertices)]
        for inode, node in enumerate(self):
            node.neighbors = (
                list(range(0, inode)) + list(range(inode, num_vertices))
            )
        
    def _calc_edges_random(self, num_vertices : int, edge_prob : float) -> None:
        '''Calculate edges in a totally random graph.

        Given a number of nodes, it is possible to enumerate the number of edges
        present in a full graph; then, they can be sampled directly.

        This implimentation creates a graph where the requested number of edges 
        exsit, rather than sampling edges with a fixed probability (i.e., with 
        variation in the actual number of edges).
        '''
        assert num_vertices >= 0
        assert 0 <= edge_prob <= 1
        if len(self) != num_vertices:
            self.nodes = [GraphNode() for _ in range(num_vertices)]
        else:
            for node in self:
                node.neighbors = []
        if num_vertices == 0 or edge_prob == 0:
            pass
        elif edge_prob == 1:
            self._calc_edges_full(num_vertices)
        else:
            max_edges = num_vertices * (num_vertices - 1) // 2
            num_edges = int(edge_prob * max_edges)
            max_edge = max_edges - 1
            edges = random.sample(range(max_edges), num_edges)
            for edge_number in edges:
                node1 = int(0.5 + math.sqrt(2 * edge_number + 0.25))
                node2 = edge_number - (node1 * node1 - node1) // 2
                self[node1].neighbors.append(node2)
                self[node2].neighbors.append(node1)

    def find_subgraphs(self) -> None:
        subgraphs = []
        num_nodes = len(self)
        in_a_graph = [False] * num_nodes
        while sum(in_a_graph) < num_nodes:
            new_graph = [in_a_graph.index(False)]
            in_a_graph[new_graph[0]] = True
            current_nodes = new_graph
            while len(current_nodes) > 0:
                new_neighbors = set([]).union(*[
                    self[inode].neighbors for inode in current_nodes
                ])
                new_neighbors = [
                    neighbor for neighbor in new_neighbors 
                    if not in_a_graph[neighbor]
                ]
                new_graph.extend(new_neighbors)
                for neighbor in new_neighbors:
                    in_a_graph[neighbor] = True
                current_nodes = new_neighbors
            subgraphs.append(new_graph)
        self.subgraphs = sorted(subgraphs, key = len, reverse = True)

    def calc_peaks(self, check_plateaus = False) -> None:
        '''Find all the peaks
        
        Also, find the pits.
        If `check_plateaus` is `True`, then peaks will be checked for the edge 
        case of having neighbors with equal fitness in a network of neutral 
        mutations from which there is an edge leading to higher fitness.
        '''
        self.peaks = []
        self.pits = []
        for inode, node in enumerate(self):
            uphill_neighbors = [
                jnode for jnode in node.neighbors 
                if self[jnode].fitness > node.fitness
            ]
            if len(uphill_neighbors) == len(node.neighbors):
                # Note that a node with no neighbors is also peak
                self.pits.append(inode)
            if len(uphill_neighbors) == 0:
                self.peaks.append(inode)
                node.is_peak = True
            else:
                node.is_peak = False
        if check_plateaus:
            for node in self:
                node.in_plateau = -1
            self.plateaus = []
            for inode in self.peaks.copy():
                if self[inode].in_plateau >= 0:
                    continue
                node = self[inode]
                neutral_neighbors = [
                    jnode for jnode in node.neighbors 
                    if self[jnode].fitness == node.fitness
                ]
                if len(neutral_neighbors) != 0:
                    current_plateau = set([inode, *neutral_neighbors])
                    current_nodes = set(neutral_neighbors)
                    while len(current_nodes) > 0:
                        new_nodes = set([]).union(*[[
                            neighbor for neighbor in self[jnode].neighbors 
                            if self[neighbor].fitness == node.fitness
                        ] for jnode in current_nodes])
                        current_nodes = new_nodes - current_plateau
                        current_plateau |= current_nodes
                    icurrent_plateau = len(self.plateaus)
                    for jnode in current_plateau:
                        self[jnode].in_plateau = icurrent_plateau
                    if all(self[jnode].is_peak for jnode in current_plateau):
                        # Currently, this isn't used for anything, except to 
                        # make sure each plateau is unique
                        self.plateaus.append(
                            {'escapable' : False, 'members' : current_plateau}
                        )
                    else:
                        self.plateaus.append(
                            {'escapable' : True, 'members' : current_plateau}
                        )
                        for jnode in current_plateau:
                            if self[jnode].is_peak:
                                self[jnode].is_peak = False
                                self.peaks.remove(jnode)
            self.write_log(
                f'\nFound {len(self.plateaus)} plateaus:\n'
                + '\n'.join([
                    f'Escapable: {plateau['escapable']}, ' 
                    f'Size: {len(plateau['members'])}' 
                    for plateau in self.plateaus
                ] + ['\n\n'])
            )

    def calc_transition_matrix(
        self, evolution_type : str = 'kimura', permute : bool = False, 
        check_plateaus : bool = False, exact : bool = False, population = 1e8
    ) -> None:
        '''Calculates the transition probabilities between each pair of nodes.

        Make sure to call `Graph.calc_peaks` before this function.

        The assumption is that most pairs will have a probability of 0, so the 
        output is saved as a sparse matrix.
        `check_plateaus` allows for neutral steps within escapable plateaus. It 
        is only applied to 'kimura', 'random', and 'greedy' evolution types, not
        because I'm sure it wouldn't work for 'basin' or 'none', but because it 
        doesn't seem useful. 'basin' and 'none' are compatible with 
        `check_plateaus`, but they do not treat them as carefully.
        `exact` only applies to Kimura evolution, overrides `check_plateaus`,
        and will calculate transition probabilities for neutral and deleterious 
        mutations.
        '''
        if evolution_type == 'kimura' and exact:
            check_plateaus = False
            num_probabilities = sum(len(node.neighbors) for node in self)
            probabilities = [0] * num_probabilities
            probability_coordinates = [0] * num_probabilities
            prob_counter = 0
            for inode, node in enumerate(self):
                probabilities_this_node = [calc_transition_prob(
                    node.fitness, self[jnode].fitness, population
                ) for jnode in node.neighbors]
                # Element (y, x) should be the probability that population on 
                # node x goes to node y in the next step of the Markov chain 
                # (indexing by row first)
                coordinates_this_node = [
                    (jnode, inode) for jnode in node.neighbors
                ]
                norm = sum(probabilities_this_node)
                probabilities_this_node = [
                    prob / norm for prob in probabilities_this_node
                ]
                slice_end = prob_counter + len(probabilities_this_node)
                probabilities[prob_counter:slice_end] = probabilities_this_node
                probability_coordinates[
                    prob_counter:slice_end
                ] = coordinates_this_node
                prob_counter = slice_end
        elif evolution_type == 'kimura':
            # We're going to use a COO sparse tensor, so we need to calculate 2 
            # tensors:
            #  * A 1xN tensor of all the non-zero transition probabilities
            #  * A 2xN tensor of the coordinates of the transition probabilities 
            #    within the matrix
            # But I like lists, so I'll make them tensors afterwards
            num_probabilities = sum(len(node.neighbors) for node in self) // 2
            probabilities = [0] * num_probabilities
            probability_coordinates = [0] * num_probabilities
            prob_counter = 0
            if permute:
                transition_probs = [
                    calc_transition_prob(node1.fitness, self[inode2].fitness) 
                    for node1 in self for inode2 in node1.neighbors 
                    if self[inode2].fitness > node1.fitness
                ]
                random.shuffle(transition_probs)
                transition_probs = iter(transition_probs)
                get_transition_prob = lambda fitness1, fitness2: (
                    0 if fitness2 < fitness1 else next(transition_probs)
                )
            else:
                get_transition_prob = calc_transition_prob
            for inode, node in enumerate(self):
                probabilities_this_node = []
                coordinates_this_node = []
                for jnode in node.neighbors:
                    if node.fitness < self[jnode].fitness:
                        prob = get_transition_prob(
                            node.fitness, self[jnode].fitness
                        )
                        probabilities_this_node.append(prob)
                        # Element (y, x) should be the probability that 
                        # population  on node x goes to node y in the next step 
                        # of the Markov  chain (indexing by row first)
                        coordinates_this_node.append((jnode, inode))
                if len(probabilities_this_node) == 0:
                    # In this case the node is a peak
                    probabilities_this_node = [1.0]
                    coordinates_this_node = [(inode, inode)]
                else:
                    norm = sum(probabilities_this_node)
                    probabilities_this_node = [
                        prob / norm for prob in probabilities_this_node
                    ]
                slice_end = prob_counter + len(probabilities_this_node)
                probabilities[prob_counter:slice_end] = probabilities_this_node
                probability_coordinates[
                    prob_counter:slice_end
                ] = coordinates_this_node
                prob_counter = slice_end
        elif evolution_type == 'random':
            num_probabilities = sum(len(node.neighbors) for node in self) // 2
            probabilities = [0] * num_probabilities
            probability_coordinates = [0] * num_probabilities
            prob_counter = 0
            for inode, node in enumerate(self):
                uphill_neighbors = [
                    jnode for jnode in node.neighbors 
                    if self[jnode].fitness > node.fitness
                ]
                if len(uphill_neighbors) == 0:
                    probabilities_this_node = [1]
                    coordinates_this_node = [(inode, inode)]
                else:
                    probabilities_this_node = [1 / len(uphill_neighbors)] * len(
                        uphill_neighbors
                    )
                    # Element (y, x) should be the probability that population 
                    # on node x goes to node y in the next step of the Markov 
                    # chain (indexing by row first)
                    coordinates_this_node = [
                        (jnode, inode) for jnode in uphill_neighbors
                    ]
                slice_end = prob_counter + len(probabilities_this_node)
                probabilities[prob_counter:slice_end] = probabilities_this_node
                probability_coordinates[
                    prob_counter:slice_end
                ] = coordinates_this_node
                prob_counter = slice_end
        elif evolution_type == 'greedy':
            num_probabilities = len(self)
            probabilities = [1] * num_probabilities
            probability_coordinates = [0] * num_probabilities
            for inode, node in enumerate(self):
                # It might not be necessary to check both of these
                if len(node.neighbors) > 0 and not node.is_peak:
                    sorted_neighbors = sorted(
                        node.neighbors, key = lambda x: self[x].fitness
                    )
                    # Note that here, there's an arbitrary choice between 
                    # neighbors with equal fitness, which would not work with 
                    # plateaus, except that they are dealt with separately below
                    probability_coordinates[inode] = (
                        sorted_neighbors[-1], inode
                    )
                else:
                    probability_coordinates[inode] = (inode, inode)
        elif evolution_type == 'basin':
            num_nodes = len(self)
            sorted_nodes = sorted(
                range(num_nodes), key = lambda x: self[x].fitness
            )
            basins = []
            in_a_basin = [False] * num_nodes
            while sum(in_a_basin) < num_nodes:
                next_peak = next(
                    node for node in sorted_nodes[::-1] if not in_a_basin[node]
                )
                new_basin = [next_peak]
                in_a_basin[next_peak] = True
                current_nodes = new_basin
                while len(current_nodes) > 0:
                    # TODO It's not actually clear whether or not it could be 
                    # faster to do the lookups in `in_a_basin` first and the 
                    # unions in this line second
                    new_neighbors = set([]).union(*[[
                        # Add each node's neighbors to the basin if...
                        neighbor for neighbor in self[inode].neighbors 
                        # ...the neighbor has lower fitness...
                        if self[neighbor].fitness < self[inode].fitness or (
                            # ...or if we are concerned about plateaus...
                            check_plateaus
                            # ...it needs to have the same fitness...
                            and self[neighbor].fitness == self[inode].fitness 
                            # ...and not have uphill neighbors itself.
                            and all(
                                self[inode].fitness <= self[neighbor].fitness 
                                for inode in self[neighbor].neighbors
                            )
                        )
                    ] for inode in current_nodes])
                    # Note that in the line above, inescapable plateaus are not 
                    # treated carefully, but this algorithm is not really about 
                    # which peak a walker reaches, but what fitness it reaches.
                    # (and, this algorithm doesn't really matter.)
                    new_neighbors = [
                        neighbor for neighbor in new_neighbors 
                        if not in_a_basin[neighbor]
                    ]
                    new_basin.extend(new_neighbors)
                    for neighbor in new_neighbors:
                        in_a_basin[neighbor] = True
                    current_nodes = new_neighbors
                basins.append(new_basin)
            probabilities = [1] * num_nodes
            probability_coordinates = [
                (basin[0], node) for basin in basins for node in basin
            ]
        elif evolution_type == 'none':
            last_peak = None
            sorted_nodes = sorted(
                range(len(self)), key = lambda x: self[x].fitness
            )
            probability_coordinates = []
            for inode in sorted_nodes:
                node = self[inode]
                if node.is_peak:
                    last_peak = inode
                if last_peak is None:
                    continue
                probability_coordinates.append((last_peak, inode))
                probabilities = [1] * len(probability_coordinates)
        if probability_coordinates[-1] == 0:
            first_zero = probability_coordinates.index(0)
            warnings.warn(
                f'This graph has ~{num_probabilities - first_zero} neighboring '
                'nodes with equal fitness! Simulation results may not be '
                'accurate.', RuntimeWarning, stacklevel = 2
            )
            probabilities = probabilities[:first_zero]
            probability_coordinates = probability_coordinates[:first_zero]
        # This seems to be working great on the CR9114-B landsacpe, but it's 
        # rather slow... probably just because of all the `pop`s and `extend`s
        if check_plateaus and evolution_type in ('kimura', 'random', 'greedy'):
            for plateau in self.plateaus:
                if plateau['escapable'] == True:
                    for inode in plateau['members']:
                        uphill_neighbors = [
                            jnode for jnode in self[inode].neighbors 
                            if self[jnode].fitness > self[inode].fitness
                        ]
                        neutral_neighbors = [
                            jnode for jnode in self[inode].neighbors 
                            if self[jnode].fitness == self[inode].fitness
                        ]
                        if (
                            len(uphill_neighbors) == 0 
                            and len(neutral_neighbors) > 0
                        ):
                            iprob = probability_coordinates.index(
                                (inode, inode)
                            )
                            probability_coordinates.pop(iprob)
                            probabilities.pop(iprob)
                            probabilities.extend([
                                1 / len(neutral_neighbors)
                            ] * len(neutral_neighbors))
                            probability_coordinates.extend([
                                (jnode, inode) for jnode in neutral_neighbors
                            ])

        probabilities = numpy.array(probabilities)
        probability_coordinates = numpy.array(
            list(zip(*probability_coordinates)) 
        )
        self.transition_matrix = scipy.sparse.coo_array(
            (probabilities, probability_coordinates), 
            shape = (len(self),) * 2
        )
        # The COO layout is bad for slicing, CSC and CSR are fast for returning 
        # columns and rows, respectively. Converting to a different format seems
        # to be doing a deepcopy, but I think that for most graphs this will not
        # be the largest use of RAM
        self.transition_matrix_csr = self.transition_matrix.tocsr()
        self.transition_matrix_csc = self.transition_matrix.tocsc()

    def markov_equilibrium(self, initial_state : list = None) -> list:
        '''Propagate an initial population through the graph.

        The default inital state is a uniform distribution.
        Make sure to call `Graph.calc_transition_matrix` before this function.
        '''
        start = time.time()
        current_pits = self.pits
        probabilities_cols = self.transition_matrix_csc
        probabilities_rows = self.transition_matrix_csr
        initial_state = initial_state if initial_state is not None else (
            [1.0 / len(self)] * len(self)
        )
        for node, state in zip(self, initial_state):
            node.population = state
        iteration = 0
        while len(current_pits) > 0:
            iteration += 1
            next_pits = set()
            for inode in current_pits:
                # To find what is going out of node, check its column
                going_to = probabilities_cols[:, [inode]]
                for ineighbor, prob in zip(
                    going_to.nonzero()[0], going_to.data
                ):
                    self[ineighbor].population += self[inode].population * prob
                self[inode].population = 0
                # Rather than taking the union every iteration, we could make a 
                # big list and then make it a set at the end of the loop; a 
                # little testing suggests that repeated unions is potentially 
                # ~20% faster (and lighter on RAM as well)
                next_pits |= set(going_to.nonzero()[0])
            current_pits = []
            for inode in next_pits:
                # ...To find what is coming into a node, check its row
                coming_from = probabilities_rows[[inode], :]
                # I think I got very lucky here: when `inode` is a peak, the 
                # list of neighbors we get back from `coming_from.nonzero` 
                # includes `inode` itself! But it's population never becomes 
                # zero, so it also never gets appended to the `current_pits` 
                # list (which would result in it's population arbitrarily 
                # getting doubled). So it's the right behavior, but entirely by 
                # accident I think
                for ineighbor in coming_from.nonzero()[1]:
                    if self[ineighbor].population != 0:
                        break
                else:
                    current_pits.append(inode)
        print(
            f'Markov equilibrium in {time.time() - start} seconds after '
            f'{iteration} iterations'
        )
        return [node.population for node in self]

    def fast_markov_equilibrium(self, initial_state : list = None) -> list:
        '''Propagate an initial population through the graph, but do it quickly.

        Instead of messing with sets and trying to find all of the pits over and
        over again, we can just sort all of the nodes by fitness and iterate 
        straight from the bottom to the top.
        
        The slow version is still sort of useful because the number of 
        iterations it takes to finish is the same as the number of iterations 
        the real Markov chain would take, but even that is only vaguely 
        interesting.
        
        The default inital state is a uniform distribution.
        Make sure to call `Graph.calc_transition_matrix` before this function.
        Timings: ~90 seconds compared with ~360 for the slower version: 4x 
        faster.

        Compared with random walkers, this approach is exact, and way faster. It
        has the downsides of needing to store the whole graph, and does not work
        for graphs with escapable plateaus (though how inaccurate it is will 
        depend on how many/large the plateaus are).
        '''
        start = time.time()
        probabilities_cols = self.transition_matrix_csc
        initial_state = initial_state if initial_state is not None else (
            [1.0 / len(self)] * len(self)
        )
        for node, state in zip(self, initial_state):
            node.population = state
        # Get a list of nodes from least fit to most fit
        fitness_ranking = sorted(
            range(len(self)), key = lambda inode: self[inode].fitness
        )
        for inode in fitness_ranking:
            # To find what is going out of node, check its column
            going_to = probabilities_cols[:, [inode]]
            # We have to be more careful with the peaks than I was in the slow 
            # version, to avoid doubling the population on peaks for no reason
            if (
                len(going_to[[inode]].data) == 1 
                and going_to[[inode]].data.item() == 1
            ):
                continue
            for ineighbor, prob in zip(going_to.nonzero()[0], going_to.data):
                self[ineighbor].population += self[inode].population * prob
            self[inode].population = 0
        print(f'Markov equilibrium in {time.time() - start} seconds')
        return [node.population for node in self]

    def single_markov_walker(
        self, start : int = None, logger : types.FunctionType = None
    ) -> list[int, list]:
        '''Simulate a walker on the graph

        If a `logger` is provided, it is expected to be a function that accepts 
        one integer argument as the id of the node that the walker is currently 
        at. Even somewhat complex loggers seem to only have a minimal effect on 
        timings—probably the larger bottelneck is `random.choices`, or possibly 
        the lookups in the sparse matrix
        Given a graph `foo` a logger might be something like:
            walk_logger = lambda x: [
                x, foo[x].fitness, foo[x].amino_sequence, 
                len([
                    neighbor for neighbor in foo[x].neighbors 
                    if foo[neighbor].is_peak
                ]) / len(foo[x].neighbors)
            ]
        Make sure to call `Graph.calc_transition_matrix` before this function.

        Return value: A tuple of the id of the last node in the walk, and the 
        output of the logging function, which is an empty list when 
        `logger = None`. Exceptionally, the id of the last node in the walk may 
        be `None` if the transition matrix contains no transitions for a certain
        node.

        Benchmark: 
            foo = rugged_maps.make_graph(
                graph_type='hamming,9,4', fitness_distribution='random'
            )[0]
            foo.calc_transition_matrix('kimura')
            walk_logger = lambda x: [
                x, foo[x].fitness, foo[x].amino_sequence, 
                len([
                    neighbor for neighbor in foo[x].neighbors 
                    if foo[neighbor].is_peak
                ]) / len(foo[x].neighbors)
            ]
            start = time.time() ; bar = [
                foo.single_markov_walker(logger = walk_logger) 
                for _ in range(1000)
            ] ; time.time() - start
        18.90735101699829 seconds
        '''
        icurrent_node = start if start is not None else random.sample(
            range(len(self)), k = 1
        )[0]
        probabilities_cols = self.transition_matrix_csc
        if logger is None:
            log = []
            while not self[icurrent_node].is_peak:
                going_to = probabilities_cols[:, [icurrent_node]]
                if going_to.nnz != 0:
                    # Cast to a python int, otherwise we end up with numpy ints
                    icurrent_node = int(random.choices(
                        going_to.nonzero()[0], weights = going_to.data
                    )[0])
                else:
                    icurrent_node = None
                    break
        else:
            log = [logger(icurrent_node)]
            while not self[icurrent_node].is_peak:
                going_to = probabilities_cols[:, [icurrent_node]]
                if going_to.nnz != 0:
                    icurrent_node = int(random.choices(
                        going_to.nonzero()[0], weights = going_to.data
                    )[0])
                    log.append(logger(icurrent_node))
                else:
                    icurrent_node = None
                    break
        return (icurrent_node, log)

    def random_walkers(
        self, 
        num_walkers : int = 1000, 
        initial_state : list = None,
        auto_cutoff : bool = False,
        trans_only_low_fit : bool = True
    ) -> list:
        # `weights` vs `cum_weights` makes no difference in a big drawing
        walker_starts = random.choices(
            range(len(self)), cum_weights = initial_state, k = num_walkers
        ) if initial_state is not None else random.choices(
            range(len(self)), k = num_walkers
        )
        walk_logger = (
            self.walk_analyzer.log_step if self.walk_analyzer is not None 
            else lambda inode: [
                inode, self[inode].fitness, self[inode].amino_sequence, (len([
                    neighbor for neighbor in self[inode].neighbors 
                    if self[neighbor].is_peak
                ]) / len([
                    neighbor for neighbor in self[inode].neighbors 
                    if self[inode].fitness < self[neighbor].fitness
                ])) if not self[inode].is_peak else 1.0
            ]
        )
        walks = [
            self.single_markov_walker(start, walk_logger) for start 
            in walker_starts
        ]
        walks = [walk for walk in walks if walk[0] is not None]
        population_fraction = 1 / num_walkers
        for node in self:
            node.population = 0
        for walk in walks:
            self[walk[0]].population += population_fraction
        if isinstance(self.walk_analyzer, walk_analysis.HoCAnalyzer): 
            self.walk_analyzer.reinit(auto = auto_cutoff)
            walks = self.walk_analyzer.relog(
                walks, trans_only_low_fit = trans_only_low_fit
            )

        info = {'walk' : walks}
        if self.walk_analyzer is not None:
            info |= self.walk_analyzer.analyze_walks(walks)
            
        return ([node.population for node in self], info)

    def calc_basins(self):
        '''This version has high RAM usage, but is much faster.

        It will only work when ignoring escapable plateaus
        '''
        sorted_nodes = sorted(
            range(len(self)), key = lambda x: self[x].fitness
        )
        for inode in sorted_nodes:
            self[inode].basin = set([inode]).union(*[
                self[neighbor].basin for neighbor in self[inode].neighbors 
                if self[neighbor].fitness < self[inode].fitness
            ])

    def calc_basins_slow(self, file_name = None):
        '''This version is absurdly slow, but probably won't get OOM'ed.

        Benchmark: folA landscape ~224 minutes, and ~17.8 GB RAM
            Basins written to disk as text: 3.7 GB
            (261333 nodes, 4055 peaks, mean: 128421 nodes/basin)
        '''
        if self.transition_matrix is None:
            self.calc_transition_matrix(evolution_type = 'random')
        trans_mat_csr = self.transition_matrix_csr
        # If there's a file name, attempt a restart
        # Note that this feature might be a little fragile for randomly 
        # generated graphs, or graphs where the sequences are remapped 
        # internally
        if file_name is not None and os.path.isfile(file_name):
            file_handle = open(file_name, 'rt')
            finished_sequences = [
                line.strip().split('\t')[0] for line in file_handle
            ]
            finished_peaks = set(
                self.inode_lookup(seq) for seq in finished_sequences
            )
            peaks_to_do = set(self.peaks) - finished_peaks
            #print(
            #    f'found {len(finished_peaks)} basins, '
            #    f'calculating {len(peaks_to_do)} more!\n'
            #)
            file_handle.close()
            file_handle = open(file_name, 'at')
        elif file_name is not None:
            file_handle = open(file_name, 'wt')
            peaks_to_do = self.peaks
        else:
            peaks_to_do = self.peaks
        for inode in peaks_to_do:
            current_nodes = set([inode])
            current_basin = current_nodes.copy()
            while len(current_nodes) > 0:
                # It would be nice to index the array with a list of rows, but 
                # making a `set` out of it explodes the RAM... but `union` 
                # doesn't
                new_nodes = set([]).union(*[tuple(
                    *trans_mat_csr[jnode, :].coords
                ) for jnode in current_nodes])
                current_nodes = new_nodes - current_basin
                current_basin |= current_nodes
            if file_name is None:
                self[inode].basin = current_basin
            else:
                current_basin = set(int(inode) for inode in current_basin)
                file_handle.write(
                    f'{self.get_sequence(inode)}\t{current_basin}\n'
                )
        if file_name is not None:
            file_handle.close()

    def calc_stats(self, subgraph : list = None) -> str:
        '''Describe some basic statistics of a graph, and its subgraphs'''
        if subgraph is not None:
            graph = Graph([self[inode] for inode in subgraph])
            trans_mat_csr = self.transition_matrix_csr
            graph.peaks = [
                peak for peak in self.peaks if peak in subgraph
            ]
            graph.pits = [
                pit for pit in self.pits if pit in subgraph
            ]
        else:
            graph = self
        num_neighbors = [len(node.neighbors) for node in graph]
        neighbor_dist = [
            num_neighbors.count(val) / len(num_neighbors) 
            for val in range(max(num_neighbors) + 1)
        ]
        avg_neighbors = sum(
            degree * prob for degree, prob in enumerate(neighbor_dist)
        )
        avg_peak_prob = sum(
            prob / (degree + 1) for degree, prob in enumerate(neighbor_dist)
        )
        expected_peaks = avg_peak_prob * len(num_neighbors)
        return (
            f'\n{'G' if subgraph is None else 'Subg'}raph of {len(graph)}'
            f' {graph.dtype[0]}s, of genotypes with length {graph.dtype[1]}\n'
            f'Total peaks:          {len(graph.peaks)}\n'
            f'Total pits:           {len(graph.pits)}\n'
            f'Fitness range:        [{min(node.fitness for node in graph)},'
            f' {max(node.fitness for node in graph)}]\n'
            f'Dist. of node degree: {neighbor_dist}\n'
            f'Avg. neighbors/node:  {avg_neighbors}\n'
            f'Avg. peak prob.:      {avg_peak_prob}\n'
            f'Peaks expected:       {expected_peaks}\n'
            + (
                f'Composed of {len(graph.subgraphs)} subgraphs with lengths '
                f'{[len(val) for val in graph.subgraphs]}\n' 
                if subgraph is None else ''
            )
        )

    def open_log(self, filename : str = None) -> None:
        if self._log_handle is not None:
            self.close_log()
        if filename is not None: 
            self._log_handle = open(filename, 'w')
        else:
            self._log_handle = sys.stdout
        self.write_log = self._log_handle.write
        self.write_log(
            'New run started at\n'
            + ' '.join([time.asctime(), time.tzname[1]]) + '\n'
        )
        self.start_time = time.time()
        self.write_log(f'Random seed this run: {the_seed}\n')

    def log_stats(self) -> None:
        self.write_log(self.calc_stats())
        for subgraph in self.subgraphs:
            self.write_log(self.calc_stats(subgraph)) 

    def close_log(self) -> None:
        self.write_log(
            'Run finished at\n'
            + ' '.join([time.asctime(), time.tzname[1]]) + '\n'
            f'Total time: {time.time() - self.start_time} seconds\n'
        )
        self._log_handle = None
        self.write_log = lambda x: (self.open_log(), self.write_log(x))

def rds_to_csv(file_name_in : str, file_name_out : str = None) -> list[list]:
    '''Convert a dataframe stored as an RDS file to a CSV file.

    Also returns the data as a list of lists.
    '''
    file_name_out = file_name_out if file_name_out is not None else (
        file_name_in[:-4] + '.csv'
    )
    # RDS -> Pandas dataframe -> CSV
    data = pyreadr.read_r(file_name_in)[None]
    labels = [val for val in data.keys()]
    columns = [list(data[label]) for label in labels]
    rows = list(zip(*columns))
    rows.insert(0, labels)
    open(file_name_out, 'w').write(
        '\n'.join([','.join([str(val) for val in row]) for row in rows])
    )
    return rows

def read_csv(file_name : str, dtypes : list[type] = None) -> list[list]:
    '''Parse a CSV to return a list of lists.

    Optionally provide a list of types corresponding to the type of each element
    in a row, otherwise they will all default to strings.
    '''
    if dtypes is not None:
        data = [
            [dtype(val) for val, dtype in zip(line.strip().split(','), dtypes)] 
            for line in open(file_name)
        ]
    else:
        data = [line.strip().split(',') for line in open(file_name)]
    return data

def read_tsv(file_name : str, dtypes : list[type] = None) -> list[list]:
    '''Parse a TSV to return a list of lists.

    Optionally provide a list of types corresponding to the type of each element
    in a row, otherwise they will all default to strings.
    '''
    if dtypes is not None:
        data = [
            [dtype(val) for val, dtype in zip(line.strip().split('\t'), dtypes)] 
            for line in open(file_name)
        ]
    else:
        data = [line.strip().split('\t') for line in open(file_name)]
    return data

def genetic_distance(seq1 : str, seq2 : str) -> int:
    '''Calculate the number of point mutations separating `seq1` into `seq2`.

    Actually, that's not what it does. This is optimized just to check if two 
    sequences are exactly 1 point mutation different.
    '''
    #assert len(seq1) == len(seq2)
    #return sum((base1 != base2 for base1, base2 in zip(seq1, seq2)))
    distance = 0
    for base1, base2 in zip(seq1, seq2):
        if base1 != base2:
            distance += 1
            if distance == 2:
                break
    return distance

def int_to_base_n(number : int, base : int) -> str:
    '''Convert an integer to a base-n enconded string

    `base` can't be larger than 94. `number = -0` returns `'0'`.
    '''
    if number == 0:
        output = '0'
    else:
        if number < 0:
            is_negative = True
            number = -number
        else:
            is_negative = False
        output = ''
        while number:
            number, remainder = divmod(number, base)
            output += base_n_digits[remainder]
        if is_negative:
            output += '-'
        output = output[::-1]
    return output

def base_n_increment_in_place(number : list[int], base = 2) -> list[int]:
    '''Increment a base-n integer by 1

    The input `number` is a `list` of `int`s, a little-endian vector of digits.
    '''
    number[0] += 1
    for idigit in range(len(number) - 1):
        if number[idigit] >= base:
            number[idigit] -= base
            number[idigit + 1] += 1
        else:
            break
    else:
        if number[-1] >= base:
            raise OverflowError(
                f'This custom base-{base} number has overflowed: {number}'
            )
    #return number

def calc_hamming_edges(d, q):
    '''Optimized, with output (probably) compatible with the igraph package'''
    assert d >= 0
    assert q >= 1
    num_vertices = q ** d
    vertex_degree = d * (q - 1)
    num_edges = (num_vertices * vertex_degree) // 2
    edge_list = [() for _ in range(num_edges)]
    iedge = 0
    base_q_vertex = [0] * (d + 1)
    base_q_vertex[0] = -1
    # Pregenerate some values for later
    base_q_digit_values = [[0] * (q - 1) for _ in range(d)]
    temp_val1 = 1
    temp_val2 = temp_val1
    for iplace in range(d):
        for idigit in range(q - 1):
            base_q_digit_values[iplace][idigit] = temp_val2
            temp_val2 += temp_val1
        temp_val1 *= q
    # Iterate over every vertex
    for ivertex in range(num_vertices):
        base_n_increment_in_place(base_q_vertex, base = q)
        # Iterate over all `d` of the base-`q` digits of `ivertex`
        for iplace in range(d):
            digit_value = base_q_vertex[iplace]
            # Slower alternative for if the vertex were a plain integer: 
            #digit_value = (ivertex // (q ** idigit)) % q
            # Iterate over values of the digit greater than the current value
            for digit_increase in range(q - digit_value - 1):
                edge_list[iedge] = (
                    ivertex, 
                    ivertex + base_q_digit_values[iplace][digit_increase]
                    #ivertex + digit_increase * q ** iplace
                )
                iedge += 1
    return num_vertices, edge_list

def sequence_to_int(sequence : str, length = 'fixed'):
    '''Calculate the base 4 equivalent of a DNA (or RNA) sequence.

    `length` can be set to either 'fixed' or 'any', where 'fixed' will generate 
    unique numbers from sequences of a given length, while setting 'any' ensures
    that sequences of any length return unique numbers.
    '''
    value = sum(
        (4 ** ibase) * base_to_int[base] for ibase, base in enumerate(sequence)
    )
    if length != 'fixed':
        value += sum(4 ** ibase for ibase in range(1, len(sequence)))
    return value

def amino_sequence_to_int(sequence : str, length = 'fixed'):
    '''Calculate the base 21 equivalent of an amino acid sequence.

    `length` can be set to either 'fixed' or 'any', where 'fixed' will generate 
    unique numbers from sequences of a given length, while setting 'any' ensures
    that sequences of any length return unique numbers.
    '''
    value = sum(
        (21 ** ibase) * amino_to_int_1[base] 
        for ibase, base in enumerate(sequence)
    )
    if length != 'fixed':
        value += sum(21 ** ibase for ibase in range(1, len(sequence)))
    return value

def base_n_sequence_to_int(sequence : str, base : int = 2, length = 'fixed'):
    '''Calculate the base n equivalent of a base-n string.

    `length` can be set to either 'fixed' or 'any', where 'fixed' will generate 
    unique numbers from sequences of a given length, while setting 'any' ensures
    that sequences of any length return unique numbers.

    There's no check to prevent digits with values larger than should be allowed
    by `base`, but this would be bad to do because the strings would not produce
    unique numebrs.
    '''
    value = sum(
        (base ** ilocus) * base_n_digits_to_int[locus] 
        for ilocus, locus in enumerate(sequence)
    )
    if length != 'fixed':
        value += sum(base ** ilocus for ilocus in range(1, len(sequence)))
    return value
    
def histogram(
    data : list or list[tuple], dimension : int = 1, **keyword_args
) -> tuple[list or numpy.ndarray, list, list]:
    '''Calculate a histogram of `data`, which goes to 0 at both ends.

    It's just a wrapper for `matplotlib.pyplot.hist`.
    `data` is a list of the values, either numbers or tuples for a 2D 
    historgram.
    `bins` should probably be either an int or a list, but it's passed directly 
    into matplotlib.pyplot.hist, so any valid argument to that function is fine.
    `dimension` is either 1 or 2.
    1D histograms are padded with one extra empty bin on both sides.
    The return value is a tuple of the histogram, the bins, and the x values 
    associated with the histogram, i.e., the midponts of the bins.
    '''
    if 'cumulative' in keyword_args.keys():
        hist_data, _, _ = matplotlib.pyplot.hist(data, **keyword_args)
        matplotlib.pyplot.close()
    else:
        hist_data, _ = numpy.histogram(data, **keyword_args)
    return hist_data

def calc_transition_prob(
    fitness1, fitness2, population = 1e8, version = 'papkou_like_greenbury'
):
    '''Calculate the probability of a mutation from initial and final fitnesses.

    For now this is just 'Kimura’s fixation probability', with some variations 
    due to some disagreement in different sources.
    The mutation frequency is assumed to be `1 / population`.
    '''
    if version == 'papkou_like_greenbury':
        # Adds an extra expoential to Papkou (2023), more like in Greenbury
        fitness_difference = math.exp(fitness2 - fitness1) - 1
    elif version == 'greenbury':
        # In Greenbury (2022)
        fitness_difference = (fitness2 / fitness1) - 1
    elif version == 'papkou':
        # In Papkou (2023)
        fitness_difference = fitness2 - fitness1 
    temp_val = math.exp(-2 * fitness_difference)
    probability = (1 - temp_val) / (1 - temp_val ** population)
    # To account for mutation frequency:
    #probability = (1 - temp_val ** (population * mutation_frequency)) / (
    #    1 - temp_val ** population
    #)
    return probability

def house_of_cards(
    alphabet_size : int, 
    sequence_length : int, 
    prob_dist_function : list[2, list],
    num_nodes : int = None,
    missing_nodes : str = 'random'
) -> dict:
    '''Calculate the expectation value for some properties of a HOC graph.

    The probability distribution should be a list of lists,
        `[[x], [P(x)]]`,
    the `x` values should be evenly spaced, `P(x)` should integrate to 1, and it
    should have `P(x)=0` at both ends, but only the evenly-spaced x values is 
    assumed.
    When interested in a graph with fewer nodes than the maximum for a given 
    alphabet/sequence length, `num_nodes` sets the number of nodes in the 
    reduced graph, and `missing_nodes` specifies which nodes have been removed: 
    either 'random', 'low_fitness', 'high_fitness', or 'no_neighbors_bottom'.
    '''
    max_nodes = alphabet_size ** sequence_length
    num_nodes = num_nodes if num_nodes != max_nodes else None

    pdf_x_step = prob_dist_function[0][1] - prob_dist_function[0][0]
    if prob_dist_function[1][0] != 0:
        prob_dist_function = [
            [prob_dist_function[0][0] - pdf_x_step] + prob_dist_function[0], 
            [0] + prob_dist_function[1]
        ]
    if prob_dist_function[1][-1] != 0:
        prob_dist_function = [
            prob_dist_function[0] + [prob_dist_function[0][-1] + pdf_x_step], 
            prob_dist_function[1] + [0]
        ]
    # Trapezoidal integration
    #pdf_norm = sum(pdf_x_step * (p1 + p2) / 2 for p1, p2 in zip(prob_dist_function[1][:-1], prob_dist_function[1][1:]))
    # Equivalent
    pdf_norm = pdf_x_step * sum(prob_dist_function[1])
    prob_dist_function[1] = [val / pdf_norm for val in prob_dist_function[1]]
    cuml_dist_function = [0.0] + [
        sum(prob_dist_function[1][:iprob] + prob_dist_function[1][1:iprob + 1]) 
        * pdf_x_step / 2 for iprob in range(1, len(prob_dist_function[1]))
    ]

    # This gives the number of pits or peaks observed, but this includes ones 
    # created by a cutoff at the bottom or top of the graph
    num_neighbors = sequence_length * (alphabet_size - 1) #* graph_fraction
    if num_nodes is not None:
        graph_fraction = num_nodes / max_nodes
        if missing_nodes.lower() == 'low_fitness':
            graph_cutoff = 1 - graph_fraction
            cuml_dist_function = [
                val * graph_fraction + graph_cutoff 
                for val in cuml_dist_function
            ]
            cdf_m1 = [1 - val for val in cuml_dist_function]
            pits_prob_dist_func = [
                p * (graph_cutoff ** num_neighbors) * sum(
                    (cm1 / graph_cutoff) ** neighbors * (neighbors + 1) 
                    * math.comb(num_neighbors, neighbors) 
                    for neighbors in range(num_neighbors + 1)
                )
                for p, cm1 in zip(prob_dist_function[1], cdf_m1)
            ]
            peaks_prob_dist_func = [
                p * c ** num_neighbors
                for p, c in zip(prob_dist_function[1], cuml_dist_function)
            ]
        elif missing_nodes.lower() == 'high_fitness':
            cuml_dist_function = [
                val * graph_fraction for val in cuml_dist_function
            ]
            cdf_m1 = [1 - val for val in cuml_dist_function]
            pits_prob_dist_func = [
                p * cm1 ** num_neighbors
                for p, cm1 in zip(prob_dist_function[1], cdf_m1)
            ]
            peaks_prob_dist_func = [
                p * graph_fraction ** num_neighbors * sum(
                    (c / graph_fraction) ** neighbors * (neighbors + 1)
                    * math.comb(num_neighbors, neighbors) 
                    for neighbors in range(num_neighbors + 1)
                )
                for p, c in zip(prob_dist_function[1], cuml_dist_function)
            ]
        elif missing_nodes.lower() == 'random':
            graph_cutoff = 1 - graph_fraction
            cdf_m1 = [1 - val for val in cuml_dist_function]
            pits_prob_dist_func = [
                p * graph_cutoff ** num_neighbors * sum(
                    (cm1 * graph_fraction / graph_cutoff) ** neighbors 
                    * math.comb(num_neighbors, neighbors) * (neighbors + 1)
                    for neighbors in range(num_neighbors + 1)
                )
                for p, cm1 in zip(prob_dist_function[1], cdf_m1)
            ]
            peaks_prob_dist_func = [
                p * graph_cutoff ** num_neighbors * sum(
                    (c * graph_fraction / graph_cutoff) ** neighbors 
                    * math.comb(num_neighbors, neighbors) * (neighbors + 1)
                    for neighbors in range(num_neighbors + 1)
                )
                for p, c in zip(prob_dist_function[1], cuml_dist_function)
            ]
        elif missing_nodes.lower() == 'low_fitness_neighbors':
            graph_cutoff = (1 - graph_fraction) ** (1 / (num_neighbors + 1))
            eff_graph_cutoff = (
                num_nodes - (1 - graph_cutoff) * max_nodes
            ) / num_nodes
            graph_fraction = 1 - graph_cutoff
            eff_graph_fraction = 1 - eff_graph_cutoff
            cuml_dist_function = [
                val / eff_graph_cutoff * graph_cutoff 
                if val < eff_graph_cutoff 
                else 1 - ((1 - val) / eff_graph_fraction * graph_fraction) 
                for val in cuml_dist_function
            ]
            cdf_m1 = [1 - val for val in cuml_dist_function]
            pits_prob_dist_func = [
                p * (1 - graph_cutoff ** num_neighbors) * (
                    cm1 ** (num_neighbors - 1) 
                    * (1 - graph_cutoff ** (num_neighbors - 1))
                ) if (1 - cm1) < graph_cutoff else p * cm1 ** num_neighbors
                for p, cm1 in zip(prob_dist_function[1], cdf_m1)
            ]
            peaks_prob_dist_func = [
                0 if c < graph_cutoff else p * c ** num_neighbors 
                for p, c in zip(prob_dist_function[1], cuml_dist_function)
            ]
        else:
            raise ValueError(
                f'{missing_nodes} is not a valid value for argument '
                '`missing_nodes` of house_of_cards'
            )
    else:
        cdf_m1 = [1 - val for val in cuml_dist_function]
        pits_prob_dist_func = [
            p * cm1 ** num_neighbors
            for p, cm1 in zip(prob_dist_function[1], cdf_m1)
        ]
        peaks_prob_dist_func = [
            p * c ** num_neighbors
            for p, c in zip(prob_dist_function[1], cuml_dist_function)
        ]
    # I didn't think it would be a problem, but these distributions don't come 
    # out normalized
    pits_norm = pdf_x_step * sum(pits_prob_dist_func)
    pits_prob_dist_func = [val / pits_norm for val in pits_prob_dist_func]
    peaks_norm = pdf_x_step * sum(peaks_prob_dist_func)
    peaks_prob_dist_func = [val / peaks_norm for val in peaks_prob_dist_func]
    # Outer product
    self_joint_dist = (
        numpy.array([prob_dist_function[1]]).T 
        @ numpy.array([prob_dist_function[1]])
    )
    # Technically a convolution, but since we already have the joint dist, why 
    # bother?
    self_difference_dist = list(zip(*[
        [idiag * pdf_x_step, self_joint_dist.diagonal(idiag).sum() * pdf_x_step]
        for idiag in
        range(-1 * len(prob_dist_function[0]), len(prob_dist_function[0]) + 1)
    ]))
    return {
        'neighbors' : num_neighbors, 
        'pdf_xs' : prob_dist_function[0],
        'pits_pdf' : pits_prob_dist_func, 'peaks_pdf' : peaks_prob_dist_func, 
        'null_joint_dist' : self_joint_dist, 
        'self_difference_dist' : self_difference_dist
    }

def noras_rough_mount_fuji_greedy_transition_matrix(
    length : int, probability : float
) -> list:
    assert length >= 1
    assert 0 <= probability <= 1
    qrobability = 1 - probability
    transition_matrix = [[0] * (length * 3 + 3) for _ in range(length * 3 + 3)]
    transition_matrix[5][0] = 1
    transition_matrix[1][1] = 1
    for i in range(2, len(transition_matrix), 3):
        transition_matrix[i][i] = 1
    transition_matrix[0][3] = qrobability * (1 - qrobability ** length)
    transition_matrix[1][3] = qrobability ** (length + 1)
    transition_matrix[2][3] = probability
    transition_matrix[2][4] = 1
    for dist in range(2, length + 1):
        i = dist * 3
        q_to_the_d = qrobability ** dist
        q_to_the_dm1 = qrobability ** (dist - 1)
        transition_matrix[i - 3][i] = q_to_the_d
        transition_matrix[i - 2][i] = (1 - q_to_the_d) * (1 - q_to_the_dm1)
        transition_matrix[i - 2][i + 1] = 1 - q_to_the_dm1
        transition_matrix[i - 1][i] = (1 - q_to_the_d) * q_to_the_dm1
        transition_matrix[i - 1][i + 1] = q_to_the_dm1
    return transition_matrix

def noras_rough_mount_fuji_greedy_transition_matrix_sparse(
    length : int, probability : float
) -> list:
    assert length >= 1
    assert 0 <= probability <= 1
    probability = numpy.float128(probability)
    qrobability = 1 - probability
    probabilities = [1] * (length + 4) + [
        qrobability * (1 - qrobability ** length), 
        qrobability ** (length + 1), 
        probability
    ] + [0] * ((length - 1) * 5)
    coordinates = [
        (5, 0), (1, 1), (2, 4), *[(i, i) for i in range(2, length * 3 + 3, 3)],
        (0, 3), (1, 3), (2, 3)
    ] + [0] * ((length - 1) * 5)
    for dist in range(2, length + 1):
        i = dist * 3
        slice_start = length + 7 + (dist - 2) * 5
        q_to_the_d = qrobability ** dist
        q_to_the_dm1 = qrobability ** (dist - 1)
        probabilities[slice_start:slice_start + 5] = [
            q_to_the_d, 
            (1 - q_to_the_d) * (1 - q_to_the_dm1), 
            1 - q_to_the_dm1, 
            (1 - q_to_the_d) * q_to_the_dm1, 
            q_to_the_dm1
        ]
        coordinates[slice_start:slice_start + 5] = [
            (i - 3, i), (i - 2, i), (i - 2, i + 1), (i - 1, i), (i - 1, i + 1)
        ]
    probabilities = numpy.array(probabilities)
    coordinates = numpy.array(
        list(zip(*coordinates)) 
    )
    transition_matrix = scipy.sparse.coo_array(
        (probabilities, coordinates), 
        shape = ((length + 1) * 3,) * 2
    )
    return transition_matrix

def noras_rough_mount_fuji_inital_state(
    length : int, probability : float
) -> list:
    assert length >= 1
    assert 0 <= probability <= 1
    probability = numpy.float128(probability)
    qrobability = 1 - probability
    norm = numpy.float128(2) ** (-1 * length)
    initial_state = [
        qrobability * (1 - qrobability ** length) * norm, 
        qrobability ** (length + 1) * norm, 
        probability * norm
    ]
    for dist in range(1, length + 1):
        binom = math.comb(length, dist) * norm
        initial_state += [
            qrobability * binom, 
            probability * (1 - qrobability ** dist) * binom, 
            probability * qrobability ** dist * binom
        ]
    return initial_state

def noras_rough_mount_fuji_final_state(
    length : int, probability : float
) -> list:
    assert length >= 1
    assert 0 <= probability <= 1
    probability = numpy.float128(probability)
    initial_state = noras_rough_mount_fuji_inital_state(length, probability)
    transition_matrix = noras_rough_mount_fuji_greedy_transition_matrix_sparse(
        length, probability
    )
    #equilibrium_transitions = scipy.sparse.linalg.matrix_power(
    #    transition_matrix, length + 1
    #)
    #final_state = list(equilibrium_transitions @ initial_state)
    final_state = initial_state.copy()
    for _ in range(length + 1):
        final_state = transition_matrix @ final_state
    final_state = [final_state[1]] + final_state[2::3]
    return final_state

def noras_rough_mount_fuji_greedy_final_state_fast(
    length : int, probability : float
) -> list:
    '''
    Find the final distribution of population evolving on an NRMF landscape.

    This function will exactly propigate a population on an NRMF fitness 
    landscape defined by the input parameters, and find the final distribution 
    trapped at each shell of the landscape (One would have to divide by the 
    expected number of peaks in a shell to find how much was trapped at an 
    specific peak). The algorithm is essentially just an optimized 
    implimentation of an absorbing Markov chain, with some numerical tricks. 
    Quad precision floats are used, but even so a lot of values will underflow 
    as the length becomes very long; however, it should be safe to assume that 
    any value which has underflowed is truly inconsequential, and there 
    shouldn't be any kind of catastrophic acumulation of errors. The runtime 
    should scale linearly with the `length`, and even `length=1e6` should only 
    take ~4 seconds (The limiting factor is more likely to become the RAM).
    '''
    assert length >= 1
    assert 0 <= probability <= 1
    probability = numpy.float128(probability)
    prob_m_1 = 1 - probability
    prob_m_1_exp = [1] * (length + 2)
    for i in range(1, len(prob_m_1_exp)):
        prob_m_1_exp[i] = prob_m_1_exp[i - 1] * prob_m_1

    # This is just calculating normalized binomial coefficients in a more 
    # efficient way than `math.comb`
    # In the end, the speedup from precalculating all of these might not be that
    # large, especially considering how large the RAM usage becomes.
    precomp_val = (
        -1 * numpy.log(numpy.float128(length + 1)) 
        - length * numpy.log(numpy.float128(2))
    )
    # For some reason, `scipy.special.betaln` doesn't support quad precision, 
    # but I think it doesn't make a huge difference, since the main benefit to 
    # quad precision is the larger range to the exponents
    binoms = [
        numpy.exp(precomp_val - scipy.special.betaln(length - i + 1, i + 1)) 
        for i in range(length + 1)
    ]

    peak_pops = [0] * (length + 1)
    current_pops = [
        prob_m_1 * binoms[length], 
        probability * (1 - prob_m_1_exp[length]) * binoms[length]
    ]
    peak_pops[length] = probability * prob_m_1_exp[length] * binoms[length]
    for dist in range(length - 1, 0, -1):
        current_pops[1] += current_pops[0] * (1 - prob_m_1_exp[dist + 1])
        current_pops[0] = (
            current_pops[0] * prob_m_1_exp[dist + 1] + prob_m_1 * binoms[dist]
        )
        current_pops[1] += probability * binoms[dist]
        peak_pops[dist] = current_pops[1] * prob_m_1_exp[dist]
        current_pops[1] *= (1 - prob_m_1_exp[dist])
    current_pops[1] += current_pops[0] * (1 - prob_m_1_exp[1])
    current_pops[0] += binoms[0]
    peak_pops[0] = current_pops[0] * prob_m_1_exp[length + 1]
    peak_pops[1] += current_pops[0] * prob_m_1 * (1 - prob_m_1_exp[length])
    return [current_pops[1]] + peak_pops

def noras_rough_mount_fuji(length : int, probability : float) -> tuple:
    prob_m_1 = 1 - probability
    prob_m_1_exp = [1] * (length + 2)
    for i in range(1, len(prob_m_1_exp)):
        prob_m_1_exp[i] = prob_m_1_exp[i - 1] * prob_m_1
    precomp_val = (
        -1 * numpy.log(numpy.float128(length + 1)) 
        - length * numpy.log(numpy.float128(2))
    )
    binoms = [
        numpy.exp(precomp_val - scipy.special.betaln(length - i + 1, i + 1)) 
        for i in range(length + 1)
    ]
    fitness_x_values = [0] + [
        val / 2 for val in range(2, length * 2 + 1)
    ] + [length + 0.5, length + 1.5]
    genotype_distribution = [
        binoms[int(val)] * (prob_m_1) if val % 1 == 0 
        else (binoms[int(val - 1.5)] * probability) for val in fitness_x_values
    ] 
    total_peaks = (
        prob_m_1 ** (length + 1) + probability * (2 - probability) ** length
    )
    peak_density_distribution = [
        0 if val % 1 == 0 else (
            probability * prob_m_1_exp[int(length - (val - 1.5))] / total_peaks
        ) for val in fitness_x_values
    ]
    peak_density_distribution[-3] = prob_m_1_exp[length + 1] / total_peaks
    peak_distribution = [
        val if x % 1 == 0 else (val * binoms[int(x - 1.5)] * 2 ** length) 
        for x, val in zip(fitness_x_values, peak_density_distribution)
    ]
    peak_distribution[-3] *= binoms[-1] * 2 ** length
    final_population_distribution_iter = iter(
        noras_rough_mount_fuji_greedy_final_state_fast(
            length, probability
        )[::-1]
    )
    final_population_distribution = [
        0 if val % 1 == 0 else next(final_population_distribution_iter) 
        for val in fitness_x_values[:-3]
    ]
    temp_val = [
        next(final_population_distribution_iter), 
        next(final_population_distribution_iter), 
        next(final_population_distribution_iter)
    ]
    final_population_distribution += [temp_val[1], temp_val[0], temp_val[2]]
    final_peak_populaiton_distribution = [
        pop / peak if peak != 0 else 0 for pop, peak 
        in zip(final_population_distribution, peak_density_distribution)
    ]
    # This stuff with the basins is really just a guesstimation
    basin_factor = max(probability, 1 - probability)
    basin_size_distribution = [
        0 if val % 1 == 0 else (probability ** (length - val + 1.5) * prob_m_1) 
        for val in fitness_x_values
    ]
    return {
        'pdf_xs' : fitness_x_values, 'genotypes' : genotype_distribution, 
        'peak_density' : peak_density_distribution, 
        'peaks_pdf' : peak_distribution,
        'population' : final_population_distribution, 
        'population_density' : final_peak_populaiton_distribution,
        'basins' : basin_size_distribution
    }

def make_graph(
    graph_type : str = 'point_mutations,papkou_2023_folA.csv',
    fitness_distribution : str or types.NoneType = None,
    fitness_cutoff : float = None, 
    eta_cutoff : float = None, 
    cutoff_type : str = 'low_fitness', 
    permute : bool or str = False
) -> list:
    '''Make a graph based on the input parameters.

    Build a graph: either reading it from a file, or picking one of a few 
    different common structures and assigning fitnesses according to one of the 
    distributions in the `random` module.
    '''
    match graph_type.strip().split(','):
        case [
            'point_mutations', 'papkou_2023_folA.csv' as file_name
        ] as graph_type:
            input_data = read_csv(file_name)[1:]
            genotype_graph = Graph([
                GraphNode(row[0], row[5] + row[7] + row[9], float(row[1])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'folA'
            )
        case [
            'point_mutations', 'fitness_data_wt.rds' as file_name
        ] as graph_type:
            input_data = rds_to_csv(
                file_name, file_name.lower().strip('.rds') + '.csv'
            )
            genotype_graph = Graph([
                GraphNode(row[0], row[5] + row[7] + row[9], float(row[1])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'folA'
            )
        case [
            'point_mutations', 'TetR_landscape.txt' as file_name
        ] as graph_type:
            input_data = read_tsv(file_name)[1:]
            genotype_graph = Graph([
                GraphNode(row[0][4:8] + row[0][11:15], None, float(row[1])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'TetR'
            )
        case ['point_mutations', 'wu_2016_1.csv' as file_name] as graph_type:
            input_data = read_csv(file_name)[1:]
            genotype_graph = Graph([
                GraphNode('', row[0], float(row[4])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'GB1'
            )
        case ['point_mutations', 'domingo_2018.tsv' as file_name] as graph_type:
            input_data = read_tsv(file_name)[6:]
            genotype_graph = Graph([GraphNode(
                row[0][0:2] + row[0][5] + row[0][26] + row[0][42] + row[0][45] 
                + row[0][65] + row[0][68:71], 
                None, 
                float(row[14])
            ) for row in input_data])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'tRNA'
            )
        case ['point_mutations', 'poelwijk_2019_FP.tsv' as file_name] as graph_type:
            input_data = read_tsv(file_name)
            genotype_graph = Graph([
                GraphNode(row[0].replace(',', ''), None, float(row[1])
            ) for row in input_data])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'FP'
            )
        case ['point_mutations', 'CR9114-B.tsv' as file_name] as graph_type:
            input_data = read_tsv(file_name)
            genotype_graph = Graph([
                GraphNode(row[0].replace(',', ''), None, float(row[1])
            ) for row in input_data])
            #genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
            #    genotype_graph, 'TetR'
            #)
        case ['point_mutations', 'ParD3-ParE3.tsv' as file_name] as graph_type:
            input_data = read_tsv(file_name)
            genotype_graph = Graph([GraphNode('', ''.join(
                amino_list_1[int(base) - 1] for base in row[0].split(',')
            ), float(row[1])) for row in input_data])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'ParD3-ParE3'
            )
        case ['point_mutations', (
            'kuo_2020_SD_FL_arti.csv' | 'kuo_2020_SD_FL_arti_Y.csv' 
            | 'kuo_2020_SD_FL_dmsC.csv' | 'kuo_2020_SD_FL_fepB.csv'
        ) as file_name] as graph_type:
            input_data = read_csv(file_name)
            genotype_graph = Graph([
                GraphNode(row[0], None, float(row[1])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'SD-arti'
            )
        case [
            'point_mutations', 'moulana_2022_RBD_ACE2.tsv' as file_name
        ] as graph_type:
            input_data = read_tsv(file_name)
            genotype_graph = Graph([
                GraphNode(row[0], None, float(row[1])) 
                for row in input_data
            ])
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'RBD-ACE2'
            )
        case (
            ['point_mutations', file_name] as graph_type 
        ) if file_name.lower().endswith('.rds'):
            warnings.warn(
                f'Reading data from {file_name} using the default method for '
                '.rds files (probably a bad idea)', RuntimeWarning, 
                stacklevel = 2
            )
            input_data = rds_to_csv(
                file_name, file_name.lower().strip('.rds') + '.csv'
            )
            genotype_graph = Graph([
                GraphNode(row[0], row[5] + row[7] + row[9], float(row[1])) 
                for row in input_data
            ])
        case (
            ['point_mutations', file_name] as graph_type 
        ) if file_name.lower().endswith('.csv'):
            warnings.warn(
                f'Reading data from {file_name} using the default method for '
                '.csv files (probably a bad idea)', RuntimeWarning, 
                stacklevel = 2
            )
            input_data = read_csv(file_name)[1:]
            genotype_graph = Graph([
                GraphNode(row[0], row[5] + row[7] + row[9], float(row[1])) 
                for row in input_data
            ])
        case (
            ['point_mutations', file_name] as graph_type 
        ) if file_name.lower().endswith('.tsv'):
            warnings.warn(
                f'Reading data from {file_name} using the default method for '
                '.tsv files (probably a bad idea)', RuntimeWarning, 
                stacklevel = 2
            )
            input_data = read_tsv(file_name)[1:]
            genotype_graph = Graph([
                GraphNode(row[0].replace(',', ''), None, float(row[1])
            ) for row in input_data])
        case ['hamming', dim, alphabet]:
            graph_type = ['hamming', int(dim), int(alphabet)]
            genotype_graph = Graph(
                [GraphNode() for _ in range(graph_type[2] ** graph_type[1])]
            )
            genotype_graph.dtype[1] = graph_type[1]
            genotype_graph.dtype[2] = graph_type[2]
        case ['full', num_nodes]:
            graph_type = ['full', int(num_nodes)]
            genotype_graph = Graph([GraphNode() for _ in range(graph_type[1])])
            genotype_graph.dtype[1] = 1
            genotype_graph.dtype[1] = graph_type[1]
        case ['random' | 'erdos_renyi', num_nodes, edge_prob]:
            graph_type = ['random', int(num_nodes), float(degree)]
            genotype_graph = Graph([GraphNode() for _ in range(graph_type[1])])
            genotype_graph.dtype[1] = int(graph_type[1] * edge_prob)
        case ['regular' | 'random_regular', num_nodes, degree]:
            graph_type = ['regular', int(num_nodes), int(degree)]
            genotype_graph = Graph([GraphNode() for _ in range(graph_type[1])])
            genotype_graph.dtype[1] = graph_type[2]
        case [
            'power_law' | 'scale_free' | 'barabasi' | 'barabasi_albert', 
            num_nodes, power
        ]:
            graph_type = ['power_law', int(num_nodes), float(power)]
            genotype_graph = Graph([GraphNode() for _ in range(graph_type[1])])
            ...
        case _:
            raise ValueError(
                f'"{graph_type}" is not a valid graph type!\n'
                 'Choose one of "point_mutations", "hamming", "full", "random",'
                 ' "regular", or "power_law"'
            )
    match (
        fitness_distribution.strip().split(',') 
        if fitness_distribution is not None else fitness_distribution
    ):
        case None:
            # In this case it's assumed that fitnesses were read in with the 
            # sequences for a point-mutation graph
            pass
        case ['read', file_name]:
            ...
        case ['rmf', *args]:
            genotype_graph.calc_edges(*graph_type)
            slope = float(args[0])
            dist_name = args[1]
            fitness_sampler = lambda : random.__dict__[dist_name](*[
                float(arg) for arg in args[2:]
            ])
            genotype_graph.random_part = [
                fitness_sampler() for _ in range(len(genotype_graph))
            ]
            genotype_graph[0].fitness = genotype_graph.random_part[0]
            genotype_graph[0].distance = 0
            for node, rand in zip(
                genotype_graph[1:], genotype_graph.random_part[1:]
            ):
                # Next line assumes that the list of neighbors is sorted... 
                # which it is, for now at least
                node.distance = genotype_graph[node.neighbors[0]].distance + 1
                node.fitness = rand - slope * node.distance
            genotype_graph.nodes_by_distance = [[
                inode for inode, node in enumerate(genotype_graph) 
                if node.distance == dist
            ] for dist in range(graph_type[1] + 1)]
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'RMF'
            )
        case ['noras_rmf', *args]:
            genotype_graph.calc_edges(*graph_type)
            slope = float(args[0])
            genotype_graph.random_part = random.choices(
                (float(args[1]), float(args[2])), 
                (float(args[3]), float(args[4])),
                k = len(genotype_graph)
            )
            genotype_graph[0].fitness = genotype_graph.random_part[0]
            genotype_graph[0].distance = 0
            for node, rand in zip(
                genotype_graph[1:], genotype_graph.random_part[1:]
            ):
                # Next line assumes that the list of neighbors is sorted... 
                # which it is, for now at least
                node.distance = genotype_graph[node.neighbors[0]].distance + 1
                node.fitness = rand - slope * node.distance
            if graph_type[0] == 'hamming':
                for node in genotype_graph:
                    node.fitness += graph_type[1]
            genotype_graph.nodes_by_distance = [[
                inode for inode, node in enumerate(genotype_graph) 
                if node.distance == dist
            ] for dist in range(graph_type[1] + 1)]
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'RMF'
            )
        case ['noras_other_rmf', *args]:
            genotype_graph.calc_edges(*graph_type)
            slope = float(args[1])
            rough_section_cutoff = int(args[0])
            rough_section_size = round(scipy.special.betainc(
                graph_type[1] - rough_section_cutoff, 
                rough_section_cutoff + 1, 0.5
            ) * 2 ** graph_type[1])
            genotype_graph.random_part = random.choices(
                (float(args[2]), float(args[3])), 
                (float(args[4]), float(args[5])),
                k = rough_section_size
            )
            random_iter = iter(genotype_graph.random_part)
            genotype_graph[0].fitness = next(random_iter)
            genotype_graph[0].distance = 0
            for node in genotype_graph[1:]:
                # Next line assumes that the list of neighbors is sorted... 
                # which it is, for now at least
                node.distance = genotype_graph[node.neighbors[0]].distance + 1
                node.fitness = (
                    next(random_iter) if node.distance <= rough_section_cutoff 
                    else 0
                ) - slope * node.distance
            if graph_type[0] == 'hamming':
                for node in genotype_graph:
                    node.fitness += graph_type[1]
            genotype_graph.nodes_by_distance = [[
                inode for inode, node in enumerate(genotype_graph) 
                if node.distance == dist
            ] for dist in range(graph_type[1] + 1)]
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'RMF'
            )
        case [dist_name, *args]:
            genotype_graph.calc_edges(*graph_type)
            fitness_sampler = lambda : random.__dict__[dist_name](*[
                float(arg) for arg in args
            ])
            for node in genotype_graph:
                node.fitness = fitness_sampler()
            genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
                genotype_graph, 'HoC'
            )

    # If there's still no walk analyzer, put the HoC:
    if genotype_graph.walk_analyzer is None or permute != False:
        genotype_graph.walk_analyzer = walk_analysis.WalkAnalyzer(
            genotype_graph, 'HoC'
        )
            
    # Make sure the cutoff values are consistent
    fitnesses = sorted(node.fitness for node in genotype_graph)
    if fitness_cutoff is not None:
        eta_cutoff = numpy.searchsorted(
            fitnesses, fitness_cutoff
        ) / len(fitnesses)
    elif eta_cutoff is not None:
        eta_cutoff = min(max(eta_cutoff, 0), 1)
        fitness_cutoff = fitnesses[int(len(fitnesses) * eta_cutoff)]
        eta_cutoff = numpy.searchsorted(
            fitnesses, fitness_cutoff
        ) / len(fitnesses)
    else:
        fitness_cutoff = float('-inf')
        eta_cutoff = 0.0

    # Do the permutation and cutoff, and calculate neighbors
    if permute in (True, False, 'aa27'):
        genotype_graph.permute_fitnesses(permute)
    if cutoff_type in ('low_fitness', 'high_fitness'):
        genotype_graph.remove_nodes(
            cutoff = fitness_cutoff, method = cutoff_type
        )
    elif cutoff_type == 'random':
        genotype_graph.remove_nodes(cutoff = eta_cutoff, method = cutoff_type)
    genotype_graph.calc_edges(*graph_type)
    if 'neighbor' in cutoff_type:
        genotype_graph.remove_nodes(
            cutoff = fitness_cutoff, method = cutoff_type
        )
        genotype_graph.calc_edges(*graph_type)
    if permute == 'after_cutoff':
        genotype_graph.permute_fitnesses(permute)
    fitnesses = sorted(node.fitness for node in genotype_graph)
    fitness_min = fitnesses[0]
    fitness_max = fitnesses[-1]

    return (
        genotype_graph, graph_type, fitnesses, fitness_min, fitness_max, 
        fitness_cutoff, eta_cutoff
    )

def main(
    graph_type : str = 'point_mutations,papkou_2023_folA.csv',
    fitness_distribution : str or types.NoneType = None,
    evolution_type : str = 'exact',
    evolution_algo : str = 'all',
    fitness_cutoff : float = None, 
    eta_cutoff : float = None, 
    cutoff_type : str = 'low_fitness', 
    permute : bool or str = False,
    population_distribution = 'uniform'
) -> dict:
    '''
    '''
    # Set the same seed each run, rather than just each time the module is 
    # loaded (I'm not convinced that this is really beneficial though)
    random.seed(the_seed)
    # Any data returned from this function should go in this dictionary
    return_value = {}
    # Build the graph, and get some other info too.
    (
        genotype_graph, graph_type, fitnesses, fitness_min, fitness_max, 
        fitness_cutoff, eta_cutoff
    ) = make_graph(
        graph_type, fitness_distribution, fitness_cutoff, eta_cutoff, 
        cutoff_type, permute
    )

    genotype_graph.open_log(
        f'graph_stats_cutoff_{cutoff_type}_{fitness_cutoff}_{eta_cutoff}_'
        f'permute_{permute}.txt'
    )
    genotype_graph.write_log(
        'Parameter values:\n'
        f'Graph type:           {graph_type}\n'
        f'Evolution type:       {evolution_type}\n'
        f'Evolution algorithm:  {evolution_algo}\n'
        f'Fitness distribution: {fitness_distribution}\n'
        f'Fitness cutoff:       {fitness_cutoff}\n'
        f'Eta cutoff:           {eta_cutoff}\n'
        f'Cutoff type:          {cutoff_type}\n'
        f'Fitness permutation:  {permute}\n'
        f'Initial population:   {population_distribution}\n'
    )

    # Make a step function of 1 - CDF of fitness; this is the minimum possible 
    # "final" fitness distribution (because it's the initial distribution), if 
    # evolution didn't happen
    fitnesses2 = [val for val in fitnesses for _ in range(2)]
    node_probability = 1 / len(genotype_graph)
    min_accessibility = [
        ival * node_probability for ival in range(len(genotype_graph), 0, -1)
    ]
    min_accessibility2 = [
        val for val in min_accessibility for _ in range(2)
    ][1:] + [0]
    # We have to be careful with these integrals because the discrete measure is
    # not a regular grid
    access_int_min = sum(
        access * (fit2 - fit1) for access, fit1, fit2 in 
        zip(min_accessibility[1:], fitnesses[:-1], fitnesses[1:])
    )
    # Calculate the transistion matrix and do the evolution
    if evolution_type == 'exact':
        # This approach relies on not allowing neutral mutations
        check_plateaus = False
    else:
        # In this case, there will be random walkers instead
        check_plateaus = True
    genotype_graph.calc_peaks(check_plateaus = check_plateaus)
    if evolution_algo == 'all':
        evolution_types = ['random', 'kimura', 'greedy', 'basin', 'none']
    else:
        evolution_types = evolution_algo.split(',')
    accessibilities = {}
    peak_accessibilities = {}
    access_cdf = {}
    access_cdf2 = {}
    access_int = {}
    cdf_access_int = {}
    unique_peak_accessibilities = {}
    unique_access_cdf = {}
    unique_access_cdf2 = {}
    unique_access_cdf3 = {}
    unique_access_int = {}
    for evo_type in evolution_types:
        if permute == 'matrix':
            genotype_graph.calc_transition_matrix(
                evolution_type = evo_type, permute = True, 
                check_plateaus = check_plateaus
            )
        else:
            genotype_graph.calc_transition_matrix(
                evolution_type = evo_type, check_plateaus = check_plateaus
            ) 
        # Permuting the peaks requires calculating the transition matrix twice, 
        # the first time is just to find the peaks
        if permute == 'peaks':
            peak_fitnesses = [
                genotype_graph[ipeak].fitness for ipeak in genotype_graph.peaks
            ]
            random.shuffle(peak_fitnesses)
            for ipeak, fitness in zip(genotype_graph.peaks, peak_fitnesses):
                genotype_graph[ipeak].fitness = fitness
            genotype_graph.calc_peaks(check_plateaus = check_plateaus)
            genotype_graph.calc_transition_matrix(
                evolution_type = evo_type, check_plateaus = check_plateaus
            )
        initial_state = None
        if population_distribution == 'lowest':
            initial_state = [0] * len(genotype_graph)
            initial_state[genotype_graph.nodes.index(
                min(genotype_graph.nodes, key = lambda x: x.fitness)
            )] = 1
        if evolution_type == 'exact':
            accessibilities[evo_type] = genotype_graph.fast_markov_equilibrium(
                initial_state = initial_state
            )
        elif evolution_type.strip().split(',')[0] == 'walkers':
            num_walkers = int(evolution_type.strip().split(',')[1])
            accessibilities[evo_type], info = genotype_graph.random_walkers(
                num_walkers = num_walkers, initial_state = initial_state
            )
            return_value |= {'walker_info_' + evo_type : info}
        # Analyze the distribuions of peaks and final populations
        fitness_vs_accessibility = sorted((
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[evo_type]) if node.is_peak
        ), key = lambda x: x[0])
        sorted_peak_fitnesses, peak_accessibilities[evo_type] = [
            list(val) for val in zip(*fitness_vs_accessibility)
        ]
        sorted_peak_fitnesses2 = [fitness_min] + [
            val for val in sorted_peak_fitnesses for _ in range(2)
        ]
        access_cdf[evo_type] = peak_accessibilities[evo_type].copy()
        for ipeak in range(len(access_cdf[evo_type]) - 1, 0, -1):
            access_cdf[evo_type][ipeak - 1] += access_cdf[evo_type][ipeak]
        access_cdf2[evo_type] = [
            val for val in access_cdf[evo_type] for _ in range(2)
        ] + [0]
        access_int[evo_type] = sum(
            access * (fit2 - fit1) for access, fit1, fit2 in zip(
                access_cdf[evo_type], 
                sorted_peak_fitnesses2[:1] + sorted_peak_fitnesses[:-1], 
                sorted_peak_fitnesses
            )
        )
        # Same, but as a function of the CDF of peaks
        cdf_access_int[evo_type] = sum(access_cdf[evo_type]) / len(
            peak_accessibilities[evo_type]
        )
        # Check to see if any peaks have the same fitness, sum them, and repeat 
        # the analyses
        unique_peak_fitnesses = sorted(set(sorted_peak_fitnesses))
        unique_peak_degeneracies = [
            sorted_peak_fitnesses.count(val) for val in unique_peak_fitnesses
        ]
        unique_peak_weights = [
            val / len(sorted_peak_fitnesses) for val in unique_peak_degeneracies
        ]
        unique_peak_accessibilities[evo_type] = [0] * len(unique_peak_fitnesses)
        for fitness, access in zip(
            sorted_peak_fitnesses, peak_accessibilities[evo_type]
        ):
            unique_peak_accessibilities[evo_type][
                unique_peak_fitnesses.index(fitness)
            ] += access
        unique_access_cdf[
            evo_type
        ] = unique_peak_accessibilities[evo_type].copy()
        for ipeak in range(len(unique_access_cdf[evo_type]) - 1, 0, -1):
            unique_access_cdf[evo_type][ipeak - 1] += unique_access_cdf[
                evo_type
            ][ipeak]
        unique_access_cdf2[evo_type] = [
            val for val in unique_access_cdf[evo_type] for _ in range(2)
        ] + [0]
        unique_access_cdf3[evo_type] = unique_access_cdf2[evo_type][1:] + [0]
        unique_access_int[evo_type] = sum(
            access * weight for access, weight 
            in zip(unique_access_cdf[evo_type], unique_peak_weights)
        )
            
    # We also want to compare with arriving at each peak randomly
    num_peaks = sum(node.is_peak for node in genotype_graph)
    random_access_cdf = [
        val / num_peaks for val in range(1, num_peaks + 1)
    ][::-1]
    random_access_cdf2 = [
        val for val in random_access_cdf for _ in range(2)
    ] + [0]
    access_int_random = sum(
        access * (fit2 - fit1) for access, fit1, fit2 in zip(
            random_access_cdf, 
            sorted_peak_fitnesses2[:1] + sorted_peak_fitnesses[:-1], 
            sorted_peak_fitnesses
        )
    )
    access_int_max = fitness_max - fitness_min
    # Again, transformed to the CDF of the peaks
    cdf_access_int_random = sum(random_access_cdf) / num_peaks
    # Again, accounting for peaks with equal fitness
    unique_random_access_cdf = unique_peak_weights.copy()
    for ipeak in range(len(unique_access_cdf[evo_type]) - 1, 0, -1):
        unique_random_access_cdf[ipeak - 1] += unique_random_access_cdf[ipeak]
    unique_random_access_cdf2 = [
        val for val in unique_random_access_cdf for _ in range(2)
    ] + [0]
    unique_random_access_cdf2_compliment = [
        1 - val for val in unique_random_access_cdf2
    ][1:] + [1.0]
    unique_access_int_random = sum(
        access * weight for access, weight 
        in zip(unique_random_access_cdf, unique_peak_weights)
    )

    # I think this should be equivalent to something like the "difference 
    # between the average initial fitness and the average final fitness of each 
    # walker, normalized to the range of possible fitnesses"
    delta_fs = {evo_type : [
        (
            (access_int[evo_type] - access_int_min) 
            / (access_int_max - access_int_min)
        ) if access_int_max != access_int_min else 1, (
            (access_int[evo_type] - access_int_random) 
            / (access_int_max - access_int_random)
        ) if access_int_max != access_int_random else 1
    ] for evo_type in evolution_types}

    genotype_graph.write_log(
        'Delta-fs:\n'
        '┌──────┬───────────────┬────────────┐\n'
        '│      │vs. Fitness CDF│vs. Peak CDF│\n'
        '├──────┼───────────────┼────────────┤\n'
        + '\n'.join([
            f'│{evo_type.title():6}│{delta_fs[evo_type][0]:^15.5f}│'
            f'{delta_fs[evo_type][1]:^12.5f}│' for evo_type in evolution_types
        ]) + '\n'
        '└──────┴───────────────┴────────────┘\n'
    )

    # Again, accounting for peaks with equal fitness
    peak_cdf_delta_fs = {evo_type : [(
        (cdf_access_int[evo_type] - cdf_access_int_random)
        / (1 - cdf_access_int_random)
    ) if cdf_access_int_random != 1 else 1, (
        (unique_access_int[evo_type] - unique_access_int_random) 
        / (1 - unique_access_int_random) 
    ) if unique_access_int_random != 1 else 1] for evo_type in evolution_types}

    genotype_graph.write_log(
        'Peak-CDF delta-fs:\n'
        '┌──────┬──────────────┬─────────────────┐\n'
        '│      │Assume no ties│Checking for ties│\n'
        '├──────┼──────────────┼─────────────────┤\n'
        + '\n'.join([
            f'│{evo_type.title():6}│{peak_cdf_delta_fs[evo_type][0]:^14.5f}│'
            f'{peak_cdf_delta_fs[evo_type][1]:^17.5f}│' for evo_type in evolution_types
        ]) + '\n'
        '└──────┴──────────────┴─────────────────┘\n'
    )

    # Calculate fraction of the population that reached the X% of peaks with the
    # highest fitness; the number of peaks above the cutoff is always rounded 
    # up, so that the behavior is consistent when the number of peaks above the 
    # cutoff would otherwise be less than 1
    percentiles = [0.01, 0.05, 0.1, 0.14, 0.25, 0.5]
    high_fitness_population = hfp = {evo_type : {percent : sum(
        peak_accessibilities[evo_type][-1 * (num_peaks * percent).__ceil__():]
    ) for percent in percentiles} for evo_type in evolution_types}

    genotype_graph.write_log(
        'Population at high-fitness peaks (continuous estimate):\n'
        f'┌───┬{'┬'.join(['─' * 7] * len(evolution_types))}┐\n' 
        f'│   │{'│'.join([
            f'{evo_type.title():^7}' for evo_type in evolution_types
        ])}│\n'
        f'├───┼{'┼'.join(['─' * 7] * len(evolution_types))}┤\n' 
        + '\n'.join([f'│{percent:3.0%}│{'│'.join([
            f'{hfp[evo_type][percent]:^7.5f}' for evo_type in evolution_types
        ])}│' for percent in percentiles]) + '\n'
        f'└───┴{'┴'.join(['─' * 7] * len(evolution_types))}┘\n' 
    )

    # Again, now optimistically assuming that a block of tied peaks can't be 
    # split, then the same, but pessimistically. In these two, the value 
    # calculated for the slice should always be a whole number of peaks, so 
    # `round` is used to correct for floating point error
    optimistic_high_fitness_population = ohfp = {evo_type : {percent : sum(
        peak_accessibilities[evo_type][-1 * round(
            num_peaks * next(
                val for val in unique_random_access_cdf[::-1] if val >= percent
            )
        ):]
    ) for percent in percentiles} for evo_type in evolution_types}

    genotype_graph.write_log(
        'Population at high-fitness peaks (optimistic estimate):\n'
        f'┌───┬{'┬'.join(['─' * 7] * len(evolution_types))}┐\n' 
        f'│   │{'│'.join([
            f'{evo_type.title():^7}' for evo_type in evolution_types
        ])}│\n'
        f'├───┼{'┼'.join(['─' * 7] * len(evolution_types))}┤\n' 
        + '\n'.join([f'│{percent:3.0%}│{'│'.join([
            f'{ohfp[evo_type][percent]:^7.5f}' for evo_type in evolution_types
        ])}│' for percent in percentiles]) + '\n'
        f'└───┴{'┴'.join(['─' * 7] * len(evolution_types))}┘\n' 
    )

    pessimistic_high_fitness_population = phfp = {evo_type : {percent : sum(
        peak_accessibilities[evo_type][-1 * round(
            num_peaks * next((
                val for val in unique_random_access_cdf if val <= percent
            ), 1 / num_peaks)
        ):]
    ) for percent in percentiles} for evo_type in evolution_types}
    genotype_graph.write_log(
        'Population at high-fitness peaks (pessimistic estimate):\n'
        f'┌───┬{'┬'.join(['─' * 7] * len(evolution_types))}┐\n' 
        f'│   │{'│'.join([
            f'{evo_type.title():^7}' for evo_type in evolution_types
        ])}│\n'
        f'├───┼{'┼'.join(['─' * 7] * len(evolution_types))}┤\n' 
        + '\n'.join([f'│{percent:3.0%}│{'│'.join([
            f'{phfp[evo_type][percent]:^7.5f}' for evo_type in evolution_types
        ])}│' for percent in percentiles]) + '\n'
        f'└───┴{'┴'.join(['─' * 7] * len(evolution_types))}┘\n' 
    )

    genotype_graph.find_subgraphs()
    genotype_graph.log_stats()

    # Get stats of a comparable random graph, for comparison
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        p = float(fitness_distribution.split(',')[-1])
        null_model = noras_rough_mount_fuji(genotype_graph.dtype[1], p)
        fitness_x_vals = null_model['pdf_xs']
        bin_step = 0.5 # Actually, it's not a constant...
        bin_shift = 0.25
        fitness_bins = [fitness_x_vals[0] - bin_shift] + [
            val + bin_shift for val in fitness_x_vals
        ]
        fitness_hist = histogram(
            [node.fitness for node in genotype_graph], bins = fitness_bins, 
            density = True
        )
    else:
        fitness_bins = list(numpy.histogram_bin_edges(
            [], bins = 100, range = (fitness_min, fitness_max)
        ))
        bin_step = fitness_bins[1] - fitness_bins[0]
        bin_shift = bin_step / 2
        fitness_x_vals = [fitness_bins[0] - bin_shift] + [
            val + bin_shift for val in fitness_bins
        ]
        fitness_bins = (
            [fitness_bins[0] - bin_step] + fitness_bins 
            + [fitness_bins[-1] + bin_step]
        )
        fitness_bins[-2] += bin_step * 1e-6
        fitness_hist = histogram(
            [node.fitness for node in genotype_graph], bins = fitness_bins, 
            density = True
        )
        null_model = house_of_cards(
            4, genotype_graph.dtype[1], [fitness_x_vals, fitness_hist], 
            len(genotype_graph), missing_nodes = cutoff_type
        )

    ### It's all plotting from here (I should maybe move them out to functions)
    # Plot a scatter plot of peak accessibility
    plot_evo = 'kimura' if 'kimura' in evolution_types else evolution_types[0]
    if (
        graph_type[0] == 'point_mutations' 
        and len(genotype_graph[0].amino_sequence) > 2
    ):
        fitness_vs_accessibility_cys = [
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[plot_evo]) 
            if access != 0 and node.amino_sequence[1] == 'C'
        ]
        fitness_vs_accessibility_asp = [
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[plot_evo])
            if access != 0 and node.amino_sequence[1] == 'D'
        ]
        fitness_vs_accessibility_glu = [
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[plot_evo])
            if access != 0 and node.amino_sequence[1] == 'E'
        ]
        fitness_vs_accessibility_other = [
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[plot_evo]) 
            if access != 0 and node.amino_sequence not in ('C', 'D', 'E')
        ]
        matplotlib.pyplot.plot(
            *list(zip(*fitness_vs_accessibility_other)), '.', c = 'grey'
        )
        matplotlib.pyplot.plot(
            *list(zip(*fitness_vs_accessibility_cys)), '.', c = 'magenta'
        )
        matplotlib.pyplot.plot(
            *list(zip(*fitness_vs_accessibility_asp)), '.', c = 'limegreen'
        )
        matplotlib.pyplot.plot(
            *list(zip(*fitness_vs_accessibility_glu)), '.', c = 'brown'
        )
        matplotlib.pyplot.legend(
            ['Other', 'Cys', 'Asp', 'Glu'], title = 'Amino acid 27'
        )
    else:
        fitness_vs_accessibility = [
            (node.fitness, access) for node, access 
            in zip(genotype_graph, accessibilities[plot_evo]) if access != 0
        ]
        matplotlib.pyplot.plot(
            *list(zip(*fitness_vs_accessibility)), '.', c = 'grey'
        )
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('Accessibility (relative probability)')
    if 'kimura' in evolution_types:
        matplotlib.pyplot.title(
            "Accessibility vs Fitness\nKimura's fixation probability w/ N=1e8"
        )
    else:
        matplotlib.pyplot.title(
            f'Accessibility vs Fitness\n{plot_evo.title()} evolution'
        )
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'peak_accessibility_cutoff_{cutoff_type}_{fitness_cutoff}_{eta_cutoff}'
        f'_permute_{permute}.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()

    # Plot 1 - CDF for accessibility as a function of fitness; it could also be 
    # interesting to look at it as a function of the fitness's CDF (or the 
    # peak's CDF?)
    matplotlib.pyplot.plot(fitnesses2, min_accessibility2)
    matplotlib.pyplot.plot(sorted_peak_fitnesses2, random_access_cdf2)
    for evo_type in evolution_types:
        matplotlib.pyplot.plot(sorted_peak_fitnesses2, access_cdf2[evo_type])
    matplotlib.pyplot.plot([fitness_min, fitness_max, fitness_max], [1, 1, 0])
    matplotlib.pyplot.legend(
        ['Initial state', 'CDF of peaks', *evolution_types, 'Ideal result']
    )
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('Cumulative probability')
    matplotlib.pyplot.title(
        'Cumulative distribution function of accessibility before and after '
        f'evolution\nCutoff type: {cutoff_type}, f={fitness_cutoff:4.7f}, '
        f'eta={eta_cutoff:4.7f}; and permutation type: {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'graph_accessibility_cutoff_{cutoff_type}_{fitness_cutoff}_'
        f'{eta_cutoff}_permute_{permute}.pdf', 
        format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()
    # Plot 1 - CDF of final peak populations as a function of the CDF of peaks
    # Version 1, where peaks with equal fitness have a random order
    Fs = [val / 1000 for val in range(1001)]
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        # Some analytical results for nrmf
        Ds = list(range(genotype_graph.dtype[1] + 1))
        Xs = [scipy.special.betaincc(
            genotype_graph.dtype[1] - dist, dist + 1, 1 / (2 - p)
        ) for dist in Ds]
        Ys_greedy = [
            scipy.special.betainc(genotype_graph.dtype[1] - dist, dist + 1, 0.5) 
            + sum(
                scipy.special.binom(genotype_graph.dtype[1], dist + i + 1) 
                * (1 - p) ** (i / 2 * (2 * dist + i + 3))
                for i in range(genotype_graph.dtype[1] - dist)
            ) * (1 - p) / 2 ** genotype_graph.dtype[1]
            for dist in Ds
        ]
        Ys_random = [
            scipy.special.betainc(genotype_graph.dtype[1] - dist + 1, dist, 0.5) 
            + scipy.special.binom(genotype_graph.dtype[1], dist) 
            / 2 ** genotype_graph.dtype[1] * (
                p + (dist * (1 - p)) 
                / (dist * (1 - p) + genotype_graph.dtype[1] * p)
            ) + (dist * (dist + 1) * (1 - p)) / (
                dist * (dist + 1) * (1 - p) ** 2 
                + (genotype_graph.dtype[1] - 1) ** 2 * p ** 2 
                + p * (1 - p) * (genotype_graph.dtype[1] - 1) * (2 * dist + 1)
            ) * scipy.special.binom(genotype_graph.dtype[1], dist + 1) 
            / 2 ** genotype_graph.dtype[1]
            * scipy.special.hyp2f1(
                1, dist + 1 - genotype_graph.dtype[1], 
                dist + 2 + ((genotype_graph.dtype[1] - 1) * p) / (1 - p), -1
            ) for dist in Ds
        ]
        Ys_lol = [math.prod(
            1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p)))
            for j in range(dist + 1, genotype_graph.dtype[1] + 1)
        ) for dist in Ds]
        matplotlib.pyplot.plot(Xs, Ys_greedy)
        matplotlib.pyplot.plot(Xs, Ys_random)
        matplotlib.pyplot.plot(Xs, Ys_lol)
    else:
        Xs = [F ** genotype_graph.dtype[1] for F in Fs]
        # Here's what can be calculated for greedy walks in the HoC based on Orr
        # (2002); it's clearly not very accurate for the size of landscapes that
        # could plausibly be simulated with this code, but oh well
        # Since it's a matter of summing over walks of different lengths, the 
        # maximum length considered is arbitrary; 100 is probably fine.
        # Then for random walks it has to be normalized and integrated as well
        Ys_greedy = [1 - sum(
            walk / math.factorial(walk + 1) 
            * F ** (genotype_graph.dtype[1] * (walk + 1)) 
            for walk in range(101)
        ) for F in Fs]
        Ys_random = [
            F ** (genotype_graph.dtype[1] - 1) 
            * math.exp(sum(
                F ** i / i for i in range(1, genotype_graph.dtype[1])
            )) for F in Fs
        ]
        Ys_norm = sum(Ys_random) #/ len(Ys_random)
        Ys_random = [val / Ys_norm for val in Ys_random]
        for ival in range(1, len(Ys_random)):
            Ys_random[ival] += Ys_random[ival - 1]
        Ys_random = [1 - val for val in Ys_random]
        matplotlib.pyplot.plot(Xs, Ys_greedy)
        matplotlib.pyplot.plot(Xs, Ys_random)
    matplotlib.pyplot.plot(random_access_cdf2[::-1], random_access_cdf2)
    for evo_type in evolution_types:
        matplotlib.pyplot.plot(random_access_cdf2[::-1], access_cdf2[evo_type])
    matplotlib.pyplot.plot([0, 1, 1], [1, 1, 0])
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        matplotlib.pyplot.legend(
            [
                'greedy walk analytical', 'random walk analytical', 
                'simple lower bound', 'CDF of peaks', *evolution_types, 
                'Ideal result'
            ]
        )
    else:
        matplotlib.pyplot.legend(
            [
                'greedy walk analytical', 'random walk analytical', 
                'CDF of peaks', *evolution_types, 'Ideal result'
            ]
        )
    matplotlib.pyplot.xlabel('CDF of peaks')
    matplotlib.pyplot.ylabel('Cumulative probability')
    matplotlib.pyplot.title(
        'Cumulative distribution function of accessibility before and after '
        f'evolution\nCutoff type: {cutoff_type}, f={fitness_cutoff:4.7f}, '
        f'eta={eta_cutoff:4.7f}; and permutation type: {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'peak_cdf_graph_accessibility_cutoff_{cutoff_type}_{fitness_cutoff}_'
        f'{eta_cutoff}_permute_{permute}.pdf', 
        format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()
    peak_cdf_plot_data = {
        'base_line' : [random_access_cdf2[::-1], random_access_cdf2],
        'greedy_analytical' : [Xs, Ys_greedy],
        'random_analytical' : [Xs, Ys_random],
        'upper_bound' : [[0, 1, 1], [1, 1, 0]]
    }
    if 'greedy' in evolution_types:
        peak_cdf_plot_data |= {'greedy_simulation' : [
            random_access_cdf2[::-1], access_cdf2['greedy']
        ]}
    if 'random' in evolution_types:
        peak_cdf_plot_data |= {'random_simulation' : [
            random_access_cdf2[::-1], access_cdf2['random']
        ]}
    if 'none' in evolution_types:
        peak_cdf_plot_data |= {'no_simulation' : [
            random_access_cdf2[::-1], access_cdf2['none']
        ]}
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        peak_cdf_plot_data |= {'random_lower_bound' : [Xs, Ys_lol]}

    # Plot 1 - CDF of final peak populations as a function of the CDF of peaks
    # Version 2, where peaks with equal fitness are grouped together
    matplotlib.pyplot.plot(
        unique_random_access_cdf2_compliment, unique_random_access_cdf2
    )
    for evo_type in evolution_types:
        matplotlib.pyplot.plot(
            unique_random_access_cdf2_compliment, unique_access_cdf2[evo_type]
        )
    matplotlib.pyplot.plot([0, 1, 1], [1, 1, 0])
    matplotlib.pyplot.legend(
        ['CDF of peaks', *evolution_types, 'Ideal result']
    )
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('Cumulative probability')
    matplotlib.pyplot.title(
        'Cumulative distribution function of accessibility before and after '
        f'evolution\nCutoff type: {cutoff_type}, f={fitness_cutoff:4.7f}, '
        f'eta={eta_cutoff:4.7f}; and permutation type: {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'unique_peak_cdf_graph_accessibility_cutoff_{cutoff_type}_'
        f'{fitness_cutoff}_{eta_cutoff}_permute_{permute}.pdf', 
        format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()

    # Plot the averaged random walk transition probability of going from a 
    # non-peak to a non-peak, binned on the CDF of fitnesses
    num_nodes = len(genotype_graph)
    sorted_nodes = sorted(
        range(num_nodes), key = lambda x: genotype_graph[x].fitness
    )
    binned_non_peaks = [[inode for inode in sorted_nodes[
        int(num_nodes / 100 * ibin):int(num_nodes / 100 * (ibin + 1))
    ] if not genotype_graph[inode].is_peak] for ibin in range(100)]
    genotype_graph.calc_transition_matrix(
        evolution_type = 'random', check_plateaus = check_plateaus
    )
    average_transition_probabilities = [
        1 - sum(
            len([
                neighbor for neighbor in genotype_graph[inode].neighbors 
                if genotype_graph[neighbor].is_peak
            ]) / genotype_graph.transition_matrix_csc[:, inode].nnz 
            for inode in bin_
        ) / len(bin_) for bin_ in binned_non_peaks
    ]
    matplotlib.pyplot.plot(
        [val / 100 + 0.005 for val in range(100)], 
        average_transition_probabilities
    )
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        Xs = [
            scipy.special.betaincc(genotype_graph.dtype[1] - d, d + 1, 0.5) 
            for d in Ds
        ]
        Ys = [
            1 / (1 + ((genotype_graph.dtype[1] - 1) * p) / (d * (1 - p))) 
            if d != 0 else 0 for d in Ds
        ]
        matplotlib.pyplot.plot(Xs, Ys)
    else:
        # HoC result
        alphabet = 3 if graph_type[0] == 'point_mutations' else 1
        Ys = [
            1 - (1 - f ** (genotype_graph.dtype[1] * alphabet + 1)) 
            / ((genotype_graph.dtype[1] * alphabet + 1) * (1 - f)) 
            if f != 1 else 0 for f in Fs
        ]
        matplotlib.pyplot.plot(Fs, Ys)

    if graph_type[0] == 'point_mutations':
        matplotlib.pyplot.hlines(
            [
                (genotype_graph.dtype[1] * 3 - 1) 
                / (genotype_graph.dtype[1] * 3 + 1)
            ], 
            xmin = 0, xmax = 1
        )
    else:
        matplotlib.pyplot.hlines(
            [(genotype_graph.dtype[1] - 1) / (genotype_graph.dtype[1] + 1)], 
            xmin = 0, xmax = 1
        )
    peak_fitness_ranks = [
        inode for inode in range(num_nodes) 
        if sorted_nodes[inode] in genotype_graph.peaks
    ]
    matplotlib.pyplot.vlines(
        [sum(peak_fitness_ranks) / len(peak_fitness_ranks) / num_nodes], 
        ymin = 0, ymax = 1
    )
    matplotlib.pyplot.xlabel('CDF of fitness')
    matplotlib.pyplot.ylabel('probability')
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.title('Probability of not transitioning to a peak')
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        'transition_probabilities.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()

    # Plot the average rank of the peaks neighboring each node, binned on the 
    # CDF of fitness
    binned_non_peaks_neighboring_peaks = [[
        inode for inode in bin_ 
        if sum(
            genotype_graph[neighbor].is_peak 
            for neighbor in genotype_graph[inode].neighbors
        ) > 0
        ] for bin_ in binned_non_peaks]
    average_neighboring_peak_ranks = [sum(walrus) / len(walrus) if len(
        walrus := [
            sorted_peak_fitnesses.index(genotype_graph[neighbor].fitness) 
            / num_peaks 
            for inode in bin_ 
            for neighbor in genotype_graph[inode].neighbors 
            if genotype_graph[neighbor].is_peak
        ]
    ) != 0 else 0 for bin_ in binned_non_peaks]
    matplotlib.pyplot.plot(
        [val / 100 + 0.005 for val in range(100)], 
        average_neighboring_peak_ranks
    )
    alphabet = 3 if graph_type[0] == 'point_mutations' else 1
    Ys_hoc = [
        (1 - F ** (2 * (genotype_graph.dtype[1] * alphabet -1))) 
        / 2 / (1 - F ** (genotype_graph.dtype[1] * alphabet - 1)) 
        if F != 1 else 1 for F in Fs
    ]
    matplotlib.pyplot.plot(Fs, Ys_hoc)
    matplotlib.pyplot.plot(Fs, Fs)
    matplotlib.pyplot.xlabel('CDF of fitness')
    matplotlib.pyplot.ylabel('Peak rank')
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.title('Average rank of neighboring peaks')
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        'neighboring_peak_ranks.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()

    # Plot the probability distribution functions of pits and peaks
    peaks_hist = histogram(
        [genotype_graph[inode].fitness for inode in genotype_graph.peaks], 
        bins = fitness_bins, density = True
    )
    pits_hist = histogram(
        [genotype_graph[inode].fitness for inode in genotype_graph.pits], 
        bins = fitness_bins, density = True
    )
    accessibility_hist = histogram(
        sorted_peak_fitnesses, bins = fitness_bins, density = True, 
        weights = peak_accessibilities[plot_evo]
    )
    peak_accessibility_hist = [
        a / p if p != 0 else 0 for p, a in zip(peaks_hist, accessibility_hist)
    ]
    peak_accessibility_norm = sum(peak_accessibility_hist) * bin_step
    peak_accessibility_hist = [
        val / peak_accessibility_norm for val in peak_accessibility_hist
    ]
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        if 'basin' in evolution_types:
            basin_hist = histogram(
                sorted_peak_fitnesses, bins = fitness_bins, #density = True, 
                weights = peak_accessibilities['basin']
            )
            matplotlib.pyplot.fill_between(
                fitness_x_vals, null_model['basins'], color = 'red', alpha = 0.3
            )
            matplotlib.pyplot.plot(fitness_x_vals, basin_hist, color = 'red')
        matplotlib.pyplot.fill_between(
            fitness_x_vals, [2 * val for val in null_model['genotypes']], 
            color = 'brown', alpha = 0.3
        )
        matplotlib.pyplot.fill_between(
            fitness_x_vals, [2 * val for val in null_model['peaks_pdf']], 
            color = 'limegreen', alpha = 0.3
        )
        if 'greedy' in evolution_types:
            greedy_access_hist = histogram(
                sorted_peak_fitnesses, bins = fitness_bins, density = True, 
                weights = peak_accessibilities['greedy']
            )
            greedy_peak_access_hist = [
                a / p if p != 0 else 0 
                for p, a in zip(peaks_hist, greedy_access_hist)
            ]
            greedy_peak_access_norm = sum(greedy_peak_access_hist) * bin_step
            greedy_peak_access_hist = [
                val / greedy_peak_access_norm for val in greedy_peak_access_hist
            ]
            matplotlib.pyplot.plot(
                fitness_x_vals, greedy_access_hist, color = 'magenta'
            )
        matplotlib.pyplot.fill_between(
            fitness_x_vals, [2 * val for val in null_model['population']], 
            color = 'magenta', alpha = 0.3
        )
    else:
        matplotlib.pyplot.fill_between(
            fitness_x_vals, null_model['pits_pdf'], color = 'magenta', 
            alpha = 0.3
        )
        matplotlib.pyplot.plot(fitness_x_vals, pits_hist, c = 'magenta')
        matplotlib.pyplot.fill_between(
            fitness_x_vals, null_model['peaks_pdf'], color = 'limegreen', 
            alpha = 0.3
        )
    matplotlib.pyplot.plot(fitness_x_vals, peaks_hist, c = 'limegreen')
    matplotlib.pyplot.plot(fitness_x_vals, fitness_hist, c = 'brown')
    matplotlib.pyplot.plot(fitness_x_vals, accessibility_hist, c = 'coral')
    matplotlib.pyplot.plot(fitness_x_vals, peak_accessibility_hist, c = 'blue')
    matplotlib.pyplot.legend([
        'HoC pits', 'Pits', 'HoC peaks', 'Peaks', 'All', 'Accessibility', 
        'Accessibility/peak'
    ])
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('Instantaneous probability')
    matplotlib.pyplot.title(
        'Probability distribution function of genotypes, peaks, and pits\n'
        f'Cutoff type: {cutoff_type}, f={fitness_cutoff:4.7f}, '
        f'eta={eta_cutoff:4.7f}; and permutation type: {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'distributions_cutoff_{cutoff_type}_{fitness_cutoff}_{eta_cutoff}_'
        f'permute_{permute}.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()
    # Same as above, but as functions of the CDF of fitness
    eta_bins = [val / 100 for val in range(-1, 102)]
    eta_bins[-2] += 1e-16
    eta_bin_step = eta_bins[1] - eta_bins[0]
    eta_bin_shift = eta_bin_step / 2
    eta_x_vals = [val + eta_bin_shift for val in eta_bins[:-1]]
    eta_fitness_bins = [float('-inf')] + [
        fitnesses[int(val * len(genotype_graph))] for val in eta_bins[1:-2]
    ] + [len(genotype_graph), float('inf')]
    fitness_hist_eta = histogram(
        [node.fitness for node in genotype_graph], bins = eta_fitness_bins#, 
        #density = True
    )
    fitness_norm_eta = sum(fitness_hist_eta) * eta_bin_step
    fitness_hist_eta = [val / fitness_norm_eta for val in fitness_hist_eta]
    peaks_hist_eta = histogram(
        [genotype_graph[inode].fitness for inode in genotype_graph.peaks], 
        bins = eta_fitness_bins, #density = True # these are the wrong bins to normalize with
    )
    peaks_norm_eta = sum(peaks_hist_eta) * eta_bin_step
    peaks_hist_eta = [val / peaks_norm_eta for val in peaks_hist_eta]
    pits_hist_eta = histogram(
        [genotype_graph[inode].fitness for inode in genotype_graph.pits], 
        bins = eta_fitness_bins, #density = True
    )
    pits_norm_eta = sum(pits_hist_eta) * eta_bin_step
    pits_hist_eta = [val / pits_norm_eta for val in pits_hist_eta]
    accessibility_hist_eta = histogram(
        sorted_peak_fitnesses, bins = eta_fitness_bins, #density = True, 
        weights = peak_accessibilities[plot_evo]
    )
    accessibility_norm_eta = sum(accessibility_hist_eta) * eta_bin_step
    accessibility_hist_eta = [
        val / accessibility_norm_eta for val in accessibility_hist_eta
    ]
    peak_accessibility_hist_eta = [
        a / p if p != 0 else 0 
        for p, a in zip(peaks_hist_eta, accessibility_hist_eta)
    ]
    peak_accessibility_norm_eta = sum(
        peak_accessibility_hist_eta
    ) * eta_bin_step
    peak_accessibility_hist_eta = [
        val / peak_accessibility_norm_eta for val in peak_accessibility_hist_eta
    ]
    matplotlib.pyplot.plot(eta_x_vals, pits_hist_eta, c = 'magenta')
    matplotlib.pyplot.plot(eta_x_vals, peaks_hist_eta, c = 'limegreen')
    matplotlib.pyplot.plot(eta_x_vals, fitness_hist_eta, c = 'brown')
    matplotlib.pyplot.plot(eta_x_vals, accessibility_hist_eta, c = 'coral')
    matplotlib.pyplot.plot(eta_x_vals, peak_accessibility_hist_eta, c = 'blue')
    matplotlib.pyplot.legend([
        'Pits', 'Peaks', 'All', 'Accessibility', 'Accessibility/peak'
    ])
    matplotlib.pyplot.xlabel('Eta(f) (CDF of ln(Fitness), fitness percentile)')
    matplotlib.pyplot.ylabel('Instantaneous probability')
    matplotlib.pyplot.title(
        'Probability distribution function of genotypes, peaks, and pits\n'
        f'Cutoff type: {cutoff_type}, f={fitness_cutoff:4.7f}, '
        f'eta={eta_cutoff:4.7f}; and permutation type: {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'eta_distributions_cutoff_{cutoff_type}_{fitness_cutoff}_{eta_cutoff}_'
        f'permute_{permute}.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()
    # Now for the joint distributions
    if fitness_distribution is None or not 'noras' in fitness_distribution:
        matplotlib.pyplot.imshow(
            null_model['null_joint_dist'],
            cmap = matplotlib.pyplot.cm.nipy_spectral, norm = 'log', 
            origin = 'lower', extent = [
                fitness_bins[0], fitness_bins[-1], 
                fitness_bins[0], fitness_bins[-1]
            ]
        )
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
        matplotlib.pyplot.ylabel('ln(Fitness) of neighboring nodes (1 / time)')
        matplotlib.pyplot.title(
            'Joint probability distribution function of the fitness of genotypes '
            'with their neighbors\nHouse of cards model with the Papkou fitness '
            'distribution'
        )
        matplotlib.pyplot.gcf().set_size_inches(9, 9)
        matplotlib.pyplot.savefig(
            f'joint_distribution_null_cutoff_{cutoff_type}_{fitness_cutoff}_'
            f'{eta_cutoff}_permute_{permute}.pdf', format = 'pdf', 
            bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()
    fitness_pairs = list(zip(*[
        (node1.fitness, genotype_graph[inode2].fitness) 
        for node1 in genotype_graph for inode2 in node1.neighbors
    ]))
    joint_fitness_hist, _, _, _ = matplotlib.pyplot.hist2d(
        *fitness_pairs, bins = fitness_bins
    )
    matplotlib.pyplot.close()
    joint_fitness_hist /= (joint_fitness_hist.sum() * (bin_step ** 2))
    matplotlib.pyplot.imshow(
        joint_fitness_hist, cmap = matplotlib.pyplot.cm.nipy_spectral, 
        norm = 'log', origin = 'lower', extent = [
            fitness_bins[0], fitness_bins[-1], 
            fitness_bins[0], fitness_bins[-1]
        ]
    )
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('ln(Fitness) of neighboring nodes (1 / time)')
    matplotlib.pyplot.title(
        'Joint probability distribution function of the fitness of genotypes '
        'with their neighbors\nPapkou fitness landscape'
    )
    matplotlib.pyplot.gcf().set_size_inches(9, 9)
    matplotlib.pyplot.savefig(
        f'joint_distribution_cutoff_{cutoff_type}_{fitness_cutoff}_{eta_cutoff}'
        f'_permute_{permute}.pdf', format = 'pdf', bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()
    if fitness_distribution is None or not 'noras' in fitness_distribution:
        matplotlib.pyplot.imshow(
            joint_fitness_hist / null_model['null_joint_dist'], 
            cmap = matplotlib.pyplot.cm.bwr, norm = 'log', vmin = 1e-3, vmax=1e3, 
            origin = 'lower', extent = [
                fitness_bins[0], fitness_bins[-1], 
                fitness_bins[0], fitness_bins[-1]
            ]
        )
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
        matplotlib.pyplot.ylabel('ln(Fitness) of neighboring nodes (1 / time)')
        matplotlib.pyplot.title(
            'Relative joint probability distribution function of the fitness of '
            'genotypes with their neighbors\nPapkou fitness landscape divided by '
            'the House of Cards model'
        )
        matplotlib.pyplot.gcf().set_size_inches(9, 9)
        matplotlib.pyplot.savefig(
            f'relative_joint_distribution_cutoff_{cutoff_type}_{fitness_cutoff}_'
            f'{eta_cutoff}_permute_{permute}.pdf', format = 'pdf', 
            bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()
    # And the difference distribution
    difference_dist = list(zip(*[
        [idiag * bin_step, joint_fitness_hist.diagonal(idiag).sum() * bin_step]
        for idiag in range(-1 * len(fitness_hist), len(fitness_hist) + 1)
    ]))
    if fitness_distribution is None or not 'noras' in fitness_distribution:
        matplotlib.pyplot.fill_between(
            *null_model['self_difference_dist'], color = 'limegreen', alpha = 0.3
        )
    matplotlib.pyplot.plot(*difference_dist, c = 'limegreen')
    matplotlib.pyplot.legend(['HoC', 'Papkou'])
    matplotlib.pyplot.xlabel('ln(Fitness) (1 / time)')
    matplotlib.pyplot.ylabel('Instantaneous probability')
    matplotlib.pyplot.title(
        'Probability distribution function of the fitness difference between '
        f'neighboring genotypes\nCutoff type: {cutoff_type}, '
        f'f={fitness_cutoff:4.7f}, eta={eta_cutoff:4.7f}; and permutation type:'
        f' {permute}'
    )
    matplotlib.pyplot.gcf().set_size_inches(16, 9)
    matplotlib.pyplot.savefig(
        f'difference_distribution_cutoff_{cutoff_type}_{fitness_cutoff}_'
        f'{eta_cutoff}_permute_{permute}.pdf', format = 'pdf', 
        bbox_inches = 'tight'
    )
    matplotlib.pyplot.close()

    # More plots! 
    plot_evo = 'random' if 'random' in evolution_types else evolution_types[0]
    if fitness_distribution is not None and 'noras' in fitness_distribution:
        # "number of genotypes as a function of d"
        figure, axes = matplotlib.pyplot.subplots()
        Ds = (
            [-1] + list(range(len(genotype_graph.nodes_by_distance))) 
            + [graph_type[1] + 1]
        )
        Ys = [0] + [len(val) for val in genotype_graph.nodes_by_distance] + [0]
        matplotlib.pyplot.plot(Ds, Ys)
        scale_up_factor = max(Ys)
        Ys_back_of_the_envelope_calculation = [scale_up_factor * math.prod(
            1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
            if j != 0 else 0 for j 
            in range(dist + 1, genotype_graph.dtype[1] + 1)
        ) for dist in Ds]
        matplotlib.pyplot.plot(Ds, Ys_back_of_the_envelope_calculation)
        matplotlib.pyplot.legend(
            ['Number of genotypes', 'Back of the envelope calculation']
        )
        matplotlib.pyplot.xlabel('Distance from reference genotype')
        matplotlib.pyplot.ylabel('Number of genotypes')
        matplotlib.pyplot.title(
            'Number of genotypes as a function of d \n'
            f'L = {graph_type[1]}, p = {p}'
        )
        second_yaxis = axes.secondary_yaxis('right', functions = (
            lambda x: x / scale_up_factor, 
            lambda x: x * scale_up_factor
        ))
        second_yaxis.set_ylabel('probability')
        matplotlib.pyplot.gcf().set_size_inches(16, 9)
        matplotlib.pyplot.savefig(
            f'num_geno_func_d.pdf', format = 'pdf', bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()

        # "number of peaks as a function of d"
        figure, axes = matplotlib.pyplot.subplots()
        Ys = [0] + [len([
            inode for inode in inodes if genotype_graph[inode].is_peak
        ]) for inodes in genotype_graph.nodes_by_distance] + [0]
        matplotlib.pyplot.plot(Ds, Ys)
        scale_up_factor = max(Ys)
        Ys_back_of_the_envelope_calculation = [scale_up_factor * math.prod(
            1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
            if j != 0 else 0 for j 
            in range(dist + 1, genotype_graph.dtype[1] + 1)
        ) for dist in Ds]
        matplotlib.pyplot.plot(Ds, Ys_back_of_the_envelope_calculation)
        matplotlib.pyplot.legend(
            ['Number of peaks', 'Back of the envelope calculation']
        )
        matplotlib.pyplot.xlabel('Distance from reference genotype')
        matplotlib.pyplot.ylabel('Number of peaks')
        matplotlib.pyplot.title(
            'Number of peaks as a function of d \n'
            f'L = {graph_type[1]}, p = {p}'
        )
        second_yaxis = axes.secondary_yaxis('right', functions = (
            lambda x: x / scale_up_factor, 
            lambda x: x * scale_up_factor
        ))
        second_yaxis.set_ylabel('probability')
        matplotlib.pyplot.gcf().set_size_inches(16, 9)
        matplotlib.pyplot.savefig(
            f'num_peak_func_d.pdf', format = 'pdf', bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()

        # "probability of transitioning to a peak as a function of d"
        num_nodes = len(genotype_graph)
        sorted_nodes = sorted(
            range(num_nodes), key = lambda x: genotype_graph[x].fitness
        )
        non_peaks_by_distance = [
            [inode for inode in inodes if not genotype_graph[inode].is_peak] 
            for inodes in genotype_graph.nodes_by_distance
        ]
        average_transition_probabilities_by_distance = [
            # These are transition probabilities for random walks, so checking 
            # if a neighbor is a peak also to checks that it's higher fitness 
            # and could be stepped to
            sum(
                len([
                    neighbor for neighbor in genotype_graph[inode].neighbors 
                    if genotype_graph[neighbor].is_peak
                ]) / len([
                    neighbor for neighbor in genotype_graph[inode].neighbors 
                    if (
                        genotype_graph[inode].fitness 
                        < genotype_graph[neighbor].fitness
                    )
                ]) for inode in inodes
            ) / len(inodes) if len(inodes) > 0 else 1 
            for inodes in non_peaks_by_distance
        ]
        return_value |= {
            'non_peaks' : non_peaks_by_distance,
            'transition_probs' : average_transition_probabilities_by_distance
        }
        matplotlib.pyplot.plot(
            Ds[1:-1], average_transition_probabilities_by_distance
        )
        Ys = [
            1 - 1 / (1 + ((genotype_graph.dtype[1] - 1) * p) / (d * (1 - p))) 
            if d != 0 else 1 for d in Ds[1:-1]
        ]
        matplotlib.pyplot.legend([
            'Probability of transitioning to a peak'
        ])
        matplotlib.pyplot.xlabel('Distance from reference genotype')
        matplotlib.pyplot.ylabel('probability')
        matplotlib.pyplot.ylim([-0.05, 1.05])
        matplotlib.pyplot.title(
            'Probability of transitioning to a peak\n'
            f'L = {graph_type[1]}, p = {p}'
        )
        matplotlib.pyplot.gcf().set_size_inches(16, 9)
        matplotlib.pyplot.savefig(
            'trans_prob_func_d.pdf', format = 'pdf', bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()

        # "peak percentile vs fraction of populations reaching at least this 
        # percentile"
        # "with back-of-the-envelope calculation drawn in"
        if 'other' in fitness_distribution:
            Ds = list(range(-1, int(graph_type[1] / 3) + 2))
            Xs = [(scipy.special.betaincc(
                genotype_graph.dtype[1] - dist, dist + 1, 1 / (2 - p)
            ) - scipy.special.betaincc(
                genotype_graph.dtype[1] - int(graph_type[1] / 3), 
                int(graph_type[1] / 3) + 1, 1 / (2 - p)
            )) / (1 - scipy.special.betaincc(
                genotype_graph.dtype[1] - int(graph_type[1] / 3), 
                int(graph_type[1] / 3) + 1, 1 / (2 - p)
            )) for dist in Ds[:-1]][::-1]
            Ys_back_of_the_envelope_calculation = [math.prod(
                1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
                if j != 0 else 0 for j 
                in range(dist + 1, int(graph_type[1] / 3) + 1)
            ) for dist in Ds[:-1]][::-1]
        else:
            Xs = [scipy.special.betaincc(
                genotype_graph.dtype[1] - dist, dist + 1, 1 / (2 - p)
            ) for dist in Ds[:-1]][::-1]
            Ys_back_of_the_envelope_calculation = [math.prod(
                1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
                if j != 0 else 0 for j 
                in range(dist + 1, genotype_graph.dtype[1] + 1)
            ) for dist in Ds[:-1]][::-1]
        matplotlib.pyplot.plot(
            Xs, Ys_back_of_the_envelope_calculation, c = 'blue'
        )
        matplotlib.pyplot.plot(
            random_access_cdf2[::-1], access_cdf2[plot_evo], c = 'red'
        )
        # Where peaks with equal fitness are grouped together and the percentile
        # of a peak is the percentile of the highest-percentile peak with the 
        # same fitess
        Xs2 = [val for val in Xs for _ in range(2)][1:]
        Ys2 = [
            val for val in Ys_back_of_the_envelope_calculation for _ in range(2)
        ][:-1]
        matplotlib.pyplot.plot(Xs2, Ys2, c = 'darkblue')
        matplotlib.pyplot.plot(
            unique_random_access_cdf2_compliment, unique_access_cdf2[plot_evo], 
            c = 'darkred'
        )
        # Where peaks with equal fitness are grouped together and the percentile
        # of a peak is the percentile of the lowest-percentile peak with the 
        # same fitess
        Xs3 = [0] + Xs2[:-1]
        Ys3 = Ys2[1:] + [0]
        matplotlib.pyplot.plot(Xs3, Ys3, c = 'lightblue')
        matplotlib.pyplot.plot(
            [0] + unique_random_access_cdf2_compliment[:-1], 
            unique_access_cdf3[plot_evo], c = 'lightcoral'
        )
        matplotlib.pyplot.legend([
            'Back of the envelope calculation', 'Simulation; random walk',
            'Highest percentile of group', 'Highest percentile of group',
            'Lowest percentile of group', 'Lowest percentile of group'
        ])
        matplotlib.pyplot.xlabel('Peak percentile')
        matplotlib.pyplot.ylabel(
            'Fraction of populations reaching at least this percentile'
        )
        matplotlib.pyplot.title(
            'Fraction of populations reaching at least a certain peak '
            f'percentile\nL = {graph_type[1]}, p = {p}'
        )
        matplotlib.pyplot.gcf().set_size_inches(16, 9)
        matplotlib.pyplot.savefig(
            f'pop_reach_peak_percen.pdf', format = 'pdf', bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()

        # "peak percentile vs fraction of populations reaching at least this 
        # percentile" version 2, no evolution
        # "with back-of-the-envelope calculation drawn in"
        if 'other' in fitness_distribution:
            Ds = list(range(-1, int(graph_type[1] / 3) + 2))
            Xs = [(scipy.special.betaincc(
                genotype_graph.dtype[1] - dist, dist + 1, 1 / (2 - p)
            ) - scipy.special.betaincc(
                genotype_graph.dtype[1] - int(graph_type[1] / 3), 
                int(graph_type[1] / 3) + 1, 1 / (2 - p)
            )) / (1 - scipy.special.betaincc(
                genotype_graph.dtype[1] - int(graph_type[1] / 3), 
                int(graph_type[1] / 3) + 1, 1 / (2 - p)
            )) for dist in Ds[:-1]][::-1]
            Ys_back_of_the_envelope_calculation = [math.prod(
                1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
                if j != 0 else 0 for j 
                in range(dist + 1, int(graph_type[1] / 3) + 1)
            ) for dist in Ds[:-1]][::-1]
        else:
            Xs = [scipy.special.betaincc(
                genotype_graph.dtype[1] - dist, dist + 1, 1 / (2 - p)
            ) for dist in Ds[:-1]][::-1]
            Ys_back_of_the_envelope_calculation = [math.prod(
                1 / (1 + (genotype_graph.dtype[1] - 1) * p / (j * (1 - p))) 
                if j != 0 else 0 for j 
                in range(dist + 1, genotype_graph.dtype[1] + 1)
            ) for dist in Ds[:-1]][::-1]
        matplotlib.pyplot.plot(
            Xs, Ys_back_of_the_envelope_calculation, c = 'blue'
        )
        plot_evo = (
            'none' if 'none' in evolution_types else evolution_types[0]
        )
        matplotlib.pyplot.plot(
            random_access_cdf2[::-1], access_cdf2[plot_evo], c = 'red'
        )
        # Where peaks with equal fitness are grouped together and the percentile
        # of a peak is the percentile of the highest-percentile peak with the 
        # same fitess
        Xs2 = [val for val in Xs for _ in range(2)][1:]
        Ys2 = [
            val for val in Ys_back_of_the_envelope_calculation for _ in range(2)
        ][:-1]
        matplotlib.pyplot.plot(Xs2, Ys2, c = 'darkblue')
        matplotlib.pyplot.plot(
            unique_random_access_cdf2_compliment, unique_access_cdf2[plot_evo], 
            c = 'darkred'
        )
        # Where peaks with equal fitness are grouped together and the percentile
        # of a peak is the percentile of the lowest-percentile peak with the 
        # same fitess
        Xs3 = [0] + Xs2[:-1]
        Ys3 = Ys2[1:] + [0]
        matplotlib.pyplot.plot(Xs3, Ys3, c = 'lightblue')
        matplotlib.pyplot.plot(
            [0] + unique_random_access_cdf2_compliment[:-1], 
            unique_access_cdf3[plot_evo], c = 'lightcoral'
        )
        matplotlib.pyplot.legend([
            'Back of the envelope calculation', 'Simulation; no evolution',
            'Highest percentile of group', 'Highest percentile of group',
            'Lowest percentile of group', 'Lowest percentile of group'
        ])
        matplotlib.pyplot.xlabel('Peak percentile')
        matplotlib.pyplot.ylabel(
            'Fraction of populations reaching at least this percentile'
        )
        matplotlib.pyplot.title(
            'Fraction of populations reaching at least a certain peak '
            f'percentile\nL = {graph_type[1]}, p = {p}'
        )
        matplotlib.pyplot.gcf().set_size_inches(16, 9)
        matplotlib.pyplot.savefig(
            f'pop_start_peak_percen.pdf', format = 'pdf', bbox_inches = 'tight'
        )
        matplotlib.pyplot.close()

    genotype_graph.close_log()
    # Dump more things into the return value
    return_value |= {
        'graph' : genotype_graph, 'accessibility' : accessibilities, 
        'house_of_cards' : null_model, 
        'high_fitness_population' : high_fitness_population,
        'peak_cdf_plot_data' : peak_cdf_plot_data
    }
    return return_value

if __name__ == '__main__':
    if '.py' in sys.argv[0]:
        sys.argv.pop(0)
    for arg in sys.argv:
        if len(arg) > 0 and '=' not in arg:
            raise RuntimeError(
                f'Argument "{arg}" is not valid input\n'
                + usage_str
            )
    cmd_line_args = {
        val.strip().split('=')[0] : val.strip().split('=')[1] 
        for val in sys.argv if len(val) > 0
    }
    for key, val in cmd_line_args.items():
        match key:
            case (
                'graph_type' | 'evolution_type' | 'evolution_algo' 
                | 'fitness_distribution'
            ):
                pass 
            case 'fitness_cutoff' | 'eta_cutoff':
                cmd_line_args[key] = float(val)
            case 'cutoff_type':
                match list(val.lower()):
                    case ['r', *_]:
                        cmd_line_args[key] = 'random'
                    case ['h', *_, 'r', 's']:
                        cmd_line_args[key] = 'high_fitness_neighbors'
                    case ['h', *_]:
                        cmd_line_args[key] = 'high_fitness'
                    case [*_, 'r', 's']:
                        cmd_line_args[key] = 'low_fitness_neighbors'
                    case _:
                        cmd_line_args[key] = 'low_fitness'
            case 'permute':
                match list(val.lower()):
                    case ['t', *_]:
                        cmd_line_args[key] = True
                    case ['a', 'a', *_]:
                        cmd_line_args[key] = 'aa27'
                    case ['a', *_]:
                        cmd_line_args[key] = 'after_cutoff'
                    case ['m', *_]:
                        cmd_line_args[key] = 'matrix'
                    case ['p', *_]:
                        cmd_line_args[key] = 'peaks'
                    case _:
                        cmd_line_args[key] = False
            case 'population_distribution':
                match list(val.lower()):
                    case ['u', *_]:
                        cmd_line_args[key] = 'uniform'
                    case ['l', *_]:
                        cmd_line_args[key] = 'lowest'
                    case _:
                        cmd_line_args[key] = 'uniform'
            case _:
                raise RuntimeError(
                    f'Argument "{key}" is not a valid paramter\n'
                    + usage_str
                )
    print(f'{cmd_line_args=}')
    result = main(**cmd_line_args)

