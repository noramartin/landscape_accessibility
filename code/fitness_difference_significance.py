#! /usr/bin/env python

# Written by Kye Hunter
# Last updated 20261008

# For a given variant, and a set of replicates, calculate whether the fitness 
# difference with all other variants is significant

import warnings, sys
import rdata, statsmodels, numpy, matplotlib.pyplot
import statsmodels.api, statsmodels.formula.api, statsmodels.tools.sm_exceptions
warnings.simplefilter(
    'ignore', statsmodels.tools.sm_exceptions.PerfectSeparationWarning
)

matplotlib.rcParams.update({                                                     
    'font.size':  22,                                                            
    'font.family' : 'serif',                                                     
    'font.sans-serif': 'cm',                                                     
    'mathtext.default': 'it',                                                    
    'mathtext.fontset': 'cm',                                                    
    'figure.figsize': (16,15)                                                     
}) 

## Load data
# The SI of Papkou 2023 says '3.51', but to get fitnesses that (nearly) 
# match what they report, it needs to be slightly larger (by more than 
# rounding error, surprisinly)
# In Papkou's second zenodo, it says that generation time is 3.51662; this 
# is probably also still a truncated value
generations = 3.51662
# Wildtype
variant1 = 'GCCGATCTC'
seq_wrapper = [
    'GCGGTAGATCGCGTTATCGGCATGGAAAACGCCATGCCGTGGAACCTGCCT', 
    'GCCTGGTTTAAACGCAACACCTTAAATAAACCCGTGATTATGGGCCGCCATACC'
]

counts_data = rdata.read_rds('sequence_reads_counts.RDS')

# These tables have 6 columns, one for each replicate
before_total_counts = counts_data['before'].sum(dim='dim_1')
after_total_counts = counts_data['M9+TMP'].sum(dim='dim_1')

before_var1_counts = counts_data['before'][
    :, counts_data['before'].dim_1.values.tolist().index(
        seq_wrapper[0] + variant1 + seq_wrapper[1]
    )
]
after_var1_counts = counts_data['M9+TMP'][
    :, counts_data['M9+TMP'].dim_1.values.tolist().index(
        seq_wrapper[0] + variant1 + seq_wrapper[1]
    )
]

usable_sequences = sorted(
    set(counts_data['before'].dim_1.values.tolist()) 
    & set(counts_data['M9+TMP'].dim_1.values.tolist())
)

# Precompute more stuff before the big loop
before_sequence_dict = {
    val : ival for ival, val 
    in enumerate(counts_data['before'].dim_1.values.tolist())
}
after_sequence_dict = {
    val : ival for ival, val 
    in enumerate(counts_data['M9+TMP'].dim_1.values.tolist())
}


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

def write_csv(data, filename) -> None:
    open(filename, 'w').write('\n'.join([
        ','.join(str(val) for val in row)
        for row in data
    ]))

def calc_fitness_difference_n_replicates(replicate_list):
    global generations, variant1, seq_wrapper, counts_data, before_total_counts
    global after_total_counts, before_var1_counts, after_var1_counts
    global usable_sequences, before_sequence_dict, after_sequence_dict

    num_reps = len(replicate_list)
    var1_counts = (
        *before_var1_counts[replicate_list], 
        *after_var1_counts[replicate_list]
    )
    single_total_counts = numpy.array((
        *before_total_counts[replicate_list], 
        *after_total_counts[replicate_list]
    ))
    double_total_counts = (
        *before_total_counts[replicate_list], 
        *after_total_counts[replicate_list]
    ) * 2
    growth_counts = numpy.zeros((num_reps * 4, 2))
    growth_counts[:, 1] = double_total_counts
    growth_counts[:num_reps * 2, 0] = var1_counts
    growth_counts[:num_reps * 2, 1] -= var1_counts
    
    dummy = numpy.array([0] * (num_reps * 2) + [1] * (num_reps * 2))
    time = numpy.array(([0] * num_reps + [generations] * num_reps) * 2)
    data_dict = {
        'growth_counts' : growth_counts, 'time' : time, 'dummy' : dummy
    }
    
    binomial_family = statsmodels.api.families.Binomial()
    result = [0] * len(usable_sequences)
    for iseq, seq in enumerate(usable_sequences):
        if not iseq % 10000:
            print(f'iter {iseq}')
        data_dict['growth_counts'][
            (num_reps * 2):(num_reps * 3), 0
        ] = counts_data['before'][replicate_list, before_sequence_dict[seq]]
        data_dict['growth_counts'][(num_reps * 3):, 0] = counts_data['M9+TMP'][
            replicate_list, after_sequence_dict[seq]
        ]
        data_dict['growth_counts'][(num_reps * 2):, 1] = (
            single_total_counts - data_dict['growth_counts'][(num_reps * 2):, 0]
        )
    
        model = statsmodels.formula.api.glm(
            'growth_counts ~ 1 + dummy + time + I(dummy * time)', data_dict, 
            family = binomial_family
        ).fit()
    
        fitness = model.params['I(dummy * time)']
        std_err = model.cov_params(
        )['I(dummy * time)']['I(dummy * time)'] ** 0.5
        pvalue = model.wald_test_terms().pvalues[-1].item()
        result[iseq] = (seq[51:60], fitness, std_err, pvalue)
    
    result = list(zip(*result))
    # The "Benjamini-Hochberg false discovery rate method"
    result[3] = statsmodels.stats.multitest.fdrcorrection(result[3])[1].tolist()
    result = list(zip(*result))
    print(result[0:10])
    
    write_csv(
        result, 
        f'fitnesses_own_calculation_ref_{variant1}_rep_{
            '_'.join(str(val) for val in replicate_list)
        }.csv'
    )

def main(replicate_list : list):
    global usable_sequences
    
    # Filtering for a minimum number of reads in 6 replicates was already 
    # applied to these data, but when picking a subset of replicates, some 
    # genotypes can still end up with 0 reads in the subset, and need to be 
    # removed
    usable_sequences = [seq for seq in usable_sequences if (not all(
        counts_data['before'][replicate_list, before_sequence_dict[seq]] == 0
    )) and (not all(
        counts_data['M9+TMP'][replicate_list, after_sequence_dict[seq]] == 0
    ))]

    calc_fitness_difference_n_replicates(replicate_list)

if __name__ == "__main__":
    main([int(val) for val in sys.argv[1].strip().split('_')])


