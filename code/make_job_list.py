#! /usr/bin/env python

#     make_job_list.py

# Written by Kye Hunter
# Last updated 20261021

import itertools

num_reps = 6
jobs = [
    '_'.join(str(val) for val in comb) 
    for length in range(1, num_reps + 1)
    for comb in itertools.combinations(range(num_reps), length)
]

open('job_list.txt', 'wt').write('\n'.join(jobs))

