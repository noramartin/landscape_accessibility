#! /usr/bin/env python

#     geometric_mean.py

# Calculate a geometric mean

# Written by Kye Hunter
# Last updated 20260212

import math

def geometric_mean(data):
    return math.exp(sum(math.log(val) for val in data) / len(data))

def geometric_mean_safe(data):
    '''A geommetric mean that avoids an error due to log(0).'''
    tmp = math.prod(data)
    if tmp == 0:
        return 0
    else:
        return math.exp(sum(math.log(val) for val in data) / len(data))

def geometric_mean_offset(data, offset = 0):
    return math.exp(
        sum(math.log(val + offset) for val in data) / len(data)
    ) - offset

def geometric_mean_with_zeros(data, tolerance = 1e-5):
    '''Caclculate an approximate geometric mean, giving 0s a tunable weight.

    This implimentation is not totally correct: The optimization loop is overly 
    simple.
    https://doi.org/10.48550/arXiv.1806.06403
    '''
    non_zero_data = [val for val in data if val != 0]
    standard_gm = geometric_mean(non_zero_data)
    threshold = tolerance * standard_gm
    offset = 1
    offset_gm = geometric_mean_offset(non_zero_data, offset)
    while offset_gm - standard_gm >= threshold:
        offset /= 2
        offset_gm = geometric_mean_offset(non_zero_data, offset)
    return geometric_mean_offset(data, offset), offset

