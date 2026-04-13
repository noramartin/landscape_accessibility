#! /usr/bin/env bash

# Run this script to reproduce all of the results decribed in "Using a 
# simplified Rough Mount Fuji model to disentangle how multi-peaked fitness 
# landscapes can be highly navigable"
# Note that this will probably take a long time, as all of the walks and basins 
# will be recalculated from scratch; alternatively the `*walks.json` and 
# `*basins.tsv` files can be left in place to only regenerate the plots and logs
# from the existing data.
# Recommended versions of dependencies: 
#   Python 3.14
#   SciPy 1.17
#   NetworkX 3.6 (only required for rmf_plots.py)

# Written by Kye Hunter
# Last updated 20260413

set -o errexit -o errtrace -o pipefail

if [[
  $(python --version | cut -d' ' -f 2 | cut -d'.' -f 1) -ne 3
  && $(python --version | cut -d' ' -f 2 | cut -d'.' -f 2) -lt 12
]] then
  echo "You need to use a version of Python 3 >= v3.12"
  exit 1
if

rm -rf ./paper_figures_and_data
mkdir paper_figures_and_data
cd paper_figures_and_data
ln -s ../code/*.py .
ln -s ../external_data_sets/*/*.csv .
ln -s ../external_data_sets/*/*.tsv .

./rmf_plots.py
./fig4_gen.py
./fig5_gen.py
./figS3_gen.py
./figS5_gen.py
./figS6_gen.py

exit 0

