#! /usr/bin/env bash

# Run this script to reproduce all of the results decribed in "A tractable model
# illustrates how local and stochastic adaptive dynamics can avoid low-ranking 
# peaks on navigable fitness landscapes"
# Note that this will probably take a long time, as all of the walks, basins, 
# and folA fitnesses will be recalculated from scratch; alternatively the 
# `*walks.json` and `*basins.tsv` files, and the landscapes in 
# `fola_replicate_fitnesses/` can be left in place to only regenerate the plots
# and logs from the existing data.
# Recommended versions of dependencies: 
#   Python 3.14
#   SciPy 1.17
#   NetworkX 3.6 (only required for rmf_plots.py)

# Written by Kye Hunter
# Last updated 20261118

set -o errexit -o errtrace -o pipefail

if [[
  $(python --version | cut -d' ' -f 2 | cut -d'.' -f 1) -ne 3
  || $(python --version | cut -d' ' -f 2 | cut -d'.' -f 2) -lt 12
]] then
  echo "You need to use a version of Python 3 >= v3.12"
  exit 1
fi

rm -rf ./paper_figures_and_data
mkdir paper_figures_and_data
cd paper_figures_and_data
ln -s ../code/*.py .
ln -s ../external_data_sets/*/*.csv .
ln -s ../external_data_sets/*/*.tsv .

./rmf_plots.py
./fig5_gen.py
./figS18_gen.py

# All the rest is for Fig S19
mkdir fola_replicate_fitnesses
cd fola_replicate_fitnesses
ln -s ../../code/*.py .
ln -s ../../external_data_sets/*/*.RDS .

./make_job_list.py
for replicate_set in $(cat job_list.txt); do
  ./fitness_difference_significance.py $replicate_set
done
./fola_ruggedness_inference.py


exit 0

