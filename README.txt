
Code, figures, and external data sets for the article "A tractable model illustrates how local and stochastic adaptive dynamics can avoid low-ranking peaks on navigable fitness landscapes".

Directory structure:
.
├── code                         # All code used to generate the data and figures
├── external_data_sets           # Source and reformatted data for empirical fitness landscapes
│   ├── fola                     # Papkou 2023
│   ├── fp                       # Poelwijk 2019
│   ├── gb1                      # Wu 2016
│   ├── rbd_ace2                 # Moulana 2022
│   ├── sd_arti                  # Kuo 2020
│   └── trna                     # Domingo 2018
└── paper_figures_and_data       # Logs, logged walks, and basins for each landscape; figures 5 and S18
    ├── fola_replicate_fitnesses # Landscapes calculated from subsets of experimental replicates; figure S19
    └── plots                    # All other figures

In principle, all data and figures are reproducilbe with the script `regen_all_results.sh`, however this may take a long time, and it may be better to run the analysis scripts one at a time.

Logs of the random walks and basins for each landscape studied are provided in JSON and TSV format respectively, but must first be decompressed with the `xz` program. 

All code and data in this repository are licensed CC-BY 4.0, except where other licenses apply.

