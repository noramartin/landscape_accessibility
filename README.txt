
Code, figures, and external data sets for the article "Using a simplified Rough Mount Fuji model to disentangle how multi-peaked fitness landscapes can be highly navigable".

Directory structure:
.
├── code                   # All code used to generate the data and figures
├── external_data_sets     # Source and reformatted data for empirical fitness landscapes
│   ├── fola               # Papkou 2023
│   ├── fp                 # Poelwijk 2019
│   ├── gb1                # Wu 2016
│   ├── rbd_ace2           # Moulana 2022
│   ├── sd_arti            # Kuo 2020
│   └── trna               # Domingo 2018
└── paper_figures_and_data # Figures, logs, logged walks, and basins for each landscape
    └── plots              # Figures 1A, 1C, 2, 3, S1, S2

In principle, all data and figures are reproducilbe with the script `regen_all_results.sh`, however this may take a long time, and it may be better to run the analysis scripts one at a time.

Logs of the random walks and basins for each landscape studied are provided in JSON and TSV format respectively, but must first be decompressed with the `xz` program. 

All code and data in this repository are licensed CC-BY 4.0, except where other licenses apply.

