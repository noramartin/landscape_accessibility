
Data from:
"Poelwijk, F.J., Socolich, M. & Ranganathan, R. Learning the pattern of epistasis linking genotype and phenotype in a protein. Nat Commun 10, 4213 (2019)." 
https://doi.org/10.1038/s41467-019-12130-8

The raw data can be found at: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-12130-8/MediaObjects/41467_2019_12130_MOESM8_ESM.xlsx
where it is licensed CC-BY 4.0

Reprocessed data can be found at: https://doi.org/10.5281/zenodo.8307147

The file `41467_2019_12130_MOESM8_ESM.xlsx` is provided as-is from static-content.springer.com
The files `Genotypes.txt` and `Phenotypes.txt` are provided as-is from the Zenodo repository,
while `poelwijk_2019_FP.tsv` contains the the data from `Genotypes.txt` and `Phenotypes.txt` combined in TSV format.

In `poelwijk_2019_FP.tsv`, the sequences (first column) are 13 binary loci indicating the mutations which distinguish the eqFP578/eqFP611 derivatives mTagBFP2 and mKate2, from Entacmaea quadricolor; the mutations indicate the presence or absence of the following 13 changes: D20N, V45A, L63M, T127P, F143S, N158A, S168G, A172C, A174L, Y197R, N206D, N207K, K231R. The fitness values (second column) are composite "brightness" measurements which combine the red and blue fluorescence channels as the sum of squared brightnesses of each channel, while also accounting for other differences between the red annd blue fluorescence and other sources of error.
The fitness values in `Phenotypes.txt` can be calculated from the two replicates reported by Poelwijk et al. by renormalizing the second replicate with a factor of 1.3130245, and then taking the mean of the replicates, as described in Park, Y., Metzger, B.P.H. & Thornton, J.W. Nat Commun 15, 7953 (2024). (https://doi.org/10.1038/s41467-024-51895-5).

