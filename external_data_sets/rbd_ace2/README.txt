
Data from:
"Moulana, A., Dupic, T., Phillips, A.M. et al. Compensatory epistasis maintains ACE2 affinity in SARS-CoV-2 Omicron BA.1. Nat Commun 13, 7011 (2022)."
https://doi.org/10.1038/s41467-022-34506-z

The data can be found at: https://github.com/desai-lab/compensatory_epistasis_omicron/blob/main/Titeseq/results/Kds/cleaned_Kds_RBD_ACE2.tsv
where it is licensed with the MIT license (c) 2022 Thomas Dupic.

The data in `cleaned_Kds_RBD_ACE2.tsv` is provided as-is from the GitHub repository, 
while `moulana_2022_RBD_ACE2.tsv` has been reduced to only the first ("geno") and last ("log10Kd_pinned") columns, and rows without a value in the "log10Kd_pinned" column have been removed.

The sequences (first column) are 15 binary loci which indicate the presence or absence of the following mutations in the spike protein receptor-binding domain of SARS-CoV-2: G339D, S371L, S373P, S375F, K417N, N440K, G446S, S477N, T478K, E484A, Q493R, G496S, Q498R, N501Y, Y505H. The fitness values (second column) are the logarithm of the K_d calculated for the binding of RBD with human ACE2, averaged across 3 replicates.

Note that these data should not be confused with the similar dataset published in 2023: https://github.com/desai-lab/omicron_ab_landscape

