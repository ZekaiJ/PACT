# Learned source-set fusion comparison

This snapshot records the supervised Ordered-MLP, Deep-Sets, and
Set-Transformer controls used to distinguish scene interpolation from transfer
to a jointly held-out degradation configuration. The Set Transformer attains
zero ncsAURC under scene-only holdout but 0.5579 on fixed support
`[0.10, 0.35]` under joint scene--configuration holdout. This comparison is
bounded to the controlled generator configurations and does not establish
open-world robustness.

`gate.json` is the primary claim-bearing record and `comparison_table.csv`
contains the summary rows. The scripts are retained as archival generation
records and require the broader experiment workspace.
