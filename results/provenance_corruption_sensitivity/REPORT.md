# A2 lineage-perturbation analysis

Decision: PASS

This frozen full-replay analysis covers 31,200 decision records nested in
48 scenes. Corruption is assigned at the 240
scene-cue clusters; uncertainty intervals use 1,000 scene-level
bootstrap draws with perturbation-seed resampling.

At 50% corruption, wrong-admission/all was
0.0106 after hidden-edge
deletion with an intact registered manifest,
0.0492 under false splitting,
0.0106 under false merging, and
0.0492 under internally
consistent forged independence. The no-lineage value was
0.0875.

False splitting and forged independence are numerically identical in this
three-source binary-lineage interface: both induce the all-distinct partition.
This is an observability result, not a detector. In particular, internally
consistent forged independence cannot be detected from the registered interface;
the reported curve quantifies its damage only.

Pairwise partition F1 and type-specific crossovers are reported in
`risk_coverage_by_corruption.csv` and `type_specific_crossovers.csv`. A crossover
is marked unavailable when the corrupted method does not reach the no-lineage
baseline anywhere on the prespecified rate grid.
