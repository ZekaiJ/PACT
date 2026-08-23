# Extended evidence snapshots

This directory preserves bounded summary outputs used beyond the released controlled study. Full no-media inputs and frozen logits for the balanced foundation-model panel are released separately in [`../balanced_fm_panel/`](../balanced_fm_panel/). Neither location contains model weights, third-party media, or private data.

## Public evidence topology

`n2_gates/` records the pairwise eligibility gate. Three dataset--emitter pairs passed the frozen native-protocol and per-view-export checks: HandWritten--Mfeat with TMC, HandWritten--Mfeat with RCML, and PIE with RCML. One Scene15--RCML pair failed native-performance reproduction; the remaining pairs lacked a verified processed payload and per-view export.

`n3_topology/` contains scale-free topology interventions for the three eligible pairs. Registered copies conserved the PACT evidence budget at multiplicity eight. False splits amplified it by 2.16x--3.21x, while false merges retained 0.0016x--0.030x of the view-distinct budget. These interventions test evidence accounting, not native-view independence or leaderboard performance.

Re-run an eligible frozen export with:

```bash
python experiments/topology_interventions.py --help
```

## Foundation-model boundary analyses

`n4_n5/` contains summary statistics for a fixed-image, three-family target-conditioning extension and a four-checkpoint error-dependence analysis. The target-conditioning result is a generator-specific boundary finding. The registered family partition was not preferential under the exact block-permutation test; it does not show that the declared family grouping recovers empirical dependence or family independence.

`FINAL_VERDICT.json` records the corrected bounded verdicts. The earlier 84-case diagnostic remains summary-only; the larger balanced panel provides the fully released frozen-output analysis.

## Reproduction exclusions

[`../public_pair_gates/scene15_rcml/`](../public_pair_gates/scene15_rcml/) records the final Scene15--RCML HOLD decision. The released code defines a Scene15 loader but no Scene15 training entry point or split file, and two available protocols do not reproduce the published accuracy. No topology result is reported for that pair.

## H2O aggregate evidence

[`../h2o_stage3/`](../h2o_stage3/) contains an aggregate source-role snapshot and a standard-library verifier. The retained generation scripts are archival and non-standalone; the directory does not provide command-level regeneration from H2O media and does not redistribute images, depth maps, or pose records.
