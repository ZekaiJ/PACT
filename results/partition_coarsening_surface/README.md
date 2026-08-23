# Exhaustive six-view PACT coarsening surface

This result package evaluates all 203 set partitions of the six frozen
HandWritten/Mfeat feature views for TMC and RCML. The directed partition cover
graph contains 856 one-merge edges. The analysis is fixed by
`configs/partition_coarsening_protocol.json` and executed by
`analysis/partition_coarsening_surface.py`.

## Main readout

The evidence-budget implementation gate passes on every record, frozen model
realization, and cover edge: merging two registered components never increases
the PACT budget. The empirical selective order is not fully determined by this
budget order. With native non-vacuity, coarsening improves ncsAURC on 17.1% of
TMC cover edges (95% CI 8.5--20.2%) and 18.7% of RCML edges (7.0--25.1%). The
prespecified posterior-confidence analysis gives 9.3% (4.0--12.7%) and 6.2%
(4.0--8.5%), respectively. At component count two, the native-score ncsAURC
range across partitions is 0.5385 for TMC and 0.0490 for RCML.

No exact score ties occur in any partition--realization curve. The vectorized
sweep matches the released PACT implementation within the locked float64
tolerance.

## Scope

The declared policy is the unique six-singleton (finest) partition. This study
therefore maps PACT's response to registered coarsenings; it does not select or
authenticate a provenance partition. It also does not test false splitting,
which remains the separate registered-copy intervention, and it does not extend
the 203-partition analysis to registration-blind comparators.

## Files

- `PARTITION_REGISTRY.csv`: canonical restricted-growth representation of all
  203 partitions.
- `COVER_EDGE_REGISTRY.csv`: all 856 fine-to-coarse one-merge edges.
- `PARTITION_METRICS.csv`: partition-level budget, posterior, accuracy,
  selective-risk, and tie diagnostics.
- `COVER_EDGE_OUTCOMES.csv`: point contrast for each cover edge.
- `COVER_EDGE_SUMMARY.csv`: paired record-bootstrap edge-direction summaries.
- `DISCLOSURE_SUMMARY.json`: deterministic trace of the counts, direction
  fractions, and effect magnitudes reported in the manuscript.
- `WITHIN_K_DISPERSION.csv`: within-component-count IQR, range, and MAD.
- `STRUCTURAL_GATE.json`: budget-order implementation check.
- `IMPLEMENTATION_EQUIVALENCE_GATE.json`: vectorized-sweep equivalence to the
  released PACT implementation.
- `RUN_MANIFEST.json`: frozen input, script, protocol, denominator, and runtime
  identifiers.
- `partition_coarsening_surface.pdf/png`: paper-facing visualization.
