# Provenance-assignment corruption sensitivity

This snapshot records the controlled perturbation used to quantify how PACT
responds when a declared provenance partition is corrupted. Numerical source
opinions remain fixed. Corruption is assigned at scene--cue clusters, and the
four prespecified interventions are false separation, false merging, deletion
of a recoverable edge, and internally consistent but incorrect distinct-parent
assignment.

At 50% corruption, false separation and incorrect distinct-parent assignment
produce a wrong-admission rate of 0.0492, versus 0.0106 after false merging or
recoverable-edge deletion and 0.0875 without provenance grouping. The 0.0492
value is 43.8% below the no-provenance reference. False separation and
incorrect distinct-parent assignment are observationally identical at this
three-source binary interface because both induce the all-distinct partition.

The released aggregates support a sensitivity analysis, not provenance
authentication or error detection. `run.py` is retained as an archival
generation record; it depends on the larger controlled-runtime workspace and
is not advertised as a standalone reproduction entry point in this package.

Files:

- `PREREGISTRATION.md`: frozen estimands, grid, and pass/fail rules.
- `REPORT.md`: concise result interpretation and boundary.
- `risk_coverage_by_corruption.csv`: primary rate-by-intervention estimates and
  bootstrap intervals.
- `result.json`: denominators, validation checks, and input hashes.
- `no_lineage_baseline.json`: provenance-unaware reference.
- `type_specific_crossovers.csv`: prespecified crossover diagnostics.
- `run.py`: archival generation script.
- `lineage_corruption_sensitivity.pdf/png`: paper-facing visualization.
