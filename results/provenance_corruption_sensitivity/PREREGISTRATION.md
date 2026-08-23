# A2 preregistration: lineage-topology perturbations

This file fixes the analysis before the v1956 A2 outputs are generated.

## Unit and grid

- Runtime denominator: 31,200 frozen decision records.
- Corruption assignment: 240 scene-cue clusters nested in 48 scenes.
- Corruption rates: 0%, 10%, 25%, 50%, 75%, and 100%.
- Perturbation seeds: 10.
- Uncertainty: scene bootstrap with perturbation-seed resampling.
- Comparator: the dependence-blind, no-lineage operating point from the same
  frozen runtime and fold-specific settings.

## Perturbations and directional predictions

1. **False split.** A valid shared component is declared as separate components.
   Risk and expected cost are expected to move toward the no-lineage baseline as
   the affected fraction increases.
2. **False merge.** Distinct components are incorrectly joined. Coverage is
   expected to decrease; the direction of wrong-admission/all is not prescribed.
3. **Hidden-edge deletion.** A declared edge is removed while the intact
   registered manifest remains available. The manifest is expected to recover
   the same partition and preserve the operating point.
4. **Forged independence.** Each source presents an internally consistent but
   false independent lineage. This condition is observationally unidentifiable
   at the registered interface. The experiment quantifies damage only and must
   not be described as detecting forged independence.

## Outcomes and interpretation

Primary outputs are pairwise partition F1, coverage, wrong-admission/all,
conditional wrong-admission risk, correct-admission/all, and expected operating
cost. Type-specific crossovers against no lineage are reported only when reached
on the fixed grid; otherwise they are marked not identifiable within the grid.
No threshold, corruption rate, or interpretation will be changed after results
are observed.
