# Joint scene--configuration holdout

This analysis excludes one source condition and one outer scene fold from each
evaluation cell. Score thresholds are fitted without the held condition or test
scenes, and the main-study concentration schedule is transferred without
reselection. The fixed comparison support is `[0.10, 0.35]`.

Across thirteen held-condition evaluations, aggregate ncsAURC is 0.0617 for
complete PACT and 0.1168 for nested Dirichlet composition. Four conditions have
scene-level curves spanning the complete fixed interval and therefore support
condition-wise contrasts. The signs split evenly: PACT is lower under missing
geometry with low-quality language and partial low-quality disagreement;
nested composition is lower under noisy context with missing language and
overconfident contradictory sources. The remaining conditions are retained as
unsupported for this fixed-interval contrast rather than extrapolated.

Run from the repository root:

```bash
python analysis/joint_scene_configuration_holdout.py
```

`gate.json` freezes the denominator, statistical unit, support, and input
hashes. The four CSV files contain aggregate estimates, eligible and ineligible
condition rows, paired scene-bootstrap draws, and fold-level selection records.
