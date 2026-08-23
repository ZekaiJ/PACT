# H2O aggregate evidence snapshot

This directory releases aggregate outputs for the real egocentric--allocentric
evidence-role analysis. It is not a command-level raw-data reproduction
package. It does not redistribute H2O images, depth maps, pose records,
decision-level tables, source cards, or unreleased dependencies used during
generation.

The available source payload was the H2O `subject1_ego_v1_1` head-mounted
RGB-D stream. The allocentric role was obtained by applying the dataset-native
camera-to-world transform to the same frame-level geometry and was registered
as dependent evidence. It was not a second fixed-camera image stream.

## Verify the released snapshot

From the repository root, run:

```bash
python results/h2o_stage3/verify_released_aggregates.py
```

The verifier uses only the Python standard library. It checks the local hash
manifest, required CSV schemas and row sets, denominators, monotonicity counts,
and consistency among the released JSON gates. A pass verifies integrity and
internal consistency of this aggregate snapshot only; it does not regenerate
the analysis from H2O media.

## Archival generation records

The following files are retained as provenance records of the original
pipeline and are deliberately **non-standalone** in this public release:

- `run_v1857_h2o_egocentric_allocentric_admission.py`
- `run_v1858_stage3_integrity_audit.py`
- `run_v1859_stage3_threshold_sensitivity.py`
- `run_v1860_h2o_unified_quality_shifts.py`

They reference provider-controlled H2O inputs, unreleased decision-level
intermediates, and internal modules/paths that are absent here. They are not
advertised as runnable reproduction commands.

The snapshot reports source-role admission, view-removal monotonicity,
threshold sensitivity, and unified source-quality shifts. It contains no
contract-accuracy, physical-robot, natural-command, or participant-safety
labels. Provider terms for `subject1_ego_v1_1` require human verification; see
[`../../docs/THIRD_PARTY_ASSETS.md`](../../docs/THIRD_PARTY_ASSETS.md).
