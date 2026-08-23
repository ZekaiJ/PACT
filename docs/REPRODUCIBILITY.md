# Reproducing the results

## Installation

```bash
python -m pip install -e .
```

Python 3.10 or later is required. `requirements-release.txt` records the packages used for the released analyses.

## Full reproducibility check

```bash
python analysis/verify_release.py
```

This command checks file integrity, source/label separation, record and scene counts, unit tests, controlled reference tables, matched secondary analyses, public multi-view outcomes, and released runtime summaries.

## Main controlled experiment

```bash
python experiments/pcecf_study.py --output outputs/pcecf_study
```

Thresholds are selected within scene-grouped outer folds. Selective risk is integrated over common support [0.10, 0.39], and uncertainty uses 2,000 scene-level bootstrap replicates. Reference outputs are stored in `results/reference/`.

## Stage and policy analyses

```bash
python analysis/minimal_attribution_2x2.py --output outputs/minimal_attribution_2x2
python analysis/score_verifier_factorial.py \
  --pcecf-output outputs/pcecf_study \
  --output outputs/score_verifier_factorial
python analysis/verify_secondary_results.py
```

## Topology and score-transport diagnostics

```bash
python results/topology_multiplicity/run.py
python results/operator_characterization/run.py
python results/equal_cardinality_topology/native_random_reference.py
python results/equal_cardinality_topology/run.py
python results/frozen_score_transport/full/run.py
python results/frozen_score_transport/no_count/run.py
python analysis/common_score_parity.py --repository . --output results/common_score_parity
python analysis/candidate_score_gap_decomposition.py
python analysis/joint_scene_configuration_holdout.py
python analysis/verify_recent_mechanism_results.py
```

These analyses reuse the controlled records, scene folds, and frozen study concentrations. The equal-cardinality control changes source pairing while holding the two-component count fixed. The transported shared-score models are fitted only at multiplicity one without coefficient refitting. The common-score parity analysis fits the shared correctness score on the four factorial methods and applies it to the two closest topology-aware comparators, which remain excluded from score training.

The exhaustive six-view coarsening outputs are released in `results/partition_coarsening_surface/`. Regeneration additionally requires the frozen per-view HandWritten/Mfeat evidence file whose SHA-256 is recorded in `configs/partition_coarsening_protocol.json`; that input is not duplicated in this compact release. The structural gate, implementation-equivalence check, complete 203-partition registry, 856-edge registry, and bootstrap summaries remain independently inspectable in the released result package.

## Public multi-view transfer

The released snapshot is in `results/public_outcome_closure/`. Recomputing it requires the frozen per-view evidence ledger:

```bash
python analysis/public_outcome_closure.py \
  --input /path/to/per_view_evidence.jsonl \
  --output outputs/public_outcome_closure
```

Every native view serves once as the duplicated donor. The paired bootstrap resamples dataset records and keeps all five frozen emitter realizations, donors, and intervention arms attached to each sampled record. ncsAURC is evaluated on pair-specific common support [0.10, 0.90]. PIE/RCML is not included in this outcome analysis because donor-level evidence was unavailable.

## Foundation-model and H2O evidence

The native-view foundation-model transfer is reproducible from released no-media logits, frozen prompt manifests, and episode-bootstrap analysis code; follow `results/native_view_fm_provenance_transfer/README.md`. The balanced foundation-model panel is likewise reproducible from released no-media logits; follow `results/balanced_fm_panel/README.md`. H2O media are not redistributed, but the released aggregate results can be checked with:

```bash
python analysis/habit_checkpoint_admission.py
python results/h2o_stage3/verify_released_aggregates.py
```

The HABIT checkpoint analysis consumes released no-media candidate fields and frozen conventional event-proximity outputs. It holds the 720 cases, target rule, event-proximity model, and threshold fixed across all three checkpoints; its README records the denominator and interpretation boundary.

Data provenance and redistribution boundaries are described in [`DATA.md`](DATA.md) and [`THIRD_PARTY_ASSETS.md`](THIRD_PARTY_ASSETS.md).

## Claim-to-artifact binding

`docs/CLAIM_ARTIFACT_MAP.csv` binds the principal numerical claims to frozen result files, artifact locators, denominators, statistical units, analysis scripts, and current SHA-256 values. Rebuild it after an intentional artifact change with:

```bash
python analysis/build_claim_artifact_map.py
```

The full verifier checks the frozen map rather than rebuilding it, so an unrecorded change to either a cited result file or its analysis script fails verification.
