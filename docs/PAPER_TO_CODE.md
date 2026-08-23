# From paper to code

| Manuscript component | Entry point | Outputs |
|---|---|---|
| PACT operator | `action_admission.pact_fuse` | posterior, non-vacuity, component budgets |
| Operator properties | `python -m unittest tests.test_pact -v` | exact-copy invariance, near-copy stability, component completeness, hidden-parent boundary |
| Minimal example | `python examples/pact_quickstart.py` | three-source PACT output |
| Fusion-only and matched-admission studies | `python experiments/pcecf_study.py` | `results/reference/` |
| Matched fusion/admission comparison | `python analysis/minimal_attribution_2x2.py` | `results/minimal_attribution_2x2/` |
| Score and verifier-policy sensitivity | `python analysis/score_verifier_factorial.py` | `results/score_verifier_factorial/` |
| Fixed-support and repeated-split analysis | checked by `python analysis/verify_release.py` | `results/p0_estimand_closure/v1/` |
| Topology multiplicity | `python results/topology_multiplicity/run.py` | `results/topology_multiplicity/` |
| Within-component meet-versus-join diagnostic | `python results/operator_characterization/run.py` | `results/operator_characterization/` |
| Equal-cardinality topology control and random-order reference | `python results/equal_cardinality_topology/run.py` | `results/equal_cardinality_topology/` |
| Candidate-score gap decomposition | `python analysis/candidate_score_gap_decomposition.py` | `results/equal_cardinality_topology/candidate_score_gap_decomposition.json` |
| Frozen shared-score transport | `python results/frozen_score_transport/full/run.py` and `python results/frozen_score_transport/no_count/run.py` | `results/frozen_score_transport/` |
| Joint scene--configuration holdout | `python analysis/joint_scene_configuration_holdout.py` | `results/joint_scene_configuration_holdout/` |
| Public multi-view transfer | `python analysis/public_outcome_closure.py` | `results/public_outcome_closure/` |
| Exhaustive six-view coarsening surface | released snapshot; see its README for the frozen-input boundary | `results/partition_coarsening_surface/` |
| Provenance-corruption sensitivity | released snapshot and archival analysis | `results/provenance_corruption_sensitivity/` |
| Opinion-interface and evidence-response sensitivity | released snapshots | `results/opinion_interface_sensitivity/`, `results/evidence_response_sensitivity/` |
| Learned Set Transformer comparison | released snapshot | `results/learned_set_fusion_comparison/` |
| Native-view foundation-model transfer | `results/native_view_fm_provenance_transfer/analysis/analyze.py` | `results/native_view_fm_provenance_transfer/` |
| Foundation-model panel | `results/balanced_fm_panel/protocol/analyze_fm_panel.py` | `results/balanced_fm_panel/analysis/` |
| HABIT fixed-image and sequential admission | released no-media outputs and analysis scripts | `results/habit_fixed_image_admission/` |
| HABIT checkpoint-robust target-plus-event admission | `python analysis/habit_checkpoint_admission.py` | `results/habit_checkpoint_admission/` |
| Claim-to-artifact binding | `python analysis/build_claim_artifact_map.py` | `docs/CLAIM_ARTIFACT_MAP.csv` |
| Full reproducibility check | `python analysis/verify_release.py` | generated verification JSON |

Predictions are generated before evaluation labels are joined. The 31,200 controlled records are deterministic variants nested within 48 scenes; scene is the statistical unit for the main bootstrap. The public multi-view analysis uses class-stratified record-paired resampling and keeps all frozen seeds, donors, and intervention arms attached to each sampled record.

Some frozen artifact paths retain the earlier identifier `pcecf` so recorded hashes and reproduction paths remain stable.
