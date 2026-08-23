# Data and released evidence

## Controlled source records

`data/controlled/source_records.jsonl.gz` contains 31,200 project-generated decision records derived from 48 controlled rendered scenes. The records are deterministic source-state variants nested within scenes, not 31,200 independent observations.

Each record contains only fields used by fusion or admission: identifiers, five-class source opinions, source quality and conflict, availability and currentness state, and provenance parent sets. `data/controlled/evaluation_labels.jsonl.gz` stores preferred contracts and evaluation descriptors separately. Predictions are formed before labels are joined by `record_id`.

The five contract classes are `normal`, `slow_clearance`, `hold_confirm`, `retreat_fallback`, and `bounded_urgent`. `configs/controlled_study.json` records the uncompressed stream hashes.

## Public and foundation-model snapshots

- `results/public_outcome_closure/` contains the donor-complete HandWritten/Mfeat TMC and RCML outcome readout. PIE/RCML remains budget-only because donor-level evidence was unavailable.
- `results/balanced_fm_panel/` contains no-media case identifiers, forced-choice logits, protocol locks, and derived analyses for eight checkpoints. Original HABIT images are absent.
- `results/native_view_fm_provenance_transfer/` contains frozen prompt and frame manifests, no-media Qwen3-VL forced-choice logits, episode-level analysis records, bootstrap results, and the publication figure for the five-camera native-view study. Original HABIT images are absent.
- `results/h2o_stage3/` contains aggregate H2O summaries and an integrity verifier. Original RGB-D, pose, and video records are absent.
- `results/public_pair_gates/` retains both successful and failed dataset-emitter eligibility decisions, including the Scene15/RCML HOLD.

## Redistribution boundary

Original HABIT, H2O, feature-view datasets, model weights, and upstream TMC/RCML implementations are not redistributed. Their provider terms continue to apply. See `docs/THIRD_PARTY_ASSETS.md` and `docs/RELEASE_SCOPE.md`.
