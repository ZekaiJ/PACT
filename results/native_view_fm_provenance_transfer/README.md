# Native-view foundation-model provenance transfer

This snapshot tests whether repeated estimates of one camera observation should count like evidence from additional physical cameras. It uses 1,128 held-out HABIT events from 696 episodes, six tasks, five physical views, two event windows, and four prompt surfaces. Qwen3-VL-32B is the primary checkpoint and Qwen3-VL-8B is a frozen replication.

Each prompt surface from the same event, window, and physical camera shares one registered parent. The five prespecified arms are acquisition-registered PACT, lineage-unaware singleton counting, exact output deduplication, all-view merge, and one-output-per-view native reference. Episode is the bootstrap unit. A locked analysis amendment corrects semantic ties in analytically constant-budget arms. It also labels a posterior-confidence readout and an equal-cardinality shuffled grouping as post-hoc diagnostics.

## Tie correction and primary result

Each emitted binary opinion has evidence budget two. At fixed multiplicity, lineage-unaware counting therefore assigns every record budget `10m`; one output per view assigns budget `10`. Their native non-vacuity scores are exact ties. The original implementation let floating-point summation noise order these records despite the frozen protocol's fractional tie rule. `ANALYSIS_AMENDMENT_TIE_V1.json` canonicalizes semantic budgets before ranking and changes no model output, label, coverage support, or resampling unit.

The primary contrast is lineage-unaware ncsAURC minus acquisition-registered ncsAURC on common coverage support [0.10, 0.90]. Positive values favor acquisition registration.

| Checkpoint | Prompt surfaces per view | Native-score contrast | 95% episode-bootstrap CI |
|---|---:|---:|---:|
| Qwen3-VL-32B | 1 | 0.0000 | [0.0000, 0.0000] |
| Qwen3-VL-32B | 2 | 0.0428 | [0.0079, 0.0694] |
| Qwen3-VL-32B | 4 | 0.0224 | [-0.0131, 0.0561] |
| Qwen3-VL-8B | 1 | 0.0000 | [0.0000, 0.0000] |
| Qwen3-VL-8B | 2 | 0.1728 | [0.1490, 0.1988] |
| Qwen3-VL-8B | 4 | 0.1202 | [0.0893, 0.1495] |

At four surfaces, the post-hoc posterior-confidence contrast is 0.0180 [0.0069, 0.0272] for 32B and -0.0156 [-0.0357, 0.0024] for 8B. The native contrast is not monotone in prompt multiplicity, task-level signs are heterogeneous, and checkpoint and score readouts disagree. The frozen evidence geometry changes under registration, but the snapshot does not support checkpoint-universal selective, calibration, or accuracy dominance.

The equal-cardinality shuffled grouping keeps five groups of `m` outputs while cyclically permuting camera assignments. It is a post-hoc topology control, not a sixth preregistered arm.

## Released files

- `PROTOCOL_LOCK.json`, `ANALYSIS_LOCK.json`, and `TEST_ANALYSIS_LOCK.json`: frozen design and estimands.
- `ANALYSIS_AMENDMENT_TIE_V1.json`: locked correction and post-hoc diagnostic boundary.
- `protocol/`: deterministic development and held-out prompt/frame manifests. Prompt packs are gzip-compressed without changing their uncompressed content hashes.
- `outputs/`: no-media forced-choice logits and recorded environments for both checkpoints.
- `analysis_test32/` and `analysis_test8/`: corrected record-level readouts, point estimates, bootstrap intervals, paired contrasts, and summaries.
- `analysis/`: analysis, independent audit, publication figure, and plotting scripts.
- `inference/`: the frozen-logit runner and output gate.
- `gates/`: preflight, assay, test-pack, run-status, and final-result audits.
- `MANIFEST.sha256`: checksums for this snapshot.

## Reproduce the analysis

```bash
python results/native_view_fm_provenance_transfer/analysis/analyze.py \
  --pack results/native_view_fm_provenance_transfer/protocol/test \
  --outputs results/native_view_fm_provenance_transfer/outputs/qwen3vl_32b.jsonl.gz \
  --environment results/native_view_fm_provenance_transfer/outputs/qwen3vl_32b_environment.json \
  --model-role primary \
  --output-dir /tmp/native-view-32b

python results/native_view_fm_provenance_transfer/analysis/analyze.py \
  --pack results/native_view_fm_provenance_transfer/protocol/test \
  --outputs results/native_view_fm_provenance_transfer/outputs/qwen3vl_8b.jsonl.gz \
  --environment results/native_view_fm_provenance_transfer/outputs/qwen3vl_8b_environment.json \
  --model-role replication \
  --output-dir /tmp/native-view-8b
```

With the released package and software environment, both commands reproduce all five derived output files byte for byte. Audit the corrected artifacts and regenerate the figure with:

```bash
python results/native_view_fm_provenance_transfer/analysis/audit_results.py
python results/native_view_fm_provenance_transfer/analysis/plot.py \
  --results results/native_view_fm_provenance_transfer \
  --output /tmp/native_view_fm_provenance_transfer.pdf
```

## Media and model boundary

HABIT images and videos and Qwen3-VL weights are not redistributed. The test prompt pack records image-relative paths and hashes; users with authorized HABIT access can reconstruct those images from `protocol/test/frame_manifest.csv` and run `inference/run_logits.py`. The released no-media logits are sufficient to reproduce every reported statistic and figure.
