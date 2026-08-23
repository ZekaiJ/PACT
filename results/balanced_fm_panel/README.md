# Balanced frozen foundation-model panel

This directory releases the no-media evidence used for the eight-checkpoint
PACT boundary study. It contains 2,256 early/event-proximal rows from 1,128
HABIT events, frozen forced-choice logits from four model families, the
prospectively locked analysis, and all reported summary tables. HABIT images
are not redistributed.

The panel tests two separate questions: whether registered topology changes
evidence accounting, and whether model-family labels recover measured error
association. Exact registered-copy invariance holds, while the exact partition
test does not confirm the declared family pairing. The latter remains a
conservative accounting policy rather than a dependence certificate.

## Reproduce the analysis

From the repository root:

```bash
python results/balanced_fm_panel/protocol/analyze_fm_panel.py \
  --cases results/balanced_fm_panel/protocol/FM_PANEL_FULL_CASES.csv \
  --inputs results/balanced_fm_panel/outputs \
  --analysis-lock results/balanced_fm_panel/protocol/FM_PANEL_ANALYSIS_LOCK.json \
  --pilot-gate results/balanced_fm_panel/gates/PILOT_GATE.json \
  --output /tmp/pact-balanced-panel \
  --bootstrap-replicates 2000
```

The released CSVs in `analysis/` were regenerated from these public inputs and
match the frozen private-analysis CSVs byte for byte; see
`analysis/REPRODUCTION_COMPARISON.json`. The post-hoc six-checkpoint
sensitivity is reproduced with:

```bash
python results/balanced_fm_panel/analysis/posthoc_no_llava_partition_sensitivity.py \
  --pairs results/balanced_fm_panel/analysis/CHECKPOINT_PAIR_DEPENDENCE.csv \
  --output-dir /tmp/pact-no-llava
```

## Scope and licensing

Image paths identify records in the upstream HABIT release but do not resolve
to media in this repository. Obtain HABIT from its provider and follow its
license for any media-level reproduction. Model identifiers, immutable
revisions, and upstream license metadata are frozen in
`protocol/FM_PANEL_PROTOCOL_LOCK.json`. No claim of physical safety,
foundation-model release authority, provenance authentication, or statistical
independence is licensed by this panel.
