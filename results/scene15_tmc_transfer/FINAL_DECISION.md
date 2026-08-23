# Scene15--TMC gated extension: final decision

## Decision

**PASS_NATIVE_AND_TOPOLOGY.** The prospective native gate passed, and the pre-authorized frozen-output topology checks passed. No canonical manuscript file was edited by this task.

## Native reproduction

- 30 prespecified runs: mean accuracy 0.667856, population standard deviation 0.022224.
- Published reference: 0.6774; absolute mean deviation: 0.009544; frozen tolerance: 0.0200.
- Learning rate selected from training-only five-fold CV: 0.003.
- All 30 frozen per-view exports contain 897 test instances and pass their recorded SHA-256 checks.

The mean passes the frozen eligibility rule. The observed run-to-run variability is larger than the published standard deviation, so the result supports dataset--emitter eligibility rather than an exact distributional reproduction claim.

## Frozen-output topology checks

- Exact registered copies through multiplicity 8 leave the posterior and evidence budget unchanged.
- False splitting at multiplicity 8 raises the mean evidence budget from 5.937410 to 19.738072 (3.324x).
- The paired budget increase is 13.800662 with a two-stage 95% bootstrap interval [12.796497, 14.781749].
- Merging all three views reduces the mean budget to 0.316149 (0.0532x native); the paired change is -5.621261 [-5.943896, -5.300786].
- All four prespecified directional checks pass.

## Scope

- The TMC repository does not distribute Scene15. The run uses the pinned Scene15 payload in the RCML repository, whose 4,485 instances, 15 labels, and GIST/PHOG/LBP view dimensions match the TMC paper description.
- This establishes a qualified Scene15--TMC dataset--emitter pair and a frozen-output evidence-accounting extension. It is not a new leaderboard claim.
- CUB--TMC remains excluded because no verified processed payload and per-view export are available.

## Integration status

The eligible Scene15--TMC extension is integrated into the canonical manuscript. The manuscript reports the native reproduction gate, exact-copy invariance, assignment-intervention budgets, posterior displacement, accuracy, and descriptive full-support selective ordering. The compiled canonical state has TeX SHA-256 `76CBCF47332FAAB33C64EB5DAD7E9FE177F17C785A74DA449D1E1E2679262022` and PDF SHA-256 `18E48DF78AF97F004A0C0F4E5ADCE92757842BECF5CDDC249EC9EAAB71462F89`.

## Author-review artifacts

- `BENCHMARK_INCLUSION_CANDIDATE.md` provides a 12-pair inclusion/exclusion table.
- `FINAL_GATE.json` and `FINAL_RUNS.csv` record the native gate.
- `phase2_topology/` records the frozen-output interventions.
- `ARTIFACT_MANIFEST.tsv` binds all retained artifacts to SHA-256 hashes.
