# Minimal 2x2 attribution

This secondary post-hoc analysis separates two uses of the same registered
topology:

- `F0`: singleton components in PACT fusion;
- `F1`: the registered partition in PACT fusion;
- `V0`: the shared verifier with only the registered-corroboration predicate disabled;
- `V1`: the full shared verifier.

The verifier evaluates the registered partition in both fusion arms. Thus,
`F` changes only operator aggregation, while `V` changes only the
registered-corroboration predicate. The analysis retains all cells and
contrasts regardless of direction; it reports matched contrasts, not causal
effects.

Regenerate the snapshot:

```bash
python analysis/minimal_attribution_2x2.py \
  --output outputs/minimal_attribution_2x2
```

Verify released inputs, checksums, anchors, factorial arithmetic, and route
transitions:

```bash
python analysis/verify_secondary_results.py
```

The primary support is coverage 0.10--0.39 over 31,200 records nested within
48 scene units. A fixed-support sensitivity uses 0.10--0.35 on the complete
21,600-record stratum.

