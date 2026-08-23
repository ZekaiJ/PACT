# Equal-cardinality topology control

This fixed-protocol analysis holds the 31,200 numerical opinion records, 48 scene clusters, two-component count, fold concentrations, eligibility rule, native non-vacuity score, and verifier-off protocol constant. It changes only which two source roles share a registered component.

| Registration | ncsAURC [0.10, 0.39] | Difference from registered | 95% scene-bootstrap CI |
|---|---:|---:|---:|
| `{L}; {G,R}` (registered) | 0.6294 | reference | -- |
| `{L,G}; {R}` | 0.8282 | +0.1988 | [0.1861, 0.2107] |
| `{L,R}; {G}` | 0.8307 | +0.2013 | [0.1846, 0.2188] |

All 2,000 paired bootstrap differences were positive for both equal-cardinality controls. The controls reject component count alone as an explanation within this benchmark. They do not authenticate ancestry or identify a general causal effect of provenance registration.

The eligible-random-order reference for PACT is 0.6170. Native PACT ncsAURC is 0.6294, an excess of 0.0125 [0.0053, 0.0188], so the native score is weakly anti-informative on this stress mixture. Product, nested, and cautious native scores lie farther above their corresponding random references.

## Reproduce

```bash
python results/equal_cardinality_topology/native_random_reference.py
python results/equal_cardinality_topology/run.py
```

`PROTOCOL_LOCK.json` records the fixed estimand and input hashes. `gate.json` contains the non-directional integrity gate, and `MANIFEST.sha256` binds the released files.
