# Topology-multiplicity stress test

This frozen analysis holds the two genuine registered components fixed and increases only the number of exact geometry copies. It compares the registered-copy arm with a false-split arm and an all-source merge. The primary endpoint is ncsAURC on common support [0.10, 0.39]; lower is better.

```bash
python results/topology_multiplicity/run.py
```

| Geometry multiplicity | False split minus registered PACT ncsAURC | 95% scene-bootstrap CI |
|---:|---:|---:|
| 1 | 0.0243 | [0.0221, 0.0268] |
| 8 | 0.3676 | [0.3534, 0.3801] |
| 32 | 0.4338 | [0.4211, 0.4469] |

The PACT fusion operator is numerically invariant throughout the registered-copy arm. The contrast approaches 0.43 at the upper stress points; this is a multiplicity response, not a scaling law.
