# Fixed-support estimand closure

This snapshot checks the controlled fixed-support estimand, repeated scene-split sensitivity, and balanced-panel comparisons reported in the manuscript.

The `inputs/` directory contains the frozen seven-method out-of-fold predictions and fold thresholds used by this closure analysis. The current controlled runner independently verifies the primary PACT values and the later hierarchy-matched cautious comparator.

```bash
python analysis/verify_p0_estimand_closure.py
```
