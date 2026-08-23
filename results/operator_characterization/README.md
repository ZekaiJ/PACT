# Within-component operator characterization

This diagnostic compares PACT's coordinatewise meet with the coordinatewise join while keeping the registered partition, folds, concentrations, eligibility rule, and score definition fixed.

```bash
python results/operator_characterization/run.py
```

Both operators are exact-copy invariant. The join, however, exceeds at least one member's evidence on every complete scene-component record and raises the component budget after same-component insertion on 96.4% of those records. Its ncsAURC is 0.7918 versus 0.6294 for the meet, a paired difference of 0.1623 [0.1535, 0.1709]. Predictive direction is reported as a diagnostic; validity is defined by the registered common-evidence cap.
