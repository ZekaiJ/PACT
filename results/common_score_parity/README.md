# Closest-comparator common-score parity

This frozen-output analysis applies the same outer-trained logistic correctness score to registered pooling and hierarchy-matched cautious fusion. The score is fitted on the four factorial methods only; both target comparators remain excluded from training. All readouts use common coverage support [0.10, 0.39].

```bash
python analysis/common_score_parity.py --repository . --output results/common_score_parity
```

`PROTOCOL.json` records the frozen inputs and reproduction checks. `table_panel_b_rows.csv` contains the manuscript values, and `sha256.json` binds every released output.
