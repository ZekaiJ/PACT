# HABIT fixed-image admission evidence

This snapshot contains the decision-level summaries used for the fixed-image
HRC admission study. It includes 720 cases from 60 held-out HABIT episodes and
six tasks. Each episode contributes two temporal windows crossed with one
reference target and five counterfactual targets.

Two related readouts are kept separate:

1. `target_object_admission/` evaluates fixed semantic candidates under target
   identity and an event-proximity state estimate. The combined verifier keeps
   51 of 53 correct candidate admissions and removes 90 of 91 incorrect ones.
2. `task_heldout_delta_admission/` requires a positive event-proximity change
   relative to the paired early observation. It admits 53 cases, all matching
   the reference contract, under leave-one-task-out fitting.
3. `task_heldout_temporal_model/` records the six leave-one-task-out temporal
   models, their task-macro and worst-task AUROC, and the paired 59-of-60
   temporal-ordering sign test used to characterize the event-proximity source.

The study uses public HABIT images but does not redistribute media here. The
target-admission and task-heldout generation scripts require authorized HABIT
data and the feature-generation dependencies documented in the manuscript;
the paired sign test runs directly from the released tables. This snapshot
supports fixed-image admission and sequential evidence claims; it is not a
physical-robot rollout or a human-subject evaluation conducted by PACT.
