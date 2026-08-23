"""Minimal PACT example."""

import numpy as np

from action_admission import SourceEvidence, pact_fuse


def source(name: str, probabilities: list[float], parents: tuple[str, ...]) -> SourceEvidence:
    return SourceEvidence(
        source_id=name,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        quality=0.9,
        conflict=0.0,
        missing=False,
        parents=parents,
    )


def main() -> None:
    opinions = [
        source("language", [0.78, 0.08, 0.06, 0.04, 0.04], ("command",)),
        source("geometry", [0.70, 0.12, 0.08, 0.06, 0.04], ("scene",)),
        source("risk", [0.64, 0.14, 0.10, 0.08, 0.04], ("scene",)),
    ]
    result = pact_fuse(opinions, concentration=8.0)
    print("posterior:", np.round(result.posterior, 4).tolist())
    print("provenance components:", result.group_ids)
    print("selection score:", round(result.selection_score, 4))


if __name__ == "__main__":
    main()
