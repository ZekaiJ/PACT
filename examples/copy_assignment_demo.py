"""Show why an exact copy counts differently under two parent assignments."""

import numpy as np

from action_admission import SourceEvidence, pact_fuse


def source(
    name: str,
    probabilities: list[float],
    parents: tuple[str, ...],
) -> SourceEvidence:
    return SourceEvidence(
        source_id=name,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        quality=0.9,
        conflict=0.0,
        missing=False,
        parents=parents,
    )


def main() -> None:
    language = source("language", [0.78, 0.08, 0.06, 0.04, 0.04], ("command",))
    geometry = source("geometry", [0.70, 0.12, 0.08, 0.06, 0.04], ("scene",))
    risk = source("risk", [0.64, 0.14, 0.10, 0.08, 0.04], ("scene",))

    baseline = pact_fuse([language, geometry, risk], concentration=8.0)

    same_parent_copy = source(
        "geometry_copy",
        geometry.probabilities.tolist(),
        ("scene",),
    )
    retained = pact_fuse(
        [language, geometry, same_parent_copy, risk],
        concentration=8.0,
    )

    false_split_copy = source(
        "geometry_copy",
        geometry.probabilities.tolist(),
        ("new_parent",),
    )
    split = pact_fuse(
        [language, geometry, false_split_copy, risk],
        concentration=8.0,
    )

    print("baseline budget:", round(float(baseline.group_evidence.sum()), 4))
    print("same-parent copy:", round(float(retained.group_evidence.sum()), 4))
    print("false-split copy:", round(float(split.group_evidence.sum()), 4))


if __name__ == "__main__":
    main()
