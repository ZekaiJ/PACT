# Third-party sources and redistribution boundary

This inventory records the identifiers used to create released derived evidence. It is not a license grant. No third-party model weights or source media are included.

## Foundation-model checkpoints

| Model identifier | Frozen revision | Recorded metadata license | Released here |
|---|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `apache-2.0` | forced-choice logits and derived analyses |
| `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | `apache-2.0` | forced-choice logits and derived analyses |
| `OpenGVLab/InternVL3-2B-hf` | `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` | `other` | forced-choice logits and derived analyses |
| `OpenGVLab/InternVL3-8B-hf` | `259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | `other` | forced-choice logits and derived analyses |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | `74dd0bf867a4cda7950c17663794267c60cf4b40` | `apache-2.0` | forced-choice logits and derived analyses |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | `0d50680527681998e456c7b78950205bedd8a068` | `apache-2.0` | forced-choice logits and derived analyses |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | `apache-2.0` | forced-choice logits and derived analyses |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `482adb537c021c86670beed01cd58990d01e72e4` | `apache-2.0` | forced-choice logits and derived analyses |

The license fields above are frozen model-card metadata, not an independent legal determination. Reusers must consult the upstream terms for each revision.

## Datasets and implementations

| Source | Material included here | Material not included |
|---|---|---|
| HABIT | case identifiers, balanced-panel and native-view forced-choice logits, protocol locks, grouping metadata, derived analyses | images and videos |
| H2O `subject1_ego_v1_1` | aggregate CSV/JSON summaries | RGB-D, pose, and video records |
| HandWritten/Mfeat, PIE, Scene15, CUB, Caltech101, HMDB | derived gates or topology summaries | source feature archives |
| TMC commit `a3272b8746861c76a3461943b5eee51df5b5a8fe` | derived emitter outputs and summaries | upstream implementation |
| RCML commit `c9c5ab41e6fe62a85e5f6441a4dc7b568e1fa421` | derived emitter outputs and summaries | upstream implementation |

All omitted materials remain governed by their providers' terms.

### HABIT attribution and modifications

HABIT is released by Jaehwi Song, Suchae Jeong, Byeongguk Jeon, Sungdong Kim, Minjoon Seo, Hyungmok Son, and Kimin Lee under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). The source dataset is available from [Hugging Face](https://huggingface.co/datasets/configinc/HABIT), with its project page at <https://habit-dataset.github.io/> and paper at <https://arxiv.org/abs/2606.31682>. It was accessed on 2026-08-27.

PACT does not redistribute HABIT images or videos. HABIT-related files in this repository are transformed, no-media research artifacts: forced-choice model logits, case and grouping identifiers, protocol specifications, and statistical summaries created for the PACT analyses. No endorsement by the HABIT authors or their institutions is implied. See the root [`NOTICE.md`](../NOTICE.md) for the concise attribution notice.
